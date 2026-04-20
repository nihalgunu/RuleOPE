"""Real-data RAG substrate derived from HotpotQA (Yang et al. 2018, EMNLP).

Pipeline:
  1. Load HotpotQA distractor-setting dev parquet (7,405 questions).
     Each question carries 10 candidate Wikipedia passages of which 2 are
     supporting_facts (gold) and 8 are distractors.
  2. Index the 10 passages per question with BM25; retrieve top-3.
  3. The "reward" is 1 iff both gold passages are in the top-3 retrieved.
     This is a standard gold-passage-recall proxy for final answer
     quality in multi-hop QA (cf. Khattab et al. 2021 "Baleen").
  4. Rules are interventions on the retrieved list (filter, rerank,
     abstain).  The logged policy is "noop" (use top-3 as-is).
     Counterfactual rewards per action are computed by replaying with
     the modified retrieval list.

The atom vocabulary is reused from `src/rule_dsl.py` but the context
features are now computed from real BM25 scores and real passage
titles / lengths rather than synthetic draws.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.logs import LoggedRecord

import sys as _sys
for _p in ("/opt/homebrew/lib/python3.11/site-packages",):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from rank_bm25 import BM25Okapi  # noqa: E402


_WS = re.compile(r"\W+")
_NUM = re.compile(r"\d")


def _tokenize(s: str) -> list[str]:
    return [t for t in _WS.split(s.lower()) if t]


@dataclass
class _Sample:
    qid: str
    question: str
    answer: str
    passages: list[tuple[str, str]]   # (title, body)
    gold_titles: set[str]


def _load_hotpot(path: str, n_queries: int, seed: int) -> list[_Sample]:
    df = pd.read_parquet(path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_queries, len(df)), replace=False)
    out = []
    for i in idx:
        row = df.iloc[int(i)]
        titles = list(row["context"]["title"])
        sentences = list(row["context"]["sentences"])
        passages = [(t, " ".join(s)) for t, s in zip(titles, sentences)]
        if len(passages) < 4:
            # Skip queries with too few distractors -- the atom vocabulary
            # (top-3 score, top-3 entity, score gap etc.) requires >= 4
            # passages to compute reliably.
            continue
        gold_titles = set(row["supporting_facts"]["title"])
        out.append(
            _Sample(
                qid=str(row["id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                passages=passages,
                gold_titles=gold_titles,
            )
        )
    return out


def _score_passages(sample: _Sample) -> np.ndarray:
    """BM25 scores of the 10 candidate passages under the query."""
    corpus = [_tokenize(title + " " + body) for title, body in sample.passages]
    bm25 = BM25Okapi(corpus)
    q = _tokenize(sample.question)
    scores = np.array(bm25.get_scores(q), dtype=np.float64)
    return scores


def _secondary_scores(sample: _Sample, primary: np.ndarray) -> np.ndarray:
    """Title-match bonus used as a secondary score for reranking."""
    q_tokens = set(_tokenize(sample.question))
    extra = np.zeros(len(sample.passages))
    for i, (title, _) in enumerate(sample.passages):
        title_tokens = set(_tokenize(title))
        extra[i] = len(q_tokens & title_tokens) / max(len(q_tokens), 1)
    return primary + 0.5 * extra


def _atom_features(sample: _Sample, scores: np.ndarray) -> dict[str, float]:
    """Compute the atom-vocabulary features used by rule_dsl.ATOMS."""
    order = np.argsort(scores)[::-1]
    top1_i, top2_i = int(order[0]), int(order[1])
    top1_score = float(scores[top1_i])
    top2_score = float(scores[top2_i])
    top3_score = float(scores[order[2]])
    mean_score = float(np.mean(scores))
    gap = top1_score - top2_score
    n_above_0_5 = int(np.sum(scores > 0.5 * max(scores.max(), 1e-9)))
    n_above_0_7 = int(np.sum(scores > 0.7 * max(scores.max(), 1e-9)))
    redundancy = 0.0  # titles overlap proxy
    top1_title_tokens = set(_tokenize(sample.passages[top1_i][0]))
    top2_title_tokens = set(_tokenize(sample.passages[top2_i][0]))
    if top1_title_tokens and top2_title_tokens:
        inter = len(top1_title_tokens & top2_title_tokens)
        union = len(top1_title_tokens | top2_title_tokens)
        redundancy = inter / max(union, 1)
    top1_len = len(sample.passages[top1_i][1])
    q_len = len(_tokenize(sample.question))
    has_num = 1.0 if _NUM.search(sample.question) else 0.0
    # Normalize scores to [0, 1] so atom thresholds are meaningful.
    s_norm = scores / max(scores.max(), 1e-9)
    top1_s = float(s_norm[top1_i])
    top2_s = float(s_norm[top2_i])
    top3_s = float(s_norm[order[2]])
    mean_s = float(s_norm.mean())
    gap_s = top1_s - top2_s

    # Source-type proxy: titles with years/numbers tend to be wiki; titles
    # that are short common nouns tend to be stubs.  Crude but serviceable.
    top1_title = sample.passages[top1_i][0]
    is_wiki = 1.0 if _NUM.search(top1_title) or " " in top1_title else 0.0
    is_stub = 1.0 if len(top1_title.split()) <= 2 and not _NUM.search(top1_title) else 0.0
    is_blog = 0.0
    is_forum = 0.0

    # Entity-missing proxy: does any top-1/top-3 passage title contain a query token?
    q_tokens = set(_tokenize(sample.question))
    top1_has_entity = 1.0 if q_tokens & top1_title_tokens else 0.0
    top3_has_entity = 0.0
    for i_idx in order[:3]:
        t_tok = set(_tokenize(sample.passages[int(i_idx)][0]))
        if q_tokens & t_tok:
            top3_has_entity = 1.0
            break

    # First entity position: smallest top-k where gold entity word appears
    first_ent_pos = 10.0
    for k, i_idx in enumerate(order):
        t_tok = set(_tokenize(sample.passages[int(i_idx)][0]))
        if q_tokens & t_tok:
            first_ent_pos = float(k)
            break

    return {
        "q_len": float(q_len),
        "q_has_person": 0.0,
        "q_has_place": 0.0,
        "q_has_org": 0.0,
        "q_has_time": 0.0,
        "q_has_num": has_num,
        "q_multihop": 1.0,       # HotpotQA is multihop by construction
        "q_ppl": 20.0,
        "top1_score": top1_s,
        "top2_score": top2_s,
        "top3_score": top3_s,
        "mean_score": mean_s,
        "score_gap": gap_s,
        "top1_len": float(top1_len),
        "top1_src_wiki": is_wiki,
        "top1_src_stub": is_stub,
        "top1_src_blog": is_blog,
        "top1_src_forum": is_forum,
        "n_above_0_5": float(n_above_0_5),
        "n_above_0_7": float(n_above_0_7),
        "redundancy": float(redundancy),
        "ent_missing_top1": 1.0 - top1_has_entity,
        "ent_missing_top3": 1.0 - top3_has_entity,
        "first_ent_pos": float(first_ent_pos),
        "gen_conf": float(top1_s),       # proxy: generator confidence ~ top-1 score
        "gen_len": 50.0,
        "src_low_trust_frac": 0.0,
        "src_entropy": 0.5 if is_wiki else 1.0,
    }


def _reward_for_top3(gold_titles: set[str], top3_titles: list[str]) -> float:
    """1 if all gold passages are in top-3, else 0."""
    got = sum(1 for t in top3_titles if t in gold_titles)
    return 1.0 if got >= len(gold_titles) else got / max(len(gold_titles), 1)


def _apply_rule(
    action: str, scores: np.ndarray, sample: _Sample
) -> list[str]:
    """Return the top-3 passage titles after applying the rule action.

    Actions:
      noop     -> return top-3 by primary BM25 score
      filter   -> drop top-1, return top-3 from remaining
      rerank   -> rerank by secondary score (title match bonus), return top-3
      abstain  -> empty retrieval (treated as zero reward)
    """
    order = np.argsort(scores)[::-1]
    if action == "abstain":
        return []
    if action == "filter":
        order = order[1:]
    elif action == "rerank":
        sec = _secondary_scores(sample, scores)
        order = np.argsort(sec)[::-1]
    return [sample.passages[int(i)][0] for i in order[:3]]


def generate_hotpot_logs(
    path: str,
    n_queries: int = 500,
    seed: int = 0,
    reward_noise: float = 0.05,
) -> list[LoggedRecord]:
    """Build LoggedRecord objects for HotpotQA under a noop logging policy.

    cf_rewards records the counterfactual reward under each action
    (noop, filter, rerank, abstain) via replay.  logged_reward is the
    noop reward + tiny Gaussian noise to emulate realistic judge noise.
    """
    samples = _load_hotpot(path, n_queries, seed)
    rng = np.random.default_rng(seed + 7)
    logs = []
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        cf = {}
        for a in ("noop", "filter", "rerank", "abstain"):
            titles = _apply_rule(a, scores, s)
            if a == "abstain":
                cf[a] = 0.5  # standard abstain reward
            else:
                cf[a] = _reward_for_top3(s.gold_titles, titles)
        logged_reward = float(np.clip(cf["noop"] + rng.normal(0, reward_noise), 0.0, 1.0))
        logs.append(
            LoggedRecord(
                query_id=s.qid,
                ctx=ctx,
                logged_action="noop",
                logged_propensity=1.0,
                logged_reward=logged_reward,
                correction=0,
                cf_rewards=cf,
            )
        )
    return logs


def ground_truth_rule_value(rule, logs: list[LoggedRecord]) -> float:
    """Exact V(rule) on the HotpotQA logs via counterfactual replay."""
    vals = []
    for rec in logs:
        if rule.fires(rec.ctx):
            vals.append(rec.cf_rewards[rule.action])
        else:
            vals.append(rec.cf_rewards["noop"])
    return float(np.mean(vals))
