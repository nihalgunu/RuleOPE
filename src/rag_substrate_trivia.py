"""Real-data RAG substrate derived from TriviaQA rc.wikipedia
(Joshi et al. 2017, ACL).

Pipeline:
  1. Load the rc.wikipedia validation parquet (7,993 questions).
  2. For each question, entity_pages provides one or more full
     Wikipedia articles.  We chunk each article into ~300-word
     segments and keep up to 10 chunks per query (padding by
     truncating to the first 10 if the article is long).
  3. BM25 over the chunked passage pool; retrieve top-3.
  4. Reward = 1 if any normalised answer alias appears in the
     concatenated top-3 passages; else 0 (soft-normalised F1
     between the generated answer and the gold aliases when an
     LLM generator is used).
  5. Rules are interventions on the retrieved list (filter,
     rerank, abstain), same as HotpotQA.

The atom vocabulary is shared with HotpotQA and the synthetic
substrate so the same 500-rule pool can be evaluated without
feature retraining.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

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
    answer_aliases: list[str]
    passages: list[tuple[str, str]]   # (title, body) chunks
    gold_phrase: str                  # primary normalised answer


def _chunk(text: str, chunk_size: int = 300) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _load_trivia(path: str, n_queries: int, seed: int) -> list[_Sample]:
    df = pd.read_parquet(path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_queries, len(df)), replace=False)
    out = []
    for i in idx:
        row = df.iloc[int(i)]
        ep = row["entity_pages"]
        titles = list(ep["title"])
        contexts = list(ep["wiki_context"])
        passages = []
        for t, c in zip(titles, contexts):
            chunks = _chunk(str(c), chunk_size=300)
            for ci, chunk in enumerate(chunks):
                passages.append((f"{t}#{ci}", chunk))
                if len(passages) >= 10:
                    break
            if len(passages) >= 10:
                break
        if len(passages) < 4:
            continue
        answers = row["answer"]
        aliases = list(answers["normalized_aliases"])
        primary = str(answers["value"])
        out.append(
            _Sample(
                qid=str(row["question_id"]),
                question=str(row["question"]),
                answer_aliases=aliases,
                passages=passages,
                gold_phrase=primary,
            )
        )
    return out


def _score_passages(sample: _Sample) -> np.ndarray:
    corpus = [_tokenize(title + " " + body) for title, body in sample.passages]
    bm25 = BM25Okapi(corpus)
    return np.array(bm25.get_scores(_tokenize(sample.question)), dtype=np.float64)


def _secondary_scores(sample: _Sample, primary: np.ndarray) -> np.ndarray:
    q_tokens = set(_tokenize(sample.question))
    extra = np.zeros(len(sample.passages))
    for i, (title, _) in enumerate(sample.passages):
        extra[i] = len(q_tokens & set(_tokenize(title.split("#")[0]))) / max(len(q_tokens), 1)
    return primary + 0.5 * extra


def _atom_features(sample: _Sample, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores)[::-1]
    top1_i, top2_i = int(order[0]), int(order[1])
    top1_s_raw = float(scores[top1_i])
    top2_s_raw = float(scores[top2_i])
    s_max = max(scores.max(), 1e-9)
    s_norm = scores / s_max
    top1_s = float(s_norm[top1_i])
    top2_s = float(s_norm[top2_i])
    top3_s = float(s_norm[order[2]])
    mean_s = float(s_norm.mean())
    gap_s = top1_s - top2_s
    n_above_0_5 = int(np.sum(s_norm > 0.5))
    n_above_0_7 = int(np.sum(s_norm > 0.7))
    top1_title = sample.passages[top1_i][0].split("#")[0]
    top2_title = sample.passages[top2_i][0].split("#")[0]
    top1_title_tokens = set(_tokenize(top1_title))
    top2_title_tokens = set(_tokenize(top2_title))
    redundancy = 0.0
    if top1_title_tokens and top2_title_tokens:
        inter = len(top1_title_tokens & top2_title_tokens)
        union = len(top1_title_tokens | top2_title_tokens)
        redundancy = inter / max(union, 1)
    top1_len = len(sample.passages[top1_i][1])
    q_len = len(_tokenize(sample.question))
    has_num = 1.0 if _NUM.search(sample.question) else 0.0
    is_wiki = 1.0
    is_stub = 1.0 if len(top1_title.split()) <= 2 else 0.0
    q_tokens = set(_tokenize(sample.question))
    top1_has_entity = 1.0 if q_tokens & top1_title_tokens else 0.0
    top3_has_entity = 0.0
    first_ent_pos = 10.0
    for k, i_idx in enumerate(order):
        t_tok = set(_tokenize(sample.passages[int(i_idx)][0].split("#")[0]))
        if q_tokens & t_tok:
            if top3_has_entity == 0.0 and k < 3:
                top3_has_entity = 1.0
            first_ent_pos = float(k)
            break
    return {
        "q_len": float(q_len),
        "q_has_person": 0.0, "q_has_place": 0.0, "q_has_org": 0.0,
        "q_has_time": 0.0, "q_has_num": has_num,
        "q_multihop": 0.0,
        "q_ppl": 15.0,
        "top1_score": top1_s, "top2_score": top2_s, "top3_score": top3_s,
        "mean_score": mean_s, "score_gap": gap_s,
        "top1_len": float(top1_len),
        "top1_src_wiki": is_wiki, "top1_src_stub": is_stub,
        "top1_src_blog": 0.0, "top1_src_forum": 0.0,
        "n_above_0_5": float(n_above_0_5),
        "n_above_0_7": float(n_above_0_7),
        "redundancy": float(redundancy),
        "ent_missing_top1": 1.0 - top1_has_entity,
        "ent_missing_top3": 1.0 - top3_has_entity,
        "first_ent_pos": float(first_ent_pos),
        "gen_conf": float(top1_s),
        "gen_len": 50.0,
        "src_low_trust_frac": 0.0,
        "src_entropy": 0.3,
    }


def _alias_match(passages: list[str], aliases: list[str]) -> float:
    """1 if any alias appears in the concatenated passages, else 0."""
    concat = " ".join(p.lower() for p in passages)
    for a in aliases:
        if a and a.lower() in concat:
            return 1.0
    return 0.0


def _apply_rule(action: str, scores: np.ndarray, sample: _Sample) -> tuple[list[str], list[str]]:
    order = np.argsort(scores)[::-1]
    if action == "abstain":
        return [], []
    if action == "filter":
        order = order[1:]
    elif action == "rerank":
        sec = _secondary_scores(sample, scores)
        order = np.argsort(sec)[::-1]
    titles = [sample.passages[int(i)][0] for i in order[:3]]
    bodies = [sample.passages[int(i)][1] for i in order[:3]]
    return titles, bodies


def generate_trivia_logs(
    path: str,
    n_queries: int = 600,
    seed: int = 0,
    reward_noise: float = 0.02,
) -> list[LoggedRecord]:
    samples = _load_trivia(path, n_queries, seed)
    rng = np.random.default_rng(seed + 7)
    logs = []
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        cf = {}
        for a in ("noop", "filter", "rerank", "abstain"):
            _, bodies = _apply_rule(a, scores, s)
            if a == "abstain":
                cf[a] = 0.5
            else:
                cf[a] = _alias_match(bodies, s.answer_aliases)
        logged_reward = float(np.clip(cf["noop"] + rng.normal(0, reward_noise), 0.0, 1.0))
        logs.append(
            LoggedRecord(
                query_id=s.qid, ctx=ctx,
                logged_action="noop", logged_propensity=1.0,
                logged_reward=logged_reward, correction=0,
                cf_rewards=cf,
            )
        )
    return logs


def ground_truth_rule_value(rule, logs: list[LoggedRecord]) -> float:
    vals = []
    for rec in logs:
        if rule.fires(rec.ctx):
            vals.append(rec.cf_rewards[rule.action])
        else:
            vals.append(rec.cf_rewards["noop"])
    return float(np.mean(vals))
