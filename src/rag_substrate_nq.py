"""NQ benchmark substrate, mirrors src/rag_substrate_hotpot.py.

NQ data sourced from BeIR/nq queries + qrels (gold passages) plus
nq_open answers (matched by question text). 1500 dev queries with
~1.2 gold passages and 8-9 random distractors padded to 10 candidates
each, written to eval/nq/dev.parquet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    answer_aliases: list[str]
    passages: list[tuple[str, str]]
    gold_titles: set[str]


def _load_nq(path: str, n_queries: int, seed: int) -> list[_Sample]:
    df = pd.read_parquet(path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_queries, len(df)), replace=False)
    out = []
    for i in idx:
        row = df.iloc[int(i)]
        passages = [tuple(p) for p in row["passages"]]
        if len(passages) < 4:
            continue
        out.append(_Sample(
            qid=str(row["qid"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            answer_aliases=list(row["answer_aliases"]),
            passages=passages,
            gold_titles=set(row["gold_titles"]),
        ))
    return out


def _score_passages(sample: _Sample) -> np.ndarray:
    corpus = [_tokenize(t + " " + b) for t, b in sample.passages]
    bm25 = BM25Okapi(corpus)
    return np.array(bm25.get_scores(_tokenize(sample.question)), dtype=np.float64)


def _secondary_scores(sample: _Sample, primary: np.ndarray) -> np.ndarray:
    q_tokens = set(_tokenize(sample.question))
    extra = np.zeros(len(sample.passages))
    for i, (title, _) in enumerate(sample.passages):
        extra[i] = len(q_tokens & set(_tokenize(title))) / max(len(q_tokens), 1)
    return primary + 0.5 * extra


def _atom_features(sample: _Sample, scores: np.ndarray) -> dict[str, float]:
    """Same atom vocabulary as hotpot/trivia/musique substrates."""
    order = np.argsort(scores)[::-1]
    top1_i, top2_i = int(order[0]), int(order[1])
    s_max = max(scores.max(), 1e-9)
    s_norm = scores / s_max
    top1_s = float(s_norm[top1_i]); top2_s = float(s_norm[top2_i])
    top3_s = float(s_norm[order[2]])
    mean_s = float(s_norm.mean()); gap_s = top1_s - top2_s
    n_above_0_5 = int(np.sum(s_norm > 0.5))
    n_above_0_7 = int(np.sum(s_norm > 0.7))
    top1_title_tokens = set(_tokenize(sample.passages[top1_i][0]))
    top2_title_tokens = set(_tokenize(sample.passages[top2_i][0]))
    redundancy = 0.0
    if top1_title_tokens and top2_title_tokens:
        inter = len(top1_title_tokens & top2_title_tokens)
        union = len(top1_title_tokens | top2_title_tokens)
        redundancy = inter / max(union, 1)
    top1_len = len(sample.passages[top1_i][1])
    q_len = len(_tokenize(sample.question))
    has_num = 1.0 if _NUM.search(sample.question) else 0.0

    top1_title = sample.passages[top1_i][0]
    is_wiki = 1.0 if _NUM.search(top1_title) or " " in top1_title else 0.0
    is_stub = 1.0 if len(top1_title.split()) <= 2 and not _NUM.search(top1_title) else 0.0

    q_tokens = set(_tokenize(sample.question))
    top1_has_entity = 1.0 if q_tokens & top1_title_tokens else 0.0
    top3_has_entity = 0.0
    for i_idx in order[:3]:
        t_tok = set(_tokenize(sample.passages[int(i_idx)][0]))
        if q_tokens & t_tok:
            top3_has_entity = 1.0
            break

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
        "q_multihop": 0.0,            # NQ is single-hop
        "q_ppl": 20.0,
        "top1_score": top1_s,
        "top2_score": top2_s,
        "top3_score": top3_s,
        "mean_score": mean_s,
        "score_gap": gap_s,
        "top1_len": float(top1_len),
        "top1_src_wiki": is_wiki,
        "top1_src_stub": is_stub,
        "top1_src_blog": 0.0,
        "top1_src_forum": 0.0,
        "n_above_0_5": float(n_above_0_5),
        "n_above_0_7": float(n_above_0_7),
        "redundancy": float(redundancy),
        "ent_missing_top1": 1.0 - top1_has_entity,
        "ent_missing_top3": 1.0 - top3_has_entity,
        "first_ent_pos": float(first_ent_pos),
        "gen_conf": float(top1_s),
        "gen_len": 50.0,
        "src_low_trust_frac": 0.0,
        "src_entropy": 0.5 if is_wiki else 1.0,
    }
