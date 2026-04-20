"""HotpotQA substrate with an LLM-judge correction signal.

Extends `rag_substrate_hotpot.py` with two extra pieces that make it
the right regime for RuleOPE's correction-fusion term:

  1. Deterministic-logging policy (noop only).  Under this logging,
     classical DR has no information about non-noop actions except
     via the reward regression, which is where RuleOPE's
     correction-fusion term picks up additional signal.

  2. A correction signal `correction \in {0, 1}` produced by an LLM
     judge (or a gold-answer-match proxy when no judge is
     available).  The judge reads the simulated generator output
     (top-3 passages concatenated) and the query, and flags the
     record if the answer is likely wrong.  Under the theorem's
     A5 condition this correction signal is informative about
     V(action) when action != noop.

The LLM-judge path calls an HTTP endpoint (Lambda-hosted
vLLM serving Llama-3-8B-Instruct).  If the endpoint is unreachable
or the judge key is missing, we fall back to a gold-answer-match
proxy: correction = 1 iff the gold-answer string does not appear in
the top-1 retrieved passage.  The proxy is the standard
quality-flag used in RAG benchmarks (RAGBench, FIRST).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

import sys as _sys
for _p in ("/opt/homebrew/lib/python3.11/site-packages",):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from src.logs import LoggedRecord
from src.rag_substrate_hotpot import (
    _atom_features,
    _load_hotpot,
    _reward_for_top3,
    _score_passages,
    _secondary_scores,
    _apply_rule,
)


# -----------------------------------------------------------------------------
# Correction signal
# -----------------------------------------------------------------------------

def _gold_match_correction(sample, top3_titles: list[str]) -> int:
    """Correction fires iff none of the gold-supporting-fact titles
    are in the top-3 retrieved list.  Standard RAG quality proxy."""
    if any(t in sample.gold_titles for t in top3_titles):
        return 0
    return 1


def _llm_judge_correction(
    judge_fn: Callable[[str, list[str]], int],
    sample,
    top3_titles: list[str],
    top3_passages: list[str],
) -> int:
    """LLM-judge correction: ask the judge model "is this answerable
    from the given passages?"  Return 1 if judge says no (correction
    fires), 0 otherwise."""
    try:
        return int(judge_fn(sample.question, top3_passages))
    except Exception:
        # Fall back to gold-match on judge failure.
        return _gold_match_correction(sample, top3_titles)


# -----------------------------------------------------------------------------
# Log generation
# -----------------------------------------------------------------------------

def generate_hotpot_logs_deterministic(
    path: str,
    n_queries: int = 1200,
    seed: int = 0,
    reward_noise: float = 0.05,
    judge_fn: Optional[Callable[[str, list[str]], int]] = None,
) -> list[LoggedRecord]:
    """HotpotQA logs under deterministic (noop) logging with a
    correction signal.  If judge_fn is provided we call it; otherwise
    we use the gold-answer-match proxy.
    """
    samples = _load_hotpot(path, n_queries, seed)
    rng = np.random.default_rng(seed + 7)
    logs = []
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        # Counterfactual rewards per action via replay.
        cf = {}
        for a in ("noop", "filter", "rerank", "abstain"):
            titles = _apply_rule(a, scores, s)
            if a == "abstain":
                cf[a] = 0.5
            else:
                cf[a] = _reward_for_top3(s.gold_titles, titles)
        # Correction signal: evaluated on the *noop* retrieval (since
        # that's what the logging policy produced).
        order = np.argsort(scores)[::-1]
        top3_titles = [s.passages[int(i)][0] for i in order[:3]]
        top3_passages = [s.passages[int(i)][1][:500] for i in order[:3]]
        if judge_fn is not None:
            correction = _llm_judge_correction(judge_fn, s, top3_titles, top3_passages)
        else:
            correction = _gold_match_correction(s, top3_titles)
        logged_reward = float(np.clip(cf["noop"] + rng.normal(0, reward_noise), 0.0, 1.0))
        logs.append(
            LoggedRecord(
                query_id=s.qid,
                ctx=ctx,
                logged_action="noop",
                logged_propensity=1.0,
                logged_reward=logged_reward,
                correction=int(correction),
                cf_rewards=cf,
            )
        )
    return logs


# -----------------------------------------------------------------------------
# Lambda-hosted Llama-3 judge HTTP client
# -----------------------------------------------------------------------------

def make_lambda_judge(
    endpoint: str,
    model: str = "meta-llama/Llama-3-8B-Instruct",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> Callable[[str, list[str]], int]:
    """Return a judge function that calls a vLLM OpenAI-compatible
    endpoint at `endpoint`/v1/chat/completions.  The judge is prompted
    to output a single token in {0, 1}: 0 = answer derivable from
    passages (no correction), 1 = not derivable (correction fires).
    """
    import urllib.request

    system = (
        "You are a strict RAG-quality judge.  Given a question and a short "
        "retrieved-passage context, output exactly one character: '0' if "
        "the question's answer is directly derivable from the passages, "
        "or '1' otherwise.  Output nothing else."
    )

    def judge(question: str, passages: list[str]) -> int:
        context = "\n\n".join(f"[passage {i+1}] {p}" for i, p in enumerate(passages))
        user = f"Question: {question}\n\nPassages:\n{context}\n\nAnswer (0 or 1):"
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": 2,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["choices"][0]["message"]["content"].strip()
        return 1 if raw.startswith("1") else 0

    return judge
