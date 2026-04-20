"""Build generator prompts for HotpotQA under each retrieval action.

Each query -> 4 prompts (one per action).  Output JSONL keyed by
"<qid>__<action>".  The file is uploaded to the Lambda A10 and fed
through `lambda_generate.py` to obtain generator outputs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.rag_substrate_hotpot import (
    _apply_rule,
    _load_hotpot,
    _score_passages,
    _secondary_scores,
)


SYSTEM = (
    "You are a precise question-answering system.  Answer the user's "
    "question using ONLY information from the provided passages.  Give the "
    "shortest correct answer (one or two words or a yes/no).  If the "
    "passages do not contain the answer, output exactly the string UNKNOWN."
)


def build_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    context = "\n\n".join(
        f"[Passage {i+1}: {title}]\n{body[:400]}" for i, (title, body) in enumerate(passages)
    )
    return f"<s>[INST] {SYSTEM}\n\nQuestion: {question}\n\nPassages:\n{context}\n\nAnswer (short): [/INST]"


def passages_for_action(sample, scores, action: str) -> list[tuple[str, str]]:
    order = np.argsort(scores)[::-1]
    if action == "abstain":
        return []
    if action == "filter":
        order = order[1:]
    elif action == "rerank":
        sec = _secondary_scores(sample, scores)
        order = np.argsort(sec)[::-1]
    return [sample.passages[int(i)] for i in order[:3]]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="eval/hotpot/dev.parquet")
    ap.add_argument("--n_queries", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="eval/hotpot/prompts.jsonl")
    args = ap.parse_args()

    samples = _load_hotpot(args.hotpot, args.n_queries, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_prompts = 0
    with open(args.output, "w") as f:
        for s in samples:
            scores = _score_passages(s)
            for action in ("noop", "filter", "rerank", "abstain"):
                psgs = passages_for_action(s, scores, action)
                if action == "abstain":
                    # Skip generation; we'll treat abstain as a fixed reward.
                    continue
                prompt = build_prompt(s.question, psgs)
                pid = f"{s.qid}__{action}"
                f.write(json.dumps({"id": pid, "prompt": prompt}) + "\n")
                n_prompts += 1
    print(f"wrote {n_prompts} prompts to {args.output}")


if __name__ == "__main__":
    main()
