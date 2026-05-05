"""Unified HotpotQA prompt builder supporting any chat template.

One script in place of the per-template scripts (build_hotpot_qwen_prompts.py,
build_hotpot_phi35_prompts.py, etc.). Used for the §7C.13 multi-LLM expansion.

Usage:
    python3 experiments/build_hotpot_prompts_multi.py --template olmo
    python3 experiments/build_hotpot_prompts_multi.py --template granite
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.rag_substrate_hotpot import (
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


def build_prompt(question, passages, template):
    context = "\n\n".join(
        f"[Passage {i+1}: {title}]\n{body[:400]}"
        for i, (title, body) in enumerate(passages)
    )
    user = f"Question: {question}\n\nPassages:\n{context}\n\nAnswer (short):"
    if template == "mistral":
        return f"<s>[INST] {SYSTEM}\n\n{user} [/INST]"
    if template == "qwen":
        return (
            "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            + "<|im_start|>user\n" + user + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
    if template == "phi35":
        return f"<|system|>\n{SYSTEM}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
    if template == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if template == "olmo":
        return f"<|endoftext|><|user|>\n{SYSTEM}\n\n{user}\n<|assistant|>\n"
    if template == "granite":
        return (
            f"<|start_of_role|>system<|end_of_role|>{SYSTEM}<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{user}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>"
        )
    raise ValueError(f"unknown template: {template}")


def passages_for_action(sample, scores, action):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="eval/hotpot/dev.parquet")
    ap.add_argument("--n_queries", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template", default="mistral",
                    choices=["mistral", "qwen", "phi35", "llama3", "olmo", "granite"])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"eval/hotpot/prompts_{args.template}_1500.jsonl"

    samples = _load_hotpot(args.hotpot, args.n_queries, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.output, "w") as f:
        for s in samples:
            scores = _score_passages(s)
            for action in ("noop", "filter", "rerank"):
                psgs = passages_for_action(s, scores, action)
                prompt = build_prompt(s.question, psgs, args.template)
                pid = f"{s.qid}__{action}"
                f.write(json.dumps({"id": pid, "prompt": prompt}) + "\n")
                n += 1
    print(f"wrote {n} prompts to {args.output}")


if __name__ == "__main__":
    main()
