"""Batch-generator script to run on the Lambda A10 instance.

Reads prompts from stdin JSONL (one per line with "id" and "prompt"),
writes outputs to stdout JSONL ("id", "text").  Uses transformers
directly (no vLLM).

Usage (on the Lambda box):
    python3 lambda_generate.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --max_new_tokens 64 --input prompts.jsonl --output outputs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"loaded; total params = {sum(p.numel() for p in model.parameters())/1e9:.2f}B", file=sys.stderr, flush=True)

    with open(args.input) as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"n_prompts = {len(items)}", file=sys.stderr, flush=True)

    t0 = time.time()
    with open(args.output, "w") as fout:
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            prompts = [b["prompt"] for b in batch]
            inputs = tok(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=3500,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0.0,
                    temperature=max(args.temperature, 1e-6),
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, inputs["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for b, t in zip(batch, texts):
                fout.write(json.dumps({"id": b["id"], "text": t.strip()}) + "\n")
            if (i // args.batch_size) % 10 == 0:
                dt = time.time() - t0
                rate = (i + len(batch)) / max(dt, 1e-6)
                eta = (len(items) - (i + len(batch))) / max(rate, 1e-6)
                print(f"  {i + len(batch)}/{len(items)} done ({rate:.1f}/s, ETA {eta:.0f}s)", file=sys.stderr, flush=True)
    print(f"wrote {args.output} in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
