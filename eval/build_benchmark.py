"""Build the frozen `benchmark_v1` artifacts.

Outputs
-------
  eval/benchmark_v1.jsonl              : public logs (cf_rewards stripped)
  eval/benchmark_v1_with_cf.jsonl      : private, includes cf_rewards for oracle use
  eval/rules_v1.jsonl                  : rule set
  eval/ground_truth_rule_values.json   : V(rho) per rule, computed from cf_rewards
  eval/correction_logs_noise_{0,10,30}.jsonl : three correction-noise regimes

We use the synthetic substrate (see src/rag_substrate.py) rather than full
Llama-3 + BEIR inference because: (1) cf_rewards are required for exact MSE
evaluation and cannot be produced by counterfactual replay in a generic
generator without approximating; (2) this makes the benchmark cheap to
reproduce; (3) we validate the substrate's marginal feature statistics match
published numbers (see `tests/test_substrate_calibration.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.correction_sim import CorrectionConfig, simulate_corrections
from src.logs import LoggedRecord, save_logs
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import Rule, save_rules
from src.rule_enumeration import select_rules_from_logs


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build(
    out_dir: Path,
    n_queries: int,
    seed: int,
    target_rules: int,
    noise_regimes: Sequence[int] = (0, 10, 30),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- logs ----
    cfg = SubstrateConfig(n_queries=n_queries, seed=seed, logging="stochastic")
    logs = generate_logs(cfg)

    # ---- rules ----
    rules = select_rules_from_logs(
        [rec.ctx for rec in logs],
        max_depth=3,
        cap_per_depth=500,
        min_fires=max(20, n_queries // 200),
        target_count=target_rules,
        rng_seed=seed,
    )

    # ---- ground truth ----
    gt = ground_truth_many(rules, logs)

    # ---- correction regimes ----
    for noise in noise_regimes:
        noise_frac = noise / 100.0
        corr = simulate_corrections(
            logs,
            CorrectionConfig(
                base_rate=0.15,
                error_sensitivity=4.0,
                noise_frac=noise_frac,
                seed=seed + 100 + noise,
            ),
        )
        with open(out_dir / f"correction_logs_noise_{noise:02d}.jsonl", "w") as f:
            for rec, c in zip(logs, corr):
                f.write(json.dumps({"query_id": rec.query_id, "correction": c}) + "\n")

    # Write public logs with noise=10 corrections baked in as the default
    default_noise = 10
    default_corr = simulate_corrections(
        logs,
        CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, noise_frac=default_noise / 100.0, seed=seed + 100 + default_noise),
    )
    for rec, c in zip(logs, default_corr):
        rec.correction = c

    save_logs(logs, str(out_dir / "benchmark_v1.jsonl"), include_cf=False)
    save_logs(logs, str(out_dir / "benchmark_v1_with_cf.jsonl"), include_cf=True)
    save_rules(rules, str(out_dir / "rules_v1.jsonl"))
    with open(out_dir / "ground_truth_rule_values.json", "w") as f:
        json.dump(
            {"rules": {r.id: {"name": r.name, "value": gt[r.id]} for r in rules}},
            f,
            indent=2,
        )

    # ---- manifest ----
    manifest = {
        "version": "v1",
        "n_queries": n_queries,
        "n_rules": len(rules),
        "seed": seed,
        "noise_regimes": list(noise_regimes),
        "files": {},
    }
    for name in [
        "benchmark_v1.jsonl",
        "benchmark_v1_with_cf.jsonl",
        "rules_v1.jsonl",
        "ground_truth_rule_values.json",
    ] + [f"correction_logs_noise_{n:02d}.jsonl" for n in noise_regimes]:
        path = out_dir / name
        manifest["files"][name] = {"sha256": _sha(str(path)), "bytes": path.stat().st_size}
    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(out_dir / "FROZEN.md", "w") as f:
        f.write(
            "# Frozen benchmark\n\n"
            "This directory contains the frozen `rule-ope-benchmark-v1` artifacts.\n"
            "Do not modify any file after Phase 1 has been committed.\n"
            f"Rules: {len(rules)}.  Queries: {n_queries}.  Noise regimes: {list(noise_regimes)}.\n"
            "Ground-truth values are computed from the substrate counterfactuals;\n"
            "the `_with_cf` variant of the logs is kept private and never used by\n"
            "estimators, only by the evaluation harness.\n"
        )

    print(f"wrote benchmark to {out_dir} with {len(rules)} rules")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval")
    ap.add_argument("--n_queries", type=int, default=4000)
    ap.add_argument("--target_rules", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(Path(args.out), args.n_queries, args.seed, args.target_rules)
    return 0


if __name__ == "__main__":
    sys.exit(main())
