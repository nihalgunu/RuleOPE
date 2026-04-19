"""Rule-space scaling: compare estimator variance across |R| in {50, 500, 5000}.

This is the scaling experiment from Phase 3 task 3.  We subsample a smaller
rule set from `rules_v1.jsonl` for |R|=50 and generate a larger one from
`enumerate_rules(max_depth=3)` for |R|=5000.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.direct_method import DirectMethod
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.evaluate import all_metrics
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import enumerate_rules, load_rules
from src.rule_enumeration import select_rules_from_logs


def make_rules(target, logs_ctxs, seed=0):
    if target <= 500:
        full = load_rules("eval/rules_v1.jsonl")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(full), size=min(target, len(full)), replace=False)
        return [full[i] for i in idx]
    # For larger spaces, generate fresh.
    return select_rules_from_logs(
        logs_ctxs, max_depth=3, cap_per_depth=6000, min_fires=10,
        target_count=target, rng_seed=seed,
    )


def main() -> int:
    logs = generate_logs(SubstrateConfig(n_queries=3000, seed=55, logging="stochastic"))
    logs = assign_corrections(logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, noise_frac=0.10, seed=555))

    out = {}
    for target in (50, 500, 5000):
        print(f"=== |R| = {target} ===")
        rules = make_rules(target, [rec.ctx for rec in logs], seed=target)
        gt = ground_truth_many(rules, logs)
        row = {}
        for est in [RuleOPE(), DoublyRobust(), DirectMethod()]:
            t0 = time.time()
            if hasattr(est, "fit"):
                est.fit(logs)
            res = est.value_many(rules, logs)
            dt = time.time() - t0
            estimates = {k: v.estimate for k, v in res.items()}
            stderrs = {k: v.stderr for k, v in res.items()}
            m = all_metrics(estimates, stderrs, gt, topk=min(20, len(rules) // 2))
            row[est.name] = {**m, "time_s": dt, "n_rules": len(rules)}
            print(f"  {est.name:>8s}  MSE={m['mse']:.5f}  tau@topk={m['topk_tau']:+.3f}  t={dt:.1f}s")
        out[f"R={target}"] = row

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/scaling.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
