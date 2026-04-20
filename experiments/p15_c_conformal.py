"""15.C  Conformal Rule-OPE — distribution-free CIs.

Compare conformal vs Wald 95% intervals on the frozen benchmark in
two regimes:
  (i) compositional substrate (well-specified): both should cover.
  (ii) misspecified substrate (heavy-tailed EIF): Wald should
       under-cover; conformal should still cover.

Metric: empirical coverage at 95%, mean half-width.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.conformal_ruleope import conformal_interval, wald_interval
from src.correction_sim import CorrectionConfig, assign_corrections
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rag_substrate_misspec import (
    generate_logs_misspecified,
    ground_truth_many_misspecified,
)
from src.rule_dsl import load_rules


def evaluate(name, logs, rules, gt, delta=0.05):
    n_half = len(logs) // 2
    cal = logs[:n_half]
    ev = logs[n_half:]
    cov_c, cov_w, hw_c, hw_w = 0, 0, [], []
    for r in rules:
        ci_c = conformal_interval(r, cal, ev, delta=delta)
        ci_w = wald_interval(r, ev, delta=delta)
        truth = gt[r.id]
        if ci_c.lower <= truth <= ci_c.upper:
            cov_c += 1
        if ci_w.lower <= truth <= ci_w.upper:
            cov_w += 1
        hw_c.append(ci_c.halfwidth)
        hw_w.append(ci_w.halfwidth)
    return {
        "substrate": name,
        "n_rules": len(rules),
        "delta": delta,
        "conformal_coverage": cov_c / len(rules),
        "wald_coverage": cov_w / len(rules),
        "conformal_halfwidth_mean": float(np.mean(hw_c)),
        "wald_halfwidth_mean": float(np.mean(hw_w)),
    }


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:80]

    cfg_c = SubstrateConfig(n_queries=1500, seed=42, logging="deterministic")
    logs_c = assign_corrections(generate_logs(cfg_c), CorrectionConfig(seed=43))
    gt_c = ground_truth_many(rules, logs_c)

    cfg_m = SubstrateConfig(n_queries=1500, seed=42, logging="deterministic")
    logs_m = assign_corrections(generate_logs_misspecified(cfg_m), CorrectionConfig(seed=43))
    gt_m = ground_truth_many_misspecified(rules, logs_m)

    out = [
        evaluate("compositional", logs_c, rules, gt_c),
        evaluate("misspecified", logs_m, rules, gt_m),
    ]
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_c_conformal.json", "w") as f:
        json.dump(out, f, indent=2)
    for o in out:
        print(f"=== {o['substrate']} ===")
        print(f"  conformal cov={o['conformal_coverage']:.3f}  hw={o['conformal_halfwidth_mean']:.5f}")
        print(f"  wald      cov={o['wald_coverage']:.3f}  hw={o['wald_halfwidth_mean']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
