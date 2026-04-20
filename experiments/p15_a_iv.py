"""15.A  IV-RuleOPE — corrections-as-instrument validation.

On the compositional substrate (where DR is unbiased), test whether
the IV bridge estimator stays inside [V_L, V_U].  On the misspecified
substrate (where A5 is violated), compare bias against DR/RuleOPE.

Success criterion: IV achieves smaller |bias| than DR on the
misspecified substrate; comparable on the compositional substrate.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.iv_ruleope import iv_value
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rag_substrate_misspec import (
    generate_logs_misspecified,
    ground_truth_many_misspecified,
)
from src.rule_dsl import load_rules


def _bias(estimates, truths):
    return float(np.mean([estimates[k] - truths[k] for k in truths]))


def _abs_bias(estimates, truths):
    return float(np.mean([abs(estimates[k] - truths[k]) for k in truths]))


def run_substrate(name: str, logs, rules, gt) -> dict:
    dr = DoublyRobust().fit(logs)
    rope = RuleOPE().fit(logs)
    dr_est = {r.id: dr.value(r, logs).estimate for r in rules}
    rope_est = {r.id: rope.value(r, logs).estimate for r in rules}
    iv_est = {r.id: iv_value(r, logs).estimate for r in rules}
    return {
        "substrate": name,
        "n_rules": len(rules),
        "DR_abs_bias": _abs_bias(dr_est, gt),
        "RuleOPE_abs_bias": _abs_bias(rope_est, gt),
        "IV_abs_bias": _abs_bias(iv_est, gt),
        "DR_signed_bias": _bias(dr_est, gt),
        "RuleOPE_signed_bias": _bias(rope_est, gt),
        "IV_signed_bias": _bias(iv_est, gt),
    }


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:60]

    # Compositional substrate
    cfg_c = SubstrateConfig(n_queries=1500, seed=42, logging="deterministic")
    logs_c = assign_corrections(generate_logs(cfg_c), CorrectionConfig(seed=43))
    gt_c = ground_truth_many(rules, logs_c)

    out_c = run_substrate("compositional", logs_c, rules, gt_c)

    # Misspecified substrate
    cfg_m = SubstrateConfig(n_queries=1500, seed=42, logging="deterministic")
    logs_m = assign_corrections(generate_logs_misspecified(cfg_m), CorrectionConfig(seed=43))
    gt_m = ground_truth_many_misspecified(rules, logs_m)
    out_m = run_substrate("misspecified", logs_m, rules, gt_m)

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_a_iv.json", "w") as f:
        json.dump([out_c, out_m], f, indent=2)
    for o in (out_c, out_m):
        print(f"=== {o['substrate']} ===")
        for k in ("DR_abs_bias", "RuleOPE_abs_bias", "IV_abs_bias"):
            print(f"  {k:25s} = {o[k]:.5f}")
        for k in ("DR_signed_bias", "RuleOPE_signed_bias", "IV_signed_bias"):
            print(f"  {k:25s} = {o[k]:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
