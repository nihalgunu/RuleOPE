"""15.D  Adversarial DRO Rule-OPE.

Compare the standard plug-in to the worst-case DRO lower bound for
several KL radii eta.  Report:
  - DRO LCB coverage of true V(rho) at level (1 - eta).
  - Tightness vs Wald LCB.

Success: DRO covers truth at advertised level uniformly across rules,
including those where Wald under-covers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.rule_ope import RuleOPE
from src.minimax_ruleope import minimax_value
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rag_substrate_misspec import (
    generate_logs_misspecified,
    ground_truth_many_misspecified,
)
from src.rule_dsl import load_rules


def run_substrate(name, logs, rules, gt):
    out = {"substrate": name}
    for eta in (0.01, 0.05, 0.1, 0.25):
        widths, covers = [], 0
        for r in rules:
            res = minimax_value(r, logs, eta=eta)
            widths.append(res.estimate - res.lower)
            if res.lower <= gt[r.id]:
                covers += 1
        out[f"eta_{eta}_lower_coverage"] = covers / len(rules)
        out[f"eta_{eta}_width_mean"] = float(np.mean(widths))
    return out


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:60]

    cfg_c = SubstrateConfig(n_queries=1500, seed=42, logging="deterministic")
    logs_c = assign_corrections(generate_logs(cfg_c), CorrectionConfig(seed=43))
    gt_c = ground_truth_many(rules, logs_c)

    cfg_m = SubstrateConfig(n_queries=1500, seed=42, logging="deterministic")
    logs_m = assign_corrections(generate_logs_misspecified(cfg_m), CorrectionConfig(seed=43))
    gt_m = ground_truth_many_misspecified(rules, logs_m)

    out = [run_substrate("compositional", logs_c, rules, gt_c),
           run_substrate("misspecified", logs_m, rules, gt_m)]
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_d_minimax.json", "w") as f:
        json.dump(out, f, indent=2)
    for o in out:
        print(f"=== {o['substrate']} ===")
        for k, v in o.items():
            if k == "substrate":
                continue
            print(f"  {k:32s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
