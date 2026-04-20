"""15.J  SCM-based Rule-OPE with Rosenbaum sensitivity bounds."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.logs import load_logs
from src.rag_substrate import ground_truth_value
from src.rule_dsl import load_rules
from src.scm_ruleope import sensitivity_value


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:60]
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")

    out = {"n_rules": len(rules), "by_gamma": {}}
    for gamma in (1.5, 2.0, 3.0):
        widths = []
        covers = 0
        for r in rules:
            res = sensitivity_value(r, logs, gamma=gamma)
            gt = ground_truth_value(r, logs)
            widths.append(res.sensitivity_upper - res.sensitivity_lower)
            if res.sensitivity_lower <= gt <= res.sensitivity_upper:
                covers += 1
        out["by_gamma"][f"{gamma:.1f}"] = {
            "mean_width": float(np.mean(widths)),
            "coverage_of_truth": covers / len(rules),
        }
        print(f"  gamma={gamma}  width={np.mean(widths):.3f}  coverage={covers/len(rules):.3f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_j_scm.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
