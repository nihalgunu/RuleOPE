"""Case study: the top-20 rules by Rule-OPE on the frozen benchmark.

We print each rule, its ground-truth value, its estimate, and the fraction of
logs it fires on.  This supports the qualitative analysis in the paper.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.estimators.rule_ope import RuleOPE
from src.logs import load_logs
from src.rule_dsl import load_rules


def main() -> int:
    with open("eval/ground_truth_rule_values.json") as f:
        gt = {k: v["value"] for k, v in json.load(f)["rules"].items()}
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")

    est = RuleOPE()
    est.fit(logs)
    results = est.value_many(rules, logs)

    rows = []
    for r in rules:
        est_v = results[r.id].estimate
        gt_v = gt[r.id]
        fires = sum(1 for rec in logs if r.fires(rec.ctx))
        rows.append({
            "rule": r.name,
            "action": r.action,
            "depth": r.depth(),
            "fires_frac": fires / len(logs),
            "rope_estimate": est_v,
            "ground_truth": gt_v,
            "abs_err": abs(est_v - gt_v),
        })
    rows.sort(key=lambda x: -x["rope_estimate"])
    top20 = rows[:20]
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/case_study_top20.json", "w") as f:
        json.dump(top20, f, indent=2)
    print(f"{'rule':<80s}  {'fires':>6s}  {'est':>6s}  {'gt':>6s}  {'|err|':>6s}")
    for r in top20:
        print(
            f"{r['rule']:<80s}  "
            f"{r['fires_frac']:6.2%}  "
            f"{r['rope_estimate']:6.3f}  "
            f"{r['ground_truth']:6.3f}  "
            f"{r['abs_err']:6.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
