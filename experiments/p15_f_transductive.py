"""15.F  Transductive Rule-OPE — per-query intervals."""
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
from src.transductive_ruleope import per_query_intervals


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    est = RuleOPE().fit(logs)
    top_rule = max(rules, key=lambda r: est.value(r, logs).estimate)
    n_half = len(logs) // 2
    train, ev = logs[:n_half], logs[n_half:]

    out = {"rule": top_rule.name, "delta_levels": {}}
    for delta in (0.05, 0.10, 0.20):
        intervals = per_query_intervals(top_rule, train, ev, delta=delta)
        # Per-query ground truth: cf_reward[rule.action] when fires else cf_reward[noop]
        gt_per_q = {
            rec.query_id: rec.cf_rewards.get(top_rule.action if top_rule.fires(rec.ctx) else "noop", rec.logged_reward)
            for rec in ev
        }
        cov = sum(int(it.lower <= gt_per_q[it.query_id] <= it.upper) for it in intervals if it.query_id in gt_per_q)
        widths = [it.upper - it.lower for it in intervals]
        out["delta_levels"][f"{delta:.2f}"] = {
            "empirical_coverage": cov / len(intervals),
            "target_coverage": 1.0 - delta,
            "mean_width": float(np.mean(widths)),
            "fires_frac": float(np.mean([1.0 if it.fires else 0.0 for it in intervals])),
        }
        print(f"  delta={delta:.2f}  cov={cov/len(intervals):.3f} (target {1-delta:.2f})  width={np.mean(widths):.3f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_f_transductive.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
