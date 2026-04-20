"""15.K  Temporal-drift Rule-OPE.

Synthetic drift: target distribution upweights queries with q_multihop
> 0.5 by a factor of 2.5 relative to source.  Compare naive RuleOPE
(ignores drift) against weighted RuleOPE.  Ground truth is computed
on a reweighted log set.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.logs import load_logs
from src.rule_dsl import load_rules
from src.estimators.rule_ope import RuleOPE
from src.temporal_ruleope import temporal_value


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")

    est = RuleOPE().fit(logs)
    top_rule = max(rules, key=lambda r: est.value(r, logs).estimate)

    def w(rec):
        return 2.5 if rec.ctx.get("q_multihop", 0.0) > 0.5 else 1.0

    # Ground truth under target = reweighted average of cf rewards.
    weights = np.array([w(rec) for rec in logs])
    if top_rule.action in logs[0].cf_rewards:
        cf_target = np.array([
            rec.cf_rewards[top_rule.action] if top_rule.fires(rec.ctx) else rec.cf_rewards.get("noop", rec.logged_reward)
            for rec in logs
        ])
    else:
        cf_target = np.array([rec.logged_reward for rec in logs])
    gt_target = float(np.sum(weights * cf_target) / np.sum(weights))

    res = temporal_value(top_rule, logs, w)
    out = {
        "rule": top_rule.name,
        "naive_estimate": res.estimate_naive,
        "weighted_estimate": res.estimate_weighted,
        "ground_truth_target": gt_target,
        "naive_abs_err": abs(res.estimate_naive - gt_target),
        "weighted_abs_err": abs(res.estimate_weighted - gt_target),
        "weight_ess": res.weight_ess,
        "naive_to_weighted_err_ratio": abs(res.estimate_naive - gt_target) / max(abs(res.estimate_weighted - gt_target), 1e-9),
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_k_temporal.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"  {k:32s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
