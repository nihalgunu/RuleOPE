"""15.I  Differentiable rule discovery.

Run gradient descent over soft-rule weights on the frozen benchmark.
Compare the discovered rule against the best enumeration-based rule
(ground truth value) for each action.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diffrule import diff_discover
from src.logs import load_logs
from src.rag_substrate import ground_truth_value
from src.rule_dsl import load_rules


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")

    out = {"by_action": {}}
    for action in ("filter", "rerank"):
        rules_for_a = [r for r in rules if r.action == action]
        gt_best = max(ground_truth_value(r, logs) for r in rules_for_a)
        gt_best_rule = max(rules_for_a, key=lambda r: ground_truth_value(r, logs))
        res = diff_discover(logs, action=action, n_steps=40, lr=0.1, seed=0)
        gt_disc = ground_truth_value(res.best_rule, logs)
        out["by_action"][action] = {
            "discovered_rule": res.best_rule.name,
            "discovered_value": res.best_value,
            "discovered_lcb": res.best_lcb,
            "ground_truth_of_discovered": gt_disc,
            "best_enum_rule": gt_best_rule.name,
            "best_enum_value_gt": gt_best,
            "regret": gt_best - gt_disc,
            "history_first_last": [res.history[0], res.history[-1]],
        }
        print(f"  action={action}  discovered={res.best_rule.name}  gt={gt_disc:.4f}  best={gt_best:.4f}  regret={gt_best-gt_disc:+.4f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_i_diffrule.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
