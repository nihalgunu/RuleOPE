"""15.E  Active-query Rule-OPE.

Pool of pool_size labelled records + candidate_size unlabelled.
We can pay to label `budget` queries.  Compare random / leverage /
active (EIF score).  Metric: bootstrap variance of V_hat for the
top RuleOPE rule.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.active_ruleope import active_compare
from src.estimators.rule_ope import RuleOPE
from src.logs import load_logs
from src.rule_dsl import load_rules


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    est = RuleOPE().fit(logs)
    top_rule = max(rules, key=lambda r: est.value(r, logs).estimate)

    pool = logs[:1500]
    candidates = logs[1500:3000]
    out = {"rule": top_rule.name, "by_budget": {}}
    for budget in (50, 150, 300):
        res = active_compare(top_rule, pool, candidates, budget=budget)
        out["by_budget"][str(budget)] = {
            "var_random": res.var_random,
            "var_active": res.var_active,
            "var_leverage": res.var_leverage,
            "reduction_active_pct": res.reduction_active_pct,
            "reduction_leverage_pct": res.reduction_leverage_pct,
        }
        print(f"  budget={budget:4d}  active vs random: {res.reduction_active_pct:+.1f}%   leverage: {res.reduction_leverage_pct:+.1f}%")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_e_active.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
