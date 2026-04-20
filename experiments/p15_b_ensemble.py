"""15.B  Rule-ensemble OPE — composed-policy evaluation.

Take the top-k RuleOPE rules from the case study; evaluate the
composed policy directly and compare against (a) sum of individual
values (the naive expectation that's typically wrong) and (b) max
over individual values.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.estimators.rule_ope import RuleOPE
from src.logs import load_logs
from src.rag_substrate import ground_truth_value
from src.rule_dsl import load_rules
from src.rule_ensemble import composed_action, ensemble_value


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")

    # Use case-study top-k as the candidate ensemble
    est = RuleOPE().fit(logs)
    individual = sorted(rules, key=lambda r: -est.value(r, logs).estimate)[:5]

    # Ground truth of the composed policy: replay each record under
    # the composed action and average cf_rewards.
    gt_pi = 0.0
    n = 0
    for rec in logs:
        a = composed_action(individual, rec)
        if a in rec.cf_rewards:
            gt_pi += rec.cf_rewards[a]
        else:
            gt_pi += rec.cf_rewards.get("noop", rec.logged_reward)
        n += 1
    gt_pi /= max(n, 1)

    res = ensemble_value(individual, logs)
    individual_gts = [ground_truth_value(r, logs) for r in individual]

    out = {
        "k": len(individual),
        "rules": [r.name for r in individual],
        "ensemble_estimate": res.estimate,
        "ensemble_stderr": res.stderr,
        "sum_individual": res.sum_baseline,
        "max_individual": res.max_baseline,
        "ground_truth_ensemble": gt_pi,
        "ground_truth_individual_max": max(individual_gts),
        "interaction_gap_pct": 100.0 * (res.estimate - max(individual_gts)) / max(individual_gts),
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_b_ensemble.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"  {k:30s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
