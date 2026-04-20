"""15.N  Bandit-of-rules: warm-start UCB vs cold UCB.

Use offline-RuleOPE estimates as priors for a UCB bandit; simulate an
online environment that pays rule-specific Bernoulli rewards drawn
from the rule's true V(rho).  Compare cumulative regret.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.bandit_deployment import cumulative_regret, warm_start_ucb
from src.estimators.rule_ope import RuleOPE
from src.logs import load_logs
from src.rag_substrate import ground_truth_value
from src.rule_dsl import load_rules


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    est = RuleOPE().fit(logs)
    candidates = sorted(rules, key=lambda r: -est.value(r, logs).estimate)[:20]

    # True per-rule values (Bernoulli arm means)
    true_v = {r.id: ground_truth_value(r, logs) for r in candidates}
    best = max(true_v.values())

    rng = np.random.default_rng(0)

    def reward_fn(rule, t):
        return float(rng.binomial(1, true_v[rule.id]))

    horizon = 1500
    rewards_warm = warm_start_ucb(candidates, logs, reward_fn, horizon, cold_start=False, seed=1)
    rewards_cold = warm_start_ucb(candidates, logs, reward_fn, horizon, cold_start=True, seed=1)
    cum_warm = cumulative_regret(rewards_warm, best)
    cum_cold = cumulative_regret(rewards_cold, best)

    out = {
        "horizon": horizon,
        "best_value": best,
        "warm_final_regret": cum_warm[-1],
        "cold_final_regret": cum_cold[-1],
        "warm_to_cold_ratio": cum_warm[-1] / max(cum_cold[-1], 1e-9),
        "warm_advantage_first_100": float(cum_cold[100] - cum_warm[100]),
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_n_bandit.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"  {k:30s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
