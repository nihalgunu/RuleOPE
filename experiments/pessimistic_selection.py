"""Pessimistic rule selection: regret of compositional LCB vs union-bound LCB.

Compares three rule-selection rules:
    * argmax of point estimate (naive)
    * argmax of standard LCB (exponent = sqrt(2 log(M/delta)))
    * argmax of compositional LCB (exponent uses LASSO sparsity s_hat)

Reports the *regret* V(rho_oracle) - V(rho_hat) averaged over trials.
Theorem 4 predicts the compositional LCB has smaller regret than the
standard LCB when the rule-value function is atom-sparse.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.crrm import PessimisticConfig, PessimisticRuleSelector
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import JointRuleOPE
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import load_rules


def _oracle_rule(rules, gt):
    best = max(rules, key=lambda r: gt[r.id])
    return best, gt[best.id]


def _trial(N, seed, rules):
    cfg = SubstrateConfig(n_queries=N, seed=seed, logging="deterministic")
    logs = generate_logs(cfg)
    logs = assign_corrections(logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, seed=seed + 1))
    gt = ground_truth_many(rules, logs)

    est = JointRuleOPE()
    est.fit(logs)
    res = est.value_many(rules, logs)

    # Strategy A: naive argmax of estimate
    naive_id = max(rules, key=lambda r: res[r.id].estimate)
    # Strategy B: standard union-bound LCB
    sel_std = PessimisticRuleSelector(PessimisticConfig(atom_sparse=False))
    std_id, _ = sel_std.select(rules, res)
    # Strategy C: compositional LCB
    sel_comp = PessimisticRuleSelector(PessimisticConfig(atom_sparse=True))
    comp_id, _ = sel_comp.select(rules, res)

    oracle, oracle_v = _oracle_rule(rules, gt)
    return {
        "oracle": oracle_v,
        "naive":   gt[naive_id.id],
        "std_lcb": gt[std_id.id],
        "comp_lcb": gt[comp_id.id],
    }


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    results = {}
    for N in (300, 600, 1200, 2400):
        per_trial = []
        for seed in range(5):
            per_trial.append(_trial(N, seed=seed * 91 + N, rules=rules))
        regret = {
            "naive":    [t["oracle"] - t["naive"]   for t in per_trial],
            "std_lcb":  [t["oracle"] - t["std_lcb"] for t in per_trial],
            "comp_lcb": [t["oracle"] - t["comp_lcb"] for t in per_trial],
        }
        agg = {k: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in regret.items()}
        results[f"N={N}"] = agg
        print(f"N={N}  naive={agg['naive']['mean']:.4f}  std={agg['std_lcb']['mean']:.4f}  comp={agg['comp_lcb']['mean']:.4f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/pessimistic.json", "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
