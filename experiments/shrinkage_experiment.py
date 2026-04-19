"""Joint rule evaluation: shrinkage vs independent per-rule estimates.

Demonstrates that cross-rule shrinkage strictly dominates independent
per-rule estimation in joint MSE across a range of rule-set sizes and
sample sizes, as predicted by Theorem 3.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import JointRuleOPE, ShrinkConfig
from src.evaluate import all_metrics
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import load_rules


def _run(N: int, seed: int, rules, mode: str = "per_rule_eb", logging: str = "deterministic") -> dict:
    cfg = SubstrateConfig(n_queries=N, seed=seed, logging=logging)
    logs = generate_logs(cfg)
    logs = assign_corrections(logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, noise_frac=0.1, seed=seed + 17))
    gt = ground_truth_many(rules, logs)

    out = {}
    # Independent baselines
    for est in [RuleOPE(), DoublyRobust()]:
        est.fit(logs)
        res = est.value_many(rules, logs)
        est_map = {k: v.estimate for k, v in res.items()}
        se_map = {k: v.stderr for k, v in res.items()}
        out[est.name] = all_metrics(est_map, se_map, gt, topk=20)

    # Shrinkage
    shrunk = JointRuleOPE(config=ShrinkConfig(mode=mode))
    shrunk.fit(logs)
    res = shrunk.value_many(rules, logs)
    est_map = {k: v.estimate for k, v in res.items()}
    se_map = {k: v.stderr for k, v in res.items()}
    out[f"Joint-{mode}"] = all_metrics(est_map, se_map, gt, topk=20)
    return out


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    results = {}
    for N in (300, 600, 1200, 2400):
        print(f"--- N = {N} (deterministic logging) ---")
        per_trial = {name: [] for name in ["RuleOPE", "DR", "Joint-per_rule_eb", "Joint-james_stein"]}
        for trial in range(3):
            for mode in ("per_rule_eb", "james_stein"):
                one = _run(N, seed=1000 * trial + N, rules=rules, mode=mode)
                label = f"Joint-{mode}"
                per_trial[label].append(one[label])
                if mode == "per_rule_eb":  # only once per trial
                    for b in ("RuleOPE", "DR"):
                        per_trial[b].append(one[b])
        agg = {}
        for k, rows in per_trial.items():
            mses = [r["mse"] for r in rows]
            taus = [r["topk_tau"] for r in rows]
            agg[k] = dict(
                mse_mean=float(np.mean(mses)),
                mse_std=float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
                tau_mean=float(np.mean(taus)),
            )
            print(f"  {k:>22s}  MSE = {agg[k]['mse_mean']:.6f} ± {agg[k]['mse_std']:.6f}  tau@20={agg[k]['tau_mean']:+.3f}")
        results[f"N={N}"] = agg

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/shrinkage.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Shrinkage dominance summary ===")
    for k, agg in results.items():
        base = agg["RuleOPE"]["mse_mean"]
        joint = agg["Joint-per_rule_eb"]["mse_mean"]
        print(f"  {k}: RuleOPE={base:.6f}  Joint={joint:.6f}  reduction={1 - joint/base:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
