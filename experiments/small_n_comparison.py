"""Small-N, deterministic-logging comparison.

The main_comparison experiment uses N=3000 stochastic logging where every
DR-family estimator ties.  The production regime -- small N, deterministic
logging -- is where the rule-OPE correction-fusion term actually matters.
This script documents that regime's dramatic improvement.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import kendalltau

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.cascade_dr import CascadeDR
from src.estimators.cips import CIPS_DR
from src.estimators.direct_method import DirectMethod
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import DualShrinkOPE, JointRuleOPE
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import load_rules


def _run_trial(N, seed, rules):
    logs = generate_logs(SubstrateConfig(n_queries=N, seed=seed, logging="deterministic"))
    logs = assign_corrections(
        logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, noise_frac=0.10, seed=seed + 1)
    )
    gt = ground_truth_many(rules, logs)
    gtv = np.array([gt[r.id] for r in rules])
    out = {}
    for name, est in [
        ("DM", DirectMethod()),
        ("DR", DoublyRobust()),
        ("CIPS-DR", CIPS_DR(clip=20.0)),
        ("CascadeDR", CascadeDR()),
        ("RuleOPE", RuleOPE()),
        ("JointRuleOPE", JointRuleOPE()),
        ("DualShrinkOPE", DualShrinkOPE()),
    ]:
        est.fit(logs)
        res = est.value_many(rules, logs)
        ev = np.array([res[r.id].estimate for r in rules])
        mse = float(np.mean((ev - gtv) ** 2))
        order = np.argsort(-ev)[:20]
        tau, _ = kendalltau(ev[order], gtv[order])
        out[name] = dict(mse=mse, tau=float(tau) if np.isfinite(tau) else 0.0)
    return out


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    results = {}
    for N in (300, 600, 1200, 2400):
        per_trial = {}
        for seed in range(5):
            one = _run_trial(N, seed=seed * 101 + N, rules=rules)
            for k, v in one.items():
                per_trial.setdefault(k, []).append(v)
        agg = {}
        for k, rows in per_trial.items():
            mses = [r["mse"] for r in rows]
            taus = [r["tau"] for r in rows]
            agg[k] = dict(
                mse_mean=float(np.mean(mses)),
                mse_std=float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
                tau_mean=float(np.mean(taus)),
            )
        results[f"N={N}"] = agg
        print(f"--- N={N}  deterministic ---")
        for k, a in agg.items():
            print(f"  {k:>15s}  MSE={a['mse_mean']:.5f} ± {a['mse_std']:.5f}  tau@20={a['tau_mean']:+.3f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/small_n_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Relative-to-DR MSE reduction ===")
    for k in ("N=300", "N=600", "N=1200", "N=2400"):
        dr = results[k]["DR"]["mse_mean"]
        rope = results[k]["RuleOPE"]["mse_mean"]
        dual = results[k]["DualShrinkOPE"]["mse_mean"]
        print(f"  {k}: RuleOPE {1-rope/dr:+.1%}  DualShrink {1-dual/dr:+.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
