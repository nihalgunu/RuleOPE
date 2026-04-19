"""Phase 3 main comparison, across three regimes:

    R1: Stochastic logging + low correction noise (the benchmark-v1 default).
    R2: Deterministic logging + low correction noise (production-like).
    R3: Stochastic logging + HIGH correction noise (robustness stress).

For each regime we train the estimators on the logs and evaluate on the
frozen rule set.  We re-seed the log generator for each (regime, trial)
pair so error bars reflect sampling variability of the logs themselves.

Outputs: experiments/results/main_comparison.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.cascade_dr import CascadeDR
from src.estimators.cips import CIPS, CIPS_DR
from src.estimators.direct_method import DirectMethod
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.ips import IPS, SNIPS
from src.estimators.rule_ope import RuleOPE
from src.evaluate import all_metrics
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import load_rules


REGIMES = {
    "R1_stoch_low_noise":    dict(logging="stochastic",    noise=0.10),
    "R2_det_low_noise":      dict(logging="deterministic", noise=0.10),
    "R3_stoch_high_noise":   dict(logging="stochastic",    noise=0.30),
}


def build_estimators():
    return [
        DirectMethod(),
        IPS(),
        SNIPS(),
        DoublyRobust(),
        CIPS(clip=20.0),
        CIPS_DR(clip=20.0),
        CascadeDR(),
        RuleOPE(),
    ]


def run_trial(regime, n_queries, seed, rules):
    cfg = SubstrateConfig(n_queries=n_queries, seed=seed, logging=regime["logging"])
    logs = generate_logs(cfg)
    logs = assign_corrections(
        logs,
        CorrectionConfig(
            base_rate=0.15, error_sensitivity=4.0, noise_frac=regime["noise"],
            seed=seed + 1000,
        ),
    )
    gt = ground_truth_many(rules, logs)
    out = {}
    for est in build_estimators():
        t0 = time.time()
        if hasattr(est, "fit"):
            est.fit(logs)
        res = est.value_many(rules, logs)
        dt = time.time() - t0
        estimates = {k: v.estimate for k, v in res.items()}
        stderrs = {k: v.stderr for k, v in res.items()}
        m = all_metrics(estimates, stderrs, gt, topk=20)
        out[est.name] = {**m, "time_s": dt}
    return out


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    n_queries = 3000
    n_trials = 3

    results = {regime: {name: {"mse": [], "bias": [], "coverage_95": [], "topk_tau": [], "time_s": []} for name in [
        "DM", "IPS", "SNIPS", "DR", "CIPS", "CIPS-DR", "CascadeDR", "RuleOPE"
    ]} for regime in REGIMES}

    for regime_name, regime in REGIMES.items():
        for trial in range(n_trials):
            print(f"=== {regime_name}  trial={trial} ===")
            out = run_trial(regime, n_queries, seed=1000 * trial + 7, rules=rules)
            for est, m in out.items():
                for k, v in m.items():
                    results[regime_name][est][k].append(v)
            for est, m in out.items():
                print(f"  {est:>9s}  MSE={m['mse']:.5f}  tau@20={m['topk_tau']:+.3f}")

    # Aggregate: mean and std over trials.
    agg = {}
    for regime_name, regime_results in results.items():
        agg[regime_name] = {}
        for est, metrics in regime_results.items():
            agg[regime_name][est] = {
                k: {
                    "mean": float(np.mean(vs)),
                    "std":  float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
                }
                for k, vs in metrics.items()
            }

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "main_comparison.json", "w") as f:
        json.dump({"raw": results, "agg": agg, "n_trials": n_trials, "n_queries": n_queries}, f, indent=2)

    # Print final table
    print("\n=== aggregated ===")
    for regime_name in REGIMES:
        print(f"\n[{regime_name}]")
        print(f"  {'estimator':>10s} | {'MSE (mean ± std)':>22s} | {'tau@20':>8s}")
        for est in ["DM", "IPS", "SNIPS", "DR", "CIPS", "CIPS-DR", "CascadeDR", "RuleOPE"]:
            m = agg[regime_name][est]["mse"]
            t = agg[regime_name][est]["topk_tau"]
            print(f"  {est:>10s} | {m['mean']:.5f} ± {m['std']:.5f}    | {t['mean']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
