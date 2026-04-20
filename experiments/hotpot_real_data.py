"""Real-data evaluation on HotpotQA: RuleOPE vs SOTA baselines.

First public-data rule-OPE experiment.  Supports the theorem by showing
RuleOPE's predictions match ground-truth rule values on a real RAG
benchmark, and compares against classical estimators (DR, DM, IPS) on
per-rule MSE and top-k selection quality.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import kendalltau

from src.estimators.direct_method import DirectMethod
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.ips import IPS
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import JointRuleOPE
from src.rag_substrate_hotpot import generate_hotpot_logs, ground_truth_rule_value
from src.rule_dsl import load_rules


def stochastic_logs(logs, pi0=(0.7, 0.1, 0.1, 0.1), seed=0):
    """Convert noop-only logs into a stochastic-logging version for OPE
    with non-degenerate propensities."""
    from src.logs import LoggedRecord
    import numpy as np
    rng = np.random.default_rng(seed)
    actions = ("noop", "filter", "rerank", "abstain")
    new = []
    for rec in logs:
        a_idx = int(rng.choice(4, p=pi0))
        a = actions[a_idx]
        r = float(rec.cf_rewards[a])
        r_noisy = float(np.clip(r + rng.normal(0, 0.05), 0.0, 1.0))
        new.append(LoggedRecord(
            query_id=rec.query_id,
            ctx=rec.ctx,
            logged_action=a,
            logged_propensity=pi0[a_idx],
            logged_reward=r_noisy,
            correction=0,
            cf_rewards=rec.cf_rewards,
        ))
    return new


def main() -> int:
    print("Loading HotpotQA + scoring ...", flush=True)
    t0 = time.time()
    logs = generate_hotpot_logs("eval/hotpot/dev.parquet", n_queries=1200, seed=0)
    print(f"  built {len(logs)} logs in {time.time() - t0:.1f}s", flush=True)

    # Convert to stochastic logging for IPS/DR to be well-defined.
    logs = stochastic_logs(logs, pi0=(0.70, 0.10, 0.10, 0.10), seed=1)

    rules = load_rules("eval/rules_v1.jsonl")[:150]

    print("Computing ground truth ...", flush=True)
    gt = {r.id: ground_truth_rule_value(r, logs) for r in rules}
    print(f"  range: [{min(gt.values()):.4f}, {max(gt.values()):.4f}]", flush=True)

    # Which rules fire on at least 5 % of queries -- a minimum coverage
    # threshold so that the per-rule estimate is well-defined.
    from src.estimators._regression import fires_mask
    firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules}
    rules = [r for r in rules if firing[r.id] >= 0.05]
    print(f"  {len(rules)} rules after firing-rate filter", flush=True)

    estimators = {
        "DM": DirectMethod(),
        "IPS": IPS(),
        "DR": DoublyRobust(),
        "RuleOPE": RuleOPE(),
        "JointRuleOPE": JointRuleOPE(),
    }

    out = {"n_logs": len(logs), "n_rules": len(rules), "estimators": {}}
    for name, est in estimators.items():
        print(f"Running {name} ...", flush=True)
        t0 = time.time()
        est.fit(logs)
        results = est.value_many(rules, logs)
        runtime = time.time() - t0
        gt_vals = np.array([gt[r.id] for r in rules])
        est_vals = np.array([results[r.id].estimate for r in rules])
        mse = float(np.mean((est_vals - gt_vals) ** 2))
        bias = float(np.mean(est_vals - gt_vals))
        tau_20, _ = kendalltau(np.argsort(-est_vals)[:20], np.argsort(-gt_vals)[:20])
        # 95% CI coverage
        cov = 0
        for r in rules:
            lo = results[r.id].estimate - 1.96 * results[r.id].stderr
            hi = results[r.id].estimate + 1.96 * results[r.id].stderr
            if lo <= gt[r.id] <= hi:
                cov += 1
        cov95 = cov / len(rules)
        out["estimators"][name] = {
            "MSE": mse,
            "bias": bias,
            "tau_top20": float(tau_20) if tau_20 == tau_20 else 0.0,
            "coverage_95": cov95,
            "runtime_s": runtime,
        }
        print(f"  {name:12s}  MSE={mse:.5f}  bias={bias:+.4f}  tau@20={tau_20:+.3f}  cov95={cov95:.3f}  t={runtime:.1f}s", flush=True)

    # Top-10 rule selection quality
    for name, est in estimators.items():
        res = est.value_many(rules, logs)
        ranked = sorted(rules, key=lambda r: -res[r.id].estimate)[:10]
        top10_gt = sorted([gt[r.id] for r in ranked], reverse=True)
        avg_gt_top10 = float(np.mean(top10_gt))
        out["estimators"][name]["avg_GT_value_top10"] = avg_gt_top10
    out["oracle_top10_value"] = float(np.mean(sorted(gt.values(), reverse=True)[:10]))

    # Bootstrap CIs for tau@20 and MSE
    print("\nBootstrap CIs (60 resamples) ...", flush=True)
    rng = np.random.default_rng(42)
    n_boot = 60
    N = len(logs)
    gt_array = np.array([gt[r.id] for r in rules])
    boot_results: dict[str, dict[str, list[float]]] = {
        name: {"tau_20": [], "mse": []} for name in estimators
    }
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot_logs = [logs[int(i)] for i in idx]
        for name in estimators:
            est = estimators[name].__class__() if name != "JointRuleOPE" else JointRuleOPE()
            est.fit(boot_logs)
            res = est.value_many(rules, boot_logs)
            est_vals = np.array([res[r.id].estimate for r in rules])
            tau_b, _ = kendalltau(np.argsort(-est_vals)[:20], np.argsort(-gt_array)[:20])
            boot_results[name]["tau_20"].append(float(tau_b) if tau_b == tau_b else 0.0)
            boot_results[name]["mse"].append(float(np.mean((est_vals - gt_array) ** 2)))
    print("\nBootstrap 90% CIs:", flush=True)
    for name in estimators:
        tau_arr = np.array(boot_results[name]["tau_20"])
        mse_arr = np.array(boot_results[name]["mse"])
        out["estimators"][name]["tau_20_CI"] = [float(np.quantile(tau_arr, 0.05)), float(np.quantile(tau_arr, 0.95))]
        out["estimators"][name]["MSE_CI"] = [float(np.quantile(mse_arr, 0.05)), float(np.quantile(mse_arr, 0.95))]
        print(f"  {name:12s}  tau@20 ∈ [{np.quantile(tau_arr, 0.05):+.3f}, {np.quantile(tau_arr, 0.95):+.3f}]  MSE ∈ [{np.quantile(mse_arr, 0.05):.5f}, {np.quantile(mse_arr, 0.95):.5f}]")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/hotpot_real_data.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nOracle top-10 GT value: {out['oracle_top10_value']:.4f}")
    print("Top-10 GT value by selector:")
    for name in estimators:
        v = out["estimators"][name]["avg_GT_value_top10"]
        print(f"  {name:12s}  {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
