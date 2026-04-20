"""HotpotQA with LLM-judge correction signal, deterministic logging.

This is the regime where RuleOPE's correction-fusion term should
empirically pay off: noop-only logging deprives classical DR of
information about non-noop actions; the judge correction provides
exactly that information, which RuleOPE exploits via its
correction-fusion bridge.

Run:
    LAMBDA_JUDGE_HOST=http://<ip>:8000 \
      python3 experiments/hotpot_with_judge.py

If LAMBDA_JUDGE_HOST is unset or unreachable, falls back to the
gold-answer-match proxy (documented in the paper's §H).
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
from src.rag_substrate_hotpot import ground_truth_rule_value
from src.rag_substrate_hotpot_judge import (
    generate_hotpot_logs_deterministic,
    make_lambda_judge,
)
from src.rule_dsl import load_rules


def main() -> int:
    judge_fn = None
    judge_source = "gold_answer_match_proxy"
    host = os.environ.get("LAMBDA_JUDGE_HOST")
    if host:
        try:
            judge_fn = make_lambda_judge(
                endpoint=host,
                model=os.environ.get("LAMBDA_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            )
            # Smoke-test with 1 call.
            judge_fn("Who wrote the Iliad?", ["Homer was a Greek poet."])
            judge_source = f"llm_judge@{host}"
            print(f"  using LLM judge: {judge_source}", flush=True)
        except Exception as e:
            print(f"  LLM judge unreachable ({e}); falling back to gold-match proxy", flush=True)
            judge_fn = None

    print("Loading HotpotQA + scoring + judging ...", flush=True)
    t0 = time.time()
    logs = generate_hotpot_logs_deterministic(
        "eval/hotpot/dev.parquet",
        n_queries=1200,
        seed=0,
        judge_fn=judge_fn,
    )
    print(f"  built {len(logs)} logs in {time.time() - t0:.1f}s", flush=True)
    correction_rate = float(np.mean([r.correction for r in logs]))
    print(f"  correction rate = {correction_rate:.3f}", flush=True)

    rules = load_rules("eval/rules_v1.jsonl")[:150]

    print("Computing ground truth ...", flush=True)
    gt = {r.id: ground_truth_rule_value(r, logs) for r in rules}
    print(f"  range: [{min(gt.values()):.4f}, {max(gt.values()):.4f}]", flush=True)

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

    out = {
        "n_logs": len(logs),
        "n_rules": len(rules),
        "judge_source": judge_source,
        "correction_rate": correction_rate,
        "estimators": {},
    }
    gt_array = np.array([gt[r.id] for r in rules])
    for name, est in estimators.items():
        print(f"Running {name} ...", flush=True)
        t0 = time.time()
        est.fit(logs)
        results = est.value_many(rules, logs)
        runtime = time.time() - t0
        est_vals = np.array([results[r.id].estimate for r in rules])
        mse = float(np.mean((est_vals - gt_array) ** 2))
        bias = float(np.mean(est_vals - gt_array))
        tau_20, _ = kendalltau(np.argsort(-est_vals)[:20], np.argsort(-gt_array)[:20])
        cov = sum(
            1 for r in rules
            if results[r.id].estimate - 1.96 * results[r.id].stderr
            <= gt[r.id]
            <= results[r.id].estimate + 1.96 * results[r.id].stderr
        )
        cov95 = cov / len(rules)
        out["estimators"][name] = {
            "MSE": mse,
            "bias": bias,
            "tau_top20": float(tau_20) if tau_20 == tau_20 else 0.0,
            "coverage_95": cov95,
            "runtime_s": runtime,
        }
        print(f"  {name:12s}  MSE={mse:.5f}  bias={bias:+.4f}  tau@20={tau_20:+.3f}  cov95={cov95:.3f}  t={runtime:.1f}s", flush=True)

    # Top-10 selection
    for name, est in estimators.items():
        res = est.value_many(rules, logs)
        ranked = sorted(rules, key=lambda r: -res[r.id].estimate)[:10]
        out["estimators"][name]["avg_GT_value_top10"] = float(np.mean([gt[r.id] for r in ranked]))
    out["oracle_top10_value"] = float(np.mean(sorted(gt.values(), reverse=True)[:10]))

    # Bootstrap CIs
    print("\nBootstrap CIs (60 resamples) ...", flush=True)
    rng = np.random.default_rng(42)
    n_boot = 60
    N = len(logs)
    boot_results = {name: {"tau_20": [], "mse": []} for name in estimators}
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
    with open("experiments/results/hotpot_with_judge.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nOracle top-10 GT value: {out['oracle_top10_value']:.4f}")
    print("Top-10 GT value by selector:")
    for name in estimators:
        v = out["estimators"][name]["avg_GT_value_top10"]
        print(f"  {name:12s}  {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
