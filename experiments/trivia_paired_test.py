"""Paired bootstrap test on the TriviaQA scaling results.

The quantile-based 90% CI on per-trial percentage reductions
(used elsewhere in the paper) can be wide under heavy-tailed
MSE distributions even at large n_trials.  For TriviaQA we add
a paired-bootstrap CI on the mean log-ratio log(MSE_NC / MSE_R),
which is the proper test statistic for "is RuleOPE better on average".

Run: python3 experiments/trivia_paired_test.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import kendalltau, ttest_rel

from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import JointRuleOPE
from src.logs import LoggedRecord
from src.rag_substrate_trivia import (
    _alias_match, _apply_rule, _atom_features, _load_trivia, _score_passages,
)
from src.rule_dsl import load_rules
from experiments.ablations import NonCompositionalDR


def run(n_trials: int, Ns: list[int], pool_size: int = 1500):
    all_samples = _load_trivia("eval/trivia/dev.parquet", pool_size, 0)
    print(f"Loaded {len(all_samples)} TriviaQA samples", flush=True)

    oracle = {}
    for s in all_samples:
        scores = _score_passages(s)
        cf = {}
        for a in ("noop", "filter", "rerank"):
            _, bodies = _apply_rule(a, scores, s)
            cf[a] = _alias_match(bodies, s.answer_aliases)
        cf["abstain"] = 0.5
        oracle[s.qid] = cf

    rules = load_rules("eval/rules_v1.jsonl")
    rules = [r for r in rules if r.action in ("filter", "rerank", "abstain")]

    ACTIONS = ("noop", "filter", "rerank")
    results = {}
    for N in Ns:
        print(f"\n=== TriviaQA N={N}, n_trials={n_trials} ===", flush=True)
        mse_NC, mse_RO = [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 * N + trial)
            idx = rng.choice(len(all_samples), size=min(N, len(all_samples)), replace=False)
            samples_tr = [all_samples[int(i)] for i in idx]
            logs = []
            for s in samples_tr:
                scores = _score_passages(s); ctx = _atom_features(s, scores)
                a = ACTIONS[int(rng.integers(0, 3))]
                logs.append(LoggedRecord(
                    query_id=s.qid, ctx=ctx, logged_action=a, logged_propensity=1/3,
                    logged_reward=float(oracle[s.qid][a]),
                    correction=0, cf_rewards=dict(oracle[s.qid])
                ))
            from src.estimators._regression import fires_mask
            firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules}
            tr_rules = [r for r in rules if 0.05 <= firing[r.id] <= 0.95]
            if len(tr_rules) < 10:
                continue

            def oracle_value(rule):
                return float(np.mean([
                    rec.cf_rewards[rule.action] if rule.fires(rec.ctx) else rec.cf_rewards["noop"]
                    for rec in logs
                ]))
            gt = np.array([oracle_value(r) for r in tr_rules])

            nc = NonCompositionalDR(); nc.fit(logs)
            ro = RuleOPE(); ro.fit(logs)
            nc_map = nc.value_many(tr_rules, logs)
            ro_map = ro.value_many(tr_rules, logs)
            nc_vals = np.array([nc_map[r.id].estimate for r in tr_rules])
            ro_vals = np.array([ro_map[r.id].estimate for r in tr_rules])
            mse_NC.append(float(np.mean((nc_vals - gt) ** 2)))
            mse_RO.append(float(np.mean((ro_vals - gt) ** 2)))

        mse_NC = np.array(mse_NC); mse_RO = np.array(mse_RO)

        # Paired log-ratio
        eps = 1e-9
        log_ratio = np.log(mse_NC + eps) - np.log(mse_RO + eps)
        # Mean log-ratio bootstrap CI
        rng2 = np.random.default_rng(17)
        boots = np.array([
            log_ratio[rng2.integers(0, len(log_ratio), size=len(log_ratio))].mean()
            for _ in range(5000)
        ])
        ci_lo = float(np.quantile(boots, 0.05))
        ci_hi = float(np.quantile(boots, 0.95))
        mean_lr = float(log_ratio.mean())

        # Convert to percentage (exp(log_ratio) - 1 = MSE_NC/MSE_RO - 1)
        pct_mean = 100.0 * (np.exp(mean_lr) - 1)
        pct_lo = 100.0 * (np.exp(ci_lo) - 1)
        pct_hi = 100.0 * (np.exp(ci_hi) - 1)

        t_stat, p_val = ttest_rel(mse_NC, mse_RO)

        results[str(N)] = {
            "n_trials": int(len(mse_NC)),
            "MSE_NC_mean": float(mse_NC.mean()),
            "MSE_RO_mean": float(mse_RO.mean()),
            "mean_log_ratio": mean_lr,
            "log_ratio_CI90_bootstrap": [ci_lo, ci_hi],
            "pct_mean":  pct_mean,
            "pct_CI90_bootstrap": [pct_lo, pct_hi],
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "significance_bootstrap_CI": bool(ci_lo > 0),
            "significance_paired_t": bool(p_val < 0.05 and mse_RO.mean() < mse_NC.mean()),
        }
        print(f"  mean log-ratio    = {mean_lr:+.4f}   pct = {pct_mean:+.2f}%", flush=True)
        print(f"  bootstrap 90% CI  = [{ci_lo:+.4f}, {ci_hi:+.4f}]   "
              f"({pct_lo:+.2f}%, {pct_hi:+.2f}%)", flush=True)
        print(f"  paired t-test     t={t_stat:+.2f}, p={p_val:.3e}", flush=True)
        print(f"  sig by bootstrap? {ci_lo > 0}   sig by t-test? {p_val < 0.05 and mse_RO.mean() < mse_NC.mean()}",
              flush=True)

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/trivia_paired_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote experiments/results/trivia_paired_test.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--Ns", nargs="+", type=int, default=[150, 300, 600, 1200])
    args = ap.parse_args()
    raise SystemExit(run(args.n_trials, args.Ns))
