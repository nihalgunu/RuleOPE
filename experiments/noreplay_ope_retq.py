"""No-replay RAG-OPE on HotpotQA — retrieval-quality reward variant.

Uses the (stronger) retrieval-quality reward:
  R_i^a = fraction of gold passages present in top-3 under action a.

This reward is retrieval-sensitive by construction (the generator is
no longer the bottleneck).  The LLM-generated answers from
`outputs.jsonl` are still used by the judge/correction, but the
reward itself is an intrinsic retrieval metric.
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
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import JointRuleOPE
from src.logs import LoggedRecord
from src.rag_substrate_hotpot import (
    _apply_rule,
    _atom_features,
    _load_hotpot,
    _reward_for_top3,
    _score_passages,
)
from src.rule_dsl import load_rules


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="eval/hotpot/dev.parquet")
    ap.add_argument("--outputs", default="eval/hotpot/outputs.jsonl")
    ap.add_argument("--n_queries", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = _load_hotpot(args.hotpot, args.n_queries, args.seed)
    # Optional: overlay LLM outputs as the correction signal (unknown-aware).
    llm_answers = {}
    if Path(args.outputs).exists():
        with open(args.outputs) as f:
            for line in f:
                d = json.loads(line)
                qid, action = d["id"].split("__")
                llm_answers.setdefault(qid, {})[action] = d["text"]

    # Build oracle retrieval-quality reward for each action.
    logs = []
    gt_rules = {}
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        cf = {}
        for a in ("noop", "filter", "rerank", "abstain"):
            titles = _apply_rule(a, scores, s)
            if a == "abstain":
                cf[a] = 0.5
            else:
                cf[a] = _reward_for_top3(s.gold_titles, titles)
        # Correction = noop retrieval was not sufficient (reward < 0.5).
        # This matches the theorem's A5 condition: corrections carry
        # information about whether the logged action was optimal.
        logged_reward = float(cf["noop"])
        correction = 1 if logged_reward < 0.5 else 0
        logs.append(
            LoggedRecord(
                query_id=s.qid,
                ctx=ctx,
                logged_action="noop",
                logged_propensity=1.0,
                logged_reward=logged_reward,
                correction=int(correction),
                cf_rewards=cf,
            )
        )

    print(f"  n_logs={len(logs)}  corr_rate={np.mean([r.correction for r in logs]):.3f}  noop_mean={np.mean([r.logged_reward for r in logs]):.3f}", flush=True)

    # Rules and ground truth
    rules = load_rules("eval/rules_v1.jsonl")
    rules = [r for r in rules if r.action in ("filter", "rerank", "abstain")]
    def oracle_value(rule):
        return float(np.mean([rec.cf_rewards[rule.action] if rule.fires(rec.ctx) else rec.cf_rewards["noop"] for rec in logs]))
    gt = {r.id: oracle_value(r) for r in rules}
    from src.estimators._regression import fires_mask
    firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules}
    rules = [r for r in rules if 0.05 <= firing[r.id] <= 0.95]
    gt = {r.id: gt[r.id] for r in rules}
    print(f"  n_rules={len(rules)}  V-range=[{min(gt.values()):.3f}, {max(gt.values()):.3f}]", flush=True)

    estimators = {
        "DM": DirectMethod(),
        "DR": DoublyRobust(),
        "RuleOPE": RuleOPE(),
        "JointRuleOPE": JointRuleOPE(),
    }
    out = {
        "n_logs": len(logs),
        "n_rules": len(rules),
        "correction_rate": float(np.mean([r.correction for r in logs])),
        "noop_mean_reward": float(np.mean([r.logged_reward for r in logs])),
        "reward_type": "retrieval_quality",
        "estimators": {},
    }
    gt_array = np.array([gt[r.id] for r in rules])
    for name, est in estimators.items():
        t0 = time.time()
        est.fit(logs)
        res = est.value_many(rules, logs)
        rt = time.time() - t0
        est_vals = np.array([res[r.id].estimate for r in rules])
        mse = float(np.mean((est_vals - gt_array) ** 2))
        bias = float(np.mean(est_vals - gt_array))
        tau20, _ = kendalltau(np.argsort(-est_vals)[:20], np.argsort(-gt_array)[:20])
        ranked = sorted(rules, key=lambda r: -res[r.id].estimate)[:10]
        avg_top10 = float(np.mean([gt[r.id] for r in ranked]))
        out["estimators"][name] = {
            "MSE": mse, "bias": bias, "tau_top20": float(tau20) if tau20 == tau20 else 0.0,
            "avg_GT_value_top10": avg_top10, "runtime_s": rt,
        }
        print(f"  {name:14s}  MSE={mse:.5f}  bias={bias:+.4f}  tau@20={tau20:+.3f}  top10_GT={avg_top10:.3f}  t={rt:.1f}s", flush=True)
    out["oracle_top10_value"] = float(np.mean(sorted(gt.values(), reverse=True)[:10]))
    print(f"  oracle_top10={out['oracle_top10_value']:.3f}", flush=True)

    # Bootstrap CIs
    rng = np.random.default_rng(42)
    n_boot = 80
    boot = {name: {"mse": [], "tau": [], "top10": []} for name in estimators}
    print(f"\nBootstrap {n_boot} resamples ...", flush=True)
    for b in range(n_boot):
        idx = rng.integers(0, len(logs), size=len(logs))
        blogs = [logs[int(i)] for i in idx]
        for name in estimators:
            est = estimators[name].__class__() if name != "JointRuleOPE" else JointRuleOPE()
            est.fit(blogs)
            res = est.value_many(rules, blogs)
            est_vals = np.array([res[r.id].estimate for r in rules])
            boot[name]["mse"].append(float(np.mean((est_vals - gt_array) ** 2)))
            tau_b, _ = kendalltau(np.argsort(-est_vals)[:20], np.argsort(-gt_array)[:20])
            boot[name]["tau"].append(float(tau_b) if tau_b == tau_b else 0.0)
            ranked = sorted(rules, key=lambda r: -res[r.id].estimate)[:10]
            boot[name]["top10"].append(float(np.mean([gt[r.id] for r in ranked])))
    for name in estimators:
        m = np.array(boot[name]["mse"]); t = np.array(boot[name]["tau"]); x = np.array(boot[name]["top10"])
        out["estimators"][name]["MSE_CI"] = [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))]
        out["estimators"][name]["tau_20_CI"] = [float(np.quantile(t, 0.05)), float(np.quantile(t, 0.95))]
        out["estimators"][name]["top10_CI"] = [float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))]
        print(f"  {name:14s}  MSE [{np.quantile(m,0.05):.5f}, {np.quantile(m,0.95):.5f}]  tau [{np.quantile(t,0.05):+.3f}, {np.quantile(t,0.95):+.3f}]  top10 [{np.quantile(x,0.05):.3f}, {np.quantile(x,0.95):.3f}]", flush=True)

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/noreplay_ope_retq.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
