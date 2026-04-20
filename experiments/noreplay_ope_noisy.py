"""No-replay RAG-OPE with retrieval noise injection.

Real production retrievers are unstable: passage rankings shift across
re-indexings, embedding updates, etc.  We simulate this by injecting
retrieval noise (random top-1 drop with probability p_noise) into the
noop logging policy.  This creates the regime where:
  - noop reward is bimodal: high when stable, low when noisy
  - filter helps on noisy queries (drops the bad top-1), hurts on clean
  - rerank is moderate
  - rules that fire conditional on retrieval instability
    (e.g., score_gap_lt_0_10) should have strong V(rho) signal
  - correction signal (R < 0.5) fires at realistic rates
  - the compositional assumption A3 is tested because the reward
    structure genuinely depends on the atom features (score_gap,
    top-3 entity presence) that our vocabulary tracks

Under this regime the no-replay theorem's prediction is most testable:
RuleOPE's correction-fusion term should recover the oracle while DR's
regression-only term cannot distinguish noisy from clean queries.
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
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators.shrinkage import JointRuleOPE
from src.logs import LoggedRecord
from src.rag_substrate_hotpot import (
    _apply_rule,
    _atom_features,
    _load_hotpot,
    _reward_for_top3,
    _score_passages,
    _secondary_scores,
)
from src.rule_dsl import load_rules


def _noisy_apply(action: str, scores: np.ndarray, sample, rng: np.random.Generator, p_noise: float) -> list[str]:
    """Apply action with retrieval noise: with prob p_noise, drop the top-1
    passage as a pre-processing step (simulating a noisy retriever), THEN
    apply the action."""
    order = np.argsort(scores)[::-1]
    is_noisy = rng.random() < p_noise
    if is_noisy and len(order) > 1:
        order = order[1:]
    if action == "abstain":
        return []
    if action == "filter":
        order = order[1:]
    elif action == "rerank":
        sec = _secondary_scores(sample, scores)
        order = np.argsort(sec)[::-1]
    return [sample.passages[int(i)][0] for i in order[:3]]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="eval/hotpot/dev.parquet")
    ap.add_argument("--n_queries", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_noise", type=float, default=0.30)
    args = ap.parse_args()

    samples = _load_hotpot(args.hotpot, args.n_queries, args.seed)
    rng = np.random.default_rng(args.seed + 7)

    # Build (seeded) per-query noise draws shared across oracle and logging.
    per_query_noisy = {
        s.qid: rng.random() < args.p_noise for s in samples
    }

    logs = []
    gt_oracle = {}
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        # cf_rewards under each action: noop's noise is sampled once; other
        # actions are computed deterministically on the post-noise retrieval
        # (i.e., the noise has already happened before the rule intervenes).
        noisy = per_query_noisy[s.qid]
        def apply_with_noise(action):
            order = np.argsort(scores)[::-1]
            if noisy and len(order) > 1:
                order = order[1:]
            if action == "abstain":
                return []
            if action == "filter":
                order = order[1:]
            elif action == "rerank":
                sec = _secondary_scores(s, scores)
                order = np.argsort(sec)[::-1]
            return [s.passages[int(i)][0] for i in order[:3]]
        cf = {}
        for a in ("noop", "filter", "rerank"):
            cf[a] = _reward_for_top3(s.gold_titles, apply_with_noise(a))
        cf["abstain"] = 0.5
        gt_oracle[s.qid] = dict(cf)
        logged_reward = float(cf["noop"])
        # Correction = noop reward < 0.5 (i.e., retrieval was bad enough
        # that the answer can't be extracted reliably).  Under noise this
        # is correlated with "top-1 was critical and we lost it".
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

    print(f"  n_logs={len(logs)}  p_noise={args.p_noise}  corr_rate={np.mean([r.correction for r in logs]):.3f}  noop_mean={np.mean([r.logged_reward for r in logs]):.3f}", flush=True)

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
    gt_array = np.array([gt[r.id] for r in rules])
    out = {
        "n_logs": len(logs),
        "p_noise": args.p_noise,
        "n_rules": len(rules),
        "correction_rate": float(np.mean([r.correction for r in logs])),
        "noop_mean_reward": float(np.mean([r.logged_reward for r in logs])),
        "estimators": {},
    }
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
    rng2 = np.random.default_rng(42)
    n_boot = 80
    boot = {name: {"mse": [], "tau": [], "top10": []} for name in estimators}
    for b in range(n_boot):
        idx = rng2.integers(0, len(logs), size=len(logs))
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
    print(f"\n90% bootstrap CIs:", flush=True)
    for name in estimators:
        m = np.array(boot[name]["mse"]); t = np.array(boot[name]["tau"]); x = np.array(boot[name]["top10"])
        out["estimators"][name]["MSE_CI"] = [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))]
        out["estimators"][name]["tau_20_CI"] = [float(np.quantile(t, 0.05)), float(np.quantile(t, 0.95))]
        out["estimators"][name]["top10_CI"] = [float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))]
        print(f"  {name:14s}  MSE [{np.quantile(m,0.05):.5f}, {np.quantile(m,0.95):.5f}]  tau [{np.quantile(t,0.05):+.3f}, {np.quantile(t,0.95):+.3f}]  top10 [{np.quantile(x,0.05):.3f}, {np.quantile(x,0.95):.3f}]", flush=True)

    # Pairwise RuleOPE vs DR on MSE improvement
    mse_ratio = np.array(boot["RuleOPE"]["mse"]) / np.array(boot["DR"]["mse"])
    out["MSE_reduction_RuleOPE_vs_DR_pct_median"] = float(100 * (1 - np.median(mse_ratio)))
    out["MSE_reduction_RuleOPE_vs_DR_pct_CI"] = [float(100 * (1 - np.quantile(mse_ratio, 0.95))), float(100 * (1 - np.quantile(mse_ratio, 0.05)))]
    print(f"\nRuleOPE vs DR MSE reduction: {100*(1-np.median(mse_ratio)):+.1f}% (90% CI [{100*(1-np.quantile(mse_ratio, 0.95)):+.1f}%, {100*(1-np.quantile(mse_ratio, 0.05)):+.1f}%])")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/noreplay_ope_noisy.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
