"""No-replay RAG-OPE evaluation on HotpotQA with real Mistral-7B outputs.

Pipeline
--------
Prerequisites:
  * eval/hotpot/prompts.jsonl    -- built by build_noreplay_prompts.py
  * eval/hotpot/outputs.jsonl    -- generator outputs from lambda_generate.py

For each query and action in {noop, filter, rerank} we have a real LLM
answer and can compute the "oracle" reward via F1(generated, gold).
Abstain is treated as a fixed reward of 0.5.

The estimators' goal: estimate V(rho) from LOGGED TUPLES ONLY
(q_i, r_i = noop, y_i, R_i), WITHOUT running the generator at
counterfactual retrievals.  The oracle is the full [N x 4] cf-reward
matrix, used only to compute ground truth V(rho) and for the MSE
comparison.

Each estimator's performance tells us how closely it approximates the
oracle "re-run the generator" ground truth.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import kendalltau

from src.estimators.direct_method import DirectMethod
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.ips import IPS
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators.shrinkage import JointRuleOPE
from src.logs import LoggedRecord
from src.rag_substrate_hotpot import (
    _apply_rule,
    _atom_features,
    _load_hotpot,
    _score_passages,
)
from src.rule_dsl import load_rules


_WS = re.compile(r"[^\w\s]")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = _WS.sub(" ", s)
    s = " ".join(s.split())
    # Strip stock QA prefixes and the abstain token.
    for lead in ("answer:", "a:", "the answer is", "the answer"):
        if s.startswith(lead):
            s = s[len(lead):].strip()
    return s


def _f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def _reward_for_answer(generated: str, gold: str) -> float:
    """F1 between generated and gold answer, with UNKNOWN -> 0."""
    if _normalize(generated) in ("unknown", ""):
        return 0.0
    return _f1(generated, gold)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="eval/hotpot/dev.parquet")
    ap.add_argument("--outputs", default="eval/hotpot/outputs.jsonl")
    ap.add_argument("--n_queries", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = _load_hotpot(args.hotpot, args.n_queries, args.seed)
    by_qid = {s.qid: s for s in samples}

    print(f"Loading generator outputs from {args.outputs}", flush=True)
    answers: dict[str, dict[str, str]] = {s.qid: {} for s in samples}
    with open(args.outputs) as f:
        for line in f:
            d = json.loads(line)
            qid, action = d["id"].split("__")
            if qid in answers:
                answers[qid][action] = d["text"]
    n_have = sum(1 for q in answers if len(answers[q]) == 3)
    print(f"  {n_have}/{len(samples)} queries have all 3 action outputs", flush=True)
    # Drop partial samples
    samples = [s for s in samples if len(answers[s.qid]) == 3]

    # Build oracle reward matrix and LoggedRecord list.
    abstain_reward = 0.5
    logs = []
    oracle = {}   # qid -> {action: reward}
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        cf = {"abstain": abstain_reward}
        for action in ("noop", "filter", "rerank"):
            gen = answers[s.qid].get(action, "")
            cf[action] = _reward_for_answer(gen, s.answer)
        oracle[s.qid] = dict(cf)
        # Logged tuple = noop.  Correction: reward < 0.5.
        logged_reward = float(cf["noop"])
        correction = 1 if logged_reward < 0.5 else 0
        logs.append(
            LoggedRecord(
                query_id=s.qid,
                ctx=ctx,
                logged_action="noop",
                logged_propensity=1.0,
                logged_reward=logged_reward,
                correction=correction,
                cf_rewards=cf,
            )
        )

    print(f"  {len(logs)} logged tuples, correction rate = {np.mean([r.correction for r in logs]):.3f}", flush=True)
    print(f"  noop mean reward = {np.mean([r.logged_reward for r in logs]):.3f}", flush=True)
    # Oracle rule value: V(rule) = E[ reward under rule action on firing queries, noop reward otherwise ]
    rules = load_rules("eval/rules_v1.jsonl")
    rules = [r for r in rules if r.action in ("filter", "rerank", "abstain")]
    def oracle_value(rule):
        vals = []
        for rec in logs:
            if rule.fires(rec.ctx):
                vals.append(rec.cf_rewards[rule.action])
            else:
                vals.append(rec.cf_rewards["noop"])
        return float(np.mean(vals))
    gt = {r.id: oracle_value(r) for r in rules}

    from src.estimators._regression import fires_mask
    firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules}
    rules = [r for r in rules if 0.05 <= firing[r.id] <= 0.95]
    print(f"  {len(rules)} rules after firing-rate filter", flush=True)
    gt = {r.id: gt[r.id] for r in rules}

    print(f"  V(rho) range: [{min(gt.values()):.3f}, {max(gt.values()):.3f}]", flush=True)

    # Build STOCHASTIC version of logs for IPS/DR (they need non-degenerate
    # propensities).  This is a separate log set -- in the no-replay setting
    # we'd need real stochastic logging to use these, so we include them for
    # reference only.  The primary comparison is on deterministic logs.

    estimators = {
        "DM": DirectMethod(),
        "DR": DoublyRobust(),
        "RuleOPE": RuleOPE(),
        "JointRuleOPE": JointRuleOPE(),
    }

    gt_array = np.array([gt[r.id] for r in rules])
    out = {
        "n_logs": len(logs),
        "n_rules": len(rules),
        "correction_rate": float(np.mean([r.correction for r in logs])),
        "oracle_reward_range": [float(min(gt.values())), float(max(gt.values()))],
        "logging": "deterministic_noop",
        "estimators": {},
    }
    for name, est in estimators.items():
        print(f"Running {name} ...", flush=True)
        t0 = time.time()
        est.fit(logs)
        res = est.value_many(rules, logs)
        rt = time.time() - t0
        est_vals = np.array([res[r.id].estimate for r in rules])
        mse = float(np.mean((est_vals - gt_array) ** 2))
        bias = float(np.mean(est_vals - gt_array))
        tau20, _ = kendalltau(np.argsort(-est_vals)[:20], np.argsort(-gt_array)[:20])
        # Top-10 oracle-value of selection
        ranked = sorted(rules, key=lambda r: -res[r.id].estimate)[:10]
        avg_top10 = float(np.mean([gt[r.id] for r in ranked]))
        out["estimators"][name] = {
            "MSE": mse,
            "bias": bias,
            "tau_top20": float(tau20) if tau20 == tau20 else 0.0,
            "avg_GT_value_top10": avg_top10,
            "runtime_s": rt,
        }
        print(f"  {name:14s}  MSE={mse:.5f}  bias={bias:+.4f}  tau@20={tau20:+.3f}  top10_GT={avg_top10:.3f}  t={rt:.1f}s", flush=True)

    out["oracle_top10_value"] = float(np.mean(sorted(gt.values(), reverse=True)[:10]))
    print(f"\nOracle top-10 GT value: {out['oracle_top10_value']:.3f}", flush=True)

    # Bootstrap CIs
    print("\nBootstrap 60 resamples ...", flush=True)
    rng = np.random.default_rng(42)
    n_boot = 60
    boot_results = {name: {"mse": [], "tau_20": [], "top10": []} for name in estimators}
    for b in range(n_boot):
        idx = rng.integers(0, len(logs), size=len(logs))
        boot_logs = [logs[int(i)] for i in idx]
        for name in estimators:
            est = estimators[name].__class__() if name != "JointRuleOPE" else JointRuleOPE()
            est.fit(boot_logs)
            res = est.value_many(rules, boot_logs)
            est_vals = np.array([res[r.id].estimate for r in rules])
            boot_results[name]["mse"].append(float(np.mean((est_vals - gt_array) ** 2)))
            tau_b, _ = kendalltau(np.argsort(-est_vals)[:20], np.argsort(-gt_array)[:20])
            boot_results[name]["tau_20"].append(float(tau_b) if tau_b == tau_b else 0.0)
            ranked = sorted(rules, key=lambda r: -res[r.id].estimate)[:10]
            boot_results[name]["top10"].append(float(np.mean([gt[r.id] for r in ranked])))
    print("\n90% bootstrap CIs:", flush=True)
    for name in estimators:
        m = np.array(boot_results[name]["mse"])
        t = np.array(boot_results[name]["tau_20"])
        x = np.array(boot_results[name]["top10"])
        out["estimators"][name]["MSE_CI"] = [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))]
        out["estimators"][name]["tau_20_CI"] = [float(np.quantile(t, 0.05)), float(np.quantile(t, 0.95))]
        out["estimators"][name]["top10_CI"] = [float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))]
        print(f"  {name:14s}  MSE ∈ [{np.quantile(m, 0.05):.5f}, {np.quantile(m, 0.95):.5f}]  tau ∈ [{np.quantile(t, 0.05):+.3f}, {np.quantile(t, 0.95):+.3f}]  top10 ∈ [{np.quantile(x, 0.05):.3f}, {np.quantile(x, 0.95):.3f}]", flush=True)

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/noreplay_ope.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote experiments/results/noreplay_ope.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
