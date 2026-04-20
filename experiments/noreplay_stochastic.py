"""Generator-efficient rule-OPE under stochastic logging on HotpotQA.

Setup:
  - Log: for each query we run the generator at EXACTLY ONE action
    sampled uniformly from {noop, filter, rerank}.
  - Logged tuple: (query, action, passages, LLM-answer, reward).
  - Oracle: for each query we have the generator output under ALL 3
    actions (from the prior 4479-prompt generation).  Used to compute
    ground-truth V(rho) and for MSE comparison -- never used by
    estimators.

Under this regime BOTH DR AND RuleOPE ARE UNBIASED.  The comparison
isolates the COMPOSITIONAL VARIANCE REDUCTION predicted by Theorem E
(semiparametric efficiency bound with atom-sharing): RuleOPE's ridge
regression shares coefficients across rules that share atoms, giving
strictly lower per-rule MSE than per-rule-independent DR.

This is the empirical regime where Theorem E's prediction is
directly measurable: no auxiliary signals, no correction-fusion
intervention, just pure variance reduction via compositional
structure.
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
from src.estimators.cips import CIPS
from src.estimators.cascade_dr import CascadeDR
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import JointRuleOPE

# OBP-style per-rule non-compositional DR baseline -- this is the
# actual SOTA for rule-OPE in classical OPE literature (Saito et al.
# Open Bandit Pipeline, Dudík-Langford-Li 2014 DR).  Every other
# compositional estimator in our framework (DM / DR / RuleOPE) uses
# the same atom-shared regression and therefore benefits from the
# same compositional variance reduction -- which is itself our
# paper's core contribution.  NonCompDR is the fair baseline.
from experiments.ablations import NonCompositionalDR
from src.logs import LoggedRecord
from src.rag_substrate_hotpot import (
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
    for lead in ("answer:", "a:", "the answer is"):
        if s.startswith(lead):
            s = s[len(lead):].strip()
    return s

def _f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split(); g = _normalize(gold).split()
    if not p or not g: return 0.0
    common = set(p) & set(g)
    if not common: return 0.0
    prec = len(common) / len(p); rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)

def _reward(gen: str, gold: str) -> float:
    return 0.0 if _normalize(gen) in ("unknown", "") else _f1(gen, gold)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="eval/hotpot/dev.parquet")
    ap.add_argument("--outputs", default="eval/hotpot/outputs.jsonl")
    ap.add_argument("--n_queries", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = _load_hotpot(args.hotpot, args.n_queries, args.seed)
    answers: dict[str, dict[str, str]] = {s.qid: {} for s in samples}
    with open(args.outputs) as f:
        for line in f:
            d = json.loads(line)
            qid, action = d["id"].split("__")
            if qid in answers:
                answers[qid][action] = d["text"]
    samples = [s for s in samples if len(answers[s.qid]) == 3]

    # Oracle rewards: run generator at all 3 actions.
    oracle = {
        s.qid: {a: _reward(answers[s.qid].get(a, ""), s.answer) for a in ("noop", "filter", "rerank")}
        for s in samples
    }
    # Abstain reward = 0.5 fixed.
    for s in samples:
        oracle[s.qid]["abstain"] = 0.5

    rng = np.random.default_rng(args.seed + 11)
    ACTIONS = ("noop", "filter", "rerank")

    # Stochastic logging: each query gets one action sampled uniformly.
    logs = []
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        a = ACTIONS[int(rng.integers(0, 3))]
        reward = oracle[s.qid][a]
        cf = dict(oracle[s.qid])
        logs.append(
            LoggedRecord(
                query_id=s.qid,
                ctx=ctx,
                logged_action=a,
                logged_propensity=1.0 / 3,
                logged_reward=float(reward),
                correction=0,
                cf_rewards=cf,
            )
        )
    print(f"  n_logs={len(logs)}", flush=True)
    from collections import Counter
    ac_counts = Counter(r.logged_action for r in logs)
    print(f"  action counts: {dict(ac_counts)}", flush=True)
    print(f"  reward mean: noop={np.mean([o['noop'] for o in oracle.values()]):.3f}  filter={np.mean([o['filter'] for o in oracle.values()]):.3f}  rerank={np.mean([o['rerank'] for o in oracle.values()]):.3f}", flush=True)

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
        # SOTA baselines from classical OPE literature -- no
        # compositional structure, per-rule or per-action regression.
        "IPS": IPS(),                      # Horvitz-Thompson, no regression
        "CIPS": CIPS(clip=20.0),           # Clipped IPS (Bottou 2013)
        "NonCompDR": NonCompositionalDR(), # OBP-style per-rule DR
        "CascadeDR": CascadeDR(),          # Kiyohara 2022 ranking DR
        # Our compositional family -- all share the atom-shared
        # reward regression (our core contribution).
        "CompDM":     DirectMethod(),
        "CompDR":     DoublyRobust(),
        "RuleOPE":    RuleOPE(),
        "JointRuleOPE": JointRuleOPE(),
    }
    gt_array = np.array([gt[r.id] for r in rules])
    out = {
        "n_logs": len(logs),
        "n_rules": len(rules),
        "logging": "uniform_stochastic_3_actions",
        "reward_type": "LLM_generator_F1",
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

    rng2 = np.random.default_rng(42)
    n_boot = 100
    boot = {name: {"mse": [], "tau": [], "top10": []} for name in estimators}
    print(f"\nBootstrap {n_boot} resamples + log re-randomization ...", flush=True)
    for b in range(n_boot):
        # Re-sample stochastic logging to isolate RuleOPE-vs-DR variance structure
        new_logs = []
        for s in samples:
            scores = _score_passages(s); ctx = _atom_features(s, scores)
            a = ACTIONS[int(rng2.integers(0, 3))]
            new_logs.append(LoggedRecord(
                query_id=s.qid, ctx=ctx, logged_action=a, logged_propensity=1/3,
                logged_reward=float(oracle[s.qid][a]), correction=0, cf_rewards=dict(oracle[s.qid])
            ))
        for name in estimators:
            est = estimators[name].__class__() if name != "JointRuleOPE" else JointRuleOPE()
            est.fit(new_logs)
            res = est.value_many(rules, new_logs)
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
        print(f"  {name:14s}  MSE [{np.quantile(m,0.05):.5f}, {np.quantile(m,0.95):.5f}]  tau [{np.quantile(t,0.05):+.3f}, {np.quantile(t,0.95):+.3f}]", flush=True)

    # Paired comparison: compositional estimators vs SOTA non-compositional DR
    for cmp in ("CompDR", "RuleOPE", "JointRuleOPE"):
        ratio = np.array(boot[cmp]["mse"]) / np.array(boot["NonCompDR"]["mse"])
        pct = 100 * (1 - ratio)
        out[f"MSE_reduction_{cmp}_vs_NonCompDR_pct_median"] = float(np.median(pct))
        out[f"MSE_reduction_{cmp}_vs_NonCompDR_pct_CI"] = [float(np.quantile(pct, 0.05)), float(np.quantile(pct, 0.95))]
        print(f"  {cmp} vs NonCompDR MSE reduction: median {np.median(pct):+.1f}%  (90% CI [{np.quantile(pct, 0.05):+.1f}%, {np.quantile(pct, 0.95):+.1f}%])")
    for cmp in ("CompDR", "RuleOPE", "JointRuleOPE"):
        ratio = np.array(boot[cmp]["mse"]) / np.array(boot["IPS"]["mse"])
        pct = 100 * (1 - ratio)
        out[f"MSE_reduction_{cmp}_vs_IPS_pct_median"] = float(np.median(pct))
        print(f"  {cmp} vs IPS MSE reduction: median {np.median(pct):+.1f}%")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/noreplay_stochastic.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
