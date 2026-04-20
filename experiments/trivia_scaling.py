"""TriviaQA scaling study: RuleOPE vs SOTA NonCompDR.

Mirror of experiments/noreplay_scaling.py on the TriviaQA
rc.wikipedia benchmark.  Uses alias-match rewards
(answer string appears in top-3 retrieved passages), so no LLM
generation is required.

Hypothesis: compositional RuleOPE's advantage over OBP-style
NonCompDR grows as N shrinks, same as HotpotQA.  A consistent
trend across two benchmarks strengthens the NeurIPS claim.
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

from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import JointRuleOPE
from src.logs import LoggedRecord
from src.rag_substrate_trivia import (
    _alias_match,
    _apply_rule,
    _atom_features,
    _load_trivia,
    _score_passages,
)
from src.rule_dsl import load_rules
from experiments.ablations import NonCompositionalDR


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trivia", default="eval/trivia/dev.parquet")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--Ns", nargs="+", type=int, default=[150, 300, 600, 1200])
    ap.add_argument("--pool_size", type=int, default=1500)
    args = ap.parse_args()

    print(f"Loading TriviaQA pool ({args.pool_size}) ...", flush=True)
    all_samples = _load_trivia(args.trivia, args.pool_size, 0)
    print(f"  {len(all_samples)} usable samples (≥4 passages)", flush=True)

    # Oracle rewards: for each sample, compute cf under each action.
    oracle = {}
    for s in all_samples:
        scores = _score_passages(s)
        cf = {}
        for a in ("noop", "filter", "rerank"):
            _, bodies = _apply_rule(a, scores, s)
            cf[a] = _alias_match(bodies, s.answer_aliases)
        cf["abstain"] = 0.5
        oracle[s.qid] = cf

    print(f"  mean rewards: noop={np.mean([o['noop'] for o in oracle.values()]):.3f}  filter={np.mean([o['filter'] for o in oracle.values()]):.3f}  rerank={np.mean([o['rerank'] for o in oracle.values()]):.3f}", flush=True)

    rules = load_rules("eval/rules_v1.jsonl")
    rules = [r for r in rules if r.action in ("filter", "rerank", "abstain")]

    ACTIONS = ("noop", "filter", "rerank")
    out = {"dataset": "TriviaQA_rc_wiki", "n_samples_pool": len(all_samples), "scaling": {}}
    for N in args.Ns:
        print(f"\n=== TriviaQA N = {N} ===", flush=True)
        per_N = {name: [] for name in ("NonCompDR", "CompDR", "RuleOPE", "JointRuleOPE")}
        top10_per_N = {name: [] for name in per_N}
        for trial in range(args.n_trials):
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
                return float(np.mean([rec.cf_rewards[rule.action] if rule.fires(rec.ctx) else rec.cf_rewards["noop"] for rec in logs]))
            gt = {r.id: oracle_value(r) for r in tr_rules}
            gt_array = np.array([gt[r.id] for r in tr_rules])
            ests = {
                "NonCompDR": NonCompositionalDR(),
                "CompDR": DoublyRobust(),
                "RuleOPE": RuleOPE(),
                "JointRuleOPE": JointRuleOPE(),
            }
            for name, est in ests.items():
                est.fit(logs)
                res = est.value_many(tr_rules, logs)
                est_vals = np.array([res[r.id].estimate for r in tr_rules])
                per_N[name].append(float(np.mean((est_vals - gt_array) ** 2)))
                ranked = sorted(tr_rules, key=lambda r: -res[r.id].estimate)[:10]
                top10_per_N[name].append(float(np.mean([gt[r.id] for r in ranked])))
        out["scaling"][str(N)] = {}
        for name in per_N:
            m = np.array(per_N[name]); t10 = np.array(top10_per_N[name])
            out["scaling"][str(N)][name] = {
                "MSE_mean": float(m.mean()),
                "MSE_std": float(m.std()),
                "MSE_CI90": [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))],
                "top10_mean": float(t10.mean()),
            }
            print(f"  {name:14s}  MSE={m.mean():.5f} ± {m.std():.5f}  top10_GT={t10.mean():.3f}", flush=True)
        ncd = np.array(per_N["NonCompDR"])
        for name in ("CompDR", "RuleOPE", "JointRuleOPE"):
            m = np.array(per_N[name])
            pct = 100 * (1 - m / ncd)
            out["scaling"][str(N)][f"{name}_vs_NonCompDR_pct"] = {
                "median": float(np.median(pct)),
                "mean": float(pct.mean()),
                "CI90": [float(np.quantile(pct, 0.05)), float(np.quantile(pct, 0.95))],
            }
            sig = " ✓" if np.quantile(pct, 0.05) > 0 else ""
            print(f"    {name} vs NonCompDR: median {np.median(pct):+.1f}%  (90% CI [{np.quantile(pct, 0.05):+.1f}%, {np.quantile(pct, 0.95):+.1f}%]){sig}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/trivia_scaling.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote experiments/results/trivia_scaling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
