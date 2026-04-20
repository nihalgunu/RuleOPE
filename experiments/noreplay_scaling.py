"""RuleOPE scaling study: compositional vs non-compositional as a
function of the effective sample size per rule.

The theorem's variance-reduction bound is asymptotically of order
$O(\|\beta\|^2 / N_\text{eff})$, where $N_\text{eff}$ is the effective
sample size per rule (firing queries × propensity).  Non-compositional
DR (OBP-style per-rule ridge) has this full sample-size penalty;
compositional RuleOPE shares coefficients across rules, reducing the
effective problem dimension from M × d to d and recovering
$O(d / N)$ scaling.

We verify this on HotpotQA by varying N and measuring MSE of both
estimators against the oracle.  Expected: MSE gap widens as N shrinks.
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

from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.estimators.shrinkage import JointRuleOPE
from src.logs import LoggedRecord
from src.rag_substrate_hotpot import (
    _atom_features,
    _load_hotpot,
    _score_passages,
)
from src.rule_dsl import load_rules
from experiments.ablations import NonCompositionalDR


_WS = re.compile(r"[^\w\s]")

def _normalize(s: str) -> str:
    s = s.lower().strip(); s = _WS.sub(" ", s); s = " ".join(s.split())
    for lead in ("answer:", "a:", "the answer is"):
        if s.startswith(lead): s = s[len(lead):].strip()
    return s

def _f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split(); g = _normalize(gold).split()
    if not p or not g: return 0.0
    c = set(p) & set(g)
    if not c: return 0.0
    return 2 * len(c) / len(p) * len(c) / len(g) / (len(c) / len(p) + len(c) / len(g))

def _reward(gen: str, gold: str) -> float:
    return 0.0 if _normalize(gen) in ("unknown", "") else _f1(gen, gold)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="eval/hotpot/dev.parquet")
    ap.add_argument("--outputs", default="eval/hotpot/outputs_1500.jsonl")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--Ns", nargs="+", type=int, default=[150, 300, 600, 1200])
    args = ap.parse_args()

    # Load ALL samples and generator outputs once.
    all_samples = _load_hotpot(args.hotpot, 1500, 0)
    answers: dict[str, dict[str, str]] = {s.qid: {} for s in all_samples}
    with open(args.outputs) as f:
        for line in f:
            d = json.loads(line)
            qid, action = d["id"].split("__")
            if qid in answers:
                answers[qid][action] = d["text"]
    all_samples = [s for s in all_samples if len(answers[s.qid]) == 3]
    print(f"Loaded {len(all_samples)} samples with full generator coverage", flush=True)

    oracle = {
        s.qid: {a: _reward(answers[s.qid].get(a, ""), s.answer) for a in ("noop", "filter", "rerank")}
        for s in all_samples
    }
    for s in all_samples:
        oracle[s.qid]["abstain"] = 0.5

    rules = load_rules("eval/rules_v1.jsonl")
    rules = [r for r in rules if r.action in ("filter", "rerank", "abstain")]

    ACTIONS = ("noop", "filter", "rerank")
    out = {"n_samples_total": len(all_samples), "scaling": {}}
    for N in args.Ns:
        print(f"\n=== N = {N} ===", flush=True)
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
            def oracle_value(rule):
                return float(np.mean([rec.cf_rewards[rule.action] if rule.fires(rec.ctx) else rec.cf_rewards["noop"] for rec in logs]))
            from src.estimators._regression import fires_mask
            # Use same rules across trials of same N; filter by firing rate
            firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules}
            tr_rules = [r for r in rules if 0.05 <= firing[r.id] <= 0.95]
            if len(tr_rules) < 10:
                continue
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
        # Print MSE-reduction vs NonCompDR for each compositional method
        ncd = np.array(per_N["NonCompDR"])
        for name in ("CompDR", "RuleOPE", "JointRuleOPE"):
            m = np.array(per_N[name])
            pct = 100 * (1 - m / ncd)
            out["scaling"][str(N)][f"{name}_vs_NonCompDR_pct"] = {
                "median": float(np.median(pct)),
                "mean": float(pct.mean()),
                "CI90": [float(np.quantile(pct, 0.05)), float(np.quantile(pct, 0.95))],
            }
            print(f"    {name} vs NonCompDR: median {np.median(pct):+.1f}%  (90% CI [{np.quantile(pct, 0.05):+.1f}%, {np.quantile(pct, 0.95):+.1f}%])")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/noreplay_scaling.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
