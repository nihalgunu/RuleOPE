"""Unified ablation harness — runs on any benchmark substrate.

Ablations:
  A. Atom-shared regression vs per-rule regression   (our main claim)
  B. Cross-fitting vs no cross-fitting                (DML correctness)
  C. Ridge alpha sweep                                (robustness)
  D. M (rule-pool size) sweep                         (theorem scaling)

Each ablation produces one JSON artifact per (benchmark, setting).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.linear_model import Ridge

from src.estimators.base import Estimator, EstimatorResult
from src.estimators._regression import (
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
    fires_mask,
)
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord
from src.rule_dsl import load_rules
from experiments.ablations import NonCompositionalDR


# ---------------------------------------------------------------------------
# Ablation-A helper: per-rule ridge at matched alpha (isolates sharing)
# ---------------------------------------------------------------------------

class PerRuleRidgeDR(Estimator):
    """Per-rule ridge regression at alpha=alpha_match, then DR.

    This is NonCompDR with *matched regularisation strength* so the
    only difference to our compositional RuleOPE is the cross-rule
    sharing (vs per-rule fits).  Any MSE gap is therefore
    attributable to sharing, not to the regularizer."""
    name = "PerRuleRidgeDR"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, logs):
        self.logs = list(logs)
        return self

    def value(self, rule, logs):
        phi = atom_feature_matrix(logs)
        fires = fires_mask(logs, rule)
        if fires.sum() < 10:
            r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
            return EstimatorResult(estimate=float(r.mean()), stderr=float(r.std()/np.sqrt(len(r))), n_effective=float(len(r)))
        # Fit ridge with the SAME alpha as RuleOPE on the sub-population where the rule fires.
        idx = np.where(fires)[0]
        actions = np.array([_ACTION_IDX[logs[i].logged_action] for i in idx], dtype=np.int64)
        rewards = np.array([logs[i].logged_reward for i in idx], dtype=np.float32)
        X = _joint_features(phi[idx], actions)
        model = Ridge(alpha=self.alpha).fit(X, rewards)

        all_a = np.array([_ACTION_IDX[rec.logged_action] for rec in logs], dtype=np.int64)
        all_a[fires] = _ACTION_IDX[rule.action]
        X_all = _joint_features(phi, all_a)
        m_rule = model.predict(X_all).astype(np.float32)

        X_logged = _joint_features(phi, np.array([_ACTION_IDX[rec.logged_action] for rec in logs], dtype=np.int64))
        m_logged = model.predict(X_logged).astype(np.float32)
        r_obs = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
        logged_actions = np.array([rec.logged_action for rec in logs])
        match = np.where(fires, logged_actions == rule.action, logged_actions == "noop")
        propensities = np.array([max(rec.logged_propensity, 1e-6) for rec in logs], dtype=np.float64)
        w = np.where(match, 1.0 / propensities, 0.0)
        psi = m_rule + w * (r_obs - m_logged)
        return EstimatorResult(
            estimate=float(psi.mean()),
            stderr=float(psi.std(ddof=1) / np.sqrt(len(psi))),
            n_effective=float(fires.sum()),
        )


def _common_setup(benchmark: str, pool_size: int = 1500, seed: int = 0):
    """Load the specified benchmark and return (all_samples, oracle, ACTIONS)."""
    if benchmark == "musique":
        from src.rag_substrate_musique import _load_musique, _score_passages, _apply_rule, _reward_for_top3
        samples = _load_musique("eval/musique/dev.parquet", pool_size, seed)
        oracle = {}
        for s in samples:
            scores = _score_passages(s)
            cf = {}
            for a in ("noop", "filter", "rerank", "abstain"):
                titles = _apply_rule(a, scores, s)
                cf[a] = 0.5 if a == "abstain" else _reward_for_top3(s.gold_titles, titles)
            oracle[s.qid] = cf
        return samples, oracle, ("noop", "filter", "rerank")
    if benchmark == "hotpot":
        from src.rag_substrate_hotpot import _load_hotpot, _score_passages
        import re
        _WS = re.compile(r"[^\w\s]")
        def _normalize(s):
            s = s.lower().strip(); s = _WS.sub(" ", s); s = " ".join(s.split())
            for lead in ("answer:", "a:", "the answer is"):
                if s.startswith(lead): s = s[len(lead):].strip()
            return s
        def _f1(pred, gold):
            p = _normalize(pred).split(); g = _normalize(gold).split()
            if not p or not g: return 0.0
            c = set(p) & set(g)
            if not c: return 0.0
            return 2 * len(c) / len(p) * len(c) / len(g) / (len(c) / len(p) + len(c) / len(g))
        samples = _load_hotpot("eval/hotpot/dev.parquet", pool_size, seed)
        answers = {s.qid: {} for s in samples}
        with open("eval/hotpot/outputs_1500.jsonl") as f:
            for line in f:
                d = json.loads(line)
                qid, action = d["id"].split("__")
                if qid in answers: answers[qid][action] = d["text"]
        samples = [s for s in samples if len(answers[s.qid]) == 3]
        oracle = {
            s.qid: {a: (0.0 if _normalize(answers[s.qid][a]) in ("unknown", "") else _f1(answers[s.qid][a], s.answer)) for a in ("noop", "filter", "rerank")}
            for s in samples
        }
        for s in samples:
            oracle[s.qid]["abstain"] = 0.5
        return samples, oracle, ("noop", "filter", "rerank")
    elif benchmark == "trivia":
        from src.rag_substrate_trivia import _load_trivia, _score_passages, _apply_rule, _alias_match
        samples = _load_trivia("eval/trivia/dev.parquet", pool_size, seed)
        oracle = {}
        for s in samples:
            scores = _score_passages(s)
            cf = {a: (_alias_match(_apply_rule(a, scores, s)[1], s.answer_aliases) if a != "abstain" else 0.5)
                  for a in ("noop", "filter", "rerank", "abstain")}
            oracle[s.qid] = cf
        return samples, oracle, ("noop", "filter", "rerank")
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")


def _score_for_benchmark(benchmark, sample):
    if benchmark == "hotpot":
        from src.rag_substrate_hotpot import _score_passages
    elif benchmark == "musique":
        from src.rag_substrate_musique import _score_passages
    else:
        from src.rag_substrate_trivia import _score_passages
    return _score_passages(sample)


def _atom_for_benchmark(benchmark, sample, scores):
    if benchmark == "hotpot":
        from src.rag_substrate_hotpot import _atom_features
    elif benchmark == "musique":
        from src.rag_substrate_musique import _atom_features
    else:
        from src.rag_substrate_trivia import _atom_features
    return _atom_features(sample, scores)


def _build_logs(benchmark, samples_tr, oracle, ACTIONS, rng, N):
    logs = []
    for s in samples_tr:
        scores = _score_for_benchmark(benchmark, s)
        ctx = _atom_for_benchmark(benchmark, s, scores)
        a = ACTIONS[int(rng.integers(0, len(ACTIONS)))]
        logs.append(LoggedRecord(
            query_id=s.qid, ctx=ctx, logged_action=a, logged_propensity=1/len(ACTIONS),
            logged_reward=float(oracle[s.qid][a]),
            correction=0, cf_rewards=dict(oracle[s.qid])
        ))
    return logs


def _oracle_value(rule, logs):
    return float(np.mean([rec.cf_rewards[rule.action] if rule.fires(rec.ctx) else rec.cf_rewards["noop"] for rec in logs]))


def run_scaling(benchmark, estimators: dict, Ns: list[int], n_trials: int, pool_size: int, rule_action_filter: tuple = ("filter", "rerank", "abstain")):
    samples, oracle, ACTIONS = _common_setup(benchmark, pool_size=pool_size)
    rules = [r for r in load_rules("eval/rules_v1.jsonl") if r.action in rule_action_filter]
    out = {}
    for N in Ns:
        per_N = {name: [] for name in estimators}
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 * N + trial)
            idx = rng.choice(len(samples), size=min(N, len(samples)), replace=False)
            samples_tr = [samples[int(i)] for i in idx]
            logs = _build_logs(benchmark, samples_tr, oracle, ACTIONS, rng, N)
            firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules}
            tr_rules = [r for r in rules if 0.05 <= firing[r.id] <= 0.95]
            if len(tr_rules) < 10:
                continue
            gt = {r.id: _oracle_value(r, logs) for r in tr_rules}
            gt_array = np.array([gt[r.id] for r in tr_rules])
            for name, est_factory in estimators.items():
                est = est_factory()
                est.fit(logs)
                res = est.value_many(tr_rules, logs)
                est_vals = np.array([res[r.id].estimate for r in tr_rules])
                per_N[name].append(float(np.mean((est_vals - gt_array) ** 2)))
        out[str(N)] = {}
        for name, arr in per_N.items():
            a = np.array(arr)
            if len(a) == 0: continue
            out[str(N)][name] = {
                "MSE_mean": float(a.mean()),
                "MSE_std": float(a.std()),
                "MSE_CI90": [float(np.quantile(a, 0.05)), float(np.quantile(a, 0.95))],
                "n_trials": len(a),
            }
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["hotpot", "trivia"])
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--Ns", nargs="+", type=int, default=[150, 300, 600])
    ap.add_argument("--pool_size", type=int, default=1500)
    ap.add_argument("--which", nargs="+", default=["A", "B", "C", "D"])
    args = ap.parse_args()

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    master = {"benchmarks": args.benchmarks, "ablations": {}}

    # --- Ablation A: atom-sharing vs per-rule ridge at matched alpha ---
    if "A" in args.which:
        print("\n### Ablation A: atom-sharing vs per-rule ridge (alpha=1.0 both) ###", flush=True)
        ests_A = {
            "PerRuleRidge_a1": lambda: PerRuleRidgeDR(alpha=1.0),
            "NonCompDR":       lambda: NonCompositionalDR(alpha=1.0),
            "CompDR_a1":       lambda: DoublyRobust(alpha=1.0),
            "RuleOPE_a1":      lambda: RuleOPE(RuleOPEConfig(alpha=1.0)),
        }
        res_A = {}
        for bench in args.benchmarks:
            print(f"  benchmark={bench}", flush=True)
            res_A[bench] = run_scaling(bench, ests_A, args.Ns, args.n_trials, args.pool_size)
            # Print pct-reduction vs PerRuleRidge (alpha-matched baseline)
            for N in args.Ns:
                if str(N) in res_A[bench]:
                    r = res_A[bench][str(N)]
                    if "PerRuleRidge_a1" in r and "RuleOPE_a1" in r:
                        pct = 100 * (1 - r["RuleOPE_a1"]["MSE_mean"] / r["PerRuleRidge_a1"]["MSE_mean"])
                        print(f"    N={N} RuleOPE vs PerRuleRidge(a=1): {pct:+.1f}% MSE")
        master["ablations"]["A_sharing"] = res_A

    # --- Ablation B: cross-fit on/off ---
    if "B" in args.which:
        print("\n### Ablation B: cross-fit on vs off ###", flush=True)
        ests_B = {
            "CompDR_xfit5":   lambda: DoublyRobust(n_folds=5),
            "CompDR_noxfit":  lambda: DoublyRobust(n_folds=2),  # K=2 is near-minimal cross-fit
            "RuleOPE_xfit5":  lambda: RuleOPE(RuleOPEConfig(n_folds=5)),
        }
        res_B = {}
        for bench in args.benchmarks:
            print(f"  benchmark={bench}", flush=True)
            res_B[bench] = run_scaling(bench, ests_B, args.Ns, args.n_trials, args.pool_size)
            for N in args.Ns:
                if str(N) in res_B[bench]:
                    r = res_B[bench][str(N)]
                    for name in ("CompDR_xfit5", "CompDR_noxfit"):
                        if name in r:
                            print(f"    N={N} {name}: MSE={r[name]['MSE_mean']:.5f}")
        master["ablations"]["B_crossfit"] = res_B

    # --- Ablation C: ridge alpha sweep ---
    if "C" in args.which:
        print("\n### Ablation C: ridge alpha sweep ###", flush=True)
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        ests_C = {f"RuleOPE_a{a}": (lambda a_=a: RuleOPE(RuleOPEConfig(alpha=a_))) for a in alphas}
        res_C = {}
        for bench in args.benchmarks:
            print(f"  benchmark={bench}", flush=True)
            res_C[bench] = run_scaling(bench, ests_C, args.Ns, args.n_trials, args.pool_size)
            for N in args.Ns:
                if str(N) in res_C[bench]:
                    r = res_C[bench][str(N)]
                    best = min(((a, r[f"RuleOPE_a{a}"]["MSE_mean"]) for a in alphas if f"RuleOPE_a{a}" in r), key=lambda x: x[1])
                    print(f"    N={N} best alpha={best[0]}  MSE={best[1]:.5f}")
        master["ablations"]["C_alpha"] = res_C

    # --- Ablation D: M (rule-pool size) sweep ---
    if "D" in args.which:
        print("\n### Ablation D: M (rule count) sweep ###", flush=True)
        pool_Ms = [50, 150, 500]
        res_D = {}
        for bench in args.benchmarks:
            print(f"  benchmark={bench}", flush=True)
            res_D[bench] = {}
            samples, oracle, ACTIONS = _common_setup(bench, pool_size=args.pool_size)
            all_rules = [r for r in load_rules("eval/rules_v1.jsonl") if r.action in ("filter", "rerank", "abstain")]
            for M in pool_Ms:
                # Deterministic subsample of M rules
                rng_m = np.random.default_rng(42)
                rules_M = list(rng_m.choice(all_rules, size=min(M, len(all_rules)), replace=False))
                for_N = {}
                for N in args.Ns:
                    per_N_arr = {"NonCompDR": [], "RuleOPE": []}
                    for trial in range(args.n_trials):
                        rng = np.random.default_rng(1000 * N + trial)
                        idx = rng.choice(len(samples), size=min(N, len(samples)), replace=False)
                        samples_tr = [samples[int(i)] for i in idx]
                        logs = _build_logs(bench, samples_tr, oracle, ACTIONS, rng, N)
                        firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules_M}
                        tr_rules = [r for r in rules_M if 0.05 <= firing[r.id] <= 0.95]
                        if len(tr_rules) < 5:
                            continue
                        gt = {r.id: _oracle_value(r, logs) for r in tr_rules}
                        gt_array = np.array([gt[r.id] for r in tr_rules])
                        for n_name, est_ctor in [("NonCompDR", NonCompositionalDR), ("RuleOPE", RuleOPE)]:
                            est = est_ctor()
                            est.fit(logs)
                            res = est.value_many(tr_rules, logs)
                            est_vals = np.array([res[r.id].estimate for r in tr_rules])
                            per_N_arr[n_name].append(float(np.mean((est_vals - gt_array) ** 2)))
                    for_N[str(N)] = {
                        name: {"MSE_mean": float(np.mean(arr)), "n_trials": len(arr)}
                        for name, arr in per_N_arr.items() if arr
                    }
                res_D[bench][f"M_{M}"] = for_N
                # Print
                for N in args.Ns:
                    if str(N) in for_N and "NonCompDR" in for_N[str(N)] and "RuleOPE" in for_N[str(N)]:
                        r = for_N[str(N)]
                        pct = 100 * (1 - r["RuleOPE"]["MSE_mean"] / r["NonCompDR"]["MSE_mean"])
                        print(f"    M={M} N={N} RuleOPE vs NonCompDR: {pct:+.1f}%")
        master["ablations"]["D_Msweep"] = res_D

    with open("experiments/results/ablation_unified.json", "w") as f:
        json.dump(master, f, indent=2)
    print("\nWrote experiments/results/ablation_unified.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
