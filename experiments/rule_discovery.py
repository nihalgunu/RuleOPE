"""Rule discovery: RuleOPE as a rule-learning framework (not just an estimator).

Pipeline (E3 in the paper's strong-accept plan):

    enumerate depth-<=2 rules
        -> sample a log of N queries under stochastic logging
        -> fit RuleOPE -> per-rule (V_hat, SE_hat) for all candidates
        -> four selectors pick top-k rules each:
             * CRRM-LCB (atom-aware pessimism, src/crrm.py)
             * ERM-argmax (argmax of V_hat)
             * Random
             * HandAuthoredBest (top-k over eval/rules_v1.jsonl by V_hat)
             plus HandAuthoredOracle (skyline: top-k over hand rules by
             the oracle V*, i.e. the best that a human-curated set can
             achieve)
        -> compute oracle V*(rho) on a held-out eval split
        -> report simple regret @ k = V*(oracle_best) - V*(selected_top).

Claim tested: CRRM-LCB on the *enumerated* rule space finds rules with
oracle value >= the oracle-best *hand-authored* rule on HotpotQA, i.e.
auto-discovery beats human-written ones at matched k.

Run:
    python3 experiments/rule_discovery.py --n_trials 25 --n_train 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.crrm import PessimisticConfig, PessimisticRuleSelector
from src.estimators.rule_ope import RuleOPE
from src.logs import LoggedRecord
from src.rag_substrate_hotpot import (
    _apply_rule,
    _atom_features,
    _load_hotpot,
    _reward_for_top3,
    _score_passages,
)
from src.rule_dsl import ATOMS, Rule, enumerate_rules, load_rules


ACTIONS_LOG = ("noop", "filter", "rerank")
ACTIONS_RULE = ("filter", "rerank", "abstain")  # candidate actions for discovered rules


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _precompute_samples(pool_size: int, pool_seed: int):
    """Load HotpotQA pool once and cache per-query cf_rewards + ctx."""
    samples = _load_hotpot("eval/hotpot/dev.parquet", pool_size, pool_seed)
    rows = []
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        cf = {}
        for a in ("noop", "filter", "rerank"):
            titles = _apply_rule(a, scores, s)
            cf[a] = _reward_for_top3(s.gold_titles, titles)
        cf["abstain"] = 0.5
        rows.append({"qid": s.qid, "ctx": ctx, "cf": cf})
    return rows


def _oracle_value(rule: Rule, rows: Sequence[dict]) -> float:
    """Exact V(rho) on the given pool via counterfactual replay."""
    vals = []
    for r in rows:
        if rule.fires(r["ctx"]):
            vals.append(r["cf"][rule.action])
        else:
            vals.append(r["cf"]["noop"])
    return float(np.mean(vals))


def _build_logs(rows: Sequence[dict], rng: np.random.Generator) -> list[LoggedRecord]:
    """Uniform-stochastic logging over ACTIONS_LOG (scenario (i) of Thm C, revised)."""
    logs = []
    for r in rows:
        a = ACTIONS_LOG[int(rng.integers(0, len(ACTIONS_LOG)))]
        logs.append(
            LoggedRecord(
                query_id=r["qid"],
                ctx=r["ctx"],
                logged_action=a,
                logged_propensity=1.0 / len(ACTIONS_LOG),
                logged_reward=float(r["cf"][a]),
                correction=0,
                cf_rewards=dict(r["cf"]),
            )
        )
    return logs


def _fires_frac(rule: Rule, rows: Sequence[dict]) -> float:
    return float(np.mean([rule.fires(r["ctx"]) for r in rows]))


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def select_erm(rules, results, k: int) -> list[Rule]:
    """Argmax of V_hat (point estimate)."""
    scores = np.array([results[r.id].estimate for r in rules])
    order = np.argsort(-scores)[:k]
    return [rules[i] for i in order]


def select_random(rules, k: int, rng: np.random.Generator) -> list[Rule]:
    idx = rng.choice(len(rules), size=min(k, len(rules)), replace=False)
    return [rules[int(i)] for i in idx]


def select_crrm_lcb(rules, results, k: int, atom_sparse: bool = True, delta: float = 0.05) -> list[Rule]:
    """Pessimistic LCB selection (src/crrm.py).

    atom_sparse=True reproduces the compositional Rademacher bound
    (Theorem 5 of proofs.tex); atom_sparse=False falls back to the
    standard union-bound LCB (Bonferroni over |C|). The two
    correspond to different tradeoffs in pessimism intensity.
    """
    sel = PessimisticRuleSelector(PessimisticConfig(delta=delta, atom_sparse=atom_sparse))
    return [r for r, _ in sel.top_k(rules, results, k=k, actions=("noop", "filter", "rerank", "abstain"))]


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def _run_trial(
    rows_pool: Sequence[dict],
    candidates: Sequence[Rule],
    hand_rules: Sequence[Rule],
    n_train: int,
    min_fire_frac: float,
    trial: int,
    rng: np.random.Generator,
    ks: Sequence[int],
) -> dict:
    # Split pool: train (logs) vs eval (oracle V*)
    perm = rng.permutation(len(rows_pool))
    train_rows = [rows_pool[int(i)] for i in perm[:n_train]]
    eval_rows = [rows_pool[int(i)] for i in perm[n_train:]]

    # Support filter on candidate rules (min firing on eval split)
    eligible = [r for r in candidates if _fires_frac(r, eval_rows) >= min_fire_frac]
    if len(eligible) < 20:
        return {"trial": trial, "skipped": True, "n_eligible": len(eligible)}

    # Precompute oracle values on eval split
    oracle_v = {r.id: _oracle_value(r, eval_rows) for r in eligible}
    oracle_v_hand = {r.id: _oracle_value(r, eval_rows) for r in hand_rules}

    oracle_best_v = max(oracle_v.values())

    # Build logs and fit estimator
    logs = _build_logs(train_rows, rng)
    estimator = RuleOPE()
    estimator.fit(logs)
    results_cand = estimator.value_many(eligible, logs)
    results_hand = estimator.value_many(hand_rules, logs)

    def _oracle_regret_at_k(selected_rules: list[Rule], oracle_map: dict) -> float:
        """Simple regret: oracle_best_v - max_{r in selected} oracle_v[r.id]."""
        if not selected_rules:
            return float(oracle_best_v)
        best = max(oracle_map.get(r.id, float("-inf")) for r in selected_rules)
        return float(oracle_best_v - best)

    out = {"trial": trial, "oracle_best_v": oracle_best_v, "n_eligible": len(eligible),
           "n_hand": len(hand_rules)}
    for k in ks:
        # CRRM-LCB (atom-sparse) / CRRM-LCB (union-bound) / ERM / Random
        crrm = select_crrm_lcb(eligible, results_cand, k, atom_sparse=True)
        crrm_union = select_crrm_lcb(eligible, results_cand, k, atom_sparse=False)
        erm = select_erm(eligible, results_cand, k)
        rand = select_random(eligible, k, rng)
        # Over hand-authored only
        hand_by_vhat = select_erm(hand_rules, results_hand, k)
        hand_by_oracle = sorted(hand_rules, key=lambda r: -oracle_v_hand[r.id])[:k]

        out[f"regret_crrm@{k}"] = _oracle_regret_at_k(crrm, oracle_v)
        out[f"regret_crrm_union@{k}"] = _oracle_regret_at_k(crrm_union, oracle_v)
        out[f"regret_erm@{k}"] = _oracle_regret_at_k(erm, oracle_v)
        out[f"regret_random@{k}"] = _oracle_regret_at_k(rand, oracle_v)
        out[f"regret_hand_by_vhat@{k}"] = _oracle_regret_at_k(hand_by_vhat, oracle_v)
        out[f"regret_hand_oracle@{k}"] = _oracle_regret_at_k(hand_by_oracle, oracle_v)

        # Also record top-1 oracle value of each selector (easier to interpret)
        def _topv(rs: list[Rule], omap) -> float:
            return float(max(omap.get(r.id, float("-inf")) for r in rs)) if rs else float("nan")
        out[f"topv_crrm@{k}"] = _topv(crrm, oracle_v)
        out[f"topv_crrm_union@{k}"] = _topv(crrm_union, oracle_v)
        out[f"topv_erm@{k}"] = _topv(erm, oracle_v)
        out[f"topv_random@{k}"] = _topv(rand, oracle_v)
        out[f"topv_hand_by_vhat@{k}"] = _topv(hand_by_vhat, oracle_v_hand)
        out[f"topv_hand_oracle@{k}"] = _topv(hand_by_oracle, oracle_v_hand)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _summarize(trials: list[dict], ks: Sequence[int]) -> dict:
    trials = [t for t in trials if not t.get("skipped", False)]
    summary = {"n_trials_effective": len(trials)}
    for k in ks:
        for selector in ("crrm", "crrm_union", "erm", "random", "hand_by_vhat", "hand_oracle"):
            key = f"regret_{selector}@{k}"
            vals = np.array([t[key] for t in trials], dtype=np.float64)
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "CI90": [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))],
            }
            tkey = f"topv_{selector}@{k}"
            tvals = np.array([t[tkey] for t in trials], dtype=np.float64)
            summary[tkey] = {
                "mean": float(tvals.mean()),
                "std": float(tvals.std(ddof=1)) if len(tvals) > 1 else 0.0,
                "CI90": [float(np.quantile(tvals, 0.05)), float(np.quantile(tvals, 0.95))],
            }
        # Paired comparison: CRRM beats hand-oracle? (auto-discovery vs best human-authored)
        crrm_topv = np.array([t[f"topv_crrm@{k}"] for t in trials])
        hand_topv = np.array([t[f"topv_hand_oracle@{k}"] for t in trials])
        diff = crrm_topv - hand_topv
        n_win = int((diff > 0).sum())
        n_tie = int((diff == 0).sum())
        n_lose = int((diff < 0).sum())
        summary[f"crrm_vs_handoracle_diff@{k}"] = {
            "mean_diff": float(diff.mean()),
            "CI90": [float(np.quantile(diff, 0.05)), float(np.quantile(diff, 0.95))],
            "n_win": n_win, "n_tie": n_tie, "n_lose": n_lose,
            "win_rate": n_win / max(1, len(diff)),
        }
    return summary


def run(n_trials: int, n_train: int, pool_size: int, max_depth: int,
        min_fire_frac: float, ks: Sequence[int], out_path: str) -> int:
    print(f"Loading HotpotQA pool size={pool_size}", flush=True)
    rows_pool = _precompute_samples(pool_size, pool_seed=0)
    print(f"Loaded {len(rows_pool)} samples", flush=True)

    candidates_raw = enumerate_rules(
        max_depth=max_depth, atoms=ATOMS, actions=ACTIONS_RULE,
    )
    print(f"Enumerated {len(candidates_raw)} depth-<={max_depth} rules", flush=True)

    hand_rules = load_rules("eval/rules_v1.jsonl")
    print(f"Loaded {len(hand_rules)} hand-authored rules", flush=True)

    trials = []
    for t in range(n_trials):
        rng = np.random.default_rng(1000 + t)
        out = _run_trial(
            rows_pool, candidates_raw, hand_rules,
            n_train=n_train, min_fire_frac=min_fire_frac,
            trial=t, rng=rng, ks=ks,
        )
        trials.append(out)
        if not out.get("skipped"):
            print(f"  trial {t}: oracle_best={out['oracle_best_v']:.4f} "
                  f"regret@1 crrm={out['regret_crrm@1']:.4f} "
                  f"erm={out['regret_erm@1']:.4f} "
                  f"rand={out['regret_random@1']:.4f} "
                  f"hand*={out['regret_hand_oracle@1']:.4f}", flush=True)
        else:
            print(f"  trial {t}: skipped (n_eligible={out['n_eligible']})", flush=True)

    summary = _summarize(trials, ks)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_trials": n_trials, "n_train": n_train,
                "pool_size": pool_size, "max_depth": max_depth,
                "min_fire_frac": min_fire_frac, "ks": list(ks),
            },
            "trials": trials,
            "summary": summary,
        }, f, indent=2, default=float)

    print("\n=== summary ===", flush=True)
    for k in ks:
        cvs = summary[f"crrm_vs_handoracle_diff@{k}"]
        print(f"k={k}:  CRRM topv mean={summary[f'topv_crrm@{k}']['mean']:.4f}  "
              f"Hand-oracle topv mean={summary[f'topv_hand_oracle@{k}']['mean']:.4f}  "
              f"diff={cvs['mean_diff']:+.4f}  win_rate={cvs['win_rate']:.2f} "
              f"({cvs['n_win']}/{cvs['n_win']+cvs['n_tie']+cvs['n_lose']})", flush=True)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=25)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--pool_size", type=int, default=1500)
    ap.add_argument("--max_depth", type=int, default=2)
    ap.add_argument("--min_fire_frac", type=float, default=0.05)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--out", type=str, default="experiments/results/rule_discovery.json")
    args = ap.parse_args()
    raise SystemExit(run(
        args.n_trials, args.n_train, args.pool_size, args.max_depth,
        args.min_fire_frac, args.ks, args.out,
    ))
