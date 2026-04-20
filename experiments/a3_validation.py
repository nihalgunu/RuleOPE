"""A3 empirical validation on HotpotQA.

A3 (no-replay identifying assumption):
    E[R | q, r] = alpha(q) + phi(r)^T * beta + eta(q, r),
    with E[eta | phi(r)] = 0.

This script tests A3 directly on HotpotQA counterfactual reward data.

Design
------
For each logged HotpotQA query we have cf_rewards under {noop, filter, rerank},
each a deterministic function of the modified retrieval.  We stack these into
a long panel (query_id, action, reward) and fit three nested models:

  M0 (query-only null):      R ~ query fixed effects
  M1 (A3 additive):          R ~ query FE + (atom x action) additive
  M2 (action saturated):     R ~ query FE * action  (upper bound on within-R2)

Reported metrics
----------------
  R2 of M1 over plain mean           (overall-R2)
  Within-R2 of M1 over M0            (A3's actual explanatory power)
  F-test M1 vs M0                    (A3 significance)
  Gap R2(M2) - R2(M1)                (how far A3 is from saturation)
  Residual-vs-atom independence      (E[eta | phi_j = 1] for each atom)
  Sensitivity to atom rank d         (R2 at top-d atoms, d in {5,10,20,48})

Writes experiments/results/a3_validation.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from src.rag_substrate_hotpot import (
    _apply_rule,
    _atom_features,
    _load_hotpot,
    _reward_for_top3,
    _score_passages,
    _secondary_scores,
    _tokenize,
)
from src.rule_dsl import ATOMS


ACTIONS = ("noop", "filter", "rerank")  # abstain is constant; drop from A3 panel


def _atoms_for_action(sample, scores, action: str) -> np.ndarray:
    """Atom indicator vector phi(r) for the retrieval state AFTER applying action.

    The ATOMS vocabulary is built from features that depend on the *retrieved*
    top passages, so phi(r) genuinely changes with action.
    """
    if action == "noop":
        ctx = _atom_features(sample, scores)
    elif action == "filter":
        # After filter, the original top-2 becomes the new top-1, etc.  Build
        # a permuted score array so that _atom_features sees the post-filter
        # ordering as the ranking.  The cleanest way is to rewrite the scores
        # with the top-1 pushed to -inf.
        new_scores = scores.copy()
        top1 = int(np.argmax(new_scores))
        new_scores[top1] = -1e9
        ctx = _atom_features(sample, new_scores)
    elif action == "rerank":
        sec = _secondary_scores(sample, scores)
        ctx = _atom_features(sample, sec)
    else:
        raise ValueError(action)
    return np.array([1.0 if a.eval(ctx) else 0.0 for a in ATOMS], dtype=np.float64)


def _build_panel(n_queries: int, seed: int) -> dict:
    samples = _load_hotpot("eval/hotpot/dev.parquet", n_queries, seed)

    rows_q, rows_a, rows_y = [], [], []
    rows_phi = []
    for i, s in enumerate(samples):
        scores = _score_passages(s)
        for a in ACTIONS:
            titles = _apply_rule(a, scores, s)
            y = _reward_for_top3(s.gold_titles, titles)
            phi = _atoms_for_action(s, scores, a)
            rows_q.append(i)
            rows_a.append(a)
            rows_y.append(y)
            rows_phi.append(phi)

    return {
        "q": np.array(rows_q, dtype=np.int64),
        "a": np.array(rows_a),
        "y": np.array(rows_y, dtype=np.float64),
        "phi": np.vstack(rows_phi),  # (N_obs, d_atoms)
        "n_queries": len(samples),
        "n_atoms": len(ATOMS),
    }


def _one_hot(values: np.ndarray, categories: list) -> np.ndarray:
    N = len(values)
    out = np.zeros((N, len(categories)), dtype=np.float64)
    for j, c in enumerate(categories):
        out[values == c, j] = 1.0
    return out


def _partial_out_query(X: np.ndarray, q: np.ndarray, n_queries: int) -> np.ndarray:
    """Subtract per-query mean from each column of X. Returns within-query residual."""
    Xdm = X.copy()
    for i in range(n_queries):
        mask = q == i
        if mask.any():
            Xdm[mask] -= X[mask].mean(axis=0, keepdims=True)
    return Xdm


def _fit_with_fe(y: np.ndarray, phi_by_action: np.ndarray, q: np.ndarray,
                  n_queries: int, alpha: float = 1.0):
    """Fit y = q_FE + X * beta via within transformation + ridge on X_demeaned.

    Returns (yhat, beta, residuals, r2_total, r2_within).
    """
    # Within transformation: demean y and X by query.
    y_bar_q = np.zeros(n_queries)
    for i in range(n_queries):
        mask = q == i
        if mask.any():
            y_bar_q[i] = y[mask].mean()
    y_q = y_bar_q[q]
    y_dm = y - y_q

    X_dm = _partial_out_query(phi_by_action, q, n_queries)

    if phi_by_action.shape[1] == 0:
        beta = np.zeros(0)
        yhat_dm = np.zeros_like(y_dm)
    else:
        model = Ridge(alpha=alpha, fit_intercept=False).fit(X_dm, y_dm)
        beta = model.coef_
        yhat_dm = model.predict(X_dm)

    yhat = y_q + yhat_dm
    resid = y - yhat

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2_total = 1.0 - ss_res / max(ss_tot, 1e-12)

    ss_within = float((y_dm ** 2).sum())
    ss_res_within = float(((y_dm - yhat_dm) ** 2).sum())
    r2_within = 1.0 - ss_res_within / max(ss_within, 1e-12)

    return {
        "yhat": yhat,
        "beta": beta,
        "resid": resid,
        "r2_total": r2_total,
        "r2_within": r2_within,
        "ss_within": ss_within,
        "ss_res_within": ss_res_within,
    }


def run(n_queries: int = 1500, seed: int = 0, alpha: float = 1.0):
    print(f"Loading HotpotQA panel (n_queries={n_queries}) ...", flush=True)
    panel = _build_panel(n_queries, seed)
    y = panel["y"]; q = panel["q"]; a = panel["a"]; phi = panel["phi"]
    n_q = panel["n_queries"]
    d = panel["n_atoms"]
    N_obs = len(y)
    print(f"  panel: {n_q} queries x {len(ACTIONS)} actions = {N_obs} obs, {d} atoms", flush=True)
    print(f"  reward mean={y.mean():.3f}  std={y.std():.3f}  "
          f"noop_mean={y[a=='noop'].mean():.3f}  filter_mean={y[a=='filter'].mean():.3f}  "
          f"rerank_mean={y[a=='rerank'].mean():.3f}", flush=True)

    # Build action x atom feature matrix: per-observation phi * action_onehot(a)
    action_oh = _one_hot(a, list(ACTIONS))  # (N_obs, 3)
    # Interaction features: phi[i, k] * action_oh[i, j] reshaped
    X_inter = (phi[:, :, None] * action_oh[:, None, :]).reshape(N_obs, d * len(ACTIONS))

    # ---- M0: query-only null
    # residuals = y - ybar_q for each q
    fit0 = _fit_with_fe(y, np.zeros((N_obs, 0)), q, n_q, alpha=alpha)
    r2_M0 = fit0["r2_total"]
    within_resid_M0 = y - fit0["yhat"]
    ss_within = float((within_resid_M0 ** 2).sum())

    # ---- M1: query FE + (atom x action) additive
    fit1 = _fit_with_fe(y, X_inter, q, n_q, alpha=alpha)

    # F-test: M0 (query only) vs M1 (query + atoms x actions)
    df1 = d * len(ACTIONS)
    df_res = N_obs - n_q - df1
    if fit1["ss_res_within"] > 0 and df_res > 0:
        F = ((ss_within - fit1["ss_res_within"]) / df1) / (fit1["ss_res_within"] / df_res)
        p_val = float(1 - stats.f.cdf(F, df1, df_res))
    else:
        F = float("inf")
        p_val = 0.0

    # ---- M2: query x action saturated (upper bound on within-R2)
    # y predicted by mean(y | q, a) -- essentially a cell mean model.
    yhat_M2 = np.zeros_like(y)
    for i in range(n_q):
        for a_name in ACTIONS:
            mask = (q == i) & (a == a_name)
            if mask.any():
                yhat_M2[mask] = y[mask].mean()
    ss_res_M2 = float(((y - yhat_M2) ** 2).sum())
    r2_M2 = 1.0 - ss_res_M2 / max(float(((y - y.mean()) ** 2).sum()), 1e-12)

    # ---- residual-vs-atom independence
    # Under A3 with atom-indexed phi, E[eta | phi_j = 1] should be ~ 0 for each atom.
    # Use residuals of M1.  Bonferroni-corrected test.
    resid = fit1["resid"]
    residual_cond_means = {}
    for j, atom in enumerate(ATOMS):
        for k, a_name in enumerate(ACTIONS):
            col = phi[:, j] * (a == a_name).astype(np.float64)
            if col.sum() < 5:
                continue
            mask = col > 0.5
            mu = float(resid[mask].mean())
            se = float(resid[mask].std(ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else 0.0
            residual_cond_means[f"{atom.name}@{a_name}"] = {
                "mean_resid": mu, "se": se, "n": int(mask.sum()),
                "t": float(mu / max(se, 1e-9)),
            }
    # Count how many survive Bonferroni at alpha=0.05.
    tests = list(residual_cond_means.values())
    if tests:
        ts = np.array([abs(t["t"]) for t in tests])
        crit_bonf = stats.norm.ppf(1 - 0.025 / len(tests))
        n_violations = int((ts > crit_bonf).sum())
    else:
        crit_bonf, n_violations = float("nan"), 0

    # ---- sensitivity to atom rank d
    # Rank atoms by absolute within-correlation with residual of M0, take top-d.
    y_within = y - fit0["yhat"]
    atom_by_action_absr = []
    labels = []
    for j in range(d):
        for k, a_name in enumerate(ACTIONS):
            col = phi[:, j] * (a == a_name).astype(np.float64)
            if col.std() < 1e-9:
                atom_by_action_absr.append(0.0)
            else:
                atom_by_action_absr.append(abs(np.corrcoef(col, y_within)[0, 1]))
            labels.append(f"{ATOMS[j].name}@{a_name}")
    order = np.argsort(atom_by_action_absr)[::-1]
    sensitivity = []
    for d_use in (5, 10, 20, 40, d * len(ACTIONS)):
        sel = order[:d_use]
        X_sel = X_inter[:, sel]
        f = _fit_with_fe(y, X_sel, q, n_q, alpha=alpha)
        sensitivity.append({"d": int(d_use), "r2_within": f["r2_within"],
                            "r2_total": f["r2_total"]})
        print(f"  d={d_use:4d}  within-R2={f['r2_within']:.3f}  total-R2={f['r2_total']:.3f}",
              flush=True)

    results = {
        "n_queries": n_q,
        "n_actions": len(ACTIONS),
        "n_obs": N_obs,
        "n_atoms": d,
        "alpha_ridge": alpha,
        "M0_query_only": {"r2_total": r2_M0},
        "M1_A3_additive": {
            "r2_total": fit1["r2_total"],
            "r2_within": fit1["r2_within"],
            "ss_res_within": fit1["ss_res_within"],
        },
        "M2_saturated": {"r2_total": r2_M2},
        "A3_vs_null_F_test": {"F": float(F), "df1": df1, "df_res": df_res, "p_value": p_val},
        "A3_saturation_gap": r2_M2 - fit1["r2_total"],
        "residual_independence": {
            "n_tests": len(residual_cond_means),
            "bonferroni_critical_z": float(crit_bonf),
            "n_violations_at_0_05": n_violations,
            "max_abs_t": float(max([abs(t["t"]) for t in residual_cond_means.values()])) if residual_cond_means else 0.0,
            "mean_abs_t": float(np.mean([abs(t["t"]) for t in residual_cond_means.values()])) if residual_cond_means else 0.0,
            "worst_10": sorted(
                residual_cond_means.items(), key=lambda kv: -abs(kv[1]["t"])
            )[:10],
        },
        "sensitivity_to_d": sensitivity,
        "residual_histogram": {
            "bin_edges": [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "counts": [int(c) for c in np.histogram(
                resid, bins=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            )[0]],
        },
    }

    return results, resid, panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="experiments/results/a3_validation.json")
    args = ap.parse_args()

    results, resid, panel = run(args.n_queries, args.seed, args.alpha)

    print("\n==== A3 validation summary ====", flush=True)
    print(f"M0 (query only)   total R^2 = {results['M0_query_only']['r2_total']:.3f}", flush=True)
    print(f"M1 (A3 additive)  total R^2 = {results['M1_A3_additive']['r2_total']:.3f}"
          f"   within-R^2 = {results['M1_A3_additive']['r2_within']:.3f}", flush=True)
    print(f"M2 (saturated)    total R^2 = {results['M2_saturated']['r2_total']:.3f}", flush=True)
    print(f"A3 vs null F      F={results['A3_vs_null_F_test']['F']:.1f}  "
          f"p={results['A3_vs_null_F_test']['p_value']:.3e}", flush=True)
    print(f"A3 saturation gap = {results['A3_saturation_gap']:.3f} (smaller = better)", flush=True)
    print(f"Residual independence: {results['residual_independence']['n_violations_at_0_05']} "
          f"of {results['residual_independence']['n_tests']} Bonferroni violations "
          f"(max |t| = {results['residual_independence']['max_abs_t']:.2f})", flush=True)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    # Also save the residuals for the figure.
    np.save(args.out.replace(".json", "_residuals.npy"), resid)
    np.save(args.out.replace(".json", "_y.npy"), panel["y"])
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
