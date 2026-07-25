"""E3 — A3 (atom-level additivity) validation across benchmarks, calibration
constant on the released pool, and an A3-violation-aware interaction model.

Per (benchmark, LLM) cell, on the cached all-action replay outputs:
  M0: query fixed effects only            (within-query R^2 = 0 by construction)
  M1: query FE + atom x action additive   (the A3 working model; its
                                           within-query R^2 is "R^2(A3)")
  M2: saturated query x action FE         (R^2 = 1)
This reproduces the shipped a3_validation_{nq,trivia}_mistral.json protocol
(within-query OLS, 48 atoms x 3 actions = 144 within-design columns) and
extends it to all 12 LLMs x {hotpot, trivia, nq} + mistral on musique/2wiki.

Calibration constant c-hat (Theorem 4): per-cell ratio Delta^2 / sigma_R^2
with sigma_R^2 = tr(V_beta Sigma) and Delta^2 = Var_a(mu^T beta_a), exactly
as in scripts/build_figures.py Fig 6; reported as the empirical kernel-ratio
band (5th pct, median, 95th pct) over the 36 in-grid cells.

A3-violation-aware M3: MRDR_vs_RuleOPE_pct ~ sigma_R2 * C(benchmark)
augmented with a sigma_R2 : R2_A3 interaction, to test whether the sigma_R2
mechanism survives where A3 is weak.

Outputs: e3_a3_per_cell.csv, e3_a3_summary.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.rebuttal.common import _norm, _f1, OUT_DIR, ROOT, LLMS
from src.rule_dsl import ATOMS

ACTIONS = ("noop", "filter", "rerank")


def _reward(bench, txt, gold):
    """Sweep-protocol reward: alias-max F1 for trivia/nq, plain F1 otherwise."""
    if _norm(txt) in ("unknown", ""):
        return 0.0
    if bench in ("trivia", "nq"):
        return max((_f1(txt, g) for g in gold), default=0.0)
    return _f1(txt, gold)


def _substrate(bench):
    if bench == "hotpot":
        from src.rag_substrate_hotpot import _load_hotpot as L, _score_passages as S, _atom_features as A
    elif bench == "trivia":
        from src.rag_substrate_trivia import _load_trivia as L, _score_passages as S, _atom_features as A
    elif bench == "nq":
        from src.rag_substrate_nq import _load_nq as L, _score_passages as S, _atom_features as A
    elif bench == "musique":
        from src.rag_substrate_musique import _load_musique as L, _score_passages as S, _atom_features as A
    elif bench == "2wiki":
        from src.rag_substrate_2wiki import _load_2wiki as L, _score_passages as S, _atom_features as A
    else:
        raise ValueError(bench)
    return L, S, A


def _gold_field(bench):
    return "gold_phrase" if bench == "trivia" else "answer"


_BENCH_CACHE = {}


def bench_features(bench):
    """qid -> 48-dim binary atom vector (LLM-independent), once per benchmark.

    M1's design uses the atom INDICATORS from the rule DSL (df1 = 144 = 48x3
    in the shipped a3_validation JSONs), not the raw context features.
    Gold is the alias list for trivia/nq (sweep reward protocol), else the
    primary answer string.
    """
    if bench in _BENCH_CACHE:
        return _BENCH_CACHE[bench]
    L, S, A = _substrate(bench)
    samples = L(str(ROOT / f"eval/{bench}/dev.parquet"), 1500, 0)
    feats, gold = {}, {}
    for s in samples:
        ctx = A(s, S(s))
        feats[s.qid] = np.array([float(a.eval(ctx)) for a in ATOMS], dtype=np.float64)
        if bench in ("trivia", "nq"):
            gold[s.qid] = list(getattr(s, "answer_aliases"))
        else:
            gold[s.qid] = getattr(s, _gold_field(bench))
    _BENCH_CACHE[bench] = (feats, gold)
    return feats, gold


def cell_XR(bench, llm):
    """(X: n x d atom features, R: dict action -> n rewards) for one cell."""
    from experiments.rebuttal.common import outputs_path
    feats, gold = bench_features(bench)
    path = outputs_path(bench, llm)
    if not path.exists():
        return None
    answers = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        qid, action = r["id"].rsplit("__", 1)
        answers.setdefault(qid, {})[action] = r["text"]
    X_list, R = [], {a: [] for a in ACTIONS}
    for qid, acts in answers.items():
        if qid not in gold or qid not in feats or len(acts) < 3:
            continue
        X_list.append(feats[qid])
        for a in ACTIONS:
            R[a].append(_reward(bench, acts.get(a, ""), gold[qid]))
    return np.array(X_list), {a: np.array(R[a]) for a in ACTIONS}


def a3_within_r2(X, R):
    """Within-query R^2 of the atom x action additive model (M1)."""
    n, d = X.shape
    Rmat = np.stack([R[a] for a in ACTIONS], axis=1)          # n x 3
    Rw = Rmat - Rmat.mean(axis=1, keepdims=True)              # remove query FE
    # within-design: x_i (x) (e_a - mean_a); with 3 actions the demeaned
    # one-hot spans 2 dims per atom -> use actions' deviations directly.
    ss_tot = float((Rw ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    # Stack per action: target Rw[:, a]; features X for each action with
    # separate coefficient blocks, jointly demeaned across actions.
    Y = Rw.T.reshape(-1)                                      # 3n
    blocks = []
    for ai in range(len(ACTIONS)):
        Zi = np.zeros((len(ACTIONS), n, d))
        Zi[ai] = X
        blocks.append(Zi.reshape(len(ACTIONS) * n, d))
    Z = np.concatenate(blocks, axis=1)                        # 3n x 3d
    # demean the design within query across actions (same transform as Y)
    Zq = Z.reshape(len(ACTIONS), n, -1)
    Zq = Zq - Zq.mean(axis=0, keepdims=True)
    Z = Zq.reshape(len(ACTIONS) * n, -1)
    coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    ss_res = float(((Y - Z @ coef) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def sigma_delta(X, R, ridge_lam=10.0):
    """(sigma_R^2, Delta^2) per Fig 6 / Theorem 4 (build_figures.py protocol)."""
    n, d = X.shape
    col_std = X.std(0) + 1e-6
    Xn = (X - X.mean(0)) / col_std
    Sigma = Xn.T @ Xn / n
    A_inv = np.linalg.inv(Xn.T @ Xn + ridge_lam * np.eye(d))
    B = np.array([A_inv @ Xn.T @ R[a] for a in ACTIONS])
    Bc = B - B.mean(0)
    V_beta = (Bc.T @ Bc) / B.shape[0]
    sigma2 = float(np.trace(V_beta @ Sigma))
    proj = X.mean(0) / col_std
    delta2 = float(np.var(np.array([proj @ B[a] for a in range(len(ACTIONS))])))
    return sigma2, delta2


def main():
    cells = [(b, l) for b in ("hotpot", "trivia", "nq") for l in LLMS]
    cells += [("musique", "mistral"), ("2wiki", "mistral")]

    rows = []
    for i, (bench, llm) in enumerate(cells, 1):
        out = cell_XR(bench, llm)
        if out is None:
            continue
        X, R = out
        r2 = a3_within_r2(X, R)
        s2, d2 = sigma_delta(X, R)
        rows.append({"bench": bench, "llm": llm, "n_queries": len(X),
                     "R2_A3_within": r2, "sigma2_thm4": s2, "delta2_thm4": d2,
                     "ratio_delta2_sigma2": d2 / s2 if s2 > 0 else np.nan})
        print(f"[{i}/{len(cells)}] {bench}/{llm}: R2(A3)={r2:.3f}  "
              f"sigma2={s2:.4f} delta2={d2:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "e3_a3_per_cell.csv", index=False)

    per_bench = df.groupby("bench")["R2_A3_within"].agg(["mean", "min", "max"])
    print("\nR2(A3) by benchmark:\n", per_bench.round(3))

    grid = df[df["bench"].isin(["hotpot", "trivia", "nq"])]
    ratios = grid["ratio_delta2_sigma2"].dropna()
    c_hat = {"pct5": float(np.percentile(ratios, 5)),
             "median": float(np.median(ratios)),
             "pct95": float(np.percentile(ratios, 95)),
             "min": float(ratios.min()), "max": float(ratios.max())}
    print(f"\nc-hat (Delta^2/sigma_R^2 kernel ratio, 36 cells): "
          f"median {c_hat['median']:.3f}, 5-95pct [{c_hat['pct5']:.3f}, {c_hat['pct95']:.3f}]")

    # A3-violation-aware M3
    import statsmodels.formula.api as smf
    pairs = pd.read_csv(ROOT / "experiments/results/full36_phenomenon_pairs.csv")
    m = pairs.merge(grid[["bench", "llm", "R2_A3_within"]],
                    left_on=["benchmark", "llm"], right_on=["bench", "llm"])
    m3 = smf.ols("MRDR_vs_RuleOPE_pct ~ sigma_R2 * C(benchmark)", data=m).fit()
    m3a = smf.ols("MRDR_vs_RuleOPE_pct ~ sigma_R2 * C(benchmark) + sigma_R2:R2_A3_within",
                  data=m).fit()
    print(f"\nM3  (sigma_R2 x benchmark):            adj-R2 = {m3.rsquared_adj:.3f}")
    print(f"M3a (+ sigma_R2 : R2_A3 interaction):  adj-R2 = {m3a.rsquared_adj:.3f}")
    inter_p = m3a.pvalues.get("sigma_R2:R2_A3_within", np.nan)
    print(f"    sigma_R2:R2_A3 coefficient = {m3a.params.get('sigma_R2:R2_A3_within', np.nan):.2f} "
          f"(p = {inter_p:.3g})")

    summary = {
        "R2_A3_by_benchmark": {b: {"mean": float(g["R2_A3_within"].mean()),
                                   "min": float(g["R2_A3_within"].min()),
                                   "max": float(g["R2_A3_within"].max()),
                                   "n_cells": int(len(g))}
                               for b, g in df.groupby("bench")},
        "c_hat_kernel_ratio_36cells": c_hat,
        "M3_adj_r2": float(m3.rsquared_adj),
        "M3a_adj_r2": float(m3a.rsquared_adj),
        "M3a_sigma_x_R2A3_coef": float(m3a.params.get("sigma_R2:R2_A3_within", np.nan)),
        "M3a_sigma_x_R2A3_pvalue": float(inter_p),
    }
    with open(OUT_DIR / "e3_a3_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {OUT_DIR}/e3_a3_per_cell.csv, e3_a3_summary.json")


if __name__ == "__main__":
    main()
