"""E9 — Numerical checks for the corrected Theorem 4 (commensurability).

Verifies, on the 36 in-grid cells, the facts the corrected statement rests
on [w7Aq Q1; PAT App-C.4 algebra]:

  1. rank(V_beta) <= K-1 = 2 always (V_beta is built from 3 centred
     coefficient vectors), so V_beta is never positive definite in d=48;
     the corrected theorem must not (and does not) assume it.
  2. Sigma = E[phi phi^T] (standardised atoms) is the kernel whose
     positive-definiteness carries the vanishing direction
     sigma^2 = tr(V_beta Sigma) = 0  =>  V_beta = 0  =>  Delta^2 = 0.
     We report lambda_min(Sigma) and its effective rank per cell.
  3. Delta^2 = tr(V_beta q q^T) with q = mean(X)/std(X): a RANK-ONE kernel,
     hence (a) Delta^2 = 0 does NOT imply sigma^2 = 0 (w7Aq's counterexample
     direction is real), and (b) no universal two-sided sandwich constant
     exists — the empirical kernel ratio c-hat spans orders of magnitude.
  4. The deployment statistic sigma-hat^2_R (audit protocol: variance of the
     three realised per-action rewards) equals the signal functional
     tr(V_beta Sigma) only up to a noise floor and ridge shrinkage; we
     report their per-cell values and Spearman correlation, which is what
     the selector actually relies on.

Output: e9_kernel_check.csv + stdout summary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.rebuttal.common import LLMS, OUT_DIR
from experiments.rebuttal.e3_a3_validation import cell_XR, ACTIONS

RIDGE_LAM = 10.0
BENCHES = ("hotpot", "trivia", "nq")


def kernel_stats(X, R):
    n, d = X.shape
    col_std = X.std(0) + 1e-6
    Xn = (X - X.mean(0)) / col_std
    Sigma = Xn.T @ Xn / n
    evals = np.linalg.eigvalsh(Sigma)
    nz = evals[evals > evals.max() * 1e-10]
    n_const = int((X.std(0) < 1e-12).sum())
    A_inv = np.linalg.inv(Xn.T @ Xn + RIDGE_LAM * np.eye(d))
    B = np.array([A_inv @ Xn.T @ R[a] for a in ACTIONS])
    Bc = B - B.mean(0)
    V_beta = (Bc.T @ Bc) / B.shape[0]
    v_evals = np.linalg.eigvalsh(V_beta)
    rank_V = int((v_evals > v_evals.max() * 1e-10).sum()) if v_evals.max() > 0 else 0
    sigma2 = float(np.trace(V_beta @ Sigma))
    q = X.mean(0) / col_std
    delta2 = float(q @ V_beta @ q)          # == tr(V_beta q q^T) == Var_a(q.B)
    delta2_var = float(np.var(np.array([q @ B[a] for a in range(len(ACTIONS))])))
    assert abs(delta2 - delta2_var) < 1e-9 * max(1.0, abs(delta2))
    # noise floor: within-query variance of realised rewards minus the
    # model-explained action variance, averaged over queries
    Rmat = np.stack([R[a] for a in ACTIONS], axis=1)
    audit_sigma2 = float(np.mean(np.var(Rmat, axis=1)))
    fitted = Xn @ B.T                        # n x 3 conditional means
    model_sigma2 = float(np.mean(np.var(fitted, axis=1)))
    return {
        "d": d, "n_queries": n,
        "lambda_min_Sigma": float(evals[0]), "lambda_max_Sigma": float(evals[-1]),
        "rank_Sigma": int(len(nz)), "min_nonzero_eig_Sigma": float(nz.min()),
        "n_constant_atoms": n_const,
        "rank_V_beta": rank_V,
        "sigma2_tr": sigma2, "delta2_tr": delta2,
        "c_hat": delta2 / sigma2 if sigma2 > 0 else np.nan,
        "audit_sigma2_R": audit_sigma2, "model_sigma2": model_sigma2,
        "noise_floor": audit_sigma2 - model_sigma2,
    }


def main():
    rows = []
    for bench in BENCHES:
        for llm in LLMS:
            out = cell_XR(bench, llm)
            if out is None:
                continue
            X, R = out
            st = kernel_stats(X, R)
            rows.append({"bench": bench, "llm": llm, **st})
            print(f"{bench}/{llm}: rank(V)={st['rank_V_beta']} "
                  f"lam_min(Sigma)={st['lambda_min_Sigma']:.2e} "
                  f"sigma2={st['sigma2_tr']:.4f} delta2={st['delta2_tr']:.4f} "
                  f"c_hat={st['c_hat']:.3f} audit={st['audit_sigma2_R']:.4f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "e9_kernel_check.csv", index=False)

    print("\n=== E9 summary (36 in-grid cells) ===")
    print(f"rank(V_beta) <= 2 in {int((df.rank_V_beta <= 2).sum())}/{len(df)} cells "
          f"(max observed rank {int(df.rank_V_beta.max())}; d = {int(df.d.iloc[0])})")
    print(f"Sigma singular in ambient d=48 (constant atoms: "
          f"{int(df.n_constant_atoms.min())}-{int(df.n_constant_atoms.max())} per bench) "
          f"but PD ON ITS RANGE in {len(df)}/{len(df)} cells: "
          f"rank {int(df.rank_Sigma.min())}-{int(df.rank_Sigma.max())}, "
          f"min nonzero eigenvalue {df.min_nonzero_eig_Sigma.min():.3e}. "
          f"Ridge estimates lie in range(Sigma), so the corrected vanishing "
          f"direction (sigma2=0 => V_beta=0 => Delta2=0) holds on-range.")
    print(f"c_hat = delta2/sigma2: min {df.c_hat.min():.3f}, median {df.c_hat.median():.3f}, "
          f"max {df.c_hat.max():.1f}")
    rho, p = spearmanr(df.audit_sigma2_R, df.sigma2_tr)
    print(f"Spearman(audit sigma-hat^2_R, tr(V_beta Sigma)) = {rho:.3f} (p = {p:.2e})")
    rho2, p2 = spearmanr(df.audit_sigma2_R, df.noise_floor)
    print(f"noise floor share of audit stat: median "
          f"{(df.noise_floor / df.audit_sigma2_R).median():.2%} "
          f"(Spearman with audit stat {rho2:.3f}, p = {p2:.2e})")
    print(f"\nwrote {OUT_DIR}/e9_kernel_check.csv")


if __name__ == "__main__":
    main()
