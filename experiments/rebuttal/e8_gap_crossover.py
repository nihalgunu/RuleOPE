"""E8 — Bias-variance crossover fit for the NonCompDR-RuleOPE gap N-profile.

Corrected-Theorem-3 mechanism check [N3HW Q1; PAT "Thm 3 vs empirics sign
contradiction"]. The small-K/N expansion keeps a positive variance-geometry
term that decays as 1/N and discards two N-independent squared-bias terms.
Keeping both gives the two-parameter form

    gap(N) := MSE_NonCompDR(N) - MSE_RuleOPE(N)  ~  a + b/N,

with b > 0 the variance-geometry coefficient (atom-sharing advantage,
dominant at pilot scale) and a = bias^2_NC - bias^2_RO the asymptote: a < 0
means the shared-regression bias exceeds the per-rule-refit bias, so the
gap must turn negative (RuleOPE "falls below baseline") for N > N* = -b/a
and DEEPEN toward a as N grows — exactly the reported NQ behaviour. This
script fits (a, b) per cell on the five cached sample sizes and tests the
prediction that deepening cells are the a < 0 cells.

Output: e8_gap_crossover.csv + stdout summary.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.rebuttal.common import OUT_DIR, RESULTS

PAIR = ("NonCompDR", "RuleOPE")


def load_gaps():
    j4 = json.load(open(RESULTS / "full_36cell_4N_5estimator.json"))
    j24 = json.load(open(RESULTS / "full_36cell_n2400_5estimator.json"))
    rows = []
    for cell, payload in j4["cells"].items():
        bench, llm = cell.split("__")
        prof = {}
        for n_str, ests in payload["scaling"].items():
            prof[int(n_str)] = (ests[PAIR[0]]["MSE_mean"], ests[PAIR[1]]["MSE_mean"])
        c24 = j24["cells"].get(cell)
        if c24 is not None:
            for n_str, ests in c24["scaling"].items():
                prof[int(n_str)] = (ests[PAIR[0]]["MSE_mean"], ests[PAIR[1]]["MSE_mean"])
        rows.append((bench, llm, dict(sorted(prof.items()))))
    return rows


def fit_cell(prof):
    ns = np.array(sorted(prof))
    gap = np.array([prof[n][0] - prof[n][1] for n in ns])
    X = np.column_stack([np.ones_like(ns, dtype=float), 1.0 / ns])
    coef, *_ = np.linalg.lstsq(X, gap, rcond=None)
    a, b = coef
    pred = X @ coef
    ss_res = float(((gap - pred) ** 2).sum())
    ss_tot = float(((gap - gap.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return ns, gap, float(a), float(b), r2


def main():
    recs = []
    for bench, llm, prof in load_gaps():
        ns, gap, a, b, r2 = fit_cell(prof)
        n_star = -b / a if (a < 0 and b > 0) else np.nan
        recs.append({
            "bench": bench, "llm": llm,
            **{f"gap_{n}": g for n, g in zip(ns, gap)},
            "a_asymptote": a, "b_variance": b, "fit_r2": r2,
            "crossover_N": n_star,
            "gap_negative_at_2400": bool(gap[-1] < 0) if 2400 in prof else np.nan,
            "deepening": bool(len(gap) >= 2 and gap[-1] < gap[-2] < 0)
            if 2400 in prof else np.nan,
        })
    df = pd.DataFrame(recs)
    df.to_csv(OUT_DIR / "e8_gap_crossover.csv", index=False)

    print("=== E8: gap(N) = a + b/N per cell (NonCompDR - RuleOPE MSE) ===")
    print(f"cells: {len(df)}   median fit R^2: {df.fit_r2.median():.3f}   "
          f"b > 0 (variance term, Thm 3 leading order): {(df.b_variance > 0).sum()}/{len(df)}")
    for bench, g in df.groupby("bench"):
        neg24 = g[g.gap_negative_at_2400 == True]  # noqa: E712
        print(f"\n{bench}: a<0 in {(g.a_asymptote < 0).sum()}/{len(g)} cells; "
              f"gap<0 at N=2400 in {len(neg24)}/{len(g)}")
        agree = ((g.a_asymptote < 0) == (g.gap_negative_at_2400 == True)).sum()  # noqa: E712
        print(f"  sign(a) predicts sign(gap@2400): {agree}/{len(g)}")
        for _, r in g[(g.a_asymptote < 0) & (g.b_variance > 0)].iterrows():
            print(f"  {r.llm:<12} a={r.a_asymptote:+.5f} b={r.b_variance:+.3f} "
                  f"R2={r.fit_r2:.3f} N*={r.crossover_N:,.0f} "
                  f"gap@2400={r.gap_2400:+.5f}")
    print(f"\nwrote {OUT_DIR}/e8_gap_crossover.csv")


if __name__ == "__main__":
    main()
