"""E1 — Out-of-sample validation of the σ_R²-thresholded pilot selector.

(a) Leave-one-LLM-out CV on the 36 in-grid cells (12 folds): thresholds are
    re-fit on the 33 training cells of the other 11 LLMs, evaluated on the
    held-out LLM's 3 cells.
(b) Leave-one-benchmark-out CV (3 folds), same protocol.
(c) Frozen-selector transfer: thresholds fit ONCE on the 36 in-grid cells
    (and, separately, the paper's fixed (0.05, 0.10)), evaluated untouched on
    every held-out cell: MuSiQue (12 LLMs), 2Wiki (12 LLMs + both anchors),
    and the qwen14b/qwen32b anchors on the in-grid benchmarks.

Baselines: always-RuleOPE, always-MRDR, and the single-threshold rule
(no pilot) at the fold-fit HI.

Outputs: e1_lolo.csv, e1_lobo.csv, e1_frozen_transfer.csv (+ printed summary).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.rebuttal.common import (
    OUT_DIR, PAPER_LO, PAPER_HI, load_cell_table,
    predict_proc, predict_rule, fit_thresholds,
)


def evaluate_fold(test_df, lo, hi):
    rows = []
    for _, r in test_df.iterrows():
        rows.append({
            "bench": r["bench"], "llm": r["llm"], "sigma_R2": r["sigma_R2"],
            "truth": r["truth"],
            "pred_proc": predict_proc(r, lo, hi),
            "pred_rule": predict_rule(r, hi),
            "pred_always_ruleope": "RuleOPE",
            "pred_always_mrdr": "MRDR",
        })
    return rows


def run_cv(df, fold_col, out_name):
    folds = sorted(df[fold_col].unique())
    all_rows = []
    for held in folds:
        train = df[df[fold_col] != held]
        test = df[df[fold_col] == held]
        lo, hi, train_acc = fit_thresholds(train)
        rows = evaluate_fold(test, lo, hi)
        for r in rows:
            r["fold"] = held
            r["fit_lo"] = lo
            r["fit_hi"] = hi
            r["train_acc"] = round(train_acc, 4)
        all_rows.extend(rows)
        print(f"  fold={held:12s} fit (LO,HI)=({lo:.2f},{hi:.2f}) "
              f"train_acc={train_acc:.3f} test_n={len(rows)}")
    out = pd.DataFrame(all_rows)
    for col in ("pred_proc", "pred_rule", "pred_always_ruleope", "pred_always_mrdr"):
        out[col.replace("pred", "ok")] = out[col] == out["truth"]
    out.to_csv(OUT_DIR / out_name, index=False)
    return out


def summarize(name, out):
    n = len(out)
    print(f"\n{name}: {n} held-out cells")
    for col, label in [("ok_proc", "full procedure (CV-fit thresholds)"),
                       ("ok_rule", "single-threshold rule (CV-fit HI)"),
                       ("ok_always_ruleope", "always-RuleOPE baseline"),
                       ("ok_always_mrdr", "always-MRDR baseline")]:
        k = int(out[col].sum())
        print(f"  {label:38s} {k}/{n} = {100 * k / n:.1f}%")


def main():
    df = load_cell_table()
    in_grid = df[df["group"] == "in_grid"].copy()
    held_out = df[df["group"] != "in_grid"].copy()
    print(f"cells: {len(in_grid)} in-grid, {len(held_out)} held-out "
          f"({dict(held_out['group'].value_counts())})")

    print("\n=== E1(a) leave-one-LLM-out (12 folds, 36 in-grid cells) ===")
    lolo = run_cv(in_grid, "llm", "e1_lolo.csv")
    summarize("LOLO", lolo)

    print("\n=== E1(b) leave-one-benchmark-out (3 folds) ===")
    lobo = run_cv(in_grid, "bench", "e1_lobo.csv")
    summarize("LOBO", lobo)

    print("\n=== E1(c) frozen-selector transfer to held-out cells ===")
    lo_f, hi_f, acc_f = fit_thresholds(in_grid)
    print(f"  thresholds fit on 36 in-grid cells: (LO,HI)=({lo_f:.2f},{hi_f:.2f}), "
          f"in-grid acc={acc_f:.3f}")
    rows = []
    for label, (lo, hi) in [("fit_in_grid", (lo_f, hi_f)),
                            ("paper_fixed", (PAPER_LO, PAPER_HI))]:
        for r in evaluate_fold(held_out, lo, hi):
            r["selector"] = label
            r["lo"] = lo
            r["hi"] = hi
            rows.append(r)
    out = pd.DataFrame(rows)
    for col in ("pred_proc", "pred_rule", "pred_always_ruleope", "pred_always_mrdr"):
        out[col.replace("pred", "ok")] = out[col] == out["truth"]
    # per-group accuracy for the paper-fixed frozen selector
    for label in ("fit_in_grid", "paper_fixed"):
        sub = out[out["selector"] == label]
        summarize(f"frozen transfer [{label}] — all held-out", sub)
        for grp, gsub in sub.merge(
                held_out[["bench", "llm", "group"]], on=["bench", "llm"]).groupby("group"):
            k = int(gsub["ok_proc"].sum())
            print(f"    {grp:10s} proc: {k}/{len(gsub)} = {100 * k / len(gsub):.1f}%")
    out.to_csv(OUT_DIR / "e1_frozen_transfer.csv", index=False)
    print(f"\nwrote {OUT_DIR}/e1_lolo.csv, e1_lobo.csv, e1_frozen_transfer.csv")


if __name__ == "__main__":
    main()
