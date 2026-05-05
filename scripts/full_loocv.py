"""Full-procedure leave-one-out cross-validation for the §7C.14 interaction models.

Three CV protocols on the 36-cell grid:

  1. LOOCV (leave-one-cell-out): 36 folds, remove 1 (LLM, benchmark) cell at a time.
     Strictest cell-level generalization test.
  2. LOLO (leave-one-LLM-out): 12 folds, remove all 3 cells of one LLM.
     Strict LLM-level generalization test (already in /tmp/lolo_cv.py).
  3. LOBO (leave-one-benchmark-out): 3 folds, remove all 12 cells of one benchmark.
     Strict benchmark-level generalization test.

Plus baselines:
  - Intercept-only model (predict the dataset mean)
  - Benchmark-mean only (predict each cell by its training-fold benchmark mean)
  - LLM-mean only (predict each cell by its training-fold LLM mean)

Reports R², RMSE, and the gain over the baseline for every (response, predictor, CV) combo.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "experiments/results/full36_phenomenon_pairs.csv")
n = len(df)
llms = df["llm"].unique()
benchmarks = df["benchmark"].unique()
print(f"Dataset: {n} cells, {len(llms)} LLMs × {len(benchmarks)} benchmarks\n")


def cv_r2(actuals, preds):
    valid = ~np.isnan(preds)
    if valid.sum() < 2: return float("nan"), float("nan")
    ss_res = float(np.sum((actuals[valid] - preds[valid]) ** 2))
    ss_tot = float(np.sum((actuals[valid] - actuals[valid].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / valid.sum()))
    return r2, rmse


def loocv(formula, df, response):
    actuals = df[response].values
    preds = np.full(n, np.nan)
    for i in range(n):
        train = df.drop(df.index[i])
        test = df.iloc[[i]]
        try:
            m = smf.ols(formula, data=train).fit()
            preds[i] = float(m.predict(test).iloc[0])
        except Exception:
            preds[i] = np.nan
    return cv_r2(actuals, preds)


def lolo(formula, df, response):
    actuals = df[response].values
    preds = np.full(n, np.nan)
    for held in llms:
        train = df[df["llm"] != held]
        test = df[df["llm"] == held]
        try:
            m = smf.ols(formula, data=train).fit()
            preds[df["llm"] == held] = m.predict(test).values
        except Exception:
            preds[df["llm"] == held] = np.nan
    return cv_r2(actuals, preds)


def lobo(formula, df, response):
    actuals = df[response].values
    preds = np.full(n, np.nan)
    for held in benchmarks:
        train = df[df["benchmark"] != held]
        test = df[df["benchmark"] == held]
        try:
            m = smf.ols(formula, data=train).fit()
            preds[df["benchmark"] == held] = m.predict(test).values
        except Exception:
            preds[df["benchmark"] == held] = np.nan
    return cv_r2(actuals, preds)


def baseline_intercept(df, response):
    """Predict by mean of training set."""
    actuals = df[response].values
    preds = np.full(n, np.nan)
    for i in range(n):
        train = df.drop(df.index[i])
        preds[i] = train[response].mean()
    return cv_r2(actuals, preds)


def baseline_bench_mean(df, response, cv="loocv"):
    """Predict each cell by training-fold benchmark mean."""
    actuals = df[response].values
    preds = np.full(n, np.nan)
    if cv == "loocv":
        for i in range(n):
            train = df.drop(df.index[i])
            test = df.iloc[i]
            m = train.groupby("benchmark")[response].mean()
            preds[i] = m.get(test["benchmark"], np.nan)
    elif cv == "lolo":
        for held in llms:
            train = df[df["llm"] != held]
            test = df[df["llm"] == held]
            m = train.groupby("benchmark")[response].mean()
            preds[df["llm"] == held] = test["benchmark"].map(m).values
    elif cv == "lobo":
        for held in benchmarks:
            train = df[df["benchmark"] != held]
            test = df[df["benchmark"] == held]
            m = train.groupby("benchmark")[response].mean()
            # held-out benchmark has NO benchmark-mean to fall back to → use overall train mean
            preds[df["benchmark"] == held] = train[response].mean()
    return cv_r2(actuals, preds)


def baseline_llm_mean(df, response, cv="loocv"):
    """Predict each cell by training-fold LLM mean."""
    actuals = df[response].values
    preds = np.full(n, np.nan)
    if cv == "loocv":
        for i in range(n):
            train = df.drop(df.index[i])
            test = df.iloc[i]
            m = train.groupby("llm")[response].mean()
            preds[i] = m.get(test["llm"], np.nan)
    elif cv == "lolo":
        for held in llms:
            train = df[df["llm"] != held]
            preds[df["llm"] == held] = train[response].mean()
    elif cv == "lobo":
        for held in benchmarks:
            train = df[df["benchmark"] != held]
            test = df[df["benchmark"] == held]
            m = train.groupby("llm")[response].mean()
            preds[df["benchmark"] == held] = test["llm"].map(m).values
    return cv_r2(actuals, preds)


# === Run all combos ===
RESPONSES = ["RuleOPE_pct", "DR_pct", "MRDR_pct", "MRDR_vs_RuleOPE_pct"]
PREDICTORS = ["noop_F1", "sigma_R2", "bridge_rate"]

print("=" * 110)
print(f"{'response':22s} {'predictor':12s} {'in-sample':>11s} {'LOOCV':>11s} {'LOLO':>11s} {'LOBO':>11s} {'rmse@LOO':>10s}")
print("-" * 110)
for response in RESPONSES:
    for predictor in PREDICTORS:
        formula = f"{response} ~ {predictor} * C(benchmark)"
        m_full = smf.ols(formula, data=df).fit()
        loo_r2, loo_rmse = loocv(formula, df, response)
        lolo_r2, _ = lolo(formula, df, response)
        lobo_r2, _ = lobo(formula, df, response)
        print(f"{response:22s} {predictor:12s} {m_full.rsquared_adj:>11.3f} {loo_r2:>11.3f} {lolo_r2:>11.3f} {lobo_r2:>11.3f} {loo_rmse:>10.1f}")
print()

print("=" * 110)
print("Baselines (LOOCV):")
print("-" * 110)
print(f"{'response':22s} {'baseline':30s} {'LOOCV R²':>12s} {'LOOCV RMSE':>12s}")
print("-" * 110)
for response in RESPONSES:
    r2_int, rmse_int = baseline_intercept(df, response)
    r2_bm, rmse_bm = baseline_bench_mean(df, response, "loocv")
    r2_lm, rmse_lm = baseline_llm_mean(df, response, "loocv")
    print(f"{response:22s} {'intercept-only':30s} {r2_int:>12.3f} {rmse_int:>12.1f}")
    print(f"{response:22s} {'benchmark-mean':30s} {r2_bm:>12.3f} {rmse_bm:>12.1f}")
    print(f"{response:22s} {'LLM-mean':30s} {r2_lm:>12.3f} {rmse_lm:>12.1f}")
print()

print("=" * 110)
print("Baselines (LOBO — predict held-out benchmark by training-overall mean):")
print("-" * 110)
print(f"{'response':22s} {'baseline':30s} {'LOBO R²':>12s} {'LOBO RMSE':>12s}")
print("-" * 110)
for response in RESPONSES:
    r2_bm_lobo, rmse_bm_lobo = baseline_bench_mean(df, response, "lobo")
    r2_lm_lobo, rmse_lm_lobo = baseline_llm_mean(df, response, "lobo")
    print(f"{response:22s} {'training-overall mean':30s} {r2_bm_lobo:>12.3f} {rmse_bm_lobo:>12.1f}")
    print(f"{response:22s} {'training-LLM mean (×bench)':30s} {r2_lm_lobo:>12.3f} {rmse_lm_lobo:>12.1f}")

# Headline summary of how much the interaction model adds over the right baseline
print()
print("=" * 110)
print("LOOCV: how much does the (predictor × benchmark) interaction add over benchmark-mean baseline?")
print("-" * 110)
for response in RESPONSES:
    r2_baseline, _ = baseline_bench_mean(df, response, "loocv")
    print(f"\n{response}:")
    print(f"  benchmark-mean baseline (LOOCV):                       {r2_baseline:+.3f}")
    for predictor in PREDICTORS:
        formula = f"{response} ~ {predictor} * C(benchmark)"
        loo_r2, _ = loocv(formula, df, response)
        gain = loo_r2 - r2_baseline
        print(f"  {predictor:12s} × benchmark interaction (LOOCV): {loo_r2:+.3f}  (Δ over baseline: {gain:+.3f})")
