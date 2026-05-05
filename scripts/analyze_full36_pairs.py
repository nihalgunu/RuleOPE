"""Analyze the second estimator pair on the 36-cell grid.

Reads experiments/results/full_36cell_5estimator.json and computes:
  1. Per-benchmark Spearman ρ for each estimator pair vs noop F1 (n=12)
  2. OLS interaction models for each pair across 36 cells
  3. Side-by-side comparison: does the §7C.14 phenomenon structure
     hold for MRDR-vs-DR / MRDR-vs-NonCompDR / RuleOPE-vs-MRDR?
"""
import json
import os
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load full_36 data
d = json.loads((ROOT / "experiments/results/full_36cell_5estimator.json").read_text())

# Reuse the existing per-cell observable computation from fit_interaction_model.py
# (mean noop F1 per cell). Hardcode lookup for speed.
_WS = re.compile(r"[^\w\s]")
def _norm(s):
    s = s.lower().strip(); s = _WS.sub(" ", s); s = " ".join(s.split())
    for lead in ("answer:", "a:", "the answer is"):
        if s.startswith(lead): s = s[len(lead):].strip()
    return s
def _f1(p, g):
    p=_norm(p).split(); g=_norm(g).split()
    if not p or not g: return 0.0
    c=set(p)&set(g)
    if not c: return 0.0
    return 2*len(c)/len(p)*len(c)/len(g)/(len(c)/len(p)+len(c)/len(g))


def reward_for(text, sample, bench):
    if _norm(text) in ("unknown", ""): return 0.0
    if bench == "hotpot": return _f1(text, sample.get("answer", ""))
    return max((_f1(text, a) for a in sample.get("answer_aliases", [sample.get("answer","")])), default=0.0)


def outputs_path(llm, bench):
    if llm == "mistral":
        if bench == "hotpot": return ROOT / "eval/hotpot/outputs_1500.jsonl"
        if bench == "trivia": return ROOT / "eval/trivia/outputs_1500.jsonl"
        if bench == "nq":     return ROOT / "eval/nq/outputs_mistral_1500.jsonl"
    return ROOT / f"eval/{bench}/outputs_{llm}_1500.jsonl"


def get_observables(llm, bench):
    """Returns mean noop F1 + within-rule reward variance proxy."""
    p = outputs_path(llm, bench)
    if not p.exists(): return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    by_qid = {}
    for r in rows:
        qid, action = r["id"].rsplit("__", 1)
        by_qid.setdefault(qid, {})[action] = r["text"]

    if bench == "hotpot":
        from src.rag_substrate_hotpot import _load_hotpot
        samples = _load_hotpot(str(ROOT / "eval/hotpot/dev.parquet"), 1500, 0)
        gold = {s.qid: {"answer": s.answer} for s in samples}
    elif bench == "trivia":
        from src.rag_substrate_trivia import _load_trivia
        samples = _load_trivia(str(ROOT / "eval/trivia/dev.parquet"), 1500, 0)
        gold = {s.qid: {"answer_aliases": s.answer_aliases} for s in samples}
    elif bench == "nq":
        from src.rag_substrate_nq import _load_nq
        samples = _load_nq(str(ROOT / "eval/nq/dev.parquet"), 1500, 0)
        gold = {s.qid: {"answer_aliases": s.answer_aliases} for s in samples}

    rewards_per_qid = {}
    for qid, acts in by_qid.items():
        if qid not in gold: continue
        rewards_per_qid[qid] = {a: reward_for(acts.get(a,""), gold[qid], bench)
                                for a in ("noop", "filter", "rerank")}
    noop = [v["noop"] for v in rewards_per_qid.values()]
    # σ_R² proxy: within-query variance of action rewards (noop, filter, rerank)
    sig_R2 = float(np.mean([np.var([v["noop"], v["filter"], v["rerank"]]) for v in rewards_per_qid.values()]))
    # bridge rate
    bridge = float(np.mean([1 if max(abs(v["filter"]-v["noop"]), abs(v["rerank"]-v["noop"])) > 0.3 else 0
                             for v in rewards_per_qid.values()]))
    return {"noop_F1": float(np.mean(noop)) if noop else float("nan"),
            "sigma_R2": sig_R2,
            "bridge_rate": bridge}


# Build the full DataFrame
rows = []
for cell_key, cell in d["cells"].items():
    bench, llm = cell_key.split("__")
    e = cell["scaling"]["150"]
    obs = get_observables(llm, bench) or {}
    rows.append({
        "llm": llm, "benchmark": bench,
        "noop_F1": obs.get("noop_F1"),
        "sigma_R2": obs.get("sigma_R2"),
        "bridge_rate": obs.get("bridge_rate"),
        "RuleOPE_pct": e["RuleOPE"]["pct_mean_logratio"],
        "DR_pct": e["DR"]["pct_mean_logratio"],
        "MRDR_pct": e["MRDR"]["pct_mean_logratio"],
        "SwitchDR_pct": e["SwitchDR"]["pct_mean_logratio"],
        "MRDR_vs_RuleOPE_pct": e["MRDR_vs_RuleOPE"]["pct_mean_logratio"],
        "MRDR_vs_DR_pct": e["MRDR"]["pct_mean_logratio"] - e["DR"]["pct_mean_logratio"],  # naive diff
    })
df = pd.DataFrame(rows)
df.to_csv(ROOT / "experiments/results/full36_phenomenon_pairs.csv", index=False)
print(f"wrote {ROOT / 'experiments/results/full36_phenomenon_pairs.csv'}")
print()
print("=== full_36 dataset ===")
print(df.to_string(index=False))

print("\n\n=== Per-benchmark Spearman ρ for each estimator pair (n=12 each, vs noop F1) ===")
print(f"  {'pair':30s} {'HotpotQA':>15s} {'TriviaQA':>15s} {'NQ':>15s}")
for col in ["RuleOPE_pct", "DR_pct", "MRDR_pct", "SwitchDR_pct", "MRDR_vs_RuleOPE_pct", "MRDR_vs_DR_pct"]:
    rs = []
    for bench in ("hotpot","trivia","nq"):
        sub = df[df.benchmark == bench]
        rho = spearmanr(sub["noop_F1"], sub[col])
        rs.append(f"ρ={rho.statistic:+.2f}(p={rho.pvalue:.3f})")
    print(f"  {col:30s} {rs[0]:>15s} {rs[1]:>15s} {rs[2]:>15s}")

print("\n=== Per-benchmark Spearman ρ vs σ_R² (n=12 each) ===")
print(f"  {'pair':30s} {'HotpotQA':>15s} {'TriviaQA':>15s} {'NQ':>15s}")
for col in ["RuleOPE_pct", "MRDR_pct", "MRDR_vs_RuleOPE_pct", "MRDR_vs_DR_pct"]:
    rs = []
    for bench in ("hotpot","trivia","nq"):
        sub = df[df.benchmark == bench]
        rho = spearmanr(sub["sigma_R2"], sub[col])
        rs.append(f"ρ={rho.statistic:+.2f}(p={rho.pvalue:.3f})")
    print(f"  {col:30s} {rs[0]:>15s} {rs[1]:>15s} {rs[2]:>15s}")

# Interaction models for each pair
print("\n=== Interaction models on 36 cells (full grid, OLS) ===")
for col in ["RuleOPE_pct", "MRDR_pct", "MRDR_vs_RuleOPE_pct", "MRDR_vs_DR_pct"]:
    print(f"\n--- {col} ~ noop_F1 × C(benchmark) ---")
    m = smf.ols(f"{col} ~ noop_F1 * C(benchmark)", data=df).fit()
    print(f"  R²={m.rsquared:.3f}  adj_R²={m.rsquared_adj:.3f}  F={m.fvalue:.2f} (p={m.f_pvalue:.4f})  N={m.nobs:.0f}")
    print(f"  AIC={m.aic:.1f}")
    print(f"\n--- {col} ~ sigma_R2 × C(benchmark) ---")
    m = smf.ols(f"{col} ~ sigma_R2 * C(benchmark)", data=df).fit()
    print(f"  R²={m.rsquared:.3f}  adj_R²={m.rsquared_adj:.3f}  F={m.fvalue:.2f} (p={m.f_pvalue:.4f})  N={m.nobs:.0f}")
    print(f"  AIC={m.aic:.1f}")

# Headline: the interaction model adj_R² for each pair
print("\n\n=== HEADLINE: interaction-model adj_R² for each estimator pair ===")
print(f"{'estimator pair':30s} {'noop_F1×bench':>15s} {'σ_R²×bench':>15s}")
for col in ["RuleOPE_pct", "DR_pct", "MRDR_pct", "SwitchDR_pct", "MRDR_vs_RuleOPE_pct", "MRDR_vs_DR_pct"]:
    m1 = smf.ols(f"{col} ~ noop_F1 * C(benchmark)", data=df).fit()
    m2 = smf.ols(f"{col} ~ sigma_R2 * C(benchmark)", data=df).fit()
    print(f"{col:30s}  {m1.rsquared_adj:>14.3f}  {m2.rsquared_adj:>14.3f}")
