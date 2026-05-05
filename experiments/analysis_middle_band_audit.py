"""Audit middle-band cells (σ_R² ∈ [0.05, 0.10]) across the full 51-cell calibration set:
  - 36 in-grid cells (12 LLMs × 3 benchmarks, calibrated on)
  - 12 MuSiQue cross-substrate cells (held-out)
  - 3 Qwen2.5-14B anchor cells (held-out frontier)
For each cell: σ_R², pred_rule, pred_proc, truth (at N=1200), and where rule≠proc.
"""
import json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_WS = re.compile(r"[^\w\s]")
def _norm(s):
    s = s.lower().strip(); s = _WS.sub(" ", s); s = " ".join(s.split())
    for lead in ("answer:", "a:", "the answer is"):
        if s.startswith(lead): s = s[len(lead):].strip()
    return s
def _f1(p, g):
    p = _norm(p).split(); g = _norm(g).split()
    if not p or not g: return 0.0
    c = set(p) & set(g)
    if not c: return 0.0
    return 2*len(c)/len(p) * len(c)/len(g) / (len(c)/len(p) + len(c)/len(g))

import sys; sys.path.insert(0, str(ROOT))
from src.rag_substrate_hotpot import _load_hotpot
from src.rag_substrate_trivia import _load_trivia
from src.rag_substrate_nq import _load_nq
from src.rag_substrate_musique import _load_musique
from src.rag_substrate_2wiki import _load_2wiki

LOAD = {"hotpot": _load_hotpot, "trivia": _load_trivia, "nq": _load_nq,
        "musique": _load_musique, "2wiki": _load_2wiki}

def _gold_field(bench):
    return "gold_phrase" if bench == "trivia" else "answer"

def sigma_R2(bench, llm):
    parq = str(ROOT / f"eval/{bench}/dev.parquet")
    samples = LOAD[bench](parq, 1500, 0)
    fld = _gold_field(bench)
    gold = {s.qid: getattr(s, fld) for s in samples}
    if bench == "musique" and llm == "mistral":
        path = ROOT / "eval/musique/outputs_1500.jsonl"
    elif bench in ("hotpot", "trivia") and llm == "mistral":
        path = ROOT / f"eval/{bench}/outputs_1500.jsonl"
    else:
        path = ROOT / f"eval/{bench}/outputs_{llm}_1500.jsonl"
    if not path.exists(): return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_qid = {}
    for r in rows:
        qid, action = r["id"].rsplit("__", 1)
        by_qid.setdefault(qid, {})[action] = r["text"]
    swings = []
    for qid, acts in by_qid.items():
        if qid not in gold or len(acts) < 3: continue
        rs = [(0.0 if _norm(acts[a]) in ("unknown","") else _f1(acts[a], gold[qid]))
              for a in ("noop","filter","rerank")]
        swings.append(float(np.var(rs)))
    return float(np.mean(swings)) if swings else None

LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35", "zephyr7b", "mistral",
        "qwen", "qwencoder7b", "internlm7b", "olmo7b", "granite8b", "yi15"]

# Load OPE result JSONs
d_36   = json.loads((ROOT / "experiments/results/full_36cell_4N_5estimator.json").read_text())
d_mu   = json.loads((ROOT / "experiments/results/musique_12LLM_4N_5estimator.json").read_text())
d_q14b = json.loads((ROOT / "experiments/results/qwen14b_anchor_4N_5estimator.json").read_text())
d_2w   = json.loads((ROOT / "experiments/results/2wiki_12LLM_4N_5estimator.json").read_text())
_q32_path = ROOT / "experiments/results/qwen32b_anchor_4N_5estimator.json"
d_q32b = json.loads(_q32_path.read_text()) if _q32_path.exists() else {"cells": {}}

LO, HI = 0.05, 0.10

def evaluate(bench, llm, sc, sig):
    if "1200" not in sc or "150" not in sc: return None
    ro_1200 = sc["1200"]["RuleOPE"]["MSE_mean"]; mr_1200 = sc["1200"]["MRDR"]["MSE_mean"]
    truth = "RuleOPE" if ro_1200 < mr_1200 else "MRDR"
    pred_r = "RuleOPE" if sig < HI else "MRDR"
    if sig < LO: pred_p = "RuleOPE"
    elif sig > HI: pred_p = "MRDR"
    else:
        ro_150 = sc["150"]["RuleOPE"]["MSE_mean"]; mr_150 = sc["150"]["MRDR"]["MSE_mean"]
        pred_p = "RuleOPE" if ro_150 < mr_150 else "MRDR"
    return dict(bench=bench, llm=llm, sigma=sig, pred_r=pred_r, pred_p=pred_p, truth=truth)

results = []
# 36-cell grid
for bench in ["hotpot","trivia","nq"]:
    for llm in LLMS:
        sig = sigma_R2(bench, llm)
        if sig is None: continue
        cell_key = f"{bench}__{llm}"
        if cell_key not in d_36["cells"]: continue
        sc = d_36["cells"][cell_key]["scaling"]
        r = evaluate(bench, llm, sc, sig)
        if r: results.append(r)
# MuSiQue
for llm in LLMS:
    sig = sigma_R2("musique", llm)
    if sig is None: continue
    cell_key = f"musique__{llm}"
    if cell_key not in d_mu["cells"]: continue
    sc = d_mu["cells"][cell_key]["scaling"]
    r = evaluate("musique", llm, sc, sig)
    if r: results.append(r)
# Qwen14b anchor (3 in-grid benchmarks)
for bench in ["hotpot","trivia","nq"]:
    sig = sigma_R2(bench, "qwen14b")
    if sig is None: continue
    cell_key = f"{bench}__qwen14b"
    if cell_key not in d_q14b["cells"]: continue
    sc = d_q14b["cells"][cell_key]["scaling"]
    r = evaluate(bench, "qwen14b", sc, sig)
    if r: results.append(r)
# 2Wiki cross-substrate (12 LLMs + qwen14b + qwen32b = up to 14 cells)
for llm in LLMS + ["qwen14b", "qwen32b"]:
    sig = sigma_R2("2wiki", llm)
    if sig is None: continue
    cell_key = f"2wiki__{llm}"
    if cell_key not in d_2w["cells"]: continue
    sc = d_2w["cells"][cell_key]["scaling"]
    r = evaluate("2wiki", llm, sc, sig)
    if r: results.append(r)
# Qwen32b anchor on hotpot/trivia/nq (in-grid benchmarks)
for bench in ["hotpot","trivia","nq"]:
    sig = sigma_R2(bench, "qwen32b")
    if sig is None: continue
    cell_key = f"{bench}__qwen32b"
    if cell_key not in d_q32b.get("cells", {}): continue
    sc = d_q32b["cells"][cell_key]["scaling"]
    r = evaluate(bench, "qwen32b", sc, sig)
    if r: results.append(r)

print(f"Total cells evaluated: {len(results)}")
print()

# Distribution
sigs = [r["sigma"] for r in results]
print(f"σ_R² distribution: min {min(sigs):.3f} | p25 {np.percentile(sigs,25):.3f} | "
      f"median {np.median(sigs):.3f} | p75 {np.percentile(sigs,75):.3f} | max {max(sigs):.3f}")
print()
n_lo  = sum(1 for s in sigs if s < LO)
n_mid = sum(1 for s in sigs if LO <= s <= HI)
n_hi  = sum(1 for s in sigs if s > HI)
print(f"  σ_R² < {LO}:        {n_lo} cells (single-rule says RuleOPE)")
print(f"  {LO} ≤ σ_R² ≤ {HI}: {n_mid} cells (middle band: rule says RuleOPE, proc runs pilot)")
print(f"  σ_R² > {HI}:        {n_hi} cells (single-rule says MRDR)")
print()

# Middle-band cells (where rule and proc *can* disagree)
print("=== MIDDLE-BAND CELLS (rule and proc can disagree) ===")
print(f"{'cell':28s} {'σ_R²':>7s} {'rule':>9s} {'proc':>9s} {'truth':>9s}  {'rule_ok':>8s} {'proc_ok':>8s}  {'rule≠proc':>10s}")
print("-" * 100)
mid_cells = [r for r in results if LO <= r["sigma"] <= HI]
mid_cells.sort(key=lambda r: r["sigma"])
n_mid_disagree = 0; n_mid_proc_better = 0; n_mid_rule_better = 0
for r in mid_cells:
    rule_ok = r["pred_r"] == r["truth"]
    proc_ok = r["pred_p"] == r["truth"]
    disagree = r["pred_r"] != r["pred_p"]
    if disagree: n_mid_disagree += 1
    if proc_ok and not rule_ok: n_mid_proc_better += 1
    if rule_ok and not proc_ok: n_mid_rule_better += 1
    cell = f"{r['bench']}/{r['llm']}"
    print(f"{cell:28s} {r['sigma']:>7.3f} {r['pred_r']:>9s} {r['pred_p']:>9s} {r['truth']:>9s}  "
          f"{'✓' if rule_ok else '✗':>8s} {'✓' if proc_ok else '✗':>8s}  "
          f"{'YES' if disagree else 'no':>10s}")

print()
print(f"Middle-band summary: {len(mid_cells)} cells")
print(f"  rule ≠ proc:                     {n_mid_disagree}/{len(mid_cells)}")
print(f"  proc correct AND rule wrong:     {n_mid_proc_better}/{len(mid_cells)} (pilot rescues)")
print(f"  rule correct AND proc wrong:     {n_mid_rule_better}/{len(mid_cells)} (pilot harms)")
print(f"  net pilot value:                 +{n_mid_proc_better - n_mid_rule_better}")

# Overall accuracy
n_rule_ok = sum(1 for r in results if r["pred_r"] == r["truth"])
n_proc_ok = sum(1 for r in results if r["pred_p"] == r["truth"])
print()
print(f"Overall accuracy across all {len(results)} cells:")
print(f"  Single-threshold rule:  {n_rule_ok}/{len(results)} = {100*n_rule_ok/len(results):.1f}%")
print(f"  Full procedure:         {n_proc_ok}/{len(results)} = {100*n_proc_ok/len(results):.1f}%")

# Save full table
import pandas as pd
df = pd.DataFrame(results)
df["middle_band"] = (df["sigma"] >= LO) & (df["sigma"] <= HI)
df["rule_ok"] = df["pred_r"] == df["truth"]
df["proc_ok"] = df["pred_p"] == df["truth"]
df["rule_neq_proc"] = df["pred_r"] != df["pred_p"]
out = ROOT / "experiments/results/middle_band_audit.csv"
df.to_csv(out, index=False)
print(f"\nwrote {out}")
