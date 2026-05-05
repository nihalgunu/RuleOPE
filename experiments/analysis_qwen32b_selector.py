"""Apply σ_R² selector to Qwen2.5-32B-Instruct anchor cells (≥32B frontier scale).
Cells: hotpot, trivia, nq, 2wiki. Computes σ_R² per cell and reports
selector accuracy at N=1200."""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

from src.rag_substrate_hotpot  import _load_hotpot
from src.rag_substrate_trivia  import _load_trivia
from src.rag_substrate_nq      import _load_nq
from src.rag_substrate_2wiki   import _load_2wiki
LOAD = {"hotpot": _load_hotpot, "trivia": _load_trivia, "nq": _load_nq, "2wiki": _load_2wiki}

def gold_for(bench):
    parq = str(ROOT / f"eval/{bench}/dev.parquet")
    samples = LOAD[bench](parq, 1500, 0)
    fld = "gold_phrase" if bench == "trivia" else "answer"
    return {s.qid: getattr(s, fld) for s in samples}

def sigma_R2(bench):
    gold = gold_for(bench)
    path = ROOT / f"eval/{bench}/outputs_qwen32b_1500.jsonl"
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

# Pull OPE results from both qwen32b_anchor and 2wiki_12LLM
d_anchor = json.loads((ROOT / "experiments/results/qwen32b_anchor_4N_5estimator.json").read_text())
d_2wiki  = json.loads((ROOT / "experiments/results/2wiki_12LLM_4N_5estimator.json").read_text())
LO, HI = 0.05, 0.10

print(f"{'cell':22s} {'σ_R²':>7s}  {'rule':>9s}  {'proc':>9s}  {'truth':>9s}  {'r':>2s} {'p':>2s}")
print("-"*80)

cells_data = []
for bench in ["hotpot", "trivia", "nq", "2wiki"]:
    sig = sigma_R2(bench)
    if sig is None: continue
    if bench == "2wiki":
        sc = d_2wiki["cells"].get("2wiki__qwen32b", {}).get("scaling")
    else:
        sc = d_anchor["cells"].get(f"{bench}__qwen32b", {}).get("scaling")
    if not sc or "1200" not in sc or "150" not in sc: continue
    ro_1200 = sc["1200"]["RuleOPE"]["MSE_mean"]; mr_1200 = sc["1200"]["MRDR"]["MSE_mean"]
    truth = "RuleOPE" if ro_1200 < mr_1200 else "MRDR"
    pred_r = "RuleOPE" if sig < HI else "MRDR"
    if sig < LO: pred_p = "RuleOPE"
    elif sig > HI: pred_p = "MRDR"
    else:
        ro_150 = sc["150"]["RuleOPE"]["MSE_mean"]; mr_150 = sc["150"]["MRDR"]["MSE_mean"]
        pred_p = "RuleOPE" if ro_150 < mr_150 else "MRDR"
    rok = "✓" if pred_r == truth else "✗"
    pok = "✓" if pred_p == truth else "✗"
    print(f"{bench+'/qwen32b':22s} {sig:>7.3f}  {pred_r:>9s}  {pred_p:>9s}  {truth:>9s}  {rok:>2s} {pok:>2s}")
    cells_data.append({"bench": bench, "sigma_R2": sig, "truth": truth,
                       "pred_rule": pred_r, "pred_proc": pred_p})

print("-"*80)
n_total = len(cells_data)
if n_total:
    n_rule = sum(1 for c in cells_data if c["pred_rule"] == c["truth"])
    n_proc = sum(1 for c in cells_data if c["pred_proc"] == c["truth"])
    print(f"\nQwen2.5-32B (≥32B frontier scale) selector accuracy across {n_total} cells:")
    print(f"  Single-threshold rule:   {n_rule}/{n_total}")
    print(f"  Full procedure:          {n_proc}/{n_total}")

    # Frontier scale aggregate (qwen14b + qwen32b)
    print("\nCombined frontier-scale (qwen14b 3 cells + qwen32b 4 cells):")
    d_q14 = json.loads((ROOT / "experiments/results/qwen14b_anchor_4N_5estimator.json").read_text())
    print("  See updates.md §5/§7 for combined picture.")

import pandas as pd
df = pd.DataFrame(cells_data)
out = ROOT / "experiments/results/qwen32b_selector_eval.csv"
df.to_csv(out, index=False)
print(f"\nwrote {out}")
