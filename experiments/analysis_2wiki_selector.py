"""Apply σ_R² selector to 2Wiki cells. Reports σ_R², single-rule pred,
full-procedure pred, truth (at N=1200), and cross-substrate accuracy on
the held-out 2WikiMultiHopQA benchmark.

Requires experiments/results/2wiki_*_5estimator.json to exist (run the
multi_estimator_n_sweep.py with --grid 2wiki_subset or 2wiki_12LLM first).
"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag_substrate_2wiki import _load_2wiki

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


samples = _load_2wiki(str(ROOT / "eval/2wiki/dev.parquet"), 1500, 0)
gold = {s.qid: s.answer for s in samples}

def sigma_R2(llm):
    if llm == "mistral":
        path = ROOT / "eval/2wiki/outputs_1500.jsonl"
    else:
        path = ROOT / f"eval/2wiki/outputs_{llm}_1500.jsonl"
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


# Try both potential result files
candidates = [
    ROOT / "experiments/results/2wiki_12LLM_4N_5estimator.json",
    ROOT / "experiments/results/2wiki_subset_4N_5estimator.json",
]
d = None; result_path = None
for c in candidates:
    if c.exists():
        d = json.loads(c.read_text())
        result_path = c
        break
if d is None:
    print("ERROR: no 2wiki OPE result file. Run multi_estimator_n_sweep.py --grid 2wiki_12LLM first.")
    sys.exit(1)

print(f"Using OPE results: {result_path}")
print()

LO, HI = 0.05, 0.10

print(f"{'cell':22s} {'σ_R²':>7s}  {'rule':>9s}  {'proc':>9s}  {'truth':>9s}  {'r':>2s} {'p':>2s}  {'rule≠proc':>10s}")
print("-"*90)

n_total = 0; n_rule = 0; n_proc = 0
n_lo = n_mid = n_hi = 0
results = []

for cell_key, cell in sorted(d["cells"].items()):
    bench, llm = cell_key.split("__", 1)
    if bench != "2wiki": continue
    sc = cell.get("scaling", {})
    if "1200" not in sc or "150" not in sc: continue
    sig = sigma_R2(llm)
    if sig is None: continue
    if sig < LO: n_lo += 1
    elif sig <= HI: n_mid += 1
    else: n_hi += 1
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
    disagree = "YES" if pred_r != pred_p else "no"
    n_total += 1
    n_rule += int(pred_r == truth)
    n_proc += int(pred_p == truth)
    print(f"{llm:22s} {sig:>7.3f}  {pred_r:>9s}  {pred_p:>9s}  {truth:>9s}  {rok:>2s} {pok:>2s}  {disagree:>10s}")
    results.append({"llm": llm, "sigma_R2": sig, "truth": truth,
                    "pred_rule": pred_r, "pred_proc": pred_p,
                    "rule_ok": pred_r == truth, "proc_ok": pred_p == truth})

print("-"*90)
print(f"\nσ_R² band coverage on 2WikiMultiHopQA ({n_total} cells):")
print(f"  σ_R² < {LO}:        {n_lo} cells (low)")
print(f"  {LO} ≤ σ_R² ≤ {HI}: {n_mid} cells (middle)")
print(f"  σ_R² > {HI}:        {n_hi} cells (HIGH — exercises single-rule MRDR branch)")
print()
print(f"Cross-substrate accuracy on 2WikiMultiHopQA:")
print(f"  Single-threshold rule (σ_R² < 0.10 → RuleOPE):  {n_rule}/{n_total} = {100*n_rule/n_total:.1f}%")
print(f"  Full procedure with middle-band pilot:           {n_proc}/{n_total} = {100*n_proc/n_total:.1f}%")

import pandas as pd
df = pd.DataFrame(results)
out = ROOT / "experiments/results/2wiki_selector_eval.csv"
df.to_csv(out, index=False)
print(f"\nwrote {out}")
