"""Apply the σ_R² selector to qwen14b cells. Tests whether the calibrated
threshold (σ_R² < 0.10 → RuleOPE, > 0.10 → MRDR) generalizes one step above
the 1.7B-9B grid."""
import json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "experiments/results/qwen14b_anchor_4N_5estimator.json").read_text())

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

LOAD = {"hotpot": _load_hotpot, "trivia": _load_trivia, "nq": _load_nq}

def sigma_R2(bench):
    parq = str(ROOT / f"eval/{bench}/dev.parquet")
    samples = LOAD[bench](parq, 1500, 0)
    gold = {s.qid: (s.gold_phrase if bench == "trivia" else s.answer) for s in samples}
    path = ROOT / f"eval/{bench}/outputs_qwen14b_1500.jsonl"
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

LO, HI = 0.05, 0.10
print(f"{'cell':22s} {'σ_R²':>7s}  {'rule':>8s}  {'proc':>8s}  {'truth':>8s}  {'r':>2s} {'p':>2s}")
print("-"*70)
ok_r=ok_p=tot=0
for bench in ["hotpot","trivia","nq"]:
    sig = sigma_R2(bench)
    sc = d["cells"][f"{bench}__qwen14b"]["scaling"]
    ro_1200 = sc["1200"]["RuleOPE"]["MSE_mean"]; mr_1200 = sc["1200"]["MRDR"]["MSE_mean"]
    truth = "RuleOPE" if ro_1200 < mr_1200 else "MRDR"
    pred_r = "RuleOPE" if sig < HI else "MRDR"
    if sig < LO: pred_p = "RuleOPE"
    elif sig > HI: pred_p = "MRDR"
    else:
        ro_150 = sc["150"]["RuleOPE"]["MSE_mean"]; mr_150 = sc["150"]["MRDR"]["MSE_mean"]
        pred_p = "RuleOPE" if ro_150 < mr_150 else "MRDR"
    rm = "✓" if pred_r == truth else "✗"
    pm = "✓" if pred_p == truth else "✗"
    tot += 1; ok_r += int(pred_r == truth); ok_p += int(pred_p == truth)
    print(f"{bench+'/qwen14b':22s} {sig:>7.3f}  {pred_r:>8s}  {pred_p:>8s}  {truth:>8s}  {rm:>2s} {pm:>2s}")
print("-"*70)
print(f"\nQwen2.5-14B anchor:")
print(f"  Single-threshold rule (σ_R²<0.10 → RuleOPE):  {ok_r}/{tot}")
print(f"  Full procedure with middle-band pilot:         {ok_p}/{tot}")

# Also dump the pct_mean_logratio table for write-up
print("\nQwen2.5-14B-Instruct: RuleOPE/MRDR pct vs NonCompDR (mean log-MSE ratio)")
print(f"{'cell':14s} {'N=150':>14s} {'N=300':>14s} {'N=600':>14s} {'N=1200':>14s}")
print("-"*72)
for bench in ["hotpot","trivia","nq"]:
    sc = d["cells"][f"{bench}__qwen14b"]["scaling"]
    parts = []
    for N in ("150","300","600","1200"):
        ro = sc[N]["RuleOPE"]["pct_mean_logratio"]
        mr = sc[N]["MRDR"]["pct_mean_logratio"]
        parts.append(f"{ro:+5.0f}/{mr:+5.0f}")
    print(f"{bench:14s} {parts[0]:>14s} {parts[1]:>14s} {parts[2]:>14s} {parts[3]:>14s}")
