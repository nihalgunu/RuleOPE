"""Compute σ_R² for each (LLM, 2wiki) cell from cached generator outputs.

Runs as soon as Lambda outputs land. Produces a quick distribution check
to tell us whether 2WikiMultiHopQA actually exercises the high-σ_R² branch
(reviewer (b)). Output files: eval/2wiki/outputs_<llm>_1500.jsonl.

Usage: python experiments/analysis_2wiki_sigma_R2.py
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
    return float(np.mean(swings)) if swings else None, len(swings)

LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35", "zephyr7b", "mistral",
        "qwen", "qwencoder7b", "internlm7b", "olmo7b", "granite8b", "yi15",
        "qwen14b", "qwen32b"]

print(f"{'llm':14s} {'σ_R²':>8s}  {'n_q':>5s}  {'band':>10s}  {'rule pred':>10s}")
print("-"*60)
for llm in LLMS:
    res = sigma_R2(llm)
    if res is None: continue
    sig, n_q = res
    if sig < 0.05: band = "low"
    elif sig <= 0.10: band = "middle"
    else: band = "HIGH"
    pred = "RuleOPE" if sig < 0.10 else "MRDR"
    print(f"{llm:14s} {sig:>8.4f}  {n_q:>5d}  {band:>10s}  {pred:>10s}")
