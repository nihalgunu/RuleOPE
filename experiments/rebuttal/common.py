"""Shared infrastructure for the rebuttal experiments (E1–E7).

Loads the cached sweep results, computes σ_R² per cell (cached to CSV), and
exposes the σ_R²-thresholded pilot selector plus a threshold-fitting routine
so the selector can be cross-validated out-of-sample.

Selector semantics (identical to experiments/analysis_middle_band_audit.py):
  σ_R² < LO            → RuleOPE
  σ_R² > HI            → MRDR
  LO ≤ σ_R² ≤ HI       → pilot tiebreaker: whichever of RuleOPE/MRDR has the
                          lower MSE at N=150 in that cell
  truth                → whichever has the lower MSE at N=1200
The paper's deployed thresholds are (LO, HI) = (0.05, 0.10).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments/results"
OUT_DIR = RESULTS / "rebuttal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_LO, PAPER_HI = 0.05, 0.10

LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35", "zephyr7b", "mistral",
        "qwen", "qwencoder7b", "internlm7b", "olmo7b", "granite8b", "yi15"]
IN_GRID_BENCHES = ["hotpot", "trivia", "nq"]

# ---------------------------------------------------------------------------
# Reward normalisation (identical to the sweep / audit scripts)
# ---------------------------------------------------------------------------
_WS = re.compile(r"[^\w\s]")


def _norm(s):
    s = s.lower().strip(); s = _WS.sub(" ", s); s = " ".join(s.split())
    for lead in ("answer:", "a:", "the answer is"):
        if s.startswith(lead):
            s = s[len(lead):].strip()
    return s


def _f1(p, g):
    p = _norm(p).split(); g = _norm(g).split()
    if not p or not g:
        return 0.0
    c = set(p) & set(g)
    if not c:
        return 0.0
    return 2 * len(c) / len(p) * len(c) / len(g) / (len(c) / len(p) + len(c) / len(g))


def outputs_path(bench: str, llm: str) -> Path:
    if bench == "nq":
        return ROOT / f"eval/nq/outputs_{llm}_1500.jsonl"
    if llm == "mistral":
        return ROOT / f"eval/{bench}/outputs_1500.jsonl"
    return ROOT / f"eval/{bench}/outputs_{llm}_1500.jsonl"


def _loaders():
    from src.rag_substrate_hotpot import _load_hotpot
    from src.rag_substrate_trivia import _load_trivia
    from src.rag_substrate_nq import _load_nq
    from src.rag_substrate_musique import _load_musique
    from src.rag_substrate_2wiki import _load_2wiki
    return {"hotpot": _load_hotpot, "trivia": _load_trivia, "nq": _load_nq,
            "musique": _load_musique, "2wiki": _load_2wiki}


def _gold_field(bench):
    return "gold_phrase" if bench == "trivia" else "answer"


def per_query_action_rewards(bench: str, llm: str):
    """{qid: {action: reward}} for the three replay actions, from cached outputs.

    Trivia/NQ rewards are alias-max F1 in the sweep but the σ_R² audit uses
    plain F1 against the primary gold string; we mirror the audit exactly so
    thresholds stay commensurable with the paper's.
    """
    load = _loaders()[bench]
    samples = load(str(ROOT / f"eval/{bench}/dev.parquet"), 1500, 0)
    fld = _gold_field(bench)
    gold = {s.qid: getattr(s, fld) for s in samples}
    path = outputs_path(bench, llm)
    if not path.exists():
        return None
    by_qid = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        qid, action = r["id"].rsplit("__", 1)
        by_qid.setdefault(qid, {})[action] = r["text"]
    out = {}
    for qid, acts in by_qid.items():
        if qid not in gold or len(acts) < 3:
            continue
        out[qid] = {a: (0.0 if _norm(acts[a]) in ("unknown", "") else _f1(acts[a], gold[qid]))
                    for a in ("noop", "filter", "rerank")}
    return out


def sigma_R2(bench: str, llm: str):
    rewards = per_query_action_rewards(bench, llm)
    if not rewards:
        return None
    swings = [float(np.var([rs["noop"], rs["filter"], rs["rerank"]])) for rs in rewards.values()]
    return float(np.mean(swings)) if swings else None


# ---------------------------------------------------------------------------
# Cell table: one row per (bench, llm) with σ_R² + MSEs at every N
# ---------------------------------------------------------------------------
_RESULT_FILES = {
    "in_grid":  RESULTS / "full_36cell_4N_5estimator.json",
    "musique":  RESULTS / "musique_12LLM_4N_5estimator.json",
    "2wiki":    RESULTS / "2wiki_12LLM_4N_5estimator.json",
    "qwen14b":  RESULTS / "qwen14b_anchor_4N_5estimator.json",
    "qwen32b":  RESULTS / "qwen32b_anchor_4N_5estimator.json",
    "musique_anchor": RESULTS / "rebuttal/e6b_musique_anchors.json",
}

ESTIMATORS = ["NonCompDR", "DR", "SwitchDR", "MRDR", "RuleOPE"]
N_VALUES = [150, 300, 600, 1200]

SIGMA_CACHE = OUT_DIR / "sigma_r2_cells.csv"


def load_cell_table(refresh_sigma: bool = False) -> pd.DataFrame:
    """All evaluable cells with group, σ_R², and per-estimator MSE at each N."""
    sig_cache = {}
    if SIGMA_CACHE.exists() and not refresh_sigma:
        for _, row in pd.read_csv(SIGMA_CACHE).iterrows():
            sig_cache[(row["bench"], row["llm"])] = row["sigma_R2"]

    rows = []
    for group, path in _RESULT_FILES.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for cell_key, cell in data.get("cells", {}).items():
            bench, llm = cell_key.split("__")
            sc = cell.get("scaling", {})
            if "150" not in sc or "1200" not in sc:
                continue
            if (bench, llm) not in sig_cache:
                sig = sigma_R2(bench, llm)
                if sig is None:
                    continue
                sig_cache[(bench, llm)] = sig
            row = {"bench": bench, "llm": llm, "group": group,
                   "sigma_R2": sig_cache[(bench, llm)]}
            for N in N_VALUES:
                if str(N) not in sc:
                    continue
                for est in ESTIMATORS:
                    if est in sc[str(N)]:
                        row[f"mse_{est}_{N}"] = sc[str(N)][est]["MSE_mean"]
            rows.append(row)

    pd.DataFrame(
        [{"bench": b, "llm": l, "sigma_R2": s} for (b, l), s in sorted(sig_cache.items())]
    ).to_csv(SIGMA_CACHE, index=False)

    df = pd.DataFrame(rows).drop_duplicates(subset=["bench", "llm"], keep="first")
    df["truth"] = np.where(df["mse_RuleOPE_1200"] < df["mse_MRDR_1200"], "RuleOPE", "MRDR")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

def predict_rule(row, hi=PAPER_HI):
    """Single-threshold rule (no pilot)."""
    return "RuleOPE" if row["sigma_R2"] < hi else "MRDR"


def predict_proc(row, lo=PAPER_LO, hi=PAPER_HI):
    """Full procedure: thresholds + pilot tiebreaker in the middle band."""
    s = row["sigma_R2"]
    if s < lo:
        return "RuleOPE"
    if s > hi:
        return "MRDR"
    return "RuleOPE" if row["mse_RuleOPE_150"] < row["mse_MRDR_150"] else "MRDR"


def selector_accuracy(df, lo, hi):
    preds = df.apply(lambda r: predict_proc(r, lo, hi), axis=1)
    return float((preds == df["truth"]).mean())


LO_GRID = np.round(np.arange(0.00, 0.101, 0.01), 3)
HI_GRID = np.round(np.arange(0.02, 0.201, 0.01), 3)


def fit_thresholds(train_df):
    """Grid-search (LO, HI) maximizing full-procedure accuracy on train cells.

    Ties broken by (1) narrower middle band (cheaper: fewer pilots), then
    (2) proximity to the paper's (0.05, 0.10).
    """
    best = None
    for lo in LO_GRID:
        for hi in HI_GRID:
            if hi < lo:
                continue
            acc = selector_accuracy(train_df, lo, hi)
            width = hi - lo
            dist = abs(lo - PAPER_LO) + abs(hi - PAPER_HI)
            key = (-acc, width, dist)
            if best is None or key < best[0]:
                best = (key, lo, hi, acc)
    _, lo, hi, acc = best
    return float(lo), float(hi), float(acc)
