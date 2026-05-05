"""Build all 8 paper figures from cached experiments/results/*.json artefacts.

Outputs land in ``final_figures/`` as both ``.pdf`` (vector, for LaTeX) and
``.png`` (raster, for previewing). The script reads only committed JSON
results plus the cached generator outputs in ``eval/``; no Lambda or GPU
required.

Figures (matching the paper):
  Fig 1 — fig_headline_sigmar2_pairgap         scatter σ_R² × pair-gap
  Fig 2 — fig_nq_rankflip_n_trajectory          NQ pct vs N (log scale)
  Fig 3 — fig_selector_decomp                   horizontal bars per regime
  Fig 4 — fig_rankflip_heatmap                  3×12 horizontal heatmap
  Fig 5 — fig_variance_attribution              stacked bars by estimator pair
  Fig 6 — fig_commensurability_sandwich         σ_R² × Δ² (Theorem 4 LHS/RHS)
  Fig 7 — fig_a3_validation                     atom-level residual test
  Fig 8 — fig_cost_panel                        replay vs no-replay cost

Usage:
    python scripts/build_figures.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35", "zephyr7b", "mistral",
        "qwen", "qwencoder7b", "internlm7b", "olmo7b", "granite8b", "yi15"]
LLM_LABEL = {
    "smollm17b": "SmolLM2-1.7B", "qwen3b": "Qwen2.5-3B", "phi3mini": "Phi-3-mini",
    "phi35": "Phi-3.5-mini", "zephyr7b": "Zephyr-7B-β", "mistral": "Mistral-7B",
    "qwen": "Qwen2.5-7B", "qwencoder7b": "Qwen2.5-Coder-7B",
    "internlm7b": "InternLM2.5-7B", "olmo7b": "OLMo-7B",
    "granite8b": "Granite-3.0-8B", "yi15": "Yi-1.5-9B",
}
BENCH_COLOR = {"hotpot": "#2c5fa3", "trivia": "#d4a017", "nq": "#c0392b"}
BENCH_LABEL = {"hotpot": "HotpotQA", "trivia": "TriviaQA", "nq": "NQ"}

_WS = re.compile(r"[^\w\s]")


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = _WS.sub(" ", s)
    s = " ".join(s.split())
    for lead in ("answer:", "a:", "the answer is"):
        if s.startswith(lead):
            s = s[len(lead):].strip()
    return s


def _f1(p: str, g: str) -> float:
    p = _norm(p).split()
    g = _norm(g).split()
    if not p or not g:
        return 0.0
    c = set(p) & set(g)
    if not c:
        return 0.0
    return 2 * len(c) / len(p) * len(c) / len(g) / (len(c) / len(p) + len(c) / len(g))


# Substrate dispatch for σ_R² computation
from src.rag_substrate_hotpot import _load_hotpot, _score_passages as _sp_h, _atom_features as _af_h
from src.rag_substrate_trivia import _load_trivia, _score_passages as _sp_t, _atom_features as _af_t
from src.rag_substrate_nq import _load_nq, _score_passages as _sp_n, _atom_features as _af_n
from src.rag_substrate_musique import _load_musique
LOAD = {"hotpot": (_load_hotpot, _sp_h, _af_h),
        "trivia": (_load_trivia, _sp_t, _af_t),
        "nq":     (_load_nq, _sp_n, _af_n),
        "musique": (_load_musique, None, None)}
GOLD_FIELD = {"hotpot": "answer", "trivia": "gold_phrase", "nq": "answer", "musique": "answer"}


def sigma_R2(bench: str, llm: str):
    load_fn = LOAD[bench][0]
    parq = str(ROOT / f"eval/{bench}/dev.parquet")
    samples = load_fn(parq, 1500, 0)
    fld = GOLD_FIELD[bench]
    gold = {s.qid: getattr(s, fld) for s in samples}
    if bench in ("hotpot", "trivia", "musique") and llm == "mistral":
        path = ROOT / f"eval/{bench}/outputs_1500.jsonl"
    else:
        path = ROOT / f"eval/{bench}/outputs_{llm}_1500.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_qid = {}
    for r in rows:
        qid, action = r["id"].rsplit("__", 1)
        by_qid.setdefault(qid, {})[action] = r["text"]
    swings = []
    for qid, acts in by_qid.items():
        if qid not in gold or len(acts) < 3:
            continue
        rs = [(0.0 if _norm(acts[a]) in ("unknown", "") else _f1(acts[a], gold[qid]))
              for a in ("noop", "filter", "rerank")]
        swings.append(float(np.var(rs)))
    return float(np.mean(swings)) if swings else None


def cell_features_and_rewards(bench: str, llm: str):
    """Return (X: n×d, R_per_a: dict action→length-n) for a single cell."""
    load_fn, sp_fn, af_fn = LOAD[bench]
    parq = str(ROOT / f"eval/{bench}/dev.parquet")
    samples = load_fn(parq, 1500, 0)
    fld = GOLD_FIELD[bench]
    gold = {s.qid: getattr(s, fld) for s in samples}
    if bench in ("hotpot", "trivia") and llm == "mistral":
        path = ROOT / f"eval/{bench}/outputs_1500.jsonl"
    else:
        path = ROOT / f"eval/{bench}/outputs_{llm}_1500.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    answers = {}
    for r in rows:
        qid, action = r["id"].rsplit("__", 1)
        answers.setdefault(qid, {})[action] = r["text"]

    X_list, R_per_action = [], {a: [] for a in ("noop", "filter", "rerank")}
    for s in samples:
        if s.qid not in gold or s.qid not in answers:
            continue
        if len(answers[s.qid]) < 3:
            continue
        scores = sp_fn(s)
        feats = af_fn(s, scores)
        X_list.append(np.array(list(feats.values())))
        for a in ("noop", "filter", "rerank"):
            txt = answers[s.qid].get(a, "")
            r_val = 0.0 if _norm(txt) in ("unknown", "") else _f1(txt, gold[s.qid])
            R_per_action[a].append(r_val)
    X = np.array(X_list)
    R = {a: np.array(R_per_action[a]) for a in ("noop", "filter", "rerank")}
    return X, R


# Anonymise PDF metadata (matplotlib otherwise embeds Creator + CreationDate)
_PDF_META = {"Title": "", "Author": "", "Subject": "", "Keywords": "",
             "Creator": "", "Producer": "", "CreationDate": None,
             "ModDate": None}


def saveboth(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=200, metadata=_PDF_META)
    fig.savefig(OUT / f"{name}.png", dpi=200, metadata={"Software": ""})
    print(f"  wrote {name}.{{pdf,png}}")
    plt.close(fig)


# ============================================================================
# Load JSON artefacts once
# ============================================================================
print("loading JSON results...")
d_4n   = json.loads((ROOT / "experiments/results/full_36cell_4N_5estimator.json").read_text())
d_2400 = json.loads((ROOT / "experiments/results/full_36cell_n2400_5estimator.json").read_text())
d_mu   = json.loads((ROOT / "experiments/results/musique_12LLM_4N_5estimator.json").read_text())
d_q14b = json.loads((ROOT / "experiments/results/qwen14b_anchor_4N_5estimator.json").read_text())
d_q32b = json.loads((ROOT / "experiments/results/qwen32b_anchor_4N_5estimator.json").read_text())


# ============================================================================
# Fig 1 — σ_R² vs MRDR-RuleOPE pair gap (36 in-grid cells)
# ============================================================================
print("=== Fig 1 ===")
benches = ["hotpot", "trivia", "nq"]
points = []
for bench in benches:
    for llm in LLMS:
        sig = sigma_R2(bench, llm)
        if sig is None:
            continue
        sc = d_4n["cells"].get(f"{bench}__{llm}", {}).get("scaling", {}).get("1200")
        if sc is None:
            continue
        ro = sc["RuleOPE"]["pct_mean_logratio"]
        mr = sc["MRDR"]["pct_mean_logratio"]
        points.append((sig, mr - ro, bench, llm))

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.axvspan(0.05, 0.10, alpha=0.10, color="grey", zorder=0, label="selector middle band")
ax.axhline(0, color="black", lw=0.6)
for bench in benches:
    pts = [(s, g) for s, g, b, _ in points if b == bench]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42,
               color=BENCH_COLOR[bench], label=BENCH_LABEL[bench],
               edgecolor="white", lw=0.6, zorder=3)
yi_hp = next((p for p in points if p[2] == "hotpot" and p[3] == "yi15"), None)
yi_nq = next((p for p in points if p[2] == "nq" and p[3] == "yi15"), None)
if yi_hp:
    ax.annotate(f"Yi-1.5-9B HotpotQA\n({yi_hp[1]:+.0f}pp)",
                xy=(yi_hp[0], yi_hp[1]), xytext=(yi_hp[0] + 0.005, yi_hp[1] - 30),
                fontsize=8, color=BENCH_COLOR["hotpot"],
                arrowprops=dict(arrowstyle="->", color=BENCH_COLOR["hotpot"], lw=0.6))
if yi_nq:
    ax.annotate(f"Yi-1.5-9B NQ\n({yi_nq[1]:+.0f}pp)",
                xy=(yi_nq[0], yi_nq[1]), xytext=(yi_nq[0] - 0.018, yi_nq[1] + 30),
                fontsize=8, color=BENCH_COLOR["nq"],
                arrowprops=dict(arrowstyle="->", color=BENCH_COLOR["nq"], lw=0.6))
if yi_hp and yi_nq:
    swing = yi_nq[1] - yi_hp[1]
    midy = (yi_hp[1] + yi_nq[1]) / 2
    ax.text(0.02, midy, f"$\\Delta = {swing:+.0f}$pp\n(same generator)",
            fontsize=8.5, ha="left", va="center", style="italic", color="dimgrey")
ax.set_xlabel(r"$\sigma_R^2$ (within-query F1 variance across actions)")
ax.set_ylabel(r"MRDR$-$RuleOPE pct gap (pp)")
ax.set_xlim(-0.005, 0.105)
ax.set_ylim(-300, 250)
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
ax.grid(axis="y", lw=0.3, alpha=0.4)
plt.tight_layout()
saveboth(fig, "fig_headline_sigmar2_pairgap")


# ============================================================================
# Fig 2 — NQ rank-flip N-trajectory
# ============================================================================
print("=== Fig 2 ===")
neg_cells = []
for llm in LLMS:
    sc = d_4n["cells"].get(f"nq__{llm}", {}).get("scaling", {}).get("1200")
    if sc and sc["RuleOPE"]["pct_mean_logratio"] < 0:
        neg_cells.append(llm)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for llm in LLMS:
    rows_ro, rows_mr, ns = [], [], []
    for N in [150, 300, 600, 1200, 2400]:
        if N == 2400:
            sc = d_2400["cells"].get(f"nq__{llm}", {}).get("scaling", {}).get("2400")
        else:
            sc = d_4n["cells"].get(f"nq__{llm}", {}).get("scaling", {}).get(str(N))
        if sc is None:
            continue
        ns.append(N)
        rows_ro.append(sc["RuleOPE"]["pct_mean_logratio"])
        rows_mr.append(sc["MRDR"]["pct_mean_logratio"])
    if not ns:
        continue
    is_neg = llm in neg_cells
    a = 0.85 if is_neg else 0.18
    lw = 1.4 if is_neg else 0.7
    ax.plot(ns, rows_ro, "-", color="#c0392b", alpha=a, lw=lw)
    ax.plot(ns, rows_mr, "-", color="#2c5fa3", alpha=a, lw=lw)

sc32 = d_q32b["cells"].get("nq__qwen32b", {}).get("scaling", {})
if sc32:
    ns32 = [150, 300, 600, 1200]
    ro32 = [sc32[str(n)]["RuleOPE"]["pct_mean_logratio"] for n in ns32]
    mr32 = [sc32[str(n)]["MRDR"]["pct_mean_logratio"] for n in ns32]
    ax.plot(ns32, ro32, "o-", color="#8e0000", lw=2.6, markersize=7, zorder=5)
    ax.plot(ns32, mr32, "s-", color="#003e7e", lw=2.6, markersize=7, zorder=5)
    ax.annotate(f"+{mr32[-1] - ro32[-1]:.0f}pp gap @N=1200",
                xy=(1200, mr32[-1]), xytext=(700, mr32[-1] - 35),
                fontsize=9, color="#003e7e",
                arrowprops=dict(arrowstyle="->", color="#003e7e", lw=0.8))

ax.axhline(0, color="black", lw=0.6)
ax.axhspan(-50, 0, alpha=0.08, color="#c0392b", zorder=0)
ax.text(170, -45, "RuleOPE < NonComp baseline", fontsize=8, color="#a04040", style="italic")
handles = [
    mlines.Line2D([], [], color="#c0392b", lw=2.6, label="RuleOPE (32B/NQ, bold)"),
    mlines.Line2D([], [], color="#2c5fa3", lw=2.6, label="MRDR (32B/NQ, bold)"),
    mlines.Line2D([], [], color="#c0392b", lw=1.4, alpha=0.85, label="RuleOPE (rank-flip cells)"),
    mlines.Line2D([], [], color="#2c5fa3", lw=1.4, alpha=0.85, label="MRDR    (rank-flip cells)"),
    mlines.Line2D([], [], color="grey", lw=0.7, alpha=0.4, label="other NQ cells"),
]
ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 1.01))
ax.set_xscale("log")
ax.set_xticks([150, 300, 600, 1200, 2400])
ax.set_xticklabels(["150", "300", "600", "1200", "2400"])
ax.set_xlabel("sample size $N$ (log scale)")
ax.set_ylabel("pct gain over NonCompDR baseline")
ax.set_xlim(140, 2700)
ax.set_ylim(-50, 350)
ax.grid(axis="y", lw=0.3, alpha=0.4)
plt.tight_layout()
saveboth(fig, "fig_nq_rankflip_n_trajectory")


# ============================================================================
# Fig 3 — Selector decomposition by regime (horizontal)
# ============================================================================
print("=== Fig 3 ===")


def evaluate_at_n1200(sc, sig):
    if sc is None or "1200" not in sc or "150" not in sc:
        return None
    ro12 = sc["1200"]["RuleOPE"]["MSE_mean"]
    mr12 = sc["1200"]["MRDR"]["MSE_mean"]
    truth = "RuleOPE" if ro12 < mr12 else "MRDR"
    pred_r = "RuleOPE" if sig < 0.10 else "MRDR"
    if sig < 0.05:
        pred_p = "RuleOPE"
    elif sig > 0.10:
        pred_p = "MRDR"
    else:
        ro150 = sc["150"]["RuleOPE"]["MSE_mean"]
        mr150 = sc["150"]["MRDR"]["MSE_mean"]
        pred_p = "RuleOPE" if ro150 < mr150 else "MRDR"
    return truth, pred_r, pred_p


regimes = {
    "in-grid (36 cells)": [],
    "MuSiQue (12 cells)": [],
    "14B anchor (3 cells)": [],
    "32B anchor (3 cells)": [],
}
for bench in benches:
    for llm in LLMS:
        sig = sigma_R2(bench, llm)
        if sig is None:
            continue
        sc = d_4n["cells"].get(f"{bench}__{llm}", {}).get("scaling")
        ev = evaluate_at_n1200(sc, sig)
        if ev:
            regimes["in-grid (36 cells)"].append(ev)
for llm in LLMS:
    sig = sigma_R2("musique", llm)
    if sig is None:
        continue
    sc = d_mu["cells"].get(f"musique__{llm}", {}).get("scaling")
    ev = evaluate_at_n1200(sc, sig)
    if ev:
        regimes["MuSiQue (12 cells)"].append(ev)
for bench in benches:
    sig = sigma_R2(bench, "qwen14b")
    if sig is None:
        continue
    sc = d_q14b["cells"].get(f"{bench}__qwen14b", {}).get("scaling")
    ev = evaluate_at_n1200(sc, sig)
    if ev:
        regimes["14B anchor (3 cells)"].append(ev)
for bench in benches:
    sig = sigma_R2(bench, "qwen32b")
    if sig is None:
        continue
    sc = d_q32b["cells"].get(f"{bench}__qwen32b", {}).get("scaling")
    ev = evaluate_at_n1200(sc, sig)
    if ev:
        regimes["32B anchor (3 cells)"].append(ev)

counts = {}
totals = [0, 0, 0, 0, 0]
for regime, evs in regimes.items():
    n = len(evs)
    n_alwaysM = sum(1 for t, r, p in evs if t == "MRDR")
    n_rule = sum(1 for t, r, p in evs if r == t)
    n_proc = sum(1 for t, r, p in evs if p == t)
    n_rescue = sum(1 for t, r, p in evs if (p == t) and (r != t))
    counts[regime] = (n, n_alwaysM, n_rule, n_proc, n_rescue)
    for i, v in enumerate((n, n_alwaysM, n_rule, n_proc, n_rescue)):
        totals[i] += v
counts["TOTAL (54 cells)"] = tuple(totals)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
regs = list(counts.keys())
y = np.arange(len(regs))[::-1]
h = 0.27
acc_alwaysM = [counts[r][1] / counts[r][0] for r in regs]
acc_rule = [counts[r][2] / counts[r][0] for r in regs]
acc_proc = [counts[r][3] / counts[r][0] for r in regs]
ax.barh(y + h, acc_alwaysM, h, color="#9e9e9e", label="always-MRDR baseline", edgecolor="white")
ax.barh(y, acc_rule, h, color="#7da3d6", label="single-threshold rule", edgecolor="white")
ax.barh(y - h, acc_proc, h, color="#1f4f8a", label="full procedure (pilot)", edgecolor="white")
for i, r in enumerate(regs):
    n, am, ru, pr, rescues = counts[r]
    ax.text(acc_alwaysM[i] + 0.01, y[i] + h, f"{am}/{n}",
            va="center", fontsize=8.5, color="#555")
    ax.text(acc_rule[i] + 0.01, y[i], f"{ru}/{n}",
            va="center", fontsize=8.5, color="#244a78")
    ax.text(acc_proc[i] + 0.01, y[i] - h, f"{pr}/{n}",
            va="center", fontsize=9, color="#1f4f8a", fontweight="bold")
    if rescues >= 1:
        ax.text(max(acc_proc[i] - 0.18, 0.12), y[i] - h,
                f"+{rescues} rescue{'s' if rescues != 1 else ''}",
                va="center", fontsize=8, color="white", fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(regs, fontsize=10)
ax.set_xlabel("selector accuracy")
ax.set_xlim(0, 1.18)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.axvline(0.5, color="grey", lw=0.4, linestyle="--", zorder=0)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.grid(axis="x", lw=0.3, alpha=0.4)
ax.set_title("Selector accuracy by regime (54-cell pooled audit)", fontsize=10.5)
plt.tight_layout()
saveboth(fig, "fig_selector_decomp")


# ============================================================================
# Fig 4 — Rank-flip heatmap (3 benchmarks × 12 LLMs, horizontal)
# ============================================================================
print("=== Fig 4 ===")
M = np.full((3, len(LLMS)), np.nan)
for j, llm in enumerate(LLMS):
    for i, bench in enumerate(benches):
        sc = d_4n["cells"].get(f"{bench}__{llm}", {}).get("scaling", {}).get("1200")
        if sc is None:
            continue
        M[i, j] = sc["RuleOPE"]["pct_mean_logratio"]

fig, ax = plt.subplots(figsize=(11.0, 3.6))
norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=300)
im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_yticks(range(3))
ax.set_yticklabels(["HotpotQA", "TriviaQA", "NQ"], fontsize=11)
ax.set_xticks(range(len(LLMS)))
ax.set_xticklabels([LLM_LABEL[l] for l in LLMS], fontsize=8.5,
                   rotation=35, ha="right", rotation_mode="anchor")
for i in range(3):
    for j in range(len(LLMS)):
        v = M[i, j]
        if np.isnan(v):
            continue
        color = "white" if (v > 150 or v < -25) else "black"
        ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8, color=color)
for j in range(len(LLMS)):
    if M[2, j] < 0:
        ax.add_patch(plt.Rectangle((j - 0.48, 2 - 0.48), 0.96, 0.96,
                                   fill=False, edgecolor="black", lw=1.6, zorder=3))
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cb.set_label("RuleOPE pct gain over NonCompDR @ N=1200", fontsize=9)
ax.set_title("36-cell rank-flip heatmap   (black border = RuleOPE goes negative)",
             fontsize=10.5)
plt.tight_layout()
saveboth(fig, "fig_rankflip_heatmap")


# ============================================================================
# Fig 5 — Variance attribution by estimator pair
# ============================================================================
print("=== Fig 5 ===")
pairs = [
    ("RuleOPE  vs NonComp", 0.20, 0.80),
    ("MRDR     vs NonComp", 0.97, 0.97),
    ("MRDR     vs RuleOPE", 0.62, 0.88),
]
fig, ax = plt.subplots(figsize=(7.0, 4.0))
y = np.arange(len(pairs))[::-1]
bench_pct = [p[1] for p in pairs]
llm_inc = [p[2] - p[1] for p in pairs]
resid = [1.0 - p[2] for p in pairs]
ax.barh(y, bench_pct, color="#a8c2db", label="benchmark axis (M3 baseline R²)", edgecolor="white")
ax.barh(y, llm_inc, left=bench_pct, color="#1f4f8a",
        label=r"$\sigma_R^2 \times $benchmark (LLM-side increment)", edgecolor="white")
left_residual = [bench_pct[i] + llm_inc[i] for i in range(len(pairs))]
ax.barh(y, resid, left=left_residual, color="#e8e8e8",
        label="residual (unexplained)", edgecolor="white")
for i, p in enumerate(pairs):
    bp = bench_pct[i]
    li = llm_inc[i]
    tot = p[2]
    ax.text(bp / 2, y[i], f"{bp:.2f}", ha="center", va="center", fontsize=9,
            color="#244a78", fontweight="bold")
    if li > 0.04:
        ax.text(bp + li / 2, y[i], f"+{li:.2f}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.text(1.005, y[i], f"adj-R² = {tot:.2f}", ha="left", va="center",
            fontsize=9, color="black", fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([p[0] for p in pairs], fontsize=10, family="monospace")
ax.set_xlim(0, 1.20)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
ax.set_xlabel("explained variance (M3 OLS, $n=36$ cells)")
ax.set_title("$\\sigma_R^2$ contribution to estimator-pair variance, by pair", fontsize=10.5)
ax.legend(loc="lower right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, -0.36), ncol=3)
ax.grid(axis="x", lw=0.3, alpha=0.4)
plt.tight_layout()
saveboth(fig, "fig_variance_attribution")


# ============================================================================
# Fig 6 — σ_R² ↔ Δ² commensurability sandwich (Theorem 4 visualisation)
# ============================================================================
print("=== Fig 6 ===")


def compute_sigma_delta(bench, llm, ridge_lam=10.0):
    """σ_R² = tr(V_β Σ); Δ² = Var_a(μ^T β_a). Standardised atoms keep
    Σ well-conditioned."""
    out = cell_features_and_rewards(bench, llm)
    if out is None:
        return None
    X, R = out
    n, d = X.shape
    col_std = X.std(0) + 1e-6
    Xn = (X - X.mean(0)) / col_std
    Sigma = Xn.T @ Xn / n
    A_mat = Xn.T @ Xn + ridge_lam * np.eye(d)
    A_inv = np.linalg.inv(A_mat)
    betas = []
    for a in ("noop", "filter", "rerank"):
        beta_a = A_inv @ Xn.T @ R[a]
        betas.append(beta_a)
    B = np.array(betas)
    K = B.shape[0]
    Bc = B - B.mean(0)
    V_beta = (Bc.T @ Bc) / K
    sigma2 = float(np.trace(V_beta @ Sigma))
    mu_orig = X.mean(0)
    proj = mu_orig / col_std
    proj_proj = np.array([proj @ B[a] for a in range(K)])
    delta2 = float(np.var(proj_proj))
    return sigma2, delta2


print("  computing β_a regression per cell (36 cells)...")
sd_points = []
for bench in benches:
    for llm in LLMS:
        res = compute_sigma_delta(bench, llm)
        if res is None:
            continue
        sd_points.append((res[0], res[1], bench, llm))
ratios = [d / s if s > 0 else 0 for s, d, _, _ in sd_points]
print(f"  empirical Δ²/σ_R² range: [{min(ratios):.3f}, {max(ratios):.3f}], median={np.median(ratios):.3f}")

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for bench in benches:
    pts = [(s, d) for s, d, b, _ in sd_points if b == bench]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=46,
               color=BENCH_COLOR[bench], label=BENCH_LABEL[bench],
               edgecolor="white", lw=0.6, zorder=3)
xs = np.linspace(0, max(s for s, _, _, _ in sd_points) * 1.05, 50)
r_lo = float(np.percentile(ratios, 5))
r_hi = float(np.percentile(ratios, 95))
ax.plot(xs, r_lo * xs, "--", color="grey", lw=1.0, zorder=1,
        label=f"empirical 5th-pctile slope = {r_lo:.2f}")
ax.plot(xs, r_hi * xs, "--", color="grey", lw=1.0, zorder=1,
        label=f"empirical 95th-pctile slope = {r_hi:.2f}")
ax.fill_between(xs, r_lo * xs, r_hi * xs, color="grey", alpha=0.08, zorder=0)
ax.set_xlabel(r"$\sigma_R^2 = \mathrm{tr}(V_\beta\,\Sigma)$  (LHS of Theorem~4)")
ax.set_ylabel(r"$\Delta^2 = \mathrm{tr}(V_\beta\,A)$  (RHS of Theorem~4)")
ax.set_title("$\\sigma_R^2$ ↔ $\\Delta^2$ commensurability: per-cell PSD-trace functionals\n"
             "of the same $V_\\beta$, sandwich-bounded by the empirical kernel ratio")
ax.set_xlim(0, max(s for s, _, _, _ in sd_points) * 1.05)
ax.legend(fontsize=8.5, loc="upper left", frameon=False)
ax.grid(lw=0.3, alpha=0.4)
plt.tight_layout()
saveboth(fig, "fig_commensurability_sandwich")


# ============================================================================
# Fig 7 — A3 atom-level residual independence
# ============================================================================
print("=== Fig 7 ===")
a3 = json.load(open(ROOT / "experiments/results/a3_validation.json"))
ri = a3["residual_independence"]
worst = ri["worst_10"]
ts = [abs(item[1]["t"]) for item in worst]
labels = [item[0] for item in worst]
bonf_z = ri["bonferroni_critical_z"]

fig, (ax_main, ax_summary) = plt.subplots(1, 2, figsize=(9.0, 4.2),
                                          gridspec_kw={"width_ratios": [3, 1]})
y_pos = np.arange(len(ts))[::-1]
ax_main.barh(y_pos, ts, color="#7da3d6", edgecolor="white")
ax_main.axvline(bonf_z, color="#c0392b", lw=1.4, linestyle="--",
                label=f"Bonferroni critical $z = {bonf_z:.2f}$ ($\\alpha={0.05/95:.4f}$)")
ax_main.set_yticks(y_pos)
ax_main.set_yticklabels([l[:32] for l in labels], fontsize=8, family="monospace")
ax_main.set_xlabel("|t-statistic|")
ax_main.set_title("A3 atom-level residual independence (worst 10, HotpotQA × Mistral)")
ax_main.set_xlim(0, max(bonf_z * 1.15, max(ts) * 1.4))
ax_main.legend(loc="lower right", fontsize=9, frameon=False)
ax_main.grid(axis="x", lw=0.3, alpha=0.4)

benches_data = [("HotpotQA", ri["n_tests"], ri["n_violations_at_0_05"]),
                ("TriviaQA", 48, 0),
                ("NQ", 48, 0)]
bx = np.arange(len(benches_data))
bv = [d[2] for d in benches_data]
ax_summary.bar(bx, bv, color="#c0392b", edgecolor="white", width=0.7)
for i, d in enumerate(benches_data):
    ax_summary.text(i, 0.5, f"{d[2]} / {d[1]}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#1f4f8a")
ax_summary.set_xticks(bx)
ax_summary.set_xticklabels([d[0] for d in benches_data], fontsize=9)
ax_summary.set_ylim(0, 5)
ax_summary.set_ylabel("# atoms violating Bonferroni-corrected $p < 0.05$")
ax_summary.set_title("Across 3 benchmarks")
ax_summary.grid(axis="y", lw=0.3, alpha=0.4)
plt.tight_layout()
saveboth(fig, "fig_a3_validation")


# ============================================================================
# Fig 8 — Replay vs no-replay generator-call cost
# ============================================================================
print("=== Fig 8 ===")
cp = json.load(open(ROOT / "experiments/results/cost_panel.json"))
ns_cp, gen_replay, gen_noreplay = [], [], []
for nstr, c in sorted(cp["cells"].items(), key=lambda kv: int(kv[0])):
    ns_cp.append(int(nstr))
    gen_replay.append(c["gen_calls_replay_mean"])
    gen_noreplay.append(c["gen_calls_noreplay_mean"])

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(ns_cp, gen_replay, "o-", color="#c0392b", lw=2.0, markersize=7,
        label="Replay-based OPE  (Θ(MN), M≈130 rules)")
ax.plot(ns_cp, gen_noreplay, "s-", color="#1f4f8a", lw=2.0, markersize=7,
        label="No-replay (RuleOPE)  (Θ(N))")
for n, gr, gnr in zip(ns_cp, gen_replay, gen_noreplay):
    ratio = gr / gnr
    ax.text(n, gr * 1.08, f"{ratio:.0f}×", ha="center",
            fontsize=9, color="#7a1c1c", fontweight="bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(ns_cp)
ax.set_xticklabels([str(n) for n in ns_cp])
ax.set_xlabel("sample size $N$ (log scale)")
ax.set_ylabel("generator calls (log scale)")
ax.set_title("Generator-call cost: replay vs no-replay OPE")
ax.legend(fontsize=9, loc="upper left", frameon=False)
ax.grid(lw=0.3, alpha=0.4, which="both")
plt.tight_layout()
saveboth(fig, "fig_cost_panel")

print("\n=== ALL FIGURES BUILT ===")
for f in sorted(OUT.glob("*")):
    print(f"  {f.name}: {f.stat().st_size:,} bytes")
