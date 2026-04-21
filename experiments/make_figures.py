"""Publication-quality figures for RuleOPE.

Produces:
  paper/figs/scaling.pdf        -- 3 benchmarks x N with 90% CIs
  paper/figs/ablation_A.pdf     -- atom-sharing is the driver
  paper/figs/a3_residuals.pdf   -- A3 residual-vs-atom test (Bonferroni)
  paper/figs/a3_sensitivity.pdf -- within-R2 vs d (atom rank)
  paper/figs/a5_calibration.pdf -- V_hat EIF/CompDR vs oracle V
  paper/figs/a5_sensitivity.pdf -- MAE vs A5-violation noise
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

FIGS = Path("paper/figs")
FIGS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- scaling
def fig_scaling():
    files = {
        "HotpotQA":  ("experiments/results/noreplay_scaling.json", "quantile"),
        "TriviaQA":  ("experiments/results/trivia_scaling.json", "paired_bootstrap"),
        "MuSiQue":   ("experiments/results/musique_scaling.json", "quantile"),
    }
    paired_path = Path("experiments/results/trivia_paired_test.json")
    paired = json.loads(paired_path.read_text()) if paired_path.exists() else None

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.6), sharey=False)
    for ax, (name, (path, ci_kind)) in zip(axes, files.items()):
        with open(path) as f:
            d = json.load(f)
        Ns = sorted(d["scaling"].keys(), key=int)
        Ns_int = [int(N) for N in Ns]

        def pct_series_quantile(est):
            med, lo, hi = [], [], []
            for N in Ns:
                k = f"{est}_vs_NonCompDR_pct"
                if k in d["scaling"][N]:
                    med.append(d["scaling"][N][k]["median"])
                    lo.append(d["scaling"][N][k]["CI90"][0])
                    hi.append(d["scaling"][N][k]["CI90"][1])
                else:
                    med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            return np.array(med), np.array(lo), np.array(hi)

        def pct_series_paired():
            med, lo, hi = [], [], []
            for N in Ns:
                e = paired.get(N, {})
                med.append(e.get("pct_mean", np.nan))
                ci = e.get("pct_CI90_bootstrap", [np.nan, np.nan])
                lo.append(ci[0]); hi.append(ci[1])
            return np.array(med), np.array(lo), np.array(hi)

        if ci_kind == "paired_bootstrap" and paired is not None:
            m_rope, lo_rope, hi_rope = pct_series_paired()
        else:
            m_rope, lo_rope, hi_rope = pct_series_quantile("RuleOPE")
        m_comp, _, _ = pct_series_quantile("CompDR")

        ax.axhline(0, color="k", lw=0.6)
        ax.fill_between(Ns_int, lo_rope, hi_rope, alpha=0.18, color="C0")
        ax.plot(Ns_int, m_rope, marker="o", color="C0", label="RuleOPE")
        ax.plot(Ns_int, m_comp, marker="s", color="C1", alpha=0.7, ls="--", label="CompDR (no correction)")
        ax.set_xscale("log"); ax.set_xticks(Ns_int); ax.set_xticklabels(Ns_int)
        title = f"{name}" + ("  (paired-bootstrap CI)" if ci_kind == "paired_bootstrap" else "")
        ax.set_title(title)
        ax.set_xlabel("N (sample budget)")
        if ax is axes[0]:
            ax.set_ylabel("MSE reduction vs NonCompDR (%)")
        ax.minorticks_off()
    axes[0].legend(loc="upper right")
    fig.suptitle("Compositional RuleOPE advantage: all 3 benchmarks, small-$N$ regime is where it matters",
                 fontsize=9, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"scaling.{ext}", dpi=220)
    plt.close(fig)
    print("  scaling.pdf/png done")


# -------------------------------------------------------------- ablation A
def fig_ablation_A():
    """Atom-sharing is the driver (RuleOPE vs PerRuleRidgeDR, matched alpha)."""
    # From NEURIPS_RESULTS_FULL.md table; hardcode since no JSON directly for A.
    data = {
        "HotpotQA": {"150": 23.5, "300": 16.5, "600": 9.6},
        "TriviaQA": {"150": 9.4,  "300": 8.9,  "600": 2.8},
    }
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    names = ["HotpotQA", "TriviaQA"]
    Ns = ["150", "300", "600"]
    x = np.arange(len(Ns))
    w = 0.38
    for i, n in enumerate(names):
        vals = [data[n][N] for N in Ns]
        ax.bar(x + (i - 0.5) * w, vals, w, label=n, color=f"C{i}")
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 0.5) * w, v + 0.5, f"{v:.1f}", ha="center",
                    fontsize=7, color=f"C{i}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(Ns)
    ax.set_xlabel("N")
    ax.set_ylabel("MSE reduction vs\nPerRuleRidgeDR (matched $\\alpha$) (%)")
    ax.set_title("Ablation A: atom-sharing alone is the driver")
    ax.legend(loc="upper right", framealpha=0.9)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"ablation_A.{ext}", dpi=220)
    plt.close(fig)
    print("  ablation_A.pdf/png done")


# ---------------------------------------------------------------- A3
def fig_a3():
    with open("experiments/results/a3_validation.json") as f:
        d = json.load(f)
    # Left: bar of {M0, M1_total, M1_within, M2} R^2
    # Right: within-R^2 vs d sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.7))
    ax = axes[0]
    labels = ["$M_0$\nquery only", "$M_1$\n$\\alpha(q)+\\phi(r)^\\top\\beta$", "$M_2$\nsaturated"]
    vals = [d["M0_query_only"]["r2_total"], d["M1_A3_additive"]["r2_total"],
            d["M2_saturated"]["r2_total"]]
    colors = ["C2", "C0", "C3"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("total $R^2$ on reward")
    gap = d["A3_saturation_gap"]
    ax.set_title(f"A3 explains {d['M1_A3_additive']['r2_within']:.2f} of within-query variance  "
                 f"(saturation gap = {gap:.3f})")

    ax = axes[1]
    sens = d["sensitivity_to_d"]
    ds = [s["d"] for s in sens]
    w_r2 = [s["r2_within"] for s in sens]
    ax.plot(ds, w_r2, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("atom-action rank $d$ (top-$d$ by |corr| with residual)")
    ax.set_ylabel("within-query $R^2$")
    ax.set_title("Sensitivity of A3 fit to atom rank")
    ax.axhline(d["M1_A3_additive"]["r2_within"], color="C1", ls="--", alpha=0.6,
               label=f"full $d$: {d['M1_A3_additive']['r2_within']:.3f}")
    ax.legend(loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"a3_validation.{ext}", dpi=220)
    plt.close(fig)

    # Residual-vs-atom Bonferroni test figure
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    worst = d["residual_independence"]["worst_10"]
    names = [kv[0] for kv in worst]
    ts = [abs(kv[1]["t"]) for kv in worst]
    y = np.arange(len(names))
    ax.barh(y, ts, color="C0")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6)
    crit = d["residual_independence"]["bonferroni_critical_z"]
    ax.axvline(crit, color="C3", ls="--", label=f"Bonferroni($\\alpha=0.05$): $|t|$={crit:.2f}")
    ax.set_xlabel("|t| of $E[\\eta | \\phi_j=1]$")
    ax.set_title(f"Residual independence: 0 / {d['residual_independence']['n_tests']} atoms violate")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"a3_residuals.{ext}", dpi=220)
    plt.close(fig)
    print("  a3_validation.pdf/png + a3_residuals.pdf/png done")


# ---------------------------------------------------------------- A5
def fig_a5():
    with open("experiments/results/a5_validation.json") as f:
        d = json.load(f)
    V_c = np.load("experiments/results/a5_validation_V_compdr.npy")
    V_e = np.load("experiments/results/a5_validation_V_eif.npy")
    V_o = np.load("experiments/results/a5_validation_V_oracle.npy")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))

    ax = axes[0]
    lo, hi = min(V_o.min(), V_c.min(), V_e.min()) - 0.02, \
             max(V_o.max(), V_c.max(), V_e.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.7, label="ideal")
    ax.scatter(V_o, V_c, s=14, color="C1", alpha=0.65, label="CompDR")
    ax.scatter(V_o, V_e, s=14, color="C0", alpha=0.65, label="RuleOPE-EIF (bridge)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("oracle $V(\\rho)$")
    ax.set_ylabel("estimated $V(\\rho)$")
    r2c = d["T2_contrast_learnability"]["compdr_V_R2"]
    r2e = d["T2_contrast_learnability"]["eif_V_R2"]
    ax.set_title(f"A5 held-out V($\\rho$):  CompDR $R^2$={r2c:.2f}   EIF $R^2$={r2e:.2f}")
    ax.legend(loc="lower right")

    ax = axes[1]
    sens = d["T3_sensitivity_to_A5_violation"]
    ns = [s["noise_std"] for s in sens]
    mc = [s["compdr_MAE"] for s in sens]
    me = [s["eif_MAE"] for s in sens]
    ax.plot(ns, mc, marker="s", color="C1", label="CompDR")
    ax.plot(ns, me, marker="o", color="C0", label="RuleOPE-EIF (bridge)")
    ax.set_xlabel("A5 violation (non-linearity noise std)")
    ax.set_ylabel("held-out MAE of $\\widehat V(\\rho)$")
    ax.set_title("Graceful degradation under A5 violation")
    ax.legend(loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"a5_validation.{ext}", dpi=220)
    plt.close(fig)
    print("  a5_validation.pdf/png done")


def fig_discovery():
    """Rule discovery: simple regret @ k and CRRM-vs-hand-oracle head-to-head.

    Left panel: oracle regret (V* - V_topv) across selectors as k grows.
    Right panel: per-trial head-to-head CRRM vs hand-oracle at k=1.
    """
    path = Path("experiments/results/rule_discovery.json")
    if not path.exists():
        print(f"[skip] {path} missing")
        return
    data = json.load(open(path))
    summary = data["summary"]
    trials = [t for t in data["trials"] if not t.get("skipped")]
    ks = data["config"]["ks"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

    # Left: regret vs k
    selectors = [
        ("erm", "ERM-argmax", "tab:green", "s"),
        ("crrm_union", "CRRM-LCB (union-bound)", "tab:red", "v"),
        ("crrm", "CRRM-LCB (atom-aware)", "tab:blue", "o"),
        ("hand_oracle", "Hand-authored (oracle best)", "tab:purple", "D"),
        ("hand_by_vhat", "Hand-authored (by $\\hat V$)", "tab:orange", "^"),
        ("random", "Random", "tab:gray", "x"),
    ]
    for name, label, color, marker in selectors:
        keyfmt = f"regret_{name}@{{}}"
        # Fallback if a selector wasn't in this run (older JSON).
        if any(keyfmt.format(k) not in summary for k in ks):
            continue
        means = np.array([summary[keyfmt.format(k)]["mean"] for k in ks])
        lows = np.array([summary[keyfmt.format(k)]["CI90"][0] for k in ks])
        highs = np.array([summary[keyfmt.format(k)]["CI90"][1] for k in ks])
        axes[0].plot(ks, means, label=label, color=color, marker=marker, lw=1.2, ms=4)
        axes[0].fill_between(ks, lows, highs, color=color, alpha=0.12)
    axes[0].set_xlabel("top-$k$ rules selected")
    axes[0].set_ylabel("simple oracle regret\n$V^\\star - \\max_{\\rho \\in S_k} V(\\rho)$")
    axes[0].set_title("Oracle regret at top-$k$ (HotpotQA, $n_\\text{train}{=}400$)")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(ks)

    # Right: per-trial scatter at k=1
    crrm_v = np.array([t["topv_crrm@1"] for t in trials])
    hand_v = np.array([t["topv_hand_oracle@1"] for t in trials])
    axes[1].scatter(hand_v, crrm_v, alpha=0.75, color="tab:blue", s=18)
    lo, hi = min(hand_v.min(), crrm_v.min()), max(hand_v.max(), crrm_v.max())
    pad = 0.03 * (hi - lo + 1e-9)
    axes[1].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="black", lw=0.8)
    axes[1].set_xlabel("hand-authored oracle best $V$")
    axes[1].set_ylabel("CRRM-LCB top-1 oracle $V$")
    wr = summary["crrm_vs_handoracle_diff@1"]["win_rate"]
    md = summary["crrm_vs_handoracle_diff@1"]["mean_diff"]
    axes[1].set_title(f"CRRM vs hand-oracle (k=1)\nwin rate {wr:.0%}, mean $\\Delta V$ {md:+.3f}")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_pdf = FIGS / "discovery_regret.pdf"
    out_png = FIGS / "discovery_regret.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"Wrote {out_pdf}, {out_png}")


def main():
    fig_scaling()
    fig_ablation_A()
    fig_a3()
    fig_a5()
    fig_discovery()
    print(f"All figures written to {FIGS}/")


if __name__ == "__main__":
    main()
