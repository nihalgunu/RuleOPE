"""ADAPT-OPE vs Waudby-Smith head-to-head.

Same drift sweep as p15_z_adapt.py.  Compares the two FDR-controlling
methods that handle multi-policy testing:

  ADAPT  : sample-split BH on EIF p-values (this paper).
  WS-eBH : Waudby-Smith DR betting + Wang-Ramdas e-BH.

Plus the standard Wald and Bonferroni baselines for context.

Reports per-strategy: empirical FDR, TPR, regret-per-discovery, and
the *anytime-valid lower CS coverage* of WS (a guarantee ADAPT does
not provide).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import norm

from src.active_drift_ruleope import ADAPTConfig, adapt_pipeline
from src.anytime_valid_ope import WSConfig, waudby_smith_pipeline
from src.estimators.rule_ope import RuleOPE
from src.fdr_ruleope import benjamini_hochberg
from src.logs import load_logs
from src.rule_dsl import load_rules


# ---- Drift definitions (same as p15_z_adapt.py) ----------------------------

def make_drift_fn(severity):
    def fn(rec):
        w = 1.0
        is_mh = rec.ctx.get("q_multihop", 0.0) > 0.5
        is_short = rec.ctx.get("q_len", 100.0) < 8
        low_conf = rec.ctx.get("gen_conf", 1.0) < 0.5
        if severity == "mild":
            if is_mh: w *= 1.5
            if is_short: w *= 0.7
        elif severity == "moderate":
            if is_mh: w *= 2.5
            if is_short: w *= 0.5
        elif severity == "heavy":
            if is_mh: w *= 4.0
            if is_short: w *= 0.3
            if low_conf: w *= 1.5
        return w
    return fn


def target_value(rule, logs, drift_fn):
    weights = np.array([drift_fn(rec) for rec in logs])
    cf = np.array([
        rec.cf_rewards[rule.action] if rule.fires(rec.ctx) and rule.action in rec.cf_rewards
        else rec.cf_rewards.get("noop", rec.logged_reward)
        for rec in logs
    ])
    return float(np.sum(weights * cf) / np.sum(weights))


def target_baseline(logs, drift_fn):
    weights = np.array([drift_fn(rec) for rec in logs])
    cf = np.array([rec.cf_rewards.get("noop", rec.logged_reward) for rec in logs])
    return float(np.sum(weights * cf) / np.sum(weights))


def evaluate(disc, rules, gt, baseline, delta=0.01):
    truly_better = {r.id for r in rules if gt[r.id] > baseline + delta}
    disc = set(disc)
    tp = len(disc & truly_better)
    fp = len(disc - truly_better)
    n_disc, n_pos = len(disc), len(truly_better)
    emp_fdr = (fp / n_disc) if n_disc else 0.0
    tpr = (tp / n_pos) if n_pos else 0.0
    regret = (max(gt.values()) - float(np.mean([gt[d] for d in disc]))) if n_disc else 0.0
    return {
        "n_discoveries": n_disc,
        "true_positives": tp,
        "false_positives": fp,
        "empirical_FDR": emp_fdr,
        "empirical_TPR": tpr,
        "mean_regret_per_discovery": regret,
    }


def lower_cs_coverage(lower, gt):
    """Fraction of rules whose true V_target lies above the WS lower CS."""
    n = 0; covered = 0
    for rid, l in lower.items():
        if rid in gt:
            n += 1
            if gt[rid] >= l:
                covered += 1
    return covered / max(n, 1)


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:200]
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    out = {}
    delta = 0.01
    q = 0.10
    for severity in ("mild", "moderate", "heavy"):
        print(f"\n[{severity}] computing ground truth ...", flush=True)
        drift_fn = make_drift_fn(severity)
        gt = {r.id: target_value(r, logs, drift_fn) for r in rules}
        baseline = target_baseline(logs, drift_fn)
        n_pos = sum(1 for r in rules if gt[r.id] > baseline + delta)

        print(f"[{severity}] running ADAPT pipeline ...", flush=True)
        t0 = time.time()
        adapt_res = adapt_pipeline(
            rules, logs, drift_fn,
            ADAPTConfig(n_active_rounds=3, label_budget_per_round=100, fdr_q=q),
        )
        adapt_time = time.time() - t0

        print(f"[{severity}] running Waudby-Smith pipeline ...", flush=True)
        t0 = time.time()
        ws_res = waudby_smith_pipeline(
            rules, logs, drift_fn,
            WSConfig(alpha=0.10, fdr_q=q),
        )
        ws_time = time.time() - t0

        # Wald and Bonferroni from ADAPT's p-values (apples-to-apples)
        from src.fdr_ruleope import benjamini_hochberg
        wald_disc = [r.id for r in rules if adapt_res.p_values[r.id] < 0.10]
        bonf_disc = [r.id for r in rules if adapt_res.p_values[r.id] < 0.10 / len(rules)]

        out[severity] = {
            "n_truly_better_at_delta_0_01": n_pos,
            "ADAPT":     evaluate(adapt_res.discoveries, rules, gt, baseline, delta) | {"runtime_s": adapt_time},
            "WS_eBH":    evaluate(ws_res.discoveries,    rules, gt, baseline, delta) | {"runtime_s": ws_time, "lower_CS_coverage": lower_cs_coverage(ws_res.lower_cs, gt)},
            "Wald":      evaluate(wald_disc,             rules, gt, baseline, delta),
            "Bonferroni":evaluate(bonf_disc,             rules, gt, baseline, delta),
        }
        print(f"\n=== drift = {severity}  (truly-better-at-delta-{delta}: {n_pos}/{len(rules)}) ===")
        for tag in ("Wald", "Bonferroni", "WS_eBH", "ADAPT"):
            r = out[severity][tag]
            extra = ""
            if tag == "WS_eBH":
                extra = f"  CS-cov={r['lower_CS_coverage']:.3f}"
            print(f"  {tag:12s}  disc={r['n_discoveries']:3d}  FDR={r['empirical_FDR']:.3f}  TPR={r['empirical_TPR']:.3f}  regret={r['mean_regret_per_discovery']:.4f}{extra}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_z2_adapt_vs_ws.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
