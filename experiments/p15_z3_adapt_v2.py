"""ADAPT-v2 vs ADAPT-v1 vs Wald vs Bonferroni vs Waudby-Smith.

The same drift sweep as p15_z_adapt.py and p15_z2_adapt_vs_ws.py.
ADAPT-v2 adds: cross-fit drift-weighted EIF, JointRuleOPE shrinkage on
per-rule estimates before p-values, Storey-adaptive q-values.

Hypothesis: ADAPT-v2 strictly dominates ADAPT-v1 on TPR while keeping
FDR <= q nominal, AND beats Wald on FDR validity at mild drift while
matching or exceeding Wald TPR at moderate / heavy drift.
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
from src.adapt_v2 import ADAPTv2Config, adapt_v2_pipeline
from src.fdr_ruleope import benjamini_hochberg
from src.logs import load_logs
from src.rule_dsl import load_rules


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
    return {
        "n_discoveries": n_disc,
        "true_positives": tp,
        "false_positives": fp,
        "empirical_FDR": (fp / n_disc) if n_disc else 0.0,
        "empirical_TPR": (tp / n_pos) if n_pos else 0.0,
        "mean_regret_per_discovery": (
            (max(gt.values()) - float(np.mean([gt[d] for d in disc]))) if n_disc else 0.0
        ),
    }


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:200]
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    out = {}
    delta = 0.0   # consistent: H_0 is "V <= V_noop", truly_better is V > V_noop
    q = 0.10
    for severity in ("mild", "moderate", "heavy"):
        print(f"\n[{severity}] computing ground truth ...", flush=True)
        drift_fn = make_drift_fn(severity)
        gt = {r.id: target_value(r, logs, drift_fn) for r in rules}
        baseline = target_baseline(logs, drift_fn)
        n_pos = sum(1 for r in rules if gt[r.id] > baseline + delta)

        print(f"[{severity}] running ADAPT-v1 ...", flush=True)
        t0 = time.time()
        v1 = adapt_pipeline(
            rules, logs, drift_fn,
            ADAPTConfig(n_active_rounds=3, label_budget_per_round=100, fdr_q=q),
        )
        v1_time = time.time() - t0

        print(f"[{severity}] running ADAPT-v2 ...", flush=True)
        t0 = time.time()
        v2 = adapt_v2_pipeline(
            rules, logs, drift_fn,
            ADAPTv2Config(fdr_q=q, use_storey=True, use_shrinkage=True, effect_delta=delta),
        )
        v2_time = time.time() - t0

        # Wald (uncorrected, on v2's p-values for apples-to-apples)
        wald_disc = [r.id for r in rules if v2.p_values[r.id] < q]
        bonf_disc = [r.id for r in rules if v2.p_values[r.id] < q / len(rules)]
        # BH on v2's p-values without Storey adaptation
        bh_disc_arr = benjamini_hochberg(np.array([v2.p_values[r.id] for r in rules]), q=q)
        bh_disc = [r.id for r, d in zip(rules, bh_disc_arr) if d]

        out[severity] = {
            "n_truly_better": n_pos,
            "delta_threshold": delta,
            "fdr_q_nominal": q,
            "ADAPT_v1":      evaluate(v1.discoveries, rules, gt, baseline, delta) | {"runtime_s": v1_time},
            "ADAPT_v2":      evaluate(v2.discoveries, rules, gt, baseline, delta) | {"runtime_s": v2_time, "pi_0_hat": v2.pi_0_hat},
            "BH_no_storey":  evaluate(bh_disc, rules, gt, baseline, delta),
            "Wald":          evaluate(wald_disc, rules, gt, baseline, delta),
            "Bonferroni":    evaluate(bonf_disc, rules, gt, baseline, delta),
        }
        print(f"\n=== drift = {severity}  (truly-better-at-delta-{delta}: {n_pos}/{len(rules)}) ===")
        for tag in ("Wald", "Bonferroni", "BH_no_storey", "ADAPT_v1", "ADAPT_v2"):
            r = out[severity][tag]
            extra = f"  pi_0={r['pi_0_hat']:.2f}" if "pi_0_hat" in r else ""
            print(f"  {tag:14s}  disc={r['n_discoveries']:3d}  FDR={r['empirical_FDR']:.3f}  TPR={r['empirical_TPR']:.3f}  regret={r['mean_regret_per_discovery']:.4f}{extra}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_z3_adapt_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
