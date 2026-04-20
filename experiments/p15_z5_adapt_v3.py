"""ADAPT-v3 (knockoffs) vs ADAPT-v2 vs Wald_effect vs BH_effect.

All methods test the practical-effect null H_0: V <= V_noop + delta.
Evaluation threshold matches the null (delta = 0.01).

Hypothesis: ADAPT-v3's compositional knockoffs exploit the rule-atom
correlation to gain power over BH/Storey while controlling FDR.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.adapt_v2 import ADAPTv2Config, adapt_v2_pipeline
from src.adapt_v3 import ADAPTv3Config, adapt_v3_pipeline
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


def evaluate(disc, rules, gt, baseline, eval_delta):
    truly_better = {r.id for r in rules if gt[r.id] > baseline + eval_delta}
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
    }


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:200]
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    out = {}
    eval_delta = 0.01
    q = 0.10
    for severity in ("mild", "moderate", "heavy"):
        print(f"\n[{severity}] computing ground truth ...", flush=True)
        drift_fn = make_drift_fn(severity)
        gt = {r.id: target_value(r, logs, drift_fn) for r in rules}
        baseline = target_baseline(logs, drift_fn)
        n_pos = sum(1 for r in rules if gt[r.id] > baseline + eval_delta)

        print(f"[{severity}] running ADAPT-v2 (cross-fit+shrinkage+Storey+δ) ...", flush=True)
        t0 = time.time()
        v2 = adapt_v2_pipeline(
            rules, logs, drift_fn,
            ADAPTv2Config(
                fdr_q=q, use_storey=True, use_shrinkage=True,
                effect_delta=eval_delta,
            ),
        )
        v2_time = time.time() - t0

        print(f"[{severity}] running ADAPT-v3 (compositional knockoffs) ...", flush=True)
        t0 = time.time()
        v3 = adapt_v3_pipeline(
            rules, logs, drift_fn,
            ADAPTv3Config(fdr_q=q, effect_delta=eval_delta),
        )
        v3_time = time.time() - t0

        # Unshrunk v2 for Wald/BH effect-aware baselines
        v2u = adapt_v2_pipeline(
            rules, logs, drift_fn,
            ADAPTv2Config(
                fdr_q=q, use_storey=False, use_shrinkage=False,
                effect_delta=eval_delta,
            ),
        )
        wald_eff = [r.id for r in rules if v2u.p_values[r.id] < q]
        bh_eff_arr = benjamini_hochberg(
            np.array([v2u.p_values[r.id] for r in rules]), q=q
        )
        bh_eff = [r.id for r, d in zip(rules, bh_eff_arr) if d]

        out[severity] = {
            "eval_delta": eval_delta,
            "n_truly_better": n_pos,
            "fdr_q_nominal": q,
            "Wald_effect":     evaluate(wald_eff, rules, gt, baseline, eval_delta),
            "BH_effect":       evaluate(bh_eff, rules, gt, baseline, eval_delta),
            "ADAPT_v2":        evaluate(v2.discoveries, rules, gt, baseline, eval_delta) | {"runtime_s": v2_time, "pi_0_hat": v2.pi_0_hat},
            "ADAPT_v3":        evaluate(v3.discoveries, rules, gt, baseline, eval_delta) | {"runtime_s": v3_time, "knockoff_threshold": v3.threshold},
        }
        print(f"\n=== drift = {severity}  (truly-better: {n_pos}/{len(rules)}) ===")
        for tag in ("Wald_effect", "BH_effect", "ADAPT_v2", "ADAPT_v3"):
            r = out[severity][tag]
            flag = " [FDR VIOLATES]" if r["empirical_FDR"] > q else ""
            print(f"  {tag:15s}  disc={r['n_discoveries']:3d}  FDR={r['empirical_FDR']:.3f}  TPR={r['empirical_TPR']:.3f}{flag}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_z5_adapt_v3.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
