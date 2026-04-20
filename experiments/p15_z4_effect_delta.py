"""ADAPT-v2 vs baselines under a PRACTICAL-EFFECT-SIZE null.

The key empirical claim: when the practitioner cares about shipping
rules that improve target value by at least `delta` (not just any
positive improvement), ADAPT-v2 is the only method that tests the
correct null and therefore controls FDR against the right metric.

Other methods test H_0: V <= V_noop (i.e. delta = 0), which
over-rejects when a rule's V is above V_noop but below
V_noop + delta.  These "marginally-positive" rules count as false
discoveries under the practitioner's δ-evaluation criterion,
inflating the effective FDR.

Hypothesis:  ADAPT-v2 (effect_delta = 0.01) controls FDR at the
nominal q under the δ = 0.01 evaluation criterion, while Wald,
Bonferroni, BH, and ADAPT-v1 all violate.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

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
        "mean_regret_per_discovery": (
            (max(gt.values()) - float(np.mean([gt[d] for d in disc]))) if n_disc else 0.0
        ),
    }


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:200]
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    out = {}
    eval_delta = 0.01   # practitioner's "rule is worth shipping" threshold
    q = 0.10
    for severity in ("mild", "moderate", "heavy"):
        print(f"\n[{severity}] computing ground truth ...", flush=True)
        drift_fn = make_drift_fn(severity)
        gt = {r.id: target_value(r, logs, drift_fn) for r in rules}
        baseline = target_baseline(logs, drift_fn)
        n_pos = sum(1 for r in rules if gt[r.id] > baseline + eval_delta)

        print(f"[{severity}] running ADAPT-v1 (tests H_0: V<=V_noop) ...", flush=True)
        t0 = time.time()
        v1 = adapt_pipeline(
            rules, logs, drift_fn,
            ADAPTConfig(n_active_rounds=3, label_budget_per_round=100, fdr_q=q),
        )
        v1_time = time.time() - t0

        print(f"[{severity}] running ADAPT-v2 (tests H_0: V<=V_noop + delta) ...", flush=True)
        t0 = time.time()
        v2 = adapt_v2_pipeline(
            rules, logs, drift_fn,
            ADAPTv2Config(
                fdr_q=q, use_storey=True, use_shrinkage=True,
                effect_delta=eval_delta,
            ),
        )
        v2_time = time.time() - t0

        # Naive baselines (test H_0: V<=V_noop) -- show the definitional
        # mismatch / FDR violation when the practitioner cares about δ.
        v1_p = v1.p_values
        wald_naive = [r.id for r in rules if v1_p[r.id] < q]
        bh_naive_arr = benjamini_hochberg(np.array([v1_p[r.id] for r in rules]), q=q)
        bh_naive = [r.id for r, d in zip(rules, bh_naive_arr) if d]

        # Effect-aware baselines (test H_0: V<=V_noop+delta) -- isolate the
        # contribution of v2's cross-fit + shrinkage + Storey above and beyond
        # the effect_delta itself.  We compute their p-values from v2's
        # cross-fit estimates BUT without shrinkage and without Storey.
        v2_unshrunk = adapt_v2_pipeline(
            rules, logs, drift_fn,
            ADAPTv2Config(
                fdr_q=q, use_storey=False, use_shrinkage=False,
                effect_delta=eval_delta,
            ),
        )
        # Wald-with-effect-delta: per-rule p < q
        wald_eff = [r.id for r in rules if v2_unshrunk.p_values[r.id] < q]
        bonf_eff = [r.id for r in rules if v2_unshrunk.p_values[r.id] < q / len(rules)]
        bh_eff_arr = benjamini_hochberg(
            np.array([v2_unshrunk.p_values[r.id] for r in rules]), q=q
        )
        bh_eff = [r.id for r, d in zip(rules, bh_eff_arr) if d]

        out[severity] = {
            "eval_delta": eval_delta,
            "n_truly_better_at_eval_delta": n_pos,
            "fdr_q_nominal": q,
            # Naive null (V<=V_noop): documented as definitional mismatch
            "Wald_naive_H0":         evaluate(wald_naive, rules, gt, baseline, eval_delta),
            "BH_naive_H0":           evaluate(bh_naive, rules, gt, baseline, eval_delta),
            "ADAPT_v1_naive_H0":     evaluate(v1.discoveries, rules, gt, baseline, eval_delta) | {"runtime_s": v1_time},
            # Effect-aware null (V<=V_noop+delta): apples-to-apples comparison
            "Wald_effect_H0":        evaluate(wald_eff, rules, gt, baseline, eval_delta),
            "Bonferroni_effect_H0":  evaluate(bonf_eff, rules, gt, baseline, eval_delta),
            "BH_effect_H0":          evaluate(bh_eff, rules, gt, baseline, eval_delta),
            # Full ADAPT-v2 (effect-aware + cross-fit + shrinkage + Storey)
            "ADAPT_v2_full":         evaluate(v2.discoveries, rules, gt, baseline, eval_delta) | {"runtime_s": v2_time, "pi_0_hat": v2.pi_0_hat},
        }
        print(f"\n=== drift = {severity}  (truly-better at eval_delta={eval_delta}: {n_pos}/{len(rules)}) ===")
        for tag in (
            "Wald_naive_H0", "BH_naive_H0", "ADAPT_v1_naive_H0",
            "Wald_effect_H0", "Bonferroni_effect_H0", "BH_effect_H0",
            "ADAPT_v2_full",
        ):
            r = out[severity][tag]
            flag = " [FDR VIOLATES]" if r["empirical_FDR"] > q else ""
            print(f"  {tag:25s}  disc={r['n_discoveries']:3d}  FDR={r['empirical_FDR']:.3f}  TPR={r['empirical_TPR']:.3f}  regret={r['mean_regret_per_discovery']:.4f}{flag}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_z4_effect_delta.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
