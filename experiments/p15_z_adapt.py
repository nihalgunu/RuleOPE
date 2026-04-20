"""ADAPT-OPE — joint validation of the proposed novel approach.

Compares four selection strategies under three drift magnitudes:

  S1  *naive top-k*               — no drift correction, no FDR.
  S2  *drift-only top-k*          — drift-corrected estimates, no FDR.
  S3  *drift + active top-k*      — adds active-labelling.
  S4  *ADAPT (drift + active + FDR)*  — the full proposed pipeline.

For each strategy we report:
  - discovery set size
  - empirical FDR (false / total discoveries) using the drift-corrected
    ground truth V_target as the oracle
  - empirical TPR (true discoveries / total truly-better rules)
  - cumulative regret if the practitioner ships ALL discovered rules
    and randomly assigns one per query at deployment time

We sweep three drift weights of increasing severity to characterise
robustness.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import norm

from src.active_drift_ruleope import ADAPTConfig, adapt_pipeline
from src.estimators.rule_ope import RuleOPE
from src.fdr_ruleope import benjamini_hochberg
from src.logs import load_logs
from src.rule_dsl import load_rules


# --- Drift definitions ------------------------------------------------------

def make_drift_fn(severity: str):
    """Return a drift weight function w(x) = dP_target/dP_source.

    'mild'      : 1.5x for multihop, 0.7x for short queries.
    'moderate'  : 2.5x for multihop, 0.5x for short queries.
    'heavy'     : 4.0x for multihop, 0.3x for short queries, 1.5x for low gen_conf.
    """
    def fn(rec):
        w = 1.0
        is_mh = rec.ctx.get("q_multihop", 0.0) > 0.5
        is_short = rec.ctx.get("q_len", 100.0) < 8
        low_conf = rec.ctx.get("gen_conf", 1.0) < 0.5
        if severity == "mild":
            if is_mh:
                w *= 1.5
            if is_short:
                w *= 0.7
        elif severity == "moderate":
            if is_mh:
                w *= 2.5
            if is_short:
                w *= 0.5
        elif severity == "heavy":
            if is_mh:
                w *= 4.0
            if is_short:
                w *= 0.3
            if low_conf:
                w *= 1.5
        return w
    return fn


# --- Ground truth under target ---------------------------------------------

def target_value(rule, logs, drift_fn):
    """V_target(rule) = E_{P_target}[R(x, pi_rho(x))], computed exactly
    from cf_rewards and drift weights."""
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


# --- Baseline strategies ---------------------------------------------------

def _wald_p_values(estimates, ses, baseline):
    z = (estimates - baseline) / np.maximum(ses, 1e-12)
    return 1.0 - norm.cdf(z)


def naive_topk(rules, logs, k):
    """No drift correction, no FDR; just top-k by source RuleOPE."""
    est = RuleOPE().fit(logs)
    res = est.value_many(rules, logs)
    estimates = np.array([res[r.id].estimate for r in rules])
    ranked = sorted(zip(rules, estimates), key=lambda x: -x[1])[:k]
    return [r.id for r, _ in ranked]


def drift_only_topk(rules, logs, drift_fn, k):
    """Drift-corrected top-k, no active, no FDR."""
    cfg = ADAPTConfig(n_active_rounds=0, label_budget_per_round=0, fdr_q=1.0)
    res = adapt_pipeline(rules, logs, drift_fn, cfg)
    sorted_rules = sorted(rules, key=lambda r: -res.estimates_target[r.id])[:k]
    return [r.id for r in sorted_rules]


def drift_active_topk(rules, logs, drift_fn, k, n_rounds=3, budget=100):
    """Drift + active, no FDR."""
    cfg = ADAPTConfig(
        n_active_rounds=n_rounds,
        label_budget_per_round=budget,
        fdr_q=1.0,
    )
    res = adapt_pipeline(rules, logs, drift_fn, cfg)
    sorted_rules = sorted(rules, key=lambda r: -res.estimates_target[r.id])[:k]
    return [r.id for r in sorted_rules]


def adapt_full(rules, logs, drift_fn, q):
    cfg = ADAPTConfig(n_active_rounds=3, label_budget_per_round=100, fdr_q=q)
    return adapt_pipeline(rules, logs, drift_fn, cfg)


def evaluate(discovery_ids, rules, gt_target, baseline_target, delta=0.01):
    """Discovery is a *true positive* iff V_target(rho) > V_target(noop) + delta.

    delta = 0.01 means we only count rules that beat the baseline by at
    least 1% of the reward scale -- a meaningful practical effect.
    """
    truly_better = {r.id for r in rules if gt_target[r.id] > baseline_target + delta}
    disc = set(discovery_ids)
    tp = len(disc & truly_better)
    fp = len(disc - truly_better)
    n_disc = len(disc)
    n_pos = len(truly_better)
    emp_fdr = (fp / n_disc) if n_disc else 0.0
    tpr = (tp / n_pos) if n_pos else 0.0
    if n_disc > 0:
        best = max(gt_target.values())
        mean_disc_v = float(np.mean([gt_target[d] for d in discovery_ids]))
        regret = best - mean_disc_v
    else:
        regret = 0.0
    return {
        "n_discoveries": n_disc,
        "n_truly_better": n_pos,
        "true_positives": tp,
        "false_positives": fp,
        "empirical_FDR": emp_fdr,
        "empirical_TPR": tpr,
        "mean_regret_per_discovery": regret,
    }


def wald_from_pvals(rules, p_values, alpha=0.10):
    """Per-rule Wald: ship if p < alpha. No multiplicity control (Bibaut-style
    naive multi-policy testing baseline).  Cheap given precomputed p-values."""
    return [r.id for r in rules if p_values[r.id] < alpha]


def bonferroni_from_pvals(rules, p_values, alpha=0.10):
    """Bonferroni: ship if p < alpha / M.  Tightest FWER-controlling baseline."""
    threshold = alpha / len(rules)
    return [r.id for r in rules if p_values[r.id] < threshold]


def main() -> int:
    all_rules = load_rules("eval/rules_v1.jsonl")
    # Use a 200-rule subsample for speed; deterministic order in the file.
    rules = all_rules[:200]
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    out = {}
    delta = 0.01
    for severity in ("mild", "moderate", "heavy"):
        print(f"\n[{severity}] computing ground truth ...", flush=True)
        drift_fn = make_drift_fn(severity)
        gt_target = {r.id: target_value(r, logs, drift_fn) for r in rules}
        baseline_target_v = target_baseline(logs, drift_fn)
        n_truly_better = sum(1 for r in rules if gt_target[r.id] > baseline_target_v + delta)
        print(f"[{severity}] truly-better-at-delta-{delta} = {n_truly_better}/{len(rules)}", flush=True)

        # ONE expensive call: full ADAPT pipeline.  Everything else reuses its p-values.
        print(f"[{severity}] running full ADAPT pipeline ...", flush=True)
        full_res = adapt_full(rules, logs, drift_fn, q=0.10)
        adapt_disc = full_res.discoveries
        k = max(len(adapt_disc), 5)

        # Drift-only (cheap: same pipeline with no active rounds, no BH)
        print(f"[{severity}] running drift-only and drift+active for matched-k ranking ...", flush=True)
        cfg_no_active = ADAPTConfig(n_active_rounds=0, label_budget_per_round=0, fdr_q=1.0)
        drift_only_res = adapt_pipeline(rules, logs, drift_fn, cfg_no_active)
        drift_only_sorted = sorted(rules, key=lambda r: -drift_only_res.estimates_target[r.id])
        s2 = [r.id for r in drift_only_sorted[:k]]
        # drift+active uses full_res (active rounds were already run) but takes top-k
        drift_active_sorted = sorted(rules, key=lambda r: -full_res.estimates_target[r.id])
        s3 = [r.id for r in drift_active_sorted[:k]]

        # Naive top-k = source-distribution RuleOPE ranking
        s1 = naive_topk(rules, logs, k)
        # Wald and Bonferroni reuse the FULL pipeline's p-values (they get the
        # benefit of drift correction + active labelling in their p-values, just
        # without proper multiplicity control)
        s_wald = wald_from_pvals(rules, full_res.p_values, alpha=0.10)
        s_bonf = bonferroni_from_pvals(rules, full_res.p_values, alpha=0.10)

        out[severity] = {
            "n_truly_better_at_delta_0_01": n_truly_better,
            "matched_k": k,
            "delta_threshold": delta,
            "S1_naive_topk":              evaluate(s1, rules, gt_target, baseline_target_v, delta),
            "S2_drift_topk":              evaluate(s2, rules, gt_target, baseline_target_v, delta),
            "S3_drift_active_topk":       evaluate(s3, rules, gt_target, baseline_target_v, delta),
            "B_wald_uncorrected":         evaluate(s_wald, rules, gt_target, baseline_target_v, delta),
            "B_bonferroni":               evaluate(s_bonf, rules, gt_target, baseline_target_v, delta),
            "S4_ADAPT_drift_active_FDR":  evaluate(adapt_disc, rules, gt_target, baseline_target_v, delta),
        }
        print(f"\n=== drift = {severity}  (truly-better at delta={delta}: {n_truly_better};  matched k = {k}) ===")
        for tag in ("S1_naive_topk", "S2_drift_topk", "S3_drift_active_topk",
                    "B_wald_uncorrected", "B_bonferroni", "S4_ADAPT_drift_active_FDR"):
            r = out[severity][tag]
            print(f"  {tag:32s}  disc={r['n_discoveries']:3d}  FDR={r['empirical_FDR']:.3f}  TPR={r['empirical_TPR']:.3f}  regret={r['mean_regret_per_discovery']:.4f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_z_adapt.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
