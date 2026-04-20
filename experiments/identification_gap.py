"""Identification-gap diagnostic on the frozen benchmark.

For each rule in `eval/rules_v1.jsonl`:
1. Compute the sharp partial-identification interval [V_L, V_U] (Theorem A).
2. Fit DR and RuleOPE on deterministic-logging logs.
3. Report where each estimator falls within the interval, and the
   empirical efficiency gap (Theorem F) using a plug-in bridge.

This lets us quantify on the benchmark the theoretical claim that (a)
classical DR has no mechanism to close the identification gap, and (b)
RuleOPE's correction-fusion term moves the estimate toward the true
value by an amount proportional to the gate signal.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.identification import (
    bridge_linear,
    diagnostic_report,
    efficiency_gap,
    partial_id_bounds,
)
from src.logs import load_logs
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import load_rules


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")

    cfg = SubstrateConfig(n_queries=2000, seed=42, logging="deterministic")
    logs = generate_logs(cfg)
    logs = assign_corrections(
        logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, seed=43)
    )
    gt = ground_truth_many(rules, logs)

    dr = DoublyRobust().fit(logs)
    rope = RuleOPE().fit(logs)
    dr_res = dr.value_many(rules, logs)
    rope_res = rope.value_many(rules, logs)

    # For the efficiency-gap estimate we need a bridge and a gate.  We use
    # the correction-linearity assumption (A5 sufficient form) with beta
    # calibrated from the benchmark: beta(noop) = error_sensitivity = 4
    # and beta(target) ~= 3 (corrections are slightly less likely under
    # effective rules).  This is the assumption the benchmark implicitly
    # satisfies; see theory/proofs.tex sec. A5.
    beta_logged = 4.0
    beta_target = 3.0
    bridge_scalar = bridge_linear(beta_target=beta_target, beta_logged=beta_logged)

    # Gate estimate: empirical P(C=1 | X) from the logged data, pooled.
    from src.estimators._regression import _ACTION_IDX, _joint_features, atom_feature_matrix
    from sklearn.linear_model import LogisticRegression
    phi = atom_feature_matrix(logs)
    actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
    C = np.array([rec.correction for rec in logs], dtype=np.int32)
    X = _joint_features(phi, actions)
    if C.sum() == 0 or C.sum() == len(C):
        gate_all = np.full(len(logs), float(C.mean()))
    else:
        clf = LogisticRegression(max_iter=1000).fit(X, C)
        gate_all = clf.predict_proba(X)[:, 1]

    rows = []
    summary = {
        "n_rules": len(rules),
        "avg_id_width": 0.0,
        "avg_abs_err_DR": 0.0,
        "avg_abs_err_RuleOPE": 0.0,
        "avg_eff_gap": 0.0,
        "DR_outside_interval_frac": 0.0,
        "RuleOPE_outside_interval_frac": 0.0,
    }
    for r in rules:
        diag = diagnostic_report(r, logs, dr_res[r.id].estimate, rope_res[r.id].estimate)
        gap = efficiency_gap(r, logs, bridge_scalar, gate_all)
        row = {
            **diag,
            "gt": gt[r.id],
            "DR_est": dr_res[r.id].estimate,
            "RuleOPE_est": rope_res[r.id].estimate,
            "gt_position": (gt[r.id] - diag["V_L"]) / max(diag["id_gap_width"], 1e-12),
            "eff_gap": gap.gap,
            "bridge": gap.bridge_mean,
            "g_mean_in_fires": gap.g_mean,
        }
        rows.append(row)

    gtv = np.array([r["gt"] for r in rows])
    drv = np.array([r["DR_est"] for r in rows])
    ropev = np.array([r["RuleOPE_est"] for r in rows])
    widths = np.array([r["id_gap_width"] for r in rows])
    gaps = np.array([r["eff_gap"] for r in rows])

    summary["avg_id_width"] = float(widths.mean())
    summary["avg_abs_err_DR"] = float(np.mean(np.abs(drv - gtv)))
    summary["avg_abs_err_RuleOPE"] = float(np.mean(np.abs(ropev - gtv)))
    summary["avg_eff_gap"] = float(gaps.mean())
    summary["DR_outside_interval_frac"] = float(np.mean([not r["DR_inside_interval"] for r in rows]))
    summary["RuleOPE_outside_interval_frac"] = float(np.mean([not r["RuleOPE_inside_interval"] for r in rows]))

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/identification_gap.json", "w") as f:
        json.dump({"summary": summary, "per_rule": rows[:50]}, f, indent=2)

    print("=== Identification-gap diagnostic ===")
    print(f"  rules                  {summary['n_rules']}")
    print(f"  avg id-interval width  {summary['avg_id_width']:.4f}")
    print(f"  avg |DR  - truth|      {summary['avg_abs_err_DR']:.4f}")
    print(f"  avg |ROPE - truth|     {summary['avg_abs_err_RuleOPE']:.4f}")
    print(f"  avg efficiency gap     {summary['avg_eff_gap']:.6f}  (per-record, Thm F)")
    print(f"  DR outside [V_L, V_U]  {summary['DR_outside_interval_frac']:.1%}")
    print(f"  ROPE outside bounds    {summary['RuleOPE_outside_interval_frac']:.1%}")
    print(f"  |gt - V_L| / width (avg): {np.mean([r['gt_position'] for r in rows]):.3f}  (0=lower bound, 1=upper bound)")
    print(f"  |DR - V_L| / width (avg): {np.mean([r['DR_position'] for r in rows]):.3f}")
    print(f"  |ROPE - V_L| / width (avg): {np.mean([r['RuleOPE_position'] for r in rows]):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
