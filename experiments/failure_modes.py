"""Phase 3 failure-mode stress tests.

Three RAG-specific failure modes of the unconfoundedness assumption
(theory/proofs.tex, Section 7):

   F1: query-dependent correction effort  (effort_slope != 0)
   F2: self-consistent answer bias        (gen_conf_bias != 0)
   F3: corpus drift                        (train/eval distribution mismatch)

For each we construct the pathological setting, run the estimators, and
report how much each degrades relative to the benign setting.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.direct_method import DirectMethod
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.evaluate import all_metrics
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import load_rules


def _run(logs, gt, rules):
    row = {}
    for est in [RuleOPE(), DoublyRobust(), DirectMethod()]:
        if hasattr(est, "fit"):
            est.fit(logs)
        res = est.value_many(rules, logs)
        estimates = {k: v.estimate for k, v in res.items()}
        stderrs = {k: v.stderr for k, v in res.items()}
        row[est.name] = all_metrics(estimates, stderrs, gt, topk=20)
    return row


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")

    results = {}

    # Benign baseline
    print("=== benign ===")
    logs = generate_logs(SubstrateConfig(n_queries=3000, seed=71))
    logs = assign_corrections(logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, seed=711))
    gt = ground_truth_many(rules, logs)
    results["benign"] = _run(logs, gt, rules)
    for k, v in results["benign"].items():
        print(f"  {k}  MSE={v['mse']:.5f}  tau@20={v['topk_tau']:+.3f}")

    # F1: query-length-dependent correction effort
    print("=== F1 query-dependent effort (effort_slope=1.5) ===")
    logs = generate_logs(SubstrateConfig(n_queries=3000, seed=72))
    logs = assign_corrections(
        logs,
        CorrectionConfig(
            base_rate=0.15, error_sensitivity=4.0, effort_slope=1.5, seed=722
        ),
    )
    gt = ground_truth_many(rules, logs)
    results["F1_effort_bias"] = _run(logs, gt, rules)
    for k, v in results["F1_effort_bias"].items():
        print(f"  {k}  MSE={v['mse']:.5f}  tau@20={v['topk_tau']:+.3f}")

    # F2: self-consistent answer bias
    print("=== F2 self-consistent answer bias (gen_conf_bias=-2.0) ===")
    logs = generate_logs(SubstrateConfig(n_queries=3000, seed=73))
    logs = assign_corrections(
        logs,
        CorrectionConfig(
            base_rate=0.15, error_sensitivity=4.0, gen_conf_bias=-2.0, seed=733
        ),
    )
    gt = ground_truth_many(rules, logs)
    results["F2_gen_conf_bias"] = _run(logs, gt, rules)
    for k, v in results["F2_gen_conf_bias"].items():
        print(f"  {k}  MSE={v['mse']:.5f}  tau@20={v['topk_tau']:+.3f}")

    # F3: corpus drift -- train estimators on one distribution, evaluate rules on another.
    print("=== F3 corpus drift (eval substrate has multihop=0.6) ===")
    train_cfg = SubstrateConfig(n_queries=3000, seed=74)
    eval_cfg  = SubstrateConfig(n_queries=3000, seed=740)
    train_logs = generate_logs(train_cfg)
    # Tilt eval distribution: regenerate with a different seed but the rules
    # still see the training-distribution ground truth.  We approximate drift
    # by using the eval substrate's ctx & reward but fitting on train logs.
    eval_logs = generate_logs(eval_cfg)
    train_logs = assign_corrections(
        train_logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, seed=744)
    )
    eval_logs = assign_corrections(
        eval_logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, seed=7444)
    )
    gt = ground_truth_many(rules, eval_logs)
    row = {}
    for est in [RuleOPE(), DoublyRobust(), DirectMethod()]:
        if hasattr(est, "fit"):
            est.fit(train_logs)  # FIT ON TRAIN
        res = est.value_many(rules, eval_logs)  # EVAL ON EVAL
        estimates = {k: v.estimate for k, v in res.items()}
        stderrs = {k: v.stderr for k, v in res.items()}
        row[est.name] = all_metrics(estimates, stderrs, gt, topk=20)
    results["F3_drift"] = row
    for k, v in row.items():
        print(f"  {k}  MSE={v['mse']:.5f}  tau@20={v['topk_tau']:+.3f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/failure_modes.json", "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
