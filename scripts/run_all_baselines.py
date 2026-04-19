"""Run every estimator on the frozen benchmark and print the summary table.

This is the reproducibility entry point cited in the paper.  It assumes
`eval/benchmark_v1.jsonl` and `eval/ground_truth_rule_values.json` exist
(run `eval/build_benchmark.py` first).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.estimators.cascade_dr import CascadeDR
from src.estimators.cips import CIPS, CIPS_DR
from src.estimators.direct_method import DirectMethod
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.ips import IPS, SNIPS
from src.estimators.rule_ope import RuleOPE
from src.evaluate import all_metrics
from src.logs import load_logs
from src.rule_dsl import Rule, load_rules


def main() -> int:
    with open("eval/ground_truth_rule_values.json") as f:
        gt = {k: v["value"] for k, v in json.load(f)["rules"].items()}
    rules = load_rules("eval/rules_v1.jsonl")
    # Attach cf_rewards to logs for downstream analyses that might need them.
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")

    estimators = [
        DirectMethod(),
        IPS(),
        SNIPS(),
        DoublyRobust(),
        CIPS(clip=20.0),
        CIPS_DR(clip=20.0),
        CascadeDR(),
        RuleOPE(),
    ]

    rows = []
    header = f"{'estimator':>10s} | {'MSE':>8s} | {'bias':>8s} | {'cov95':>6s} | {'tau@20':>7s} | {'time_s':>7s}"
    print(header)
    print("-" * len(header))
    for est in estimators:
        t0 = time.time()
        if hasattr(est, "fit"):
            est.fit(logs)
        results = est.value_many(rules, logs)
        dt = time.time() - t0
        estimates = {k: v.estimate for k, v in results.items()}
        stderrs = {k: v.stderr for k, v in results.items()}
        m = all_metrics(estimates, stderrs, gt, topk=20)
        row = dict(name=est.name, time_s=dt, **m)
        rows.append(row)
        print(
            f"{est.name:>10s} | {m['mse']:8.5f} | {m['bias']:+8.4f} | {m['coverage_95']:6.3f} | {m['topk_tau']:+7.3f} | {dt:7.2f}"
        )

    with open("eval/baseline_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
