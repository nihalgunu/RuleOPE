"""Phase-0 smoke test.

Generates 100 queries, 10 candidate rules, runs DM and IPS, and prints MSE
against ground truth.  Passes if MSE is finite and top-rule identification is
nontrivial.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.direct_method import DirectMethod
from src.estimators.ips import IPS, SNIPS
from src.estimators.rule_ope import RuleOPE
from src.evaluate import all_metrics
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import enumerate_rules


def main() -> int:
    cfg = SubstrateConfig(n_queries=500, seed=0)
    logs = generate_logs(cfg)
    logs = assign_corrections(logs, CorrectionConfig(base_rate=0.2, error_sensitivity=4.0, seed=1))

    # Take 30 candidate rules from the depth=1,2 subset.
    rules = enumerate_rules(max_depth=2, cap_per_depth=60, rng_seed=0)
    # Filter to rules firing between 5% and 70% of the time.
    def fires_frac(r):
        return sum(1 for rec in logs if r.fires(rec.ctx)) / len(logs)
    rules = [r for r in rules if 0.05 < fires_frac(r) < 0.7][:30]
    assert rules, "no rules survived the firing-rate filter"

    gt = ground_truth_many(rules, logs)

    for est_cls in (DirectMethod, IPS, SNIPS, RuleOPE):
        est = est_cls()
        est.fit(logs) if hasattr(est, "fit") else None
        results = est.value_many(rules, logs)
        estimates = {k: v.estimate for k, v in results.items()}
        stderrs = {k: v.stderr for k, v in results.items()}
        metrics = all_metrics(estimates, stderrs, gt, topk=10)
        print(f"{est.name:>10s} | MSE={metrics['mse']:.5f}  bias={metrics['bias']:+.4f}  cov95={metrics['coverage_95']:.3f}  tau@10={metrics['topk_tau']:+.3f}")

    print(f"n_rules={len(rules)}  n_logs={len(logs)}  gt_mean={sum(gt.values())/len(gt):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
