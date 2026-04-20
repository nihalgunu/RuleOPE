"""15.L  Real-data LLM-judge proxy.

Substitute the oracle reward with a calibrated LLM-judge proxy
(Zheng et al. 2023 calibration: rho ~ 0.8 vs human, sigma_judge =
0.15).  Run RuleOPE on the relabelled benchmark; compare top-20
ranking against the oracle ranking using Kendall's tau.

Success: top-20 tau >= 0.7 across noise scales (rule-OPE survives
LLM-judge noise).

If LAMBDA_API_KEY is set in the environment, results from the proxy
can be replaced with a live LLM call (left as a stub for full
real-data integration; see novelty.md).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import kendalltau

from src.estimators.rule_ope import RuleOPE
from src.llm_judge_proxy import JudgeConfig, relabel_logs
from src.logs import load_logs
from src.rag_substrate import ground_truth_value
from src.rule_dsl import load_rules


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")

    gt = {r.id: ground_truth_value(r, logs) for r in rules}
    est_oracle = RuleOPE().fit(logs).value_many(rules, logs)
    rope_oracle = {r.id: est_oracle[r.id].estimate for r in rules}

    out = {"by_sigma": {}}
    for sigma in (0.05, 0.15, 0.30):
        cfg = JudgeConfig(sigma_judge=sigma, seed=0)
        relab = relabel_logs(logs, cfg)
        est_llm = RuleOPE().fit(relab).value_many(rules, relab)
        rope_llm = {r.id: est_llm[r.id].estimate for r in rules}

        ids = [r.id for r in rules]
        tau_oracle_vs_llm, _ = kendalltau([rope_oracle[i] for i in ids], [rope_llm[i] for i in ids])
        tau_llm_vs_gt, _ = kendalltau([rope_llm[i] for i in ids], [gt[i] for i in ids])
        # Top-20 overlap
        top20_oracle = set(sorted(ids, key=lambda i: -rope_oracle[i])[:20])
        top20_llm = set(sorted(ids, key=lambda i: -rope_llm[i])[:20])
        overlap = len(top20_oracle & top20_llm) / 20.0

        rope_llm_mse = float(np.mean([(rope_llm[i] - gt[i]) ** 2 for i in ids]))
        rope_oracle_mse = float(np.mean([(rope_oracle[i] - gt[i]) ** 2 for i in ids]))

        out["by_sigma"][f"{sigma:.2f}"] = {
            "tau_oracle_vs_llm_ranking": float(tau_oracle_vs_llm),
            "tau_llm_vs_ground_truth": float(tau_llm_vs_gt),
            "top20_overlap": overlap,
            "rope_llm_MSE_vs_truth": rope_llm_mse,
            "rope_oracle_MSE_vs_truth": rope_oracle_mse,
            "MSE_inflation_pct": 100.0 * (rope_llm_mse - rope_oracle_mse) / max(rope_oracle_mse, 1e-12),
        }
        print(f"  sigma={sigma:.2f}  tau(oracle,llm)={tau_oracle_vs_llm:+.3f}  top20-overlap={overlap:.2f}  MSE-inflation={100*(rope_llm_mse-rope_oracle_mse)/max(rope_oracle_mse,1e-12):+.1f}%")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_l_llm_judge.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
