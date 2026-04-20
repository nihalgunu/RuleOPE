"""15.G  FDR-controlled rule selection.

Run BH at q in {0.05, 0.10, 0.20} on the candidate rule pool.  Report:
  - Number of discoveries.
  - Empirical FDR vs ground truth (rule actually beats baseline).
  - Comparison to top-k selection at matched k.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.fdr_ruleope import fdr_select
from src.logs import load_logs
from src.rag_substrate import ground_truth_value
from src.rule_dsl import load_rules


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    baseline_v = float(np.mean([rec.logged_reward for rec in logs]))

    gt = {r.id: ground_truth_value(r, logs) for r in rules}
    truly_better = {r.id for r in rules if gt[r.id] > baseline_v}

    out = {
        "n_rules": len(rules),
        "baseline_value": baseline_v,
        "n_truly_better": len(truly_better),
        "by_q": {},
    }
    for q in (0.05, 0.10, 0.20):
        results = fdr_select(rules, logs, q=q)
        discoveries = [r for r in results if r.discovered]
        false_disc = [r for r in discoveries if r.rule_id not in truly_better]
        true_disc = [r for r in discoveries if r.rule_id in truly_better]
        emp_fdr = len(false_disc) / max(len(discoveries), 1)
        # Matched top-k baseline
        topk = sorted(results, key=lambda x: -x.estimate)[: len(discoveries)]
        topk_fdr = sum(1 for r in topk if r.rule_id not in truly_better) / max(len(topk), 1)
        out["by_q"][f"{q:.2f}"] = {
            "n_discoveries": len(discoveries),
            "empirical_FDR": emp_fdr,
            "true_positives": len(true_disc),
            "topk_FDR_matched_k": topk_fdr,
            "BH_threshold_p": discoveries[-1].bh_threshold if discoveries else None,
        }
        print(f"  q={q:.2f}  discoveries={len(discoveries):3d}  empirical_FDR={emp_fdr:.3f}  topk_FDR={topk_fdr:.3f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_g_fdr.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
