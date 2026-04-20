"""15.M  Fairness-constrained Rule-OPE.

Group records by entity type (q_has_person/place/org/time/num);
evaluate per-subgroup rule values and report the fraction of top
rules that are *fair* (no subgroup loses more than tau).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.fair_ruleope import fair_select
from src.logs import load_logs
from src.rule_dsl import load_rules


def group_fn(rec):
    for k in ("q_has_person", "q_has_place", "q_has_org", "q_has_time", "q_has_num"):
        if rec.ctx.get(k, 0.0) > 0.5:
            return k.replace("q_has_", "")
    return "other"


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:50]
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    out = {"n_rules": len(rules), "by_tau": {}}
    for tau in (0.005, 0.02, 0.05):
        results = fair_select(rules, logs, group_fn, tau=tau)
        n_feasible = sum(1 for r in results if r.feasible)
        # Best fair rule
        feasible = [r for r in results if r.feasible]
        if feasible:
            best_fair = max(feasible, key=lambda r: r.overall)
            best_overall = max(results, key=lambda r: r.overall)
            out["by_tau"][f"{tau:.3f}"] = {
                "n_feasible": n_feasible,
                "n_total": len(results),
                "best_fair_rule": best_fair.rule_name,
                "best_fair_overall": best_fair.overall,
                "best_overall_rule": best_overall.rule_name,
                "best_overall_value": best_overall.overall,
                "fairness_cost": best_overall.overall - best_fair.overall,
                "max_drop_at_best_overall": best_overall.min_group_drop,
            }
        else:
            out["by_tau"][f"{tau:.3f}"] = {
                "n_feasible": 0,
                "n_total": len(results),
                "note": "no feasible rules",
            }
        print(f"  tau={tau:.3f}  feasible={n_feasible}/{len(results)}  fairness_cost={out['by_tau'][f'{tau:.3f}'].get('fairness_cost', float('nan'))}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_m_fair.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
