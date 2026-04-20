"""15.H  Meta-learned bridge functions.

Train a single rule-conditioned bridge over all rules; compare its
held-out per-rule MSE against per-rule independent regression.

Success: meta-bridge MSE within 1.5x of per-rule fit, with O(1)
training time vs O(M) fits.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.estimators.rule_ope import RuleOPE
from src.logs import load_logs
from src.meta_bridge import MetaBridge
from src.rag_substrate import ground_truth_value
from src.rule_dsl import load_rules


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")[:80]
    logs = load_logs("eval/benchmark_v1_with_cf.jsonl")
    n_half = len(logs) // 2
    train, ev = logs[:n_half], logs[n_half:]

    gt = {r.id: ground_truth_value(r, ev) for r in rules}

    t0 = time.time()
    meta = MetaBridge(alpha=1.0)
    meta_pred = meta.fit_predict(train, rules)
    meta_t = time.time() - t0

    t0 = time.time()
    rope = RuleOPE().fit(train)
    per_rule_pred = {r.id: rope.value(r, ev).estimate for r in rules}
    per_rule_t = time.time() - t0

    meta_mse = float(np.mean([(meta_pred[r.id] - gt[r.id]) ** 2 for r in rules]))
    per_rule_mse = float(np.mean([(per_rule_pred[r.id] - gt[r.id]) ** 2 for r in rules]))

    out = {
        "n_rules": len(rules),
        "meta_MSE": meta_mse,
        "per_rule_MSE": per_rule_mse,
        "meta_to_per_rule_ratio": meta_mse / max(per_rule_mse, 1e-12),
        "meta_train_time_s": meta_t,
        "per_rule_train_time_s": per_rule_t,
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/p15_h_meta_bridge.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"  {k:30s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
