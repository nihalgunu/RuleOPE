"""CRRM: counterfactual rule risk minimisation (rule *learning* from logs).

We compare three rule learners:
   (1) ERM:    argmax_{rho in R} V_hat(rho)       -- pure empirical maximisation
   (2) LCB:    argmax_{rho in R} V_hat_LCB(rho)   -- pessimistic with union-bound exponent
   (3) C-CRRM: argmax_{rho in R} V_hat_LCB_comp(rho) -- compositional-sparsity exponent

The evaluation metric is *regret*: V(rho_oracle) - V(rho_learned), averaged
over seeds at a range of log sizes N.  Theorem 5 predicts the compositional
CRRM has sharper regret in the small-N regime when the atom-level value
function is sparse.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.correction_sim import CorrectionConfig, assign_corrections
from src.crrm import CRRM, CRRMConfig, PessimisticConfig, PessimisticRuleSelector
from src.estimators.shrinkage import JointRuleOPE
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import load_rules


def _trial(N, seed, rules):
    logs = generate_logs(SubstrateConfig(n_queries=N, seed=seed, logging="deterministic"))
    logs = assign_corrections(logs, CorrectionConfig(base_rate=0.15, error_sensitivity=4.0, seed=seed + 3))
    gt = ground_truth_many(rules, logs)
    oracle = max(rules, key=lambda r: gt[r.id])

    est = JointRuleOPE()
    est.fit(logs)
    res = est.value_many(rules, logs)

    # ERM
    erm = max(rules, key=lambda r: res[r.id].estimate)

    # LCB union-bound
    sel_lcb = PessimisticRuleSelector(PessimisticConfig(atom_sparse=False))
    lcb_rule, _ = sel_lcb.select(rules, res)

    # CRRM compositional-sparsity
    sel_comp = PessimisticRuleSelector(PessimisticConfig(atom_sparse=True))
    ccrm_rule, _ = sel_comp.select(rules, res)

    return {
        "oracle": gt[oracle.id],
        "regret_ERM":   gt[oracle.id] - gt[erm.id],
        "regret_LCB":   gt[oracle.id] - gt[lcb_rule.id],
        "regret_CRRM":  gt[oracle.id] - gt[ccrm_rule.id],
    }


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    results = {}
    for N in (300, 600, 1200, 2400):
        regrets = {"ERM": [], "LCB": [], "CRRM": []}
        for seed in range(5):
            r = _trial(N, seed=seed * 31 + N, rules=rules)
            regrets["ERM"].append(r["regret_ERM"])
            regrets["LCB"].append(r["regret_LCB"])
            regrets["CRRM"].append(r["regret_CRRM"])
        agg = {k: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in regrets.items()}
        results[f"N={N}"] = agg
        print(f"N={N}  ERM={agg['ERM']['mean']:.4f}  LCB={agg['LCB']['mean']:.4f}  CRRM={agg['CRRM']['mean']:.4f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/crrm.json", "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
