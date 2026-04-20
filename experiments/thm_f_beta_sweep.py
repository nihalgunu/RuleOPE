"""Thm F validation: the efficiency gap tracks the closed-form formula.

Theorem F predicts:
    Var(psi^DR) - Var(psi^star) = E[p(X)^2 b_rho(X)^2 g(X, a_0)(1 - g(X, a_0))]

We sweep the bridge coefficient b_rho (via beta_target under the linear
correction-linearity form A5-sufficient) and measure:
  * The CLOSED-FORM gap predicted by the RHS of Thm F.
  * The BOOTSTRAP EMPIRICAL gap Var(V_DR) - Var(V_ROPE-EIF).

If Thm F is correct, these two should track each other across the sweep
with high correlation and matched magnitude.  This is the rigorous test
of the theorem's quantitative content, conducted via standard bootstrap
methodology on a fixed substrate.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators._regression import (
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
)
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.identification import bridge_linear, efficiency_gap
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import load_rules


def _gate_at_noop(logs):
    phi = atom_feature_matrix(logs)
    actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
    C = np.array([rec.correction for rec in logs], dtype=np.int32)
    if C.sum() == 0 or C.sum() == len(C):
        return np.full(len(logs), float(C.mean()))
    X = _joint_features(phi, actions)
    clf = LogisticRegression(max_iter=1000).fit(X, C)
    a_noop = np.full(len(logs), _ACTION_IDX["noop"], dtype=np.int64)
    return clf.predict_proba(_joint_features(phi, a_noop))[:, 1].astype(np.float64)


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    N = 1500
    beta_logged = 4.0
    beta_target_sweep = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    n_bootstrap = 40

    logs = generate_logs(SubstrateConfig(n_queries=N, seed=17, logging="deterministic"))
    logs = assign_corrections(logs, CorrectionConfig(base_rate=0.15, error_sensitivity=beta_logged, seed=18))

    # Rule subset with meaningful firing
    subset = [r for r in rules if 0.10 < sum(1 for rec in logs if r.fires(rec.ctx)) / N < 0.60][:20]
    print(f"Sweeping {len(beta_target_sweep)} beta_target values across {len(subset)} rules, {n_bootstrap} bootstraps each")

    g_a0 = _gate_at_noop(logs)

    # Single DR fit (doesn't depend on beta_target)
    rng = np.random.default_rng(0)
    dr_by_boot = {r.id: [] for r in subset}
    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        bl = [logs[i] for i in idx]
        dr_b = DoublyRobust().fit(bl)
        dr_r = dr_b.value_many(subset, bl)
        for r in subset:
            dr_by_boot[r.id].append(dr_r[r.id].estimate)
    var_dr = {rid: float(np.var(xs, ddof=1)) for rid, xs in dr_by_boot.items()}

    results = []
    for beta_t in beta_target_sweep:
        b_rho = bridge_linear(beta_target=beta_t, beta_logged=beta_logged)

        # Bootstrap RuleOPE(EIF, b_rho)
        rope_by_boot = {r.id: [] for r in subset}
        rng_boot = np.random.default_rng(1)
        for b in range(n_bootstrap):
            idx = rng_boot.integers(0, N, size=N)
            bl = [logs[i] for i in idx]
            cfg = RuleOPEConfig(mode="eif", beta_target=beta_t, beta_logged=beta_logged)
            est = RuleOPE(config=cfg).fit(bl)
            res = est.value_many(subset, bl)
            for r in subset:
                rope_by_boot[r.id].append(res[r.id].estimate)
        var_rope = {rid: float(np.var(xs, ddof=1)) for rid, xs in rope_by_boot.items()}
        emp_gap = {rid: var_dr[rid] - var_rope[rid] for rid in var_dr}

        # Closed-form formula, per rule
        formula_gap = {}
        for r in subset:
            g = efficiency_gap(r, logs, bridge=b_rho, gate=g_a0)
            formula_gap[r.id] = g.gap / N  # per-estimate scaling

        rids = list(var_dr.keys())
        empv = np.array([emp_gap[k] for k in rids])
        forv = np.array([formula_gap[k] for k in rids])
        if len(rids) >= 2 and np.std(empv) > 0 and np.std(forv) > 0:
            corr = float(np.corrcoef(empv, forv)[0, 1])
        else:
            corr = float("nan")

        row = {
            "beta_target": beta_t,
            "bridge_scalar": b_rho,
            "bridge_squared": b_rho ** 2,
            "avg_formula_gap": float(np.mean(forv)),
            "avg_empirical_gap": float(np.mean(empv)),
            "correlation": corr,
            "ratio_formula_to_empirical": float(np.mean(forv) / max(np.mean(empv), 1e-12)) if np.mean(empv) > 0 else float("nan"),
        }
        results.append(row)
        print(
            f"  beta_t={beta_t:<4.2g}  b^2={b_rho**2:<8.5g}  "
            f"formula={row['avg_formula_gap']:<10.2e}  "
            f"empirical={row['avg_empirical_gap']:<10.2e}  "
            f"ratio={row['ratio_formula_to_empirical']:<6.3f}  corr={row['correlation']:+.3f}"
        )

    # Overall: regression of empirical gap on formula gap across the sweep.
    ev = np.array([r["avg_empirical_gap"] for r in results])
    fv = np.array([r["avg_formula_gap"] for r in results])
    sweep_corr = float(np.corrcoef(ev, fv)[0, 1]) if len(results) >= 2 else float("nan")
    ratio = float(np.mean(ev) / max(np.mean(fv), 1e-12))

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/thm_f_beta_sweep.json", "w") as f:
        json.dump({
            "sweep": results,
            "sweep_correlation_formula_vs_empirical": sweep_corr,
            "sweep_ratio_empirical_over_formula": ratio,
            "n_bootstrap": n_bootstrap,
            "N": N,
            "n_rules_subset": len(subset),
        }, f, indent=2)

    print("\n=== THM F VALIDATION ===")
    print(f"  correlation(formula, empirical) across sweep = {sweep_corr:+.3f}")
    print(f"  avg empirical / avg formula                  = {ratio:.3f}")
    print(f"  PASS criterion: correlation > 0.8 and ratio in [0.3, 3.0]")
    if sweep_corr > 0.8 and 0.3 < ratio < 3.0:
        print("  VERDICT: Thm F PASS")
    else:
        print("  VERDICT: Thm F needs tuning (correlation or magnitude off)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
