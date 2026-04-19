"""Phase 3 ablations.

A. Compositional factorisation vs. per-rule regression.
   `RuleOPE` already uses the compositional regressor.  A naive baseline
   refits a fresh ridge per rule on *only the logs where rule fires*, and
   uses that regression for the DM/DR term.  We call this `NonCompositional`.

B. Correction-noise sensitivity.
   We vary noise in {0, 10, 20, 30, 50}% and report MSE of RuleOPE, DR, DM.

C. Rule depth sensitivity.
   Stratify results by rule depth in {1, 2, 3}.

D. Sample efficiency.
   Vary N in {250, 500, 1000, 2000, 4000}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.linear_model import Ridge

from src.correction_sim import CorrectionConfig, assign_corrections
from src.estimators.base import Estimator, EstimatorResult
from src.estimators.direct_method import DirectMethod
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE
from src.evaluate import all_metrics
from src.logs import LoggedRecord
from src.rag_substrate import SubstrateConfig, generate_logs, ground_truth_many
from src.rule_dsl import Rule, load_rules


# ----------------------------------------------------------------------
# Non-compositional DR baseline: refits a fresh regression per rule.
# ----------------------------------------------------------------------
class NonCompositionalDR(Estimator):
    name = "NonCompDR"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, logs):
        self.logs = list(logs)
        return self

    def value(self, rule: Rule, logs):
        from src.estimators._regression import atom_feature_matrix, fires_mask, _ACTION_IDX, _joint_features
        phi = atom_feature_matrix(logs)
        fires = fires_mask(logs, rule)
        if fires.sum() < 10:
            # degenerate: return mean reward
            r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
            return EstimatorResult(estimate=float(r.mean()), stderr=float(r.std()/np.sqrt(len(r))), n_effective=float(len(r)))
        # Fit a regression *only* on records where the rule fires (this
        # kills the cross-rule sharing).
        idx = np.where(fires)[0]
        actions = np.array([_ACTION_IDX[logs[i].logged_action] for i in idx], dtype=np.int64)
        rewards = np.array([logs[i].logged_reward for i in idx], dtype=np.float32)
        X = _joint_features(phi[idx], actions)
        model = Ridge(alpha=self.alpha).fit(X, rewards)

        # DM term for rule: predict at (x, rule.action) on *all* records.
        all_a = np.array(
            [_ACTION_IDX[rec.logged_action] for rec in logs], dtype=np.int64
        )
        rule_a = _ACTION_IDX[rule.action]
        all_a[fires] = rule_a
        X_all = _joint_features(phi, all_a)
        m_rule = model.predict(X_all).astype(np.float32)

        # DR correction at logged action.
        X_logged = _joint_features(phi, np.array([_ACTION_IDX[rec.logged_action] for rec in logs], dtype=np.int64))
        m_logged = model.predict(X_logged).astype(np.float32)
        r_obs = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
        logged_actions = np.array([rec.logged_action for rec in logs])
        match = np.where(fires, logged_actions == rule.action, logged_actions == "noop")
        propensities = np.array([max(rec.logged_propensity, 1e-6) for rec in logs], dtype=np.float64)
        w = np.where(match, 1.0 / propensities, 0.0)
        psi = m_rule + w * (r_obs - m_logged)
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
        return EstimatorResult(estimate=est, stderr=se, n_effective=float(fires.sum()))


def _trial(n_queries, seed, noise, rules):
    cfg = SubstrateConfig(n_queries=n_queries, seed=seed, logging="stochastic")
    logs = generate_logs(cfg)
    logs = assign_corrections(
        logs,
        CorrectionConfig(
            base_rate=0.15, error_sensitivity=4.0, noise_frac=noise,
            seed=seed + 1000,
        ),
    )
    gt = ground_truth_many(rules, logs)
    return logs, gt


def ablation_factorisation(rules, seed=11):
    """Compare compositional RuleOPE DR regression vs. per-rule regression."""
    logs, gt = _trial(3000, seed, 0.10, rules)
    out = {}
    for est in [RuleOPE(), DoublyRobust(), NonCompositionalDR()]:
        if hasattr(est, "fit"):
            est.fit(logs)
        res = est.value_many(rules, logs)
        estimates = {k: v.estimate for k, v in res.items()}
        stderrs = {k: v.stderr for k, v in res.items()}
        out[est.name] = all_metrics(estimates, stderrs, gt, topk=20)
    return out


def ablation_noise(rules):
    out = {}
    for noise in (0.0, 0.1, 0.2, 0.3, 0.5):
        logs, gt = _trial(3000, seed=17, noise=noise, rules=rules)
        row = {}
        for est in [RuleOPE(), DoublyRobust(), DirectMethod()]:
            if hasattr(est, "fit"):
                est.fit(logs)
            res = est.value_many(rules, logs)
            estimates = {k: v.estimate for k, v in res.items()}
            stderrs = {k: v.stderr for k, v in res.items()}
            row[est.name] = all_metrics(estimates, stderrs, gt, topk=20)
        out[f"noise={noise:.2f}"] = row
        print(f"noise={noise:.2f}  " + "  ".join(f"{k}:MSE={v['mse']:.5f}" for k, v in row.items()))
    return out


def ablation_depth(rules):
    """Stratify results by rule depth."""
    logs, gt = _trial(3000, seed=23, noise=0.10, rules=rules)
    out = {}
    for est in [RuleOPE(), DoublyRobust(), DirectMethod()]:
        if hasattr(est, "fit"):
            est.fit(logs)
        res = est.value_many(rules, logs)
        estimates = {k: v.estimate for k, v in res.items()}
        stderrs = {k: v.stderr for k, v in res.items()}
        for d in (1, 2, 3):
            rule_ids = {r.id for r in rules if r.depth() == d}
            sub_est = {k: v for k, v in estimates.items() if k in rule_ids}
            sub_se = {k: v for k, v in stderrs.items() if k in rule_ids}
            sub_gt = {k: v for k, v in gt.items() if k in rule_ids}
            m = all_metrics(sub_est, sub_se, sub_gt, topk=10)
            out.setdefault(est.name, {})[f"depth={d}"] = m
            print(f"{est.name:>8s} depth={d}  MSE={m['mse']:.5f}  tau@10={m['topk_tau']:+.3f}")
    return out


def ablation_sample_efficiency(rules):
    out = {}
    for n in (250, 500, 1000, 2000, 4000):
        logs, gt = _trial(n, seed=31, noise=0.10, rules=rules)
        row = {}
        for est in [RuleOPE(), DoublyRobust(), DirectMethod()]:
            if hasattr(est, "fit"):
                est.fit(logs)
            res = est.value_many(rules, logs)
            estimates = {k: v.estimate for k, v in res.items()}
            stderrs = {k: v.stderr for k, v in res.items()}
            row[est.name] = all_metrics(estimates, stderrs, gt, topk=20)
        out[f"N={n}"] = row
        print(f"N={n}  " + "  ".join(f"{k}:MSE={v['mse']:.5f}" for k, v in row.items()))
    return out


def main() -> int:
    rules = load_rules("eval/rules_v1.jsonl")
    out = {}
    print("=== A. Compositional factorisation ablation ===")
    out["factorisation"] = ablation_factorisation(rules)
    print("=== B. Correction noise sensitivity ===")
    out["noise"] = ablation_noise(rules)
    print("=== C. Rule depth stratification ===")
    out["depth"] = ablation_depth(rules)
    print("=== D. Sample efficiency ===")
    out["sample_efficiency"] = ablation_sample_efficiency(rules)

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/ablations.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
