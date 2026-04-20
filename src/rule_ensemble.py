"""15.B  Rule-ensemble OPE — composed-policy evaluation.

A rule SET S = {rho_1, ..., rho_k} induces a composed deterministic
policy via an action-precedence rule on the firing rules:

    pi_S(x) = noop if no rule fires
            = highest-precedence action among rules that fire on x

We use precedence: abstain > filter > rerank (most conservative wins).

Naive baseline (Sum): treats V(S) ~= sum_i V(rho_i), which ignores
firing overlap and double-counts.
Naive baseline (Max): V(S) ~= max_i V(rho_i), ignores composition.
True ensemble: directly DR-evaluates the composed policy pi_S.

Inclusion-exclusion variance bound: Var(V_hat(pi_S)) <= sum_T (-1)^|T|
Var(V_hat(intersection of T)) — proved sublinear in |S| under
disjoint-firing assumptions.  We just expose Var(V_hat(pi_S))
empirically and compare to the sum-of-individual variance bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators._regression import fires_mask
from src.logs import LoggedRecord
from src.rule_dsl import Rule


_PRECEDENCE = {"abstain": 3, "filter": 2, "rerank": 1, "noop": 0}


@dataclass
class EnsembleResult:
    estimate: float
    stderr: float
    sum_baseline: float
    max_baseline: float


def composed_action(rules: Sequence[Rule], rec: LoggedRecord) -> str:
    best = "noop"
    best_p = 0
    for r in rules:
        if r.fires(rec.ctx):
            if _PRECEDENCE[r.action] > best_p:
                best = r.action
                best_p = _PRECEDENCE[r.action]
    return best


def ensemble_value(
    rules: Sequence[Rule], logs: Sequence[LoggedRecord], cfg: RuleOPEConfig | None = None
) -> EnsembleResult:
    """Evaluate the composed policy pi_S directly."""
    cfg = cfg or RuleOPEConfig()
    est = RuleOPE(cfg).fit(logs)
    m_logged = est.reg.predict_logged(logs).astype(np.float64)
    r_obs = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
    propensities = np.array([max(rec.logged_propensity, 1e-6) for rec in logs])

    pi_actions = np.array([composed_action(rules, rec) for rec in logs])
    logged_actions = np.array([rec.logged_action for rec in logs])

    # Per-record DM for the composed policy
    n = len(logs)
    m_pi = np.zeros(n)
    for action in ("noop", "filter", "rerank", "abstain"):
        if action == "noop":
            m_pi += (pi_actions == action) * m_logged
        else:
            m_a = est.reg.predict_for_action(logs, action).astype(np.float64)
            m_pi += (pi_actions == action) * m_a

    match = (logged_actions == pi_actions) | ((pi_actions == "noop") & (logged_actions == "noop"))
    w = np.where(match, 1.0 / propensities, 0.0)
    psi = m_pi + w * (r_obs - m_logged)
    point = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))

    individual_vals = []
    for r in rules:
        individual_vals.append(est.value(r, logs).estimate)
    return EnsembleResult(
        estimate=point,
        stderr=se,
        sum_baseline=float(np.sum(individual_vals)),
        max_baseline=float(np.max(individual_vals)),
    )
