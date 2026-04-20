"""15.E  Active-query Rule-OPE.

Given a budget B of additional correction labels we can collect,
which queries should we label to maximally reduce Var(V_hat(rho^*))?

EIF-gradient query scoring: under the cross-fit RuleOPE EIF, the
contribution of query i to Var(V_hat) is psi_i^2 / N^2.  If we could
*remove* the noise on query i (oracle correction), the variance would
drop by approximately psi_i^2 / N^2 * (residual_var_i / total_var_i).

We approximate this with the simpler heuristic:
    score_i = |psi_i - psi_bar|  (squared influence)
and select the top-B queries.

We compare against:
    - random sampling (uniform)
    - leverage-only (sample by row-leverage of the regression)

Success criterion: active sampling reduces Var(V_hat(rho^*)) by more
than random.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators._regression import fires_mask, atom_feature_matrix
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class ActiveResult:
    var_random: float
    var_active: float
    var_leverage: float
    reduction_active_pct: float
    reduction_leverage_pct: float


def _psi(rule: Rule, logs: Sequence[LoggedRecord], cfg: RuleOPEConfig) -> np.ndarray:
    est = RuleOPE(cfg).fit(logs)
    m_rule = est.reg.predict_for_rule(logs, rule).astype(np.float64)
    m_logged = est.reg.predict_logged(logs).astype(np.float64)
    r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
    fires = fires_mask(logs, rule)
    logged_actions = np.array([rec.logged_action for rec in logs])
    match = np.where(fires, logged_actions == rule.action, logged_actions == "noop")
    propensities = np.array([max(rec.logged_propensity, 1e-6) for rec in logs])
    w = np.where(match, 1.0 / propensities, 0.0)
    return m_rule + w * (r - m_logged)


def _bootstrap_variance(
    rule: Rule,
    logs_pool: list[LoggedRecord],
    extra_labels: list[LoggedRecord],
    n_boot: int = 60,
    cfg: RuleOPEConfig | None = None,
) -> float:
    cfg = cfg or RuleOPEConfig()
    rng = np.random.default_rng(0)
    augmented = logs_pool + extra_labels
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(augmented), size=len(augmented))
        boot = [augmented[i] for i in idx]
        est = RuleOPE(cfg).fit(boot)
        estimates.append(est.value(rule, boot).estimate)
    return float(np.var(estimates))


def active_compare(
    rule: Rule,
    pool_logs: Sequence[LoggedRecord],
    candidate_logs: Sequence[LoggedRecord],
    budget: int,
    cfg: RuleOPEConfig | None = None,
) -> ActiveResult:
    cfg = cfg or RuleOPEConfig()
    rng = np.random.default_rng(0)
    pool = list(pool_logs)
    cand = list(candidate_logs)

    psi = _psi(rule, cand, cfg)
    psi_bar = psi.mean()
    scores_active = np.abs(psi - psi_bar)
    active_idx = np.argsort(scores_active)[::-1][:budget]
    active_extra = [cand[i] for i in active_idx]

    phi = atom_feature_matrix(cand)
    leverage = (phi ** 2).sum(axis=1)
    lev_idx = np.argsort(leverage)[::-1][:budget]
    leverage_extra = [cand[i] for i in lev_idx]

    rand_idx = rng.choice(len(cand), size=budget, replace=False)
    rand_extra = [cand[int(i)] for i in rand_idx]

    var_a = _bootstrap_variance(rule, pool, active_extra, cfg=cfg)
    var_l = _bootstrap_variance(rule, pool, leverage_extra, cfg=cfg)
    var_r = _bootstrap_variance(rule, pool, rand_extra, cfg=cfg)

    return ActiveResult(
        var_random=var_r,
        var_active=var_a,
        var_leverage=var_l,
        reduction_active_pct=100.0 * (var_r - var_a) / max(var_r, 1e-12),
        reduction_leverage_pct=100.0 * (var_r - var_l) / max(var_r, 1e-12),
    )
