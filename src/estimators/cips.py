"""Clipped IPS (CIPS) and the deterministic-logging CIPS-DR of Saito et al. 2025.

Clipped IPS simply clips the importance weight above a constant M:

    V_CIPS(rho) = (1/N) sum_i min(w_i, M) * 1[a_i^0 = rho(x_i)] * r_i^0

This trades variance for bias.  For a principled comparator we also implement
the *CIPS-DR* variant that combines CIPS with a regression correction, which
is the state-of-the-art for deterministic-logging OPE.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.estimators.base import Estimator, EstimatorResult
from src.estimators._regression import RewardRegressor
from src.logs import LoggedRecord
from src.rule_dsl import Rule


class CIPS(Estimator):
    name = "CIPS"

    def __init__(self, clip: float = 20.0) -> None:
        self.clip = clip

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        N = len(logs)
        match = np.zeros(N, dtype=bool)
        w = np.zeros(N, dtype=np.float64)
        r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
        for i, rec in enumerate(logs):
            rho_a = rule.action if rule.fires(rec.ctx) else "noop"
            if rec.logged_action == rho_a:
                match[i] = True
                w[i] = min(1.0 / max(rec.logged_propensity, 1e-6), self.clip)
        psi = np.where(match, w * r, 0.0)
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(N))
        w_m = w[match]
        ess = float((w_m.sum()) ** 2 / max((w_m ** 2).sum(), 1e-12)) if w_m.size else 0.0
        return EstimatorResult(estimate=est, stderr=se, n_effective=ess)


class CIPS_DR(Estimator):
    name = "CIPS-DR"

    def __init__(self, clip: float = 20.0, alpha: float = 1.0, n_folds: int = 5, seed: int = 0) -> None:
        self.clip = clip
        self.reg = RewardRegressor(alpha=alpha, n_folds=n_folds, seed=seed)

    def fit(self, logs: Sequence[LoggedRecord]) -> "CIPS_DR":
        self.reg.fit(logs)
        return self

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        m_rule = self.reg.predict_for_rule(logs, rule)
        m_logged = self.reg.predict_logged(logs)
        r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
        N = len(logs)
        match = np.zeros(N, dtype=bool)
        w = np.zeros(N, dtype=np.float64)
        for i, rec in enumerate(logs):
            rho_a = rule.action if rule.fires(rec.ctx) else "noop"
            if rec.logged_action == rho_a:
                match[i] = True
                w[i] = min(1.0 / max(rec.logged_propensity, 1e-6), self.clip)
        psi = m_rule + np.where(match, w * (r - m_logged), 0.0)
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(N))
        w_m = w[match]
        ess = float((w_m.sum()) ** 2 / max((w_m ** 2).sum(), 1e-12)) if w_m.size else 0.0
        return EstimatorResult(estimate=est, stderr=se, n_effective=ess)
