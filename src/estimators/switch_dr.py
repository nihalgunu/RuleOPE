"""Switch-DR estimator (Wang, Agarwal, Dudík 2017).

When the importance weight w_i exceeds a threshold tau, the per-record
contribution falls back to the Direct Method (no IPS correction).
This trades a small finite-sample bias for unbounded-variance protection.

    V_switch(rho) = (1/N) sum_i [ m_hat(x_i, rho(x_i))
                                  + 1[w_i <= tau] * 1[a_i^0 = rho(x_i)]
                                    * w_i * (r_i^0 - m_hat(x_i, a_i^0)) ]

A practical alternative to clipping (CIPS), with a cleaner
bias-variance story: when w_i > tau the IPS correction is zero (not
clipped), which is unbiased under DM consistency. We default to
tau = 5.0 (modest threshold given uniform-stochastic logging gives
w in {0, 3} and skewed-stochastic gives w up to 1/0.15 ~= 6.67).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.estimators.base import Estimator, EstimatorResult
from src.estimators._regression import RewardRegressor, fires_mask
from src.logs import LoggedRecord
from src.rule_dsl import Rule


class SwitchDR(Estimator):
    name = "SwitchDR"

    def __init__(self, alpha: float = 1.0, n_folds: int = 5, seed: int = 0,
                 tau: float = 5.0) -> None:
        self.reg = RewardRegressor(alpha=alpha, n_folds=n_folds, seed=seed)
        self.tau = float(tau)

    def fit(self, logs: Sequence[LoggedRecord]) -> "SwitchDR":
        self.reg.fit(logs)
        return self

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        m_rule = self.reg.predict_for_rule(logs, rule)
        m_logged = self.reg.predict_logged(logs)
        r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
        fires = fires_mask(logs, rule)
        logged_actions = np.array([rec.logged_action for rec in logs])
        match = np.where(fires, logged_actions == rule.action, logged_actions == "noop")
        propensities = np.array(
            [max(rec.logged_propensity, 1e-6) for rec in logs], dtype=np.float64
        )
        w_raw = np.where(match, 1.0 / propensities, 0.0)
        # Switch: zero out the IPS contribution when w > tau (fall back to DM)
        w = np.where(w_raw <= self.tau, w_raw, 0.0)
        psi = m_rule + w * (r - m_logged)
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
        w_m = w[match & (w > 0)]
        ess = float((w_m.sum()) ** 2 / max((w_m ** 2).sum(), 1e-12)) if w_m.size else 0.0
        return EstimatorResult(estimate=est, stderr=se, n_effective=ess)

    def value_many(self, rules, logs):
        return {r.id: self.value(r, logs) for r in rules}
