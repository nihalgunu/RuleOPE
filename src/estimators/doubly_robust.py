"""Standard doubly-robust estimator.

    V_DR(rho) = (1/N) sum_i [ m_hat(x_i, rho(x_i))
                              + 1[a_i^0 = rho(x_i)] / pi_0(a_i^0|x_i)
                                * (r_i^0 - m_hat(x_i, a_i^0)) ]

Uses the same cross-fitted regression as DM.  Consistent whenever either the
regression or the propensity model is correct (the "doubly-robust" property,
Robins et al. 1994).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.estimators.base import Estimator, EstimatorResult
from src.estimators._regression import RewardRegressor
from src.logs import LoggedRecord
from src.rule_dsl import Rule


class DoublyRobust(Estimator):
    name = "DR"

    def __init__(self, alpha: float = 1.0, n_folds: int = 5, seed: int = 0) -> None:
        self.reg = RewardRegressor(alpha=alpha, n_folds=n_folds, seed=seed)

    def fit(self, logs: Sequence[LoggedRecord]) -> "DoublyRobust":
        self.reg.fit(logs)
        return self

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        from src.estimators._regression import fires_mask
        m_rule = self.reg.predict_for_rule(logs, rule)
        m_logged = self.reg.predict_logged(logs)
        r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
        fires = fires_mask(logs, rule)
        logged_actions = np.array([rec.logged_action for rec in logs])
        match = np.where(fires, logged_actions == rule.action, logged_actions == "noop")
        propensities = np.array(
            [max(rec.logged_propensity, 1e-6) for rec in logs], dtype=np.float64
        )
        w = np.where(match, 1.0 / propensities, 0.0)
        psi = m_rule + w * (r - m_logged)
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
        w_m = w[match]
        ess = float((w_m.sum()) ** 2 / max((w_m ** 2).sum(), 1e-12)) if w_m.size else 0.0
        return EstimatorResult(estimate=est, stderr=se, n_effective=ess)
