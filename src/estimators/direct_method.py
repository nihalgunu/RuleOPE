"""Direct Method estimator.

    V_DM(rho) = (1/N) sum_i m_hat(x_i, rho(x_i))

with rho(x_i) = rule.action if the rule fires on x_i, else logged_action.
Uses cross-fitted ridge regression (see `_regression.RewardRegressor`).

Pros: zero variance from importance weights, consistent if the regression is
well-specified.  Cons: fully model-dependent -- biased if the regression is
biased.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.estimators.base import Estimator, EstimatorResult
from src.estimators._regression import RewardRegressor
from src.logs import LoggedRecord
from src.rule_dsl import Rule


class DirectMethod(Estimator):
    name = "DM"

    def __init__(self, alpha: float = 1.0, n_folds: int = 5, seed: int = 0) -> None:
        self.reg = RewardRegressor(alpha=alpha, n_folds=n_folds, seed=seed)

    def fit(self, logs: Sequence[LoggedRecord]) -> "DirectMethod":
        self.reg.fit(logs)
        return self

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        m_rule = self.reg.predict_for_rule(logs, rule)
        est = float(m_rule.mean())
        # DM standard error is just the empirical SE of the per-record predictions.
        se = float(m_rule.std(ddof=1) / np.sqrt(len(m_rule)))
        return EstimatorResult(estimate=est, stderr=se, n_effective=float(len(m_rule)))
