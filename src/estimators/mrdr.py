"""More Robust Doubly Robust (Farajtabar, Chow, Ghavamzadeh 2018).

The DR estimator's variance depends on the regression's accuracy
*weighted by the squared importance weight*. MRDR fits the regression
to minimise that weighted error directly:

    L_MRDR(m) = sum_i [ (e(rho, x_i) / pi_0(a_i | x_i))^2
                        * (1 - pi_0(a_i | x_i) / e(rho, x_i))
                        * (r_i - m(x_i, a_i))^2 ]

where e(rho, x) = 1 if a = rho(x), 0 otherwise (deterministic target).

For a rule-OPE target rho with rule action a_rho on queries where the
rule fires (and noop elsewhere), the per-sample weight reduces to:

    w_i = match_i * (1 - pi_0(a_i | x_i)) / pi_0(a_i | x_i)^2

with match_i = 1 if (fires_i and a_i = a_rho) or (not fires_i and a_i = noop).

Because the weight depends on rho, the regression is refit per rule.
This is cheap relative to fold cross-fitting since the underlying
ridge solve is O(d^3) with d ~ 100.

Reference:
    Farajtabar, M., Chow, Y., Ghavamzadeh, M. (2018).
    "More Robust Doubly Robust Off-policy Evaluation."  ICML.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge

from src.estimators.base import Estimator, EstimatorResult
from src.estimators._regression import (
    ACTIONS,
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
    fires_mask,
)
from src.logs import LoggedRecord
from src.rule_dsl import Rule


class MRDR(Estimator):
    name = "MRDR"

    def __init__(self, alpha: float = 1.0, seed: int = 0) -> None:
        self.alpha = alpha
        self.seed = seed
        self._cached_logs: int | None = None
        self._phi: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self._propensities: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._logged_action_strs: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, logs: Sequence[LoggedRecord]) -> "MRDR":
        self._cached_logs = id(logs)
        self._phi = atom_feature_matrix(logs)
        self._actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
        self._rewards = np.array([r.logged_reward for r in logs], dtype=np.float32)
        self._propensities = np.array(
            [max(r.logged_propensity, 1e-6) for r in logs], dtype=np.float64
        )
        self._X = _joint_features(self._phi, self._actions)
        self._logged_action_strs = np.array([r.logged_action for r in logs])
        return self

    # ------------------------------------------------------------------
    def _fit_for_rule(self, rule: Rule, logs: Sequence[LoggedRecord]) -> Ridge:
        assert self._X is not None
        fires = fires_mask(logs, rule)
        # match_i = 1 when logged action equals the rule's target action on
        # the rho(x_i) policy: (fires & a==rule.action) | (~fires & a==noop)
        match = np.where(
            fires,
            self._logged_action_strs == rule.action,
            self._logged_action_strs == "noop",
        ).astype(np.float64)
        pi = self._propensities
        # MRDR weight (deterministic target): match * (1 - pi) / pi^2.
        # All-zero weights happen when no log matches the target — fall back
        # to standard DR by giving each sample weight 1.
        weights = match * (1.0 - pi) / np.maximum(pi ** 2, 1e-12)
        if weights.sum() < 1e-9:
            weights = np.ones_like(weights)
        m = Ridge(alpha=self.alpha)
        m.fit(self._X, self._rewards, sample_weight=weights)
        return m

    # ------------------------------------------------------------------
    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        assert self._phi is not None and self._X is not None
        m_per_rule = self._fit_for_rule(rule, logs)
        N = len(logs)

        # Predict m_hat at the rho-action for each x
        fires = fires_mask(logs, rule)
        rho_actions = np.where(
            fires, _ACTION_IDX[rule.action], _ACTION_IDX["noop"]
        ).astype(np.int64)
        X_rho = _joint_features(self._phi, rho_actions)
        m_rule = m_per_rule.predict(X_rho).astype(np.float64)

        # Predict m_hat at the logged action
        m_logged = m_per_rule.predict(self._X).astype(np.float64)

        r = self._rewards.astype(np.float64)
        match = np.where(
            fires,
            self._logged_action_strs == rule.action,
            self._logged_action_strs == "noop",
        )
        pi = self._propensities
        w = np.where(match, 1.0 / pi, 0.0)
        psi = m_rule + w * (r - m_logged)
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(N))
        w_m = w[match]
        ess = float((w_m.sum()) ** 2 / max((w_m ** 2).sum(), 1e-12)) if w_m.size else 0.0
        return EstimatorResult(estimate=est, stderr=se, n_effective=ess)
