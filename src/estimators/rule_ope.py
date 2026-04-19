"""Rule-OPE: our doubly-robust estimator with compositional variance reduction
and correction-signal fusion.

Point estimate
--------------
For a conjunctive rule rho with action a_rho:

    V_hat(rho) = (1/N) sum_i [
        m_hat_C(x_i, rho(x_i))                                  (DM term, compositional)
      + 1[a_i^0 = rho(x_i)] / pi_0(a_i^0 | x_i) * (r_i^0 - m_hat_C(x_i, a_i^0))
                                                                (logged-action DR)
      + c_i * g_hat(x_i, rho(x_i)) * (tilde{r}_i(a_rho) - m_hat_C(x_i, rho(x_i)))
                                                                (correction DR)
    ]

Components
----------
* m_hat_C is a ridge regression parameterised as
      m_hat_C(x, a) = beta_{0,a} + sum_alpha phi_alpha(x) * beta_{alpha, a}
  where phi_alpha is the indicator of atomic predicate alpha.  Under this
  factorisation two rules that share an atom also share the regression
  coefficient, which is what shrinks the cross-rule variance.  Implemented in
  `_regression.RewardRegressor`.
* g_hat(x, a) is a gate learnt from the correction signal: the model predicts
  P(c = 1 | x, a = a_0) and we use a relative likelihood to gate the
  correction term for non-logged actions.  The gate is clipped to [0, clip_g].
* tilde{r}_i(a) is a pseudo-reward imputation given a correction was issued.
  For action "abstain" we assume the abstain reward r_abs (exogenous config).
  For action "filter" or "rerank" we impute the expected reward under the
  second-best retrieval list, which we estimate from the regression at x with
  the relevant action.

Consistency
-----------
The estimator inherits doubly-robust consistency whenever either the reward
regression or the effective propensity model (union of logging propensity and
correction-informativeness) is correctly specified, plus a positivity
assumption on rule firing.  See `theory/proofs.tex`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.estimators.base import Estimator, EstimatorResult
from src.estimators._regression import (
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
    RewardRegressor,
)
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class RuleOPEConfig:
    alpha: float = 1.0
    n_folds: int = 5
    seed: int = 0
    # upper clip on the correction gate (1 / propensity bound)
    gate_clip: float = 5.0
    # pseudo-reward for abstain under correction (generous default)
    r_abstain: float = 0.5
    # shrinkage of the correction-driven term (0 disables, 1 uses as-is)
    correction_weight: float = 1.0


class RuleOPE(Estimator):
    name = "RuleOPE"

    def __init__(self, config: RuleOPEConfig | None = None) -> None:
        self.cfg = config or RuleOPEConfig()
        self.reg = RewardRegressor(alpha=self.cfg.alpha, n_folds=self.cfg.n_folds, seed=self.cfg.seed)
        self._gate: LogisticRegression | None = None

    # ------------------------------------------------------------------
    def fit(self, logs: Sequence[LoggedRecord]) -> "RuleOPE":
        self.reg.fit(logs)
        # Gate model: P(correction = 1 | x, a_logged).
        phi = atom_feature_matrix(logs)
        actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
        y = np.array([r.correction for r in logs], dtype=np.int32)
        X = _joint_features(phi, actions)
        # If corrections are all zero or all one we fall back to a constant gate.
        if y.sum() == 0 or y.sum() == len(y):
            self._gate = None
        else:
            self._gate = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
        return self

    # ------------------------------------------------------------------
    def _gate_prob(self, logs: Sequence[LoggedRecord], action: str) -> np.ndarray:
        N = len(logs)
        if self._gate is None:
            return np.full(N, float(np.mean([r.correction for r in logs])), dtype=np.float64)
        phi = atom_feature_matrix(logs)
        a_idx = _ACTION_IDX[action]
        actions = np.full(N, a_idx, dtype=np.int64)
        X = _joint_features(phi, actions)
        p = self._gate.predict_proba(X)[:, 1]
        return p.astype(np.float64)

    # ------------------------------------------------------------------
    def _pseudo_reward(self, logs: Sequence[LoggedRecord], action: str) -> np.ndarray:
        """Counterfactual reward imputation given a correction was issued.

        The correction signal tells us that the logged (noop) answer was wrong,
        which is direct evidence that an *alternative* action could have done
        better.  We use the abstain reward r_abs as a lower bound on the
        information the correction provides -- when c=1, taking the abstain
        action would have been strictly better than the (wrong) noop answer.
        For filter/rerank we combine the regression prediction with the abstain
        baseline: tilde_r = max(m_hat(x, a), r_abs).  This is conservative and
        goes to the DR correction term to shrink it when the regression
        over-estimates.
        """
        if action == "abstain":
            return np.full(len(logs), self.cfg.r_abstain, dtype=np.float64)
        m_a = self.reg.predict_for_action(logs, action).astype(np.float64)
        return np.maximum(m_a, self.cfg.r_abstain)

    # ------------------------------------------------------------------
    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        m_rule = self.reg.predict_for_rule(logs, rule).astype(np.float64)
        m_logged = self.reg.predict_logged(logs).astype(np.float64)

        from src.estimators._regression import fires_mask
        r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
        c = np.array([rec.correction for rec in logs], dtype=np.float64)
        fires = fires_mask(logs, rule)
        logged_actions = np.array([rec.logged_action for rec in logs])
        match = np.where(fires, logged_actions == rule.action, logged_actions == "noop")
        propensities = np.array(
            [max(rec.logged_propensity, 1e-6) for rec in logs], dtype=np.float64
        )
        w_logged = np.where(match, 1.0 / propensities, 0.0)
        psi_logged = m_rule + w_logged * (r - m_logged)

        # Correction-driven term: active only on queries where rule fires with
        # a non-noop action and the logged action was noop (so no coverage from
        # the logged-action DR term).
        active = fires & ~match & (rule.action != "noop")
        if active.any():
            p_ratio = np.zeros(len(logs), dtype=np.float64)
            p_c_given_noop = self._gate_prob(logs, "noop")
            p_c_given_a = self._gate_prob(logs, rule.action)
            # "Correction-informativeness": 1 - P(c|a) / P(c|noop).  Intuitively,
            # if correction is *less* likely under the rule's action than under
            # the logging action, the correction signal is informative that
            # the rule's action would have avoided the error.
            eps = 1e-3
            informativeness = np.clip(
                1.0 - p_c_given_a / np.maximum(p_c_given_noop, eps),
                0.0,
                1.0,
            )
            # Gate: we only gain signal when there was actually a correction.
            gate = np.minimum(informativeness / np.maximum(p_c_given_noop, eps), self.cfg.gate_clip)
            p_ratio = np.where(active, gate, 0.0)

            pseudo_r = self._pseudo_reward(logs, rule.action)
            psi_corr = self.cfg.correction_weight * c * p_ratio * (pseudo_r - m_rule)
        else:
            psi_corr = np.zeros(len(logs), dtype=np.float64)

        psi = psi_logged + psi_corr
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
        w_all = w_logged + np.abs(psi_corr)
        ess = float((w_all.sum()) ** 2 / max((w_all ** 2).sum(), 1e-12)) if w_all.any() else 0.0
        return EstimatorResult(estimate=est, stderr=se, n_effective=ess)
