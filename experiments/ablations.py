"""Per-rule (non-compositional) Doubly-Robust baseline used in the headline
estimator panel.

This file exposes one class, :class:`NonCompositionalDR`, which is the
``NonCompDR`` baseline appearing in the paper. Unlike :class:`RuleOPE`, which
shares atom-action coefficients across the entire rule pool through a single
joint regression, ``NonCompositionalDR`` refits a fresh ridge regression on
*only the logs where the rule fires*, then forms the standard DR estimator
from that per-rule regression. It is the per-rule DR baseline of Saito et al.
(OBP, 2021) instantiated on the released RAG pool.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.linear_model import Ridge

from src.estimators.base import Estimator, EstimatorResult
from src.logs import LoggedRecord  # noqa: F401  (type hint surface)
from src.rule_dsl import Rule


class NonCompositionalDR(Estimator):
    name = "NonCompDR"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, logs):
        self.logs = list(logs)
        return self

    def value(self, rule: Rule, logs):
        from src.estimators._regression import (
            atom_feature_matrix, fires_mask, _ACTION_IDX, _joint_features,
        )
        phi = atom_feature_matrix(logs)
        fires = fires_mask(logs, rule)
        if fires.sum() < 10:
            # degenerate: fall back to mean reward
            r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
            return EstimatorResult(
                estimate=float(r.mean()),
                stderr=float(r.std() / np.sqrt(len(r))),
                n_effective=float(len(r)),
            )

        # Fit a regression *only* on records where the rule fires (kills
        # cross-rule atom sharing — this is the contrast with RuleOPE).
        idx = np.where(fires)[0]
        actions = np.array(
            [_ACTION_IDX[logs[i].logged_action] for i in idx], dtype=np.int64
        )
        rewards = np.array(
            [logs[i].logged_reward for i in idx], dtype=np.float32
        )
        X = _joint_features(phi[idx], actions)
        model = Ridge(alpha=self.alpha).fit(X, rewards)

        # DM term: predict reward under the rule's intervention on every record.
        all_a = np.array(
            [_ACTION_IDX[rec.logged_action] for rec in logs], dtype=np.int64
        )
        rule_a = _ACTION_IDX[rule.action]
        all_a[fires] = rule_a
        X_all = _joint_features(phi, all_a)
        m_rule = model.predict(X_all).astype(np.float32)

        # DR correction at the logged action.
        X_logged = _joint_features(
            phi,
            np.array(
                [_ACTION_IDX[rec.logged_action] for rec in logs], dtype=np.int64
            ),
        )
        m_logged = model.predict(X_logged).astype(np.float32)
        r_obs = np.array(
            [rec.logged_reward for rec in logs], dtype=np.float64
        )
        logged_actions = np.array([rec.logged_action for rec in logs])
        match = np.where(
            fires, logged_actions == rule.action, logged_actions == "noop"
        )
        propensities = np.array(
            [max(rec.logged_propensity, 1e-6) for rec in logs],
            dtype=np.float64,
        )
        w = np.where(match, 1.0 / propensities, 0.0)
        psi = m_rule + w * (r_obs - m_logged)
        est = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
        return EstimatorResult(
            estimate=est, stderr=se, n_effective=float(fires.sum())
        )
