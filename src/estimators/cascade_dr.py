"""Cascade Doubly-Robust (Kiyohara et al. 2022) adapted to rule actions.

The original Cascade DR estimator targets slate recommendation where the
action is a ranked list and the reward decomposes as a sum over positions.
We adapt it to RAG by treating the action "filter" / "rerank" / "abstain" /
"noop" as modifying the top-k *positions* of the retrieval list and regressing
position-wise partial rewards.

Our adaptation: we fit a position-wise regression r_k(x, a) that predicts the
contribution of position k to the total reward under action a; the total
predicted reward is sum_k r_k(x, a); the DR correction uses the logged
reward minus the position-wise model at the logged positions.  In the
retrieval setting we use K = 3 top positions, which is the standard depth at
which the generator attends in RAG systems.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.estimators.base import Estimator, EstimatorResult
from src.estimators._regression import (
    ACTIONS,
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
)
from src.logs import LoggedRecord
from src.rule_dsl import Rule


def _position_weights(ctx: dict) -> np.ndarray:
    """Extract per-position signals from a context dict.

    The features `top1_score`, `top2_score`, `top3_score` give us proxy
    per-position reward contributions.  We normalise to sum to ~1.
    """
    scores = np.array(
        [ctx.get("top1_score", 0.0), ctx.get("top2_score", 0.0), ctx.get("top3_score", 0.0)],
        dtype=np.float64,
    )
    total = scores.sum()
    if total <= 0:
        return np.array([1.0, 0.0, 0.0])
    return scores / total


class CascadeDR(Estimator):
    name = "CascadeDR"

    def __init__(self, K: int = 3, alpha: float = 1.0, n_folds: int = 5, seed: int = 0) -> None:
        self.K = K
        self.alpha = alpha
        self.n_folds = n_folds
        self.seed = seed
        self._fold_models: list[list[Ridge]] = []
        self._full_models: list[Ridge] = []
        self._fold_assign: np.ndarray | None = None

    def fit(self, logs: Sequence[LoggedRecord]) -> "CascadeDR":
        phi = atom_feature_matrix(logs)
        actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
        rewards = np.array([r.logged_reward for r in logs], dtype=np.float32)
        pw = np.stack([_position_weights(rec.ctx) for rec in logs], axis=0)  # (N, K)
        X = _joint_features(phi, actions)

        self._full_models = []
        for k in range(self.K):
            y_k = rewards * pw[:, k]
            self._full_models.append(Ridge(alpha=self.alpha).fit(X, y_k))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        self._fold_assign = np.zeros(len(logs), dtype=np.int64)
        self._fold_models = [[] for _ in range(self.K)]
        for f, (tr, te) in enumerate(kf.split(X)):
            self._fold_assign[te] = f
            for k in range(self.K):
                y_k = rewards * pw[:, k]
                self._fold_models[k].append(Ridge(alpha=self.alpha).fit(X[tr], y_k[tr]))
        return self

    def _predict_total(self, logs: Sequence[LoggedRecord], action_lookup: np.ndarray, cross_fit: bool) -> np.ndarray:
        phi = atom_feature_matrix(logs)
        N = phi.shape[0]
        X = _joint_features(phi, action_lookup)
        total = np.zeros(N, dtype=np.float32)
        for k in range(self.K):
            if cross_fit:
                pred_k = np.zeros(N, dtype=np.float32)
                for f in range(self.n_folds):
                    mask = self._fold_assign == f
                    if mask.any():
                        pred_k[mask] = self._fold_models[k][f].predict(X[mask]).astype(np.float32)
            else:
                pred_k = self._full_models[k].predict(X).astype(np.float32)
            total = total + pred_k
        return total

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        logged_idx = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
        fires = np.array([rule.fires(rec.ctx) for rec in logs], dtype=bool)
        rule_idx = logged_idx.copy()
        rule_idx[fires] = _ACTION_IDX[rule.action]

        m_rule = self._predict_total(logs, rule_idx, cross_fit=True)
        m_logged = self._predict_total(logs, logged_idx, cross_fit=True)

        r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
        match = (logged_idx == rule_idx)
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
