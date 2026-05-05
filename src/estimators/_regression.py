"""Reward regression used by DM / DR / Cascade DR / Rule-OPE.

We use a *compositional* featurization: each record is encoded as a binary
vector over the fixed atom vocabulary, and each action is one-hot encoded.
The joint featurization is `phi(x) tensor e_a`.  A ridge regression over this
joint feature space is equivalent to a per-action linear model in atoms, which
is exactly the factorisation the theory exploits.

For the standard (non-rule-aware) baselines we use the same featurization so
that none of the estimators has an unfair head-start: the only difference is
how the regression is *used*.

Cross-fitting
-------------
To avoid the overfitting bias that makes DR estimators lose their consistency
guarantees in finite samples, we fit the regression on K-1 folds and evaluate
on the held-out fold; predictions are concatenated back into the original
order.  See Chernozhukov et al. 2018.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.logs import LoggedRecord
from src.rule_dsl import ATOMS


ACTIONS = ("noop", "filter", "rerank", "abstain")
_ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
_N_ACTIONS = len(ACTIONS)


_PHI_CACHE: dict[int, np.ndarray] = {}


def atom_feature_matrix(logs: Sequence[LoggedRecord]) -> np.ndarray:
    """Return (N, n_atoms) binary indicator matrix, memoised by id(logs)."""
    key = id(logs)
    cached = _PHI_CACHE.get(key)
    if cached is not None and cached.shape[0] == len(logs):
        return cached
    N = len(logs)
    d = len(ATOMS)
    M = np.zeros((N, d), dtype=np.float32)
    for i, rec in enumerate(logs):
        for j, atom in enumerate(ATOMS):
            if atom.eval(rec.ctx):
                M[i, j] = 1.0
    _PHI_CACHE[key] = M
    return M


def fires_mask(logs: Sequence[LoggedRecord], rule) -> np.ndarray:
    """Compute rule-firing mask from the (cached) atom matrix without re-evaluating atoms."""
    phi = atom_feature_matrix(logs)  # (N, d)
    idxs = [list(ATOMS).index(a) for a in rule.atoms]
    mask = np.ones(phi.shape[0], dtype=bool)
    for j in idxs:
        mask &= phi[:, j] > 0.5
    return mask


def _joint_features(phi: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Tensor product of per-record atom indicators and action one-hots.

    phi: (N, d),  actions: (N,) ints in [0, _N_ACTIONS)
    returns (N, d * _N_ACTIONS + _N_ACTIONS) with an added per-action intercept.
    """
    N, d = phi.shape
    out = np.zeros((N, _N_ACTIONS + d * _N_ACTIONS), dtype=np.float32)
    out[np.arange(N), actions] = 1.0  # action intercept
    for a_idx in range(_N_ACTIONS):
        mask = actions == a_idx
        if mask.any():
            out[mask, _N_ACTIONS + a_idx * d : _N_ACTIONS + (a_idx + 1) * d] = phi[mask]
    return out


class RewardRegressor:
    """Ridge regression of logged reward on atom*action features.

    We fit one regressor *on the logged observations only* (which were logged
    under the logging policy distribution over actions), then use it to
    *predict* the reward for any (x, a) pair we like.
    """

    def __init__(self, alpha: float = 1.0, n_folds: int = 5, seed: int = 0) -> None:
        self.alpha = alpha
        self.n_folds = n_folds
        self.seed = seed
        self._full: Ridge | None = None
        self._fold_models: list[Ridge] = []
        self._fold_assign: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, logs: Sequence[LoggedRecord]) -> "RewardRegressor":
        phi = atom_feature_matrix(logs)
        actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
        rewards = np.array([r.logged_reward for r in logs], dtype=np.float32)
        X = _joint_features(phi, actions)

        # Full-data model used for out-of-sample prediction of *new* records.
        self._full = Ridge(alpha=self.alpha).fit(X, rewards)

        # Cross-fitted models used for in-sample DR predictions.
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        self._fold_assign = np.zeros(len(logs), dtype=np.int64)
        self._fold_models = []
        for f, (train_idx, test_idx) in enumerate(kf.split(X)):
            model = Ridge(alpha=self.alpha).fit(X[train_idx], rewards[train_idx])
            self._fold_models.append(model)
            self._fold_assign[test_idx] = f
        return self

    # ------------------------------------------------------------------
    def _predict(self, phi: np.ndarray, action_idx: int, use_fold: bool, fold_ids: np.ndarray | None = None) -> np.ndarray:
        N = phi.shape[0]
        actions = np.full(N, action_idx, dtype=np.int64)
        X = _joint_features(phi, actions)
        if not use_fold:
            assert self._full is not None
            return self._full.predict(X).astype(np.float32)
        assert fold_ids is not None
        out = np.zeros(N, dtype=np.float32)
        for f, model in enumerate(self._fold_models):
            mask = fold_ids == f
            if mask.any():
                out[mask] = model.predict(X[mask]).astype(np.float32)
        return out

    def predict_for_action(self, logs: Sequence[LoggedRecord], action: str, cross_fit: bool = True) -> np.ndarray:
        phi = atom_feature_matrix(logs)
        a_idx = _ACTION_IDX[action]
        return self._predict(
            phi,
            a_idx,
            use_fold=cross_fit,
            fold_ids=self._fold_assign if cross_fit else None,
        )

    def predict_logged(self, logs: Sequence[LoggedRecord], cross_fit: bool = True) -> np.ndarray:
        """Predict m_hat(x_i, a_i^0) for each logged record."""
        phi = atom_feature_matrix(logs)
        N = phi.shape[0]
        actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
        X = _joint_features(phi, actions)
        if not cross_fit:
            assert self._full is not None
            return self._full.predict(X).astype(np.float32)
        assert self._fold_assign is not None
        out = np.zeros(N, dtype=np.float32)
        for f, model in enumerate(self._fold_models):
            mask = self._fold_assign == f
            if mask.any():
                out[mask] = model.predict(X[mask]).astype(np.float32)
        return out

    def predict_for_rule(self, logs: Sequence[LoggedRecord], rule, cross_fit: bool = True) -> np.ndarray:
        """Predict m_hat(x_i, rho(x_i)) for each logged record.

        rho(x_i) = rule.action if rule fires on x_i, else logged_action (noop).
        """
        phi = atom_feature_matrix(logs)
        N = phi.shape[0]
        fires = fires_mask(logs, rule)
        actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
        rule_a = _ACTION_IDX[rule.action]
        actions[fires] = rule_a
        X = _joint_features(phi, actions)
        if not cross_fit:
            assert self._full is not None
            return self._full.predict(X).astype(np.float32)
        assert self._fold_assign is not None
        out = np.zeros(N, dtype=np.float32)
        for f, model in enumerate(self._fold_models):
            mask = self._fold_assign == f
            if mask.any():
                out[mask] = model.predict(X[mask]).astype(np.float32)
        return out
