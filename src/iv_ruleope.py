"""15.A  IV-RuleOPE — corrections as proximal instruments.

Treats the binary correction signal C as a *proximal outcome* in the
sense of Miao--Geng--Tchetgen-Tchetgen (2018):

    Z = (X, A) -> U -> R          (target outcome)
                 \-> C             (proximal outcome)

Under (i) C _||_ A | (X, U) (exclusion), (ii) C non-degenerate in U |
X (relevance), and (iii) bridge existence, V(rho) is point-identified
even when no parametric form like A5 is assumed.

We implement a *two-stage* plug-in: stage 1 fits a bridge function
h(x, a) such that E[h(x, a) | x, U] = E[R | x, U]; stage 2 averages
h(x, rho(x)) over the empirical X distribution.  We use a linear
ridge bridge in the joint atom-action features and a ridge first-stage
regression of (R - h) onto (C - g) to identify the bridge in closed
form.  See Cui--Tchetgen-Tchetgen 2020 (Eq. 14) for the linear
identification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge

from src.estimators._regression import (
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
)
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class IVResult:
    estimate: float
    stderr: float


def _features_for_rule(logs: Sequence[LoggedRecord], rule: Rule) -> np.ndarray:
    """Stage-2 features assuming the rule's action."""
    phi = atom_feature_matrix(logs)
    a_idx = _ACTION_IDX[rule.action]
    actions = np.full(phi.shape[0], a_idx, dtype=np.int64)
    return _joint_features(phi, actions)


def iv_value(rule: Rule, logs: Sequence[LoggedRecord], alpha: float = 1.0) -> IVResult:
    phi = atom_feature_matrix(logs)
    actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
    X = _joint_features(phi, actions)
    R = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
    C = np.array([rec.correction for rec in logs], dtype=np.float64)

    # Bridge identification: under linear-bridge restriction, h(x, a) =
    # E[R | x, a] - lambda * (E[C | x, a] - C), with lambda chosen so
    # the residual is orthogonal to (C - g).  Closed-form via two
    # ridge fits + a single scalar.
    fit_R = Ridge(alpha=alpha).fit(X, R)
    fit_C = Ridge(alpha=alpha).fit(X, C)
    R_hat = fit_R.predict(X)
    C_hat = fit_C.predict(X)
    R_resid = R - R_hat
    C_resid = C - C_hat
    denom = float(np.dot(C_resid, C_resid))
    lam = float(np.dot(R_resid, C_resid) / denom) if denom > 1e-12 else 0.0

    X_target = _features_for_rule(logs, rule)
    R_pred_target = fit_R.predict(X_target)
    C_pred_target = fit_C.predict(X_target)
    h = R_pred_target - lam * (C_pred_target - C)

    est = float(h.mean())
    se = float(h.std(ddof=1) / np.sqrt(len(h)))
    return IVResult(estimate=est, stderr=se)
