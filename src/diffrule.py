"""15.I  Differentiable rule discovery.

Replace the boolean atom-indicator with a smoothed soft-indicator and
gradient-descend through the RuleOPE LCB to find the rule that
maximises V_LCB(rho).

We implement a *soft conjunction*: with relaxed atom weights
w in (0, 1)^d (one per atom in the vocabulary) and temperature tau,

    pi_w(x) = action     if  prod_alpha (1 - w_alpha + w_alpha * sigmoid((phi_alpha(x) - 0.5) / tau))  >= 0.5
            = noop       otherwise

The objective is V_LCB(pi_w) = V_DR(pi_w) - c * sigma_DR(pi_w).
We optimise via projected gradient ascent (numpy autograd-free) and
report best discrete rule found by thresholding w at 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.estimators._regression import (
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
    fires_mask,
)
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord
from src.rule_dsl import ATOMS, Rule, Atom


@dataclass
class DiffResult:
    best_rule: Rule
    best_value: float
    best_lcb: float
    history: list[float]


def _soft_value(
    w: np.ndarray, action: str, logs: Sequence[LoggedRecord], cfg: RuleOPEConfig
) -> tuple[float, float, np.ndarray]:
    """Soft-rule value and gradient w.r.t. w.

    Soft firing: f_i(w) = prod_alpha (1 - w_alpha + w_alpha * phi_alpha(x_i)).
    For phi in {0, 1}, this is exactly the indicator when w in {0, 1}.
    Gradient: d f_i / d w_alpha = prod_{beta != alpha} (...) * (phi_alpha - 1).
    """
    phi = atom_feature_matrix(logs)  # (N, d)
    one_minus = 1.0 - w + w * phi  # (N, d)
    f = np.prod(one_minus, axis=1)  # (N,)

    est = RuleOPE(cfg).fit(logs)
    m_logged = est.reg.predict_logged(logs).astype(np.float64)
    m_action = est.reg.predict_for_action(logs, action).astype(np.float64)
    r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
    propensities = np.array([max(rec.logged_propensity, 1e-6) for rec in logs])
    logged_actions = np.array([rec.logged_action for rec in logs])

    # DR contribution: f * m_action + (1 - f) * m_logged + matched-action correction.
    matched = (logged_actions == action).astype(np.float64) * f + \
              (logged_actions == "noop").astype(np.float64) * (1.0 - f)
    w_imp = matched / propensities
    psi = f * m_action + (1.0 - f) * m_logged + w_imp * (r - m_logged)
    point = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    grad = np.zeros_like(w)
    eps = 1e-3
    for j in range(len(w)):
        w_p = w.copy()
        w_p[j] = min(1.0, w[j] + eps)
        one_p = 1.0 - w_p + w_p * phi
        f_p = np.prod(one_p, axis=1)
        psi_p = f_p * m_action + (1.0 - f_p) * m_logged
        grad[j] = (float(psi_p.mean()) - point) / eps
    return point, se, grad


def diff_discover(
    logs: Sequence[LoggedRecord],
    action: str = "rerank",
    n_steps: int = 80,
    lr: float = 0.05,
    lcb_const: float = 0.5,
    seed: int = 0,
    cfg: RuleOPEConfig | None = None,
) -> DiffResult:
    cfg = cfg or RuleOPEConfig()
    rng = np.random.default_rng(seed)
    d = len(ATOMS)
    w = rng.uniform(0.05, 0.3, size=d)
    history = []
    best_lcb = -np.inf
    best_w = w.copy()
    best_point = 0.0
    for _ in range(n_steps):
        point, se, grad = _soft_value(w, action, logs, cfg)
        lcb = point - lcb_const * se
        history.append(lcb)
        if lcb > best_lcb:
            best_lcb = lcb
            best_w = w.copy()
            best_point = point
        w = np.clip(w + lr * grad, 0.0, 1.0)
    selected = [ATOMS[i] for i in range(d) if best_w[i] > 0.5]
    if not selected:
        selected = [ATOMS[int(np.argmax(best_w))]]
    if len(selected) > 3:
        order = np.argsort(best_w)[::-1][:3]
        selected = [ATOMS[i] for i in order]
    rule = Rule(atoms=tuple(selected), action=action)
    return DiffResult(best_rule=rule, best_value=best_point, best_lcb=best_lcb, history=history)
