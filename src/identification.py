"""Partial-identification bounds and efficiency-gap diagnostics.

Implements the theory of Section "Identification and efficiency under
deterministic logging" in `theory/proofs.tex`:

* `partial_id_bounds(rule, logs)` returns (V_L, V_U) -- the sharp
  identification interval under A1-A4 and deterministic logging.
* `efficiency_gap(rule, logs, bridge, gate)` returns an empirical estimate
  of the variance gap between the classical DR influence function and the
  RuleOPE efficient influence function, per Theorem F.
* `bridge_linear(beta_target, beta_logged)` gives the closed-form
  bridge function under the correction-linearity model.

The bounds are useful as diagnostics: they tell a practitioner how much
of their rule-evaluation error is identification gap versus estimation
error, and hence whether collecting more data or collecting more
corrections is the right next step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.logs import LoggedRecord
from src.rule_dsl import Rule


def partial_id_bounds(rule: Rule, logs: Sequence[LoggedRecord]) -> tuple[float, float]:
    """Return (V_L, V_U) for the rule under A1-A4 + deterministic logging.

    V_L = E[q(X) R];  V_U = V_L + E[p(X)].
    Uses the empirical distribution as the estimator; no regression is fit.
    """
    N = len(logs)
    fires = np.array([rule.fires(rec.ctx) for rec in logs], dtype=bool)
    R = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
    q = (~fires).astype(np.float64)
    p = fires.astype(np.float64)
    V_L = float((q * R).mean())
    V_U = V_L + float(p.mean())
    return V_L, V_U


def bridge_linear(beta_target: float, beta_logged: float) -> float:
    """Closed-form bridge under the correction-linearity model g(x,a) =
    alpha(x) + beta(a) * (1 - m(x,a)).

    Solving the bridge equation yields the (x-independent) scalar
        b = (beta_target - beta_logged) / beta_logged^2.
    When beta_target == beta_logged the bridge vanishes -- corrections
    carry no cross-action information.
    """
    if abs(beta_logged) < 1e-9:
        raise ValueError("beta_logged must be non-zero for the bridge to exist")
    return (beta_target - beta_logged) / (beta_logged ** 2)


@dataclass
class EfficiencyGapResult:
    gap: float           # Var(psi_DR) - Var(psi_star), population-level estimate
    gap_stderr: float    # bootstrap-style SE of the gap estimate
    fire_frac: float     # E[p(X)]
    g_mean: float        # E[g(X, a_0)]
    bridge_mean: float   # E[b_rho]


def efficiency_gap(
    rule: Rule,
    logs: Sequence[LoggedRecord],
    bridge: np.ndarray | float,
    gate: np.ndarray,
) -> EfficiencyGapResult:
    """Estimate the per-rule efficiency gap of Theorem F:

        Var(psi^DR) - Var(psi^star) = E[p(X)^2 * b_rho(X)^2 * g(X, a_0)(1 - g(X, a_0))].

    Arguments
    ---------
    bridge : array of shape (N,) or scalar -- bridge function b_rho at each record.
    gate   : array of shape (N,) giving g_hat(X_i, a_0) at each record.
    """
    fires = np.array([rule.fires(rec.ctx) for rec in logs], dtype=bool).astype(np.float64)
    N = len(logs)
    if np.isscalar(bridge):
        b = np.full(N, float(bridge))
    else:
        b = np.asarray(bridge, dtype=np.float64)
        assert b.shape[0] == N, "bridge must match the logs length"
    per_record = fires ** 2 * b ** 2 * gate * (1.0 - gate)
    gap = float(per_record.mean())
    se = float(per_record.std(ddof=1) / np.sqrt(N))
    return EfficiencyGapResult(
        gap=gap,
        gap_stderr=se,
        fire_frac=float(fires.mean()),
        g_mean=float(gate.mean()),
        bridge_mean=float(b.mean()),
    )


def diagnostic_report(
    rule: Rule,
    logs: Sequence[LoggedRecord],
    dr_estimate: float,
    rope_estimate: float,
) -> dict:
    """Bundle partial-id bounds + estimator positions into a diagnostic.

    Returns a dict that a practitioner can use to judge whether their
    estimator's gap from the ground truth is due to identification
    (reduce firing frequency, tighten A5) vs. estimation (collect more
    data).
    """
    V_L, V_U = partial_id_bounds(rule, logs)
    width = V_U - V_L
    return {
        "rule": rule.name,
        "V_L": V_L,
        "V_U": V_U,
        "id_gap_width": width,
        "DR_position": float((dr_estimate - V_L) / max(width, 1e-12)),
        "RuleOPE_position": float((rope_estimate - V_L) / max(width, 1e-12)),
        "DR_inside_interval": bool(V_L <= dr_estimate <= V_U),
        "RuleOPE_inside_interval": bool(V_L <= rope_estimate <= V_U),
    }
