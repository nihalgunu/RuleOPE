"""15.D  Adversarial minimax (DRO) Rule-OPE.

Compute the *worst-case* value of a rule over a Kullback--Leibler ball
around the empirical distribution:

    underline_V(rho) = inf_{Q : KL(Q || P_n) <= eta} E_Q[psi(rho)]

By convex duality the worst-case mean is the dual of an exponential
tilting:

    underline_V(rho) = sup_{lambda > 0} { -lambda * eta - lambda *
                          log( (1/N) sum_i exp(-psi_i / lambda) ) }

We solve the 1-D dual via golden-section search.  This is the
distributionally-robust analogue of the RuleOPE plug-in: it gives a
lower confidence bound that is valid for any distribution within an
eta-KL ball of the empirical distribution.

Connection to the protocol: corresponds to the Namkoong--Duchi /
Zhan-et-al-2024 DRO-OPE family, specialised to the per-rule EIF.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators._regression import fires_mask
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class MinimaxResult:
    estimate: float          # standard plug-in
    lower: float             # DRO lower confidence bound
    eta: float


def _eif(rule: Rule, logs: Sequence[LoggedRecord], cfg: RuleOPEConfig) -> np.ndarray:
    est = RuleOPE(cfg).fit(logs)
    m_rule = est.reg.predict_for_rule(logs, rule).astype(np.float64)
    m_logged = est.reg.predict_logged(logs).astype(np.float64)
    r = np.array([rec.logged_reward for rec in logs], dtype=np.float64)
    fires = fires_mask(logs, rule)
    logged_actions = np.array([rec.logged_action for rec in logs])
    match = np.where(fires, logged_actions == rule.action, logged_actions == "noop")
    propensities = np.array([max(rec.logged_propensity, 1e-6) for rec in logs])
    w = np.where(match, 1.0 / propensities, 0.0)
    return m_rule + w * (r - m_logged)


def _dro_dual(psi: np.ndarray, eta: float) -> float:
    """Maximize over lambda > 0 of -lambda*eta - lambda*log(mean exp(-psi/lambda))."""
    if eta <= 0:
        return float(psi.mean())

    def obj(lam: float) -> float:
        z = -psi / lam
        z_max = z.max()
        log_mean = z_max + np.log(np.mean(np.exp(z - z_max)))
        return -lam * eta - lam * log_mean

    # Golden-section search over log lambda in [-5, 5]
    lo, hi = -5.0, 5.0
    phi = (np.sqrt(5) - 1) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    for _ in range(80):
        if obj(np.exp(c)) > obj(np.exp(d)):
            b = d
        else:
            a = c
        c = b - phi * (b - a)
        d = a + phi * (b - a)
    return float(obj(np.exp((a + b) / 2.0)))


def minimax_value(
    rule: Rule,
    logs: Sequence[LoggedRecord],
    eta: float = 0.05,
    cfg: RuleOPEConfig | None = None,
) -> MinimaxResult:
    cfg = cfg or RuleOPEConfig()
    psi = _eif(rule, logs, cfg)
    point = float(psi.mean())
    lcb = _dro_dual(psi, eta)
    return MinimaxResult(estimate=point, lower=lcb, eta=eta)
