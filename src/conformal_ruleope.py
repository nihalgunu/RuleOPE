"""15.C  Conformal Rule-OPE — distribution-free per-rule CIs.

Split-conformal calibration on out-of-fold RuleOPE influence-function
contributions.  Given a rule rho and logs, we compute the per-record
EIF psi_i(rho) via cross-fit RuleOPE, treat them as exchangeable
samples of (V(rho) + noise_i), and use the empirical (1-delta)
quantile of |psi_i - mean(psi_i)| on a held-out calibration split as
the conformal half-width.

This is the OPE analogue of split-conformal mean estimation: under
exchangeability of the EIF residuals, the resulting interval has
finite-sample coverage at least 1 - delta - 1/(n_cal+1).

Comparison: standard Wald uses sigma_hat * z_{1-delta/2}, which is
asymptotically valid under CLT but may under-cover in small N or under
heavy-tailed EIF residuals (misspecification).  Conformal trades
asymptotic sharpness for finite-sample validity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators._regression import fires_mask, _ACTION_IDX
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class ConformalResult:
    estimate: float
    lower: float
    upper: float
    halfwidth: float
    method: str  # "conformal" or "wald"


def _rule_ope_psi(rule: Rule, logs: Sequence[LoggedRecord], cfg: RuleOPEConfig) -> np.ndarray:
    """Per-record influence-function contribution of RuleOPE."""
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


def conformal_interval(
    rule: Rule,
    calib_logs: Sequence[LoggedRecord],
    eval_logs: Sequence[LoggedRecord],
    delta: float = 0.05,
    cfg: RuleOPEConfig | None = None,
) -> ConformalResult:
    cfg = cfg or RuleOPEConfig()
    psi_eval = _rule_ope_psi(rule, eval_logs, cfg)
    point = float(psi_eval.mean())
    psi_cal = _rule_ope_psi(rule, calib_logs, cfg)
    cal_residuals = np.abs(psi_cal - psi_cal.mean())
    n_cal = len(cal_residuals)
    q_idx = int(np.ceil((1.0 - delta) * (n_cal + 1))) - 1
    q_idx = max(0, min(q_idx, n_cal - 1))
    halfwidth_per_record = float(np.sort(cal_residuals)[q_idx])
    halfwidth = halfwidth_per_record / np.sqrt(len(eval_logs))
    return ConformalResult(
        estimate=point,
        lower=point - halfwidth,
        upper=point + halfwidth,
        halfwidth=halfwidth,
        method="conformal",
    )


def wald_interval(
    rule: Rule,
    eval_logs: Sequence[LoggedRecord],
    delta: float = 0.05,
    cfg: RuleOPEConfig | None = None,
) -> ConformalResult:
    from scipy.stats import norm

    cfg = cfg or RuleOPEConfig()
    psi = _rule_ope_psi(rule, eval_logs, cfg)
    point = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    z = float(norm.ppf(1.0 - delta / 2.0))
    halfwidth = z * se
    return ConformalResult(
        estimate=point,
        lower=point - halfwidth,
        upper=point + halfwidth,
        halfwidth=halfwidth,
        method="wald",
    )
