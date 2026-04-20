"""15.G  FDR-controlled rule selection.

For a candidate rule pool, treat each rule as a hypothesis
    H_0(rho): V(rho) <= V(noop)
and compute a one-sided p-value from the per-record EIF residuals.
Apply Benjamini--Hochberg to control FDR at level q.

The output is a *set* of discoveries with controlled false-discovery
rate, in contrast to top-k or LCB selection which provide no formal
error guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators._regression import fires_mask
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class FDRResult:
    rule_id: str
    rule_name: str
    estimate: float
    se: float
    p_value: float
    discovered: bool
    bh_threshold: float


def _psi_per_record(
    rule: Rule, logs: Sequence[LoggedRecord], cfg: RuleOPEConfig
) -> np.ndarray:
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


def _baseline_value(logs: Sequence[LoggedRecord]) -> float:
    """V(noop) estimated by the empirical mean reward (rule = no intervention)."""
    return float(np.mean([rec.logged_reward for rec in logs]))


def benjamini_hochberg(p_values: np.ndarray, q: float) -> np.ndarray:
    """Return boolean mask of discoveries at FDR level q."""
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    k = int(np.max(np.where(below)[0])) + 1
    cutoff = sorted_p[k - 1]
    return p_values <= cutoff


def fdr_select(
    rules: Sequence[Rule],
    logs: Sequence[LoggedRecord],
    q: float = 0.05,
    cfg: RuleOPEConfig | None = None,
) -> list[FDRResult]:
    cfg = cfg or RuleOPEConfig()
    baseline = _baseline_value(logs)
    psi_each = [_psi_per_record(r, logs, cfg) for r in rules]
    estimates = np.array([float(p.mean()) for p in psi_each])
    ses = np.array([float(p.std(ddof=1) / np.sqrt(len(p))) for p in psi_each])
    z = (estimates - baseline) / np.maximum(ses, 1e-12)
    p_values = 1.0 - norm.cdf(z)  # one-sided H0: V <= baseline
    discovered = benjamini_hochberg(p_values, q)
    cutoff = p_values[discovered].max() if discovered.any() else 0.0
    return [
        FDRResult(
            rule_id=rule.id,
            rule_name=rule.name,
            estimate=float(estimates[i]),
            se=float(ses[i]),
            p_value=float(p_values[i]),
            discovered=bool(discovered[i]),
            bh_threshold=float(cutoff),
        )
        for i, rule in enumerate(rules)
    ]
