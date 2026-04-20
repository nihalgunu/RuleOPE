"""15.K  Temporal-drift Rule-OPE.

Source distribution P_0 (observed via logs at t_0) drifts to target
P_1 (deployment at t_1).  We have a parametric drift model (or a
density-ratio estimator) w(x) = dP_1/dP_0(x).

Weighted RuleOPE: replace the empirical mean by a w-weighted mean,
self-normalised to control variance under heavy-tailed weights.

This is the OPE analogue of importance-weighted target estimation
under covariate shift (Sugiyama et al. 2008), specialised to
rule-evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators._regression import fires_mask
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class TemporalResult:
    estimate_naive: float
    estimate_weighted: float
    weight_ess: float


def _psi(rule: Rule, logs: Sequence[LoggedRecord], cfg: RuleOPEConfig) -> np.ndarray:
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


def temporal_value(
    rule: Rule,
    logs: Sequence[LoggedRecord],
    drift_weight: Callable[[LoggedRecord], float],
    cfg: RuleOPEConfig | None = None,
) -> TemporalResult:
    cfg = cfg or RuleOPEConfig()
    psi = _psi(rule, logs, cfg)
    naive = float(psi.mean())
    w = np.array([drift_weight(rec) for rec in logs], dtype=np.float64)
    w = np.maximum(w, 0.0)
    if w.sum() < 1e-12:
        return TemporalResult(estimate_naive=naive, estimate_weighted=naive, weight_ess=0.0)
    w_norm = w / w.mean()  # E_P0[w] = 1
    weighted = float(np.sum(w_norm * psi) / len(psi))
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return TemporalResult(estimate_naive=naive, estimate_weighted=weighted, weight_ess=ess)
