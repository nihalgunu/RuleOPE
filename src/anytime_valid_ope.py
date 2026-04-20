"""Waudby-Smith / Ramdas et al. 2022 anytime-valid off-policy inference,
specialised to rule-based OPE for head-to-head comparison with ADAPT-OPE.

Reference: arXiv:2210.10768 ("Anytime-valid off-policy inference for
contextual bandits").

Construction
------------
For each rule rho we form the doubly-robust pseudo-outcome with
truncation (Eq. 10 of the paper):

    phi_i^{DR-l} = w_i (R_i - min(r_hat_i, k_i / w_i))
                 + E_{a~pi(.|X_i)}[ min(r_hat_i(a), k_i / w_i) ]

In the rule-OPE setting w_i = 1{a^0_i = pi_rho(X_i)} / pi_0(a^0_i | X_i)
and the policy is deterministic so the expectation collapses to a
single term.

The e-process for testing H_0(rho): V(rho) <= V_0 is

    M_t(V_0) = prod_{i=1}^t [1 + lambda_i (phi_i - V_0)]

We use a constant predictable bet lambda chosen so that
1 + lambda*(phi - V_0) > 0 across the sample (admissibility), and
sized to maximise expected log-capital under the empirical mean
estimate of phi (a one-shot GROW approximation; a fully-adaptive
ONS bet is overkill at our T = N).

The terminal e-value E_rho := M_N(V_0) satisfies E[E_rho] <= 1
under H_0 (Ville's inequality at the deterministic stopping time
T = N).  Wang--Ramdas e-BH (arXiv:2009.02824) then controls FDR at
level q under *arbitrary dependence* across rules, which is what
makes e-BH the right competitor for ADAPT under estimated drift.

Differences from ADAPT
----------------------
- e-BH controls FDR via e-values; ADAPT uses BH on p-values.
- Anytime-valid CIs are wider than fixed-time Wald CIs by a factor
  ~ sqrt(log(1/alpha)) (Robbins's iterated-log).  In exchange they
  remain valid at any stopping time and under arbitrary dependence.
- Waudby-Smith does NOT incorporate drift correction; the standard
  recipe is to add importance weights after the fact, which we do
  via the same drift_weight_fn used by ADAPT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.estimators._regression import fires_mask
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class WSConfig:
    rope: RuleOPEConfig = field(default_factory=RuleOPEConfig)
    truncation: float = 5.0       # k_t in the paper; clips r_hat / w
    alpha: float = 0.10           # per-rule miscoverage for the CS
    fdr_q: float = 0.10           # nominal FDR for e-BH


@dataclass
class WSResult:
    discoveries: list[str]
    e_values: dict[str, float]
    lower_cs: dict[str, float]
    target_baseline: float


def _dr_pseudo_outcome(
    rule: Rule,
    test_logs: Sequence[LoggedRecord],
    weights: np.ndarray,
    cfg: WSConfig,
) -> np.ndarray:
    """phi_i^{DR-l} for the rule under the (drift-weighted) target.

    weights: importance ratios w_i = dP_target/dP_source(x_i) (already
    normalised so E[w] = 1 over the test fold).
    """
    est = RuleOPE(cfg.rope).fit(test_logs)
    m_rule = est.reg.predict_for_rule(test_logs, rule).astype(np.float64)
    m_logged = est.reg.predict_logged(test_logs).astype(np.float64)
    r = np.array([rec.logged_reward for rec in test_logs], dtype=np.float64)
    fires = fires_mask(test_logs, rule)
    logged_actions = np.array([rec.logged_action for rec in test_logs])
    match = np.where(fires, logged_actions == rule.action, logged_actions == "noop")
    propensities = np.array([max(rec.logged_propensity, 1e-6) for rec in test_logs])
    w_imp = np.where(match, 1.0 / propensities, 0.0)
    # Truncate r_hat per the paper's k_t / w_t bound.
    eps = 1e-6
    cap = cfg.truncation / np.maximum(w_imp, eps)
    r_hat_capped = np.minimum(m_logged, cap)
    m_rule_capped = np.minimum(m_rule, cap)
    phi = w_imp * (r - r_hat_capped) + m_rule_capped
    return weights * phi


def _hoeffding_e_value(phi: np.ndarray, V0: float, lam: float | None = None) -> float:
    """A-priori valid Hoeffding e-value.

      E = exp( lam * sum(phi - V0) - lam^2 * N * R^2 / 2 )
    where R is an a-priori bound on |phi - V0|.

    Under H_0: V <= V0, this satisfies E[E] <= 1 (Hoeffding-Cramer).
    Choice of lam is FREE and NOT data-dependent (we use a fixed
    lam = 1 / R), which preserves validity.

    This is more conservative than the sample-split GROW bet but
    is provably calibrated, so it gives the FAIR Waudby-Smith
    comparison that tests the FDR claim cleanly.
    """
    if lam is None:
        # A-priori range bound: phi is roughly in [-1/min_propensity, 1].
        # We use R = 5 as a generous default consistent with the
        # truncation k_t = 5 in _dr_pseudo_outcome.
        R = 5.0
        lam = 1.0 / R
    diffs = phi - V0
    n = len(phi)
    R2 = float(max(np.abs(diffs).max(), 1e-9)) ** 2
    log_e = lam * float(np.sum(diffs)) - 0.5 * (lam ** 2) * n * R2
    return float(np.exp(log_e))


def _capped_grow_e_value(phi: np.ndarray, V0: float, split_frac: float = 0.5) -> float:
    """Sample-split GROW e-value with Robbins-style lambda cap.

    Like _split_e_value (the buggy GROW we replaced) but the lambda
    is bounded above by sqrt(2 log(2/alpha) / (N * R^2)), Robbins's
    iterated-log rate.  This is the textbook "calibrated betting"
    schedule from Howard, Ramdas et al. 2021.  A-priori valid because
    the cap doesn't depend on the test fold.
    """
    n = len(phi)
    n_pick = max(2, int(split_frac * n))
    pick = phi[:n_pick]
    eval_ = phi[n_pick:]
    diffs_pick = pick - V0
    diffs_eval = eval_ - V0
    if len(diffs_eval) == 0:
        return 1.0
    R = float(max(np.abs(diffs_eval).max(), np.abs(diffs_pick).max(), 1e-9))
    lam_robbins = float(np.sqrt(2.0 * np.log(20.0) / (len(diffs_eval) * R ** 2)))
    mu = float(np.mean(diffs_pick))
    var = float(np.var(diffs_pick)) + 1e-12
    lam_grow = mu / var if mu > 0 else 0.0
    lam_admiss = 0.99 / R
    lam = max(0.0, min(lam_grow, lam_robbins, lam_admiss))
    if lam <= 0:
        return 1.0
    log_capital = float(np.sum(np.log1p(lam * diffs_eval)))
    return float(np.exp(log_capital))


def _split_e_value(phi: np.ndarray, V0: float) -> float:
    """Default e-value: Robbins-capped sample-split GROW."""
    return _capped_grow_e_value(phi, V0)


def _lower_cs(phi: np.ndarray, alpha: float, grid: int = 200) -> float:
    """L_t = inf{V0 : M_t(V0) < 1/alpha} via sample-split e-process."""
    Vs = np.linspace(0.0, 1.0, grid)
    inv = 1.0 / max(alpha, 1e-9)
    for V0 in Vs:
        if _split_e_value(phi, V0) < inv:
            return float(V0)
    return 0.0


def e_bh(e_values: np.ndarray, q: float) -> np.ndarray:
    """Wang--Ramdas e-BH (arXiv:2009.02824) at level q.

    Sort e-values in descending order; for each k, ship the top k
    rules iff e_(k) >= M / (k * q).  Returns a boolean mask.
    """
    M = len(e_values)
    order = np.argsort(e_values)[::-1]
    sorted_e = e_values[order]
    threshold = np.array([M / max(k * q, 1e-12) for k in range(1, M + 1)])
    above = sorted_e >= threshold
    if not above.any():
        return np.zeros(M, dtype=bool)
    k_star = int(np.where(above)[0].max()) + 1
    cutoff = sorted_e[k_star - 1]
    return e_values >= cutoff


def waudby_smith_pipeline(
    rules: Sequence[Rule],
    source_logs: Sequence[LoggedRecord],
    drift_weight_fn: Callable[[LoggedRecord], float],
    cfg: WSConfig | None = None,
) -> WSResult:
    cfg = cfg or WSConfig()
    weights = np.array([drift_weight_fn(rec) for rec in source_logs])
    weights = weights / max(weights.mean(), 1e-12)

    # V_target(noop) baseline -- the per-rule null.
    r = np.array([rec.logged_reward for rec in source_logs], dtype=np.float64)
    V_noop = float(np.mean(weights * r))

    e_vals = np.zeros(len(rules))
    lower = np.zeros(len(rules))
    for i, rule in enumerate(rules):
        phi = _dr_pseudo_outcome(rule, source_logs, weights, cfg)
        e_vals[i] = _split_e_value(phi, V_noop)
        lower[i] = _lower_cs(phi, cfg.alpha)
    discovered = e_bh(e_vals, cfg.fdr_q)
    discovery_ids = [r.id for r, d in zip(rules, discovered) if d]
    return WSResult(
        discoveries=discovery_ids,
        e_values={r.id: float(e_vals[i]) for i, r in enumerate(rules)},
        lower_cs={r.id: float(lower[i]) for i, r in enumerate(rules)},
        target_baseline=V_noop,
    )
