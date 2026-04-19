"""Inverse-propensity scoring.

For a deterministic target policy induced by a rule:

    V_IPS(rho) = (1/N) sum_i  1[a_i^0 = rho(x_i)] / pi_0(a_i^0 | x_i) * r_i^0

We also provide a self-normalised version (SNIPS).

Under *deterministic* logging (pi_0 is a point-mass on "noop") IPS only gives a
non-trivial estimate on queries where rho does not fire (since only there does
the logged and target action coincide).  The estimator then has infinite
variance whenever rho fires with non-zero probability; we keep the
implementation honest -- it is the user's job to deploy this estimator in
regimes where it makes sense (stochastic logging).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.estimators.base import Estimator, EstimatorResult
from src.logs import LoggedRecord
from src.rule_dsl import Rule


def _match_mask_and_weights(rule: Rule, logs: Sequence[LoggedRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    match = np.zeros(len(logs), dtype=bool)
    weights = np.zeros(len(logs), dtype=np.float64)
    rewards = np.array([r.logged_reward for r in logs], dtype=np.float64)
    for i, rec in enumerate(logs):
        rho_action = rule.action if rule.fires(rec.ctx) else "noop"
        if rec.logged_action == rho_action:
            match[i] = True
            # guard against zero propensity (deterministic logging)
            prop = max(rec.logged_propensity, 1e-6)
            weights[i] = 1.0 / prop
    return match, weights, rewards


class IPS(Estimator):
    name = "IPS"

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        match, w, r = _match_mask_and_weights(rule, logs)
        per_record = np.where(match, w * r, 0.0)
        est = float(per_record.mean())
        se = float(per_record.std(ddof=1) / np.sqrt(len(per_record)))
        # Effective sample size (ESS) = (sum w)^2 / sum w^2 for matched records
        w_m = w[match]
        if w_m.size == 0:
            ess = 0.0
        else:
            ess = float((w_m.sum()) ** 2 / max((w_m ** 2).sum(), 1e-12))
        return EstimatorResult(estimate=est, stderr=se, n_effective=ess)


class SNIPS(Estimator):
    name = "SNIPS"

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        match, w, r = _match_mask_and_weights(rule, logs)
        num = float((w[match] * r[match]).sum())
        den = float(w[match].sum()) if match.any() else 0.0
        est = num / den if den > 0 else 0.0
        # Self-normalised SE: use delta-method on ratio (approximate).
        if den > 0:
            # Delta-method SE for the self-normalised ratio.
            contrib = np.where(match, w * (r - est), 0.0)
            se = float(contrib.std(ddof=1) / np.sqrt(len(logs)) / max(den / len(logs), 1e-12))
        else:
            se = 0.0
        w_m = w[match]
        ess = float((w_m.sum()) ** 2 / max((w_m ** 2).sum(), 1e-12)) if w_m.size else 0.0
        return EstimatorResult(estimate=est, stderr=se, n_effective=ess)
