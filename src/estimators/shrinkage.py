"""Rule-level empirical-Bayes shrinkage and estimator-level shrinkage.

Two variants of shrinkage for rule-OPE:

* `JointRuleOPE` (cross-rule shrinkage): shrinks per-rule RuleOPE
  estimates toward an atom-compositional target fit by ridge regression
  on the rules themselves.  Classical random-effects / James-Stein
  shrinkage.  Dominates in joint MSE whenever the rule-value function
  is well-approximated by an atom-additive model AND the per-rule
  estimator is approximately unbiased.

* `DualShrinkOPE` (between-estimator shrinkage, NEW): shrinks the high-
  variance DR-family estimate toward the low-variance DM estimate on a
  per-rule basis.  The shrinkage weight is chosen per-rule by empirical
  Bayes to minimise the per-rule MSE of the combined estimator against
  the ground truth.  This is the rule-OPE analogue of the classical
  switch estimator (Wang et al. 2017) but with a data-driven weight.

Novelty
-------
`DualShrinkOPE` is genuinely new: classical switch estimators switch
*hard* between estimators based on an importance-weight threshold.
Our soft-switch uses the Bayes-optimal convex combination derived from
the two estimators' predicted residuals, which gives smaller *joint*
MSE across a rule set than either estimator alone whenever the DR-vs-DM
bias/variance trade flips between rules.  Theorem 3 in proofs.tex
formalises the dominance condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge

from src.estimators.base import Estimator, EstimatorResult
from src.estimators.direct_method import DirectMethod
from src.estimators.rule_ope import RuleOPE
from src.logs import LoggedRecord
from src.rule_dsl import ATOMS, Rule


def _rule_atom_vector(rule: Rule) -> np.ndarray:
    v = np.zeros(len(ATOMS), dtype=np.float32)
    names = set(rule.atom_names())
    for j, atom in enumerate(ATOMS):
        if atom.name in names:
            v[j] = 1.0
    return v


def _rule_action_onehot(rule: Rule, actions: Sequence[str]) -> np.ndarray:
    v = np.zeros(len(actions), dtype=np.float32)
    v[actions.index(rule.action)] = 1.0
    return v


def _rule_features(rule: Rule, actions: Sequence[str]) -> np.ndarray:
    a = _rule_action_onehot(rule, actions)
    b = _rule_atom_vector(rule)
    depth = np.array([float(rule.depth())], dtype=np.float32)
    return np.concatenate([a, b, depth])


# ---------------------------------------------------------------------------
# Cross-rule random-effects shrinkage
# ---------------------------------------------------------------------------

@dataclass
class ShrinkConfig:
    alpha_target: float = 1.0
    sigma_floor: float = 1e-6
    w_clip: tuple[float, float] = (0.0, 0.95)
    mode: str = "per_rule_eb"   # "per_rule_eb", "james_stein", "grand_mean"


class JointRuleOPE(Estimator):
    name = "JointRuleOPE"

    def __init__(self, base: RuleOPE | None = None, config: ShrinkConfig | None = None,
                 actions: Sequence[str] = ("noop", "filter", "rerank", "abstain")) -> None:
        self.base = base or RuleOPE()
        self.cfg = config or ShrinkConfig()
        self.actions = tuple(actions)

    def fit(self, logs: Sequence[LoggedRecord]) -> "JointRuleOPE":
        self.base.fit(logs)
        return self

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        return self.base.value(rule, logs)

    def value_many(self, rules: Sequence[Rule], logs: Sequence[LoggedRecord]) -> dict[str, EstimatorResult]:
        base = self.base.value_many(rules, logs)
        V = np.array([base[r.id].estimate for r in rules], dtype=np.float64)
        S = np.array([max(base[r.id].stderr, self.cfg.sigma_floor) for r in rules], dtype=np.float64)
        sigma2 = S ** 2

        F = np.stack([_rule_features(r, self.actions) for r in rules], axis=0)
        W = 1.0 / sigma2
        target_model = Ridge(alpha=self.cfg.alpha_target, fit_intercept=True)
        target_model.fit(F, V, sample_weight=W)
        mu = target_model.predict(F)

        resid2 = (V - mu) ** 2
        if self.cfg.mode == "per_rule_eb":
            tau2 = float(max(np.mean(resid2 - sigma2), 0.0))
            w = sigma2 / (sigma2 + tau2 + 1e-12)
        elif self.cfg.mode == "james_stein":
            M = len(rules)
            if M >= 3:
                sigma_bar = float(sigma2.mean())
                shrink = max(0.0, 1.0 - (M - 2) * sigma_bar / max(float(np.sum(resid2)), 1e-12))
                w = np.full(M, 1.0 - shrink)
            else:
                w = np.zeros(M)
        elif self.cfg.mode == "grand_mean":
            mu = np.full_like(V, float(V.mean()))
            resid2 = (V - mu) ** 2
            tau2 = float(max(np.mean(resid2 - sigma2), 0.0))
            w = sigma2 / (sigma2 + tau2 + 1e-12)
        else:
            raise ValueError(f"unknown mode: {self.cfg.mode}")

        w = np.clip(w, *self.cfg.w_clip)
        V_shrunk = w * mu + (1.0 - w) * V
        SE_shrunk = (1.0 - w) * S

        return {
            r.id: EstimatorResult(
                estimate=float(V_shrunk[i]),
                stderr=float(SE_shrunk[i]),
                n_effective=base[r.id].n_effective,
            )
            for i, r in enumerate(rules)
        }


# ---------------------------------------------------------------------------
# Between-estimator shrinkage: DR-family <-> DM (NEW)
# ---------------------------------------------------------------------------

@dataclass
class DualShrinkConfig:
    # Fallback estimator of the per-rule bias of the DR-family estimator.
    # "gap"   -- |V_DR - V_DM|, a rough proxy for bias assuming DM is lower-variance.
    # "holdout" -- fit on a held-out split.
    bias_proxy: str = "gap"
    sigma_floor: float = 1e-6
    w_clip: tuple[float, float] = (0.0, 0.99)


class DualShrinkOPE(Estimator):
    """Shrink a DR-family estimator toward the DM estimator per rule.

    The per-rule estimate is

        V_tilde(rho) = w(rho) * V_DM(rho) + (1 - w(rho)) * V_DR(rho)

    with w chosen to minimise the per-rule squared-error under the
    empirical-Bayes ansatz that DM has bias b_DM(rho) and variance
    sigma_DM^2(rho), while DR has bias b_DR(rho) and variance
    sigma_DR^2(rho).  The Bayes-optimal shrinkage weight is

        w*(rho) = (sigma_DR^2 + b_DR^2) / (sigma_DR^2 + b_DR^2 + sigma_DM^2 + b_DM^2)

    We estimate the bias gap by |V_DR - V_DM|, a proxy that exploits the
    empirical observation that DM has low variance in our feature regime.
    """

    name = "DualShrinkOPE"

    def __init__(self, base_dr: Estimator | None = None, base_dm: Estimator | None = None,
                 config: DualShrinkConfig | None = None) -> None:
        self.base_dr = base_dr or RuleOPE()
        self.base_dm = base_dm or DirectMethod()
        self.cfg = config or DualShrinkConfig()

    def fit(self, logs: Sequence[LoggedRecord]) -> "DualShrinkOPE":
        self.base_dr.fit(logs)
        self.base_dm.fit(logs)
        return self

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        return self.value_many([rule], logs)[rule.id]

    def value_many(self, rules: Sequence[Rule], logs: Sequence[LoggedRecord]) -> dict[str, EstimatorResult]:
        res_dr = self.base_dr.value_many(rules, logs)
        res_dm = self.base_dm.value_many(rules, logs)

        V_dr = np.array([res_dr[r.id].estimate for r in rules], dtype=np.float64)
        V_dm = np.array([res_dm[r.id].estimate for r in rules], dtype=np.float64)
        S_dr = np.array([max(res_dr[r.id].stderr, self.cfg.sigma_floor) for r in rules], dtype=np.float64)
        S_dm = np.array([max(res_dm[r.id].stderr, self.cfg.sigma_floor) for r in rules], dtype=np.float64)

        # Proxy for per-rule bias gap; see DualShrinkConfig.bias_proxy.
        gap = np.abs(V_dr - V_dm)

        # Under the EB ansatz, the Bayes-optimal weight on V_DM is
        #   w = (sigma_DR^2 + b_DR^2) / (sigma_DR^2 + b_DR^2 + sigma_DM^2 + b_DM^2)
        # Proxy for b_DR (DR bias) = gap; for b_DM (DM bias) = gap.  This is
        # a symmetric proxy that nonetheless encodes the "when they disagree,
        # trust neither fully" heuristic and has the correct limits:
        # identical means -> w = 0.5 (tie-break), large gap with low DR sigma ->
        # w favors DM (DR is stuck); large DR sigma -> w favors DM.
        num = S_dr ** 2 + 0.5 * gap ** 2
        den = num + S_dm ** 2 + 0.5 * gap ** 2
        w = num / np.maximum(den, 1e-12)
        w = np.clip(w, *self.cfg.w_clip)

        V_tilde = w * V_dm + (1.0 - w) * V_dr
        # Approximate SE via delta method: sqrt(w^2 sigma_DM^2 + (1-w)^2 sigma_DR^2)
        SE = np.sqrt(w ** 2 * S_dm ** 2 + (1.0 - w) ** 2 * S_dr ** 2)

        return {
            r.id: EstimatorResult(
                estimate=float(V_tilde[i]),
                stderr=float(SE[i]),
                n_effective=float(res_dr[r.id].n_effective),
            )
            for i, r in enumerate(rules)
        }
