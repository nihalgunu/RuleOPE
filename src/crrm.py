"""Counterfactual Rule Risk Minimization (CRRM) and pessimistic rule selection.

Two new capabilities on top of the rule-OPE estimators:

1. `PessimisticRuleSelector` selects the rule that maximises a Lower
   Confidence Bound (LCB) on its value.  We propose an atom-aware LCB
   that is tighter than the standard union-bound LCB when the rule
   value function is approximately sparse in the atom-compositional
   basis.  Specifically, the LCB exponent depends on the *effective
   atom sparsity* s rather than the (logarithm of the) number of rules
   M.

2. `CRRM` performs rule learning: it searches the conjunctive rule space
   for the pessimistic-optimal rule, i.e. the rule that maximises
   $\hat V_\text{LCB}(rho) - lambda |rho|$ subject to a minimum support
   constraint.  We prove (Theorem 5 in theory/proofs.tex) a regret bound
   that scales with the compositional Rademacher complexity of the rule
   class rather than with |R|.

Both objects are defined in terms of a base per-rule estimator supplying
point estimate and standard error (typically `JointRuleOPE` or
`RuleOPE`).

Relation to pessimistic offline RL
----------------------------------
This mirrors the pessimism principle in offline RL (Jin et al. 2021;
Xie et al. 2021) but applied to the rule-OPE setting with a
compositional complexity control.  Classical pessimistic OPE uses
regret bounds that scale with |R| (number of candidate policies); our
bound scales with the atom sparsity s.  When many rules share a small
set of informative atoms, the compositional bound is strictly tighter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Lasso

from src.estimators.base import Estimator, EstimatorResult
from src.estimators.shrinkage import _rule_features
from src.logs import LoggedRecord
from src.rule_dsl import ATOMS, Rule


@dataclass
class PessimisticConfig:
    delta: float = 0.05           # coverage error
    # If `atom_sparse=True`, substitute the rule-size `log M` term by a
    # LASSO-estimated effective-sparsity `s_hat * log d` term (see paper
    # §5.2 discussion).
    atom_sparse: bool = True
    # per-atom L1 penalty used when estimating the atom-level value model
    lasso_alpha: float = 0.01
    # Penalty on rule depth (regularises against very long conjunctions).
    depth_penalty: float = 0.0
    # Minimum support required for a rule to be eligible.
    min_fires: int = 10


class PessimisticRuleSelector:
    """Select rules by maximising a compositionally-adaptive LCB."""

    def __init__(self, config: PessimisticConfig | None = None) -> None:
        self.cfg = config or PessimisticConfig()

    # ------------------------------------------------------------------
    def _effective_complexity(self, F: np.ndarray, V: np.ndarray) -> float:
        """Empirical-Bayes estimate of the effective atom sparsity s.

        Fits a LASSO regression of V on rule features and returns the
        number of non-zero coefficients.  The resulting s is used in the
        compositional LCB as log(d+1) * max(s, 1) in place of log(M).
        """
        if not self.cfg.atom_sparse:
            return float("nan")
        lasso = Lasso(alpha=self.cfg.lasso_alpha, max_iter=2000).fit(F, V)
        return float((np.abs(lasso.coef_) > 1e-8).sum())

    # ------------------------------------------------------------------
    def lcb(
        self,
        rules: Sequence[Rule],
        estimates: Sequence[float],
        stderrs: Sequence[float],
        actions: Sequence[str] = ("noop", "filter", "rerank", "abstain"),
    ) -> np.ndarray:
        """Per-rule lower confidence bound."""
        M = len(rules)
        if M == 0:
            return np.zeros(0)
        V = np.asarray(estimates, dtype=np.float64)
        S = np.asarray(stderrs, dtype=np.float64)
        F = np.stack([_rule_features(r, actions) for r in rules], axis=0)
        d = len(ATOMS) + len(actions) + 1  # + depth

        if self.cfg.atom_sparse:
            s_hat = self._effective_complexity(F, V)
            # Compositional exponent: log((d+1)^{s_hat+1}/delta)
            #   = (s_hat+1) log(d+1) + log(1/delta)
            c = np.sqrt(2.0 * ((s_hat + 1.0) * np.log(d + 1) + np.log(1.0 / self.cfg.delta)))
        else:
            # Standard union-bound exponent: log(M/delta)
            c = np.sqrt(2.0 * np.log(max(M, 1) / self.cfg.delta))

        depth = np.array([r.depth() for r in rules], dtype=np.float64)
        penalty = self.cfg.depth_penalty * depth
        return V - c * S - penalty

    # ------------------------------------------------------------------
    def select(
        self,
        rules: Sequence[Rule],
        results: dict[str, EstimatorResult],
        actions: Sequence[str] = ("noop", "filter", "rerank", "abstain"),
    ) -> tuple[Rule, float]:
        """Return (best rule, its LCB)."""
        est = [results[r.id].estimate for r in rules]
        ses = [results[r.id].stderr for r in rules]
        lcb = self.lcb(rules, est, ses, actions=actions)
        idx = int(np.argmax(lcb))
        return rules[idx], float(lcb[idx])

    # ------------------------------------------------------------------
    def top_k(
        self,
        rules: Sequence[Rule],
        results: dict[str, EstimatorResult],
        k: int = 10,
        actions: Sequence[str] = ("noop", "filter", "rerank", "abstain"),
    ) -> list[tuple[Rule, float]]:
        est = [results[r.id].estimate for r in rules]
        ses = [results[r.id].stderr for r in rules]
        lcb = self.lcb(rules, est, ses, actions=actions)
        order = np.argsort(-lcb)[:k]
        return [(rules[i], float(lcb[i])) for i in order]


# ----------------------------------------------------------------------
# CRRM: Counterfactual Rule Risk Minimisation
# ----------------------------------------------------------------------

@dataclass
class CRRMConfig:
    lcb_delta: float = 0.05
    atom_sparse: bool = True
    lasso_alpha: float = 0.01
    depth_penalty: float = 0.01
    min_fires_frac: float = 0.005


class CRRM:
    """Learn a rule from logged data by pessimistic value maximisation.

    The learner searches the conjunctive rule space (up to depth D) and
    returns the rule maximising V_hat_LCB(rho) - lambda * |rho|.  Support
    is enforced so selected rules fire on at least `min_fires_frac` of
    the logs.
    """

    def __init__(
        self,
        base_estimator: Estimator,
        pessimistic: PessimisticRuleSelector,
        config: CRRMConfig | None = None,
    ) -> None:
        self.base = base_estimator
        self.pessimistic = pessimistic
        self.cfg = config or CRRMConfig()

    # ------------------------------------------------------------------
    def learn(
        self,
        rules: Sequence[Rule],
        logs: Sequence[LoggedRecord],
    ) -> tuple[Rule, float, dict[str, EstimatorResult]]:
        """Return (selected rule, its LCB, raw per-rule estimates)."""
        if hasattr(self.base, "fit"):
            self.base.fit(logs)

        min_fires = int(self.cfg.min_fires_frac * len(logs))
        eligible = []
        fires = {}
        for r in rules:
            f = sum(1 for rec in logs if r.fires(rec.ctx))
            fires[r.id] = f
            if f >= min_fires:
                eligible.append(r)
        if not eligible:
            raise ValueError("no eligible rules: increase data or reduce min_fires_frac")

        results = self.base.value_many(eligible, logs)
        best, lcb = self.pessimistic.select(eligible, results)
        return best, lcb, results
