"""15.J  Causal-mechanism (SCM) Rule-OPE.

Models the RAG pipeline as an SCM:

    X (query, context)
       \-> A (action: noop / filter / rerank / abstain)  <- pi_0(X)
            \-> R (reward)
       \-> U (unobserved confounder: query difficulty)
            \-> R, C
            \-> A (via pi_0)

A rule rho is a hard intervention do(A := pi_rho(X)).  Under the
"adjustment-set" backdoor criterion, V(rho) = E_X[E[R | X, do(A)]].

Compared to the standard DR identification this SCM view requires
the extra assumption that the joint atom feature phi(X) is a valid
backdoor adjustment (i.e., U _||_ A | phi(X)).

We implement the backdoor estimator and report its sensitivity to
unobserved confounding U (Rosenbaum-style sensitivity gamma).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.estimators._regression import (
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
)
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class SCMResult:
    backdoor_estimate: float
    sensitivity_lower: float
    sensitivity_upper: float
    gamma: float


def backdoor_value(
    rule: Rule, logs: Sequence[LoggedRecord], cfg: RuleOPEConfig | None = None
) -> float:
    cfg = cfg or RuleOPEConfig()
    est = RuleOPE(cfg).fit(logs)
    m = est.reg.predict_for_rule(logs, rule).astype(np.float64)
    return float(m.mean())


def sensitivity_value(
    rule: Rule,
    logs: Sequence[LoggedRecord],
    gamma: float = 2.0,
    cfg: RuleOPEConfig | None = None,
) -> SCMResult:
    """Rosenbaum-style sensitivity bounds.

    Under unobserved confounding bounded by odds ratio gamma >= 1,
    the worst-case backdoor estimate satisfies

        E[R | X, do(A=a)]_lo = m_hat(X, a) / (1 + (gamma - 1) * (1 - m_hat))
        E[R | X, do(A=a)]_hi = gamma * m_hat / (1 + (gamma - 1) * m_hat)

    (Rosenbaum 2002, Cornfield-style bounds for binary outcomes.)
    """
    cfg = cfg or RuleOPEConfig()
    est = RuleOPE(cfg).fit(logs)
    m = est.reg.predict_for_rule(logs, rule).astype(np.float64)
    m = np.clip(m, 1e-3, 1.0 - 1e-3)
    lo = m / (1.0 + (gamma - 1.0) * (1.0 - m))
    hi = gamma * m / (1.0 + (gamma - 1.0) * m)
    return SCMResult(
        backdoor_estimate=float(m.mean()),
        sensitivity_lower=float(lo.mean()),
        sensitivity_upper=float(hi.mean()),
        gamma=gamma,
    )
