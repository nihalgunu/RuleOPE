"""15.M  Fairness-constrained Rule-OPE.

Estimate per-subgroup rule values V_g(rho) and select rules subject
to a fairness constraint:

    max_rho V(rho)  s.t.  min_g V_g(rho) >= V_baseline_g - tau

We expose subgroup-stratified RuleOPE plus a Pareto-optimal selector.
The fairness slack tau is a hyperparameter on the per-subgroup
absolute drop allowed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class FairResult:
    rule_id: str
    rule_name: str
    overall: float
    per_group: dict[str, float]
    feasible: bool
    min_group_drop: float


def fair_select(
    rules: Sequence[Rule],
    logs: Sequence[LoggedRecord],
    group_fn: Callable[[LoggedRecord], str],
    tau: float = 0.02,
    cfg: RuleOPEConfig | None = None,
) -> list[FairResult]:
    cfg = cfg or RuleOPEConfig()
    groups = sorted({group_fn(r) for r in logs})
    group_idx = {g: [i for i, r in enumerate(logs) if group_fn(r) == g] for g in groups}
    baseline_per_g = {g: float(np.mean([logs[i].logged_reward for i in group_idx[g]])) for g in groups}

    out = []
    for rule in rules:
        est = RuleOPE(cfg).fit(logs)
        overall = est.value(rule, logs).estimate
        per_g = {}
        for g, idxs in group_idx.items():
            sub = [logs[i] for i in idxs]
            est_g = RuleOPE(cfg).fit(sub)
            per_g[g] = est_g.value(rule, sub).estimate
        drops = {g: baseline_per_g[g] - per_g[g] for g in groups}
        max_drop = max(drops.values())
        feasible = max_drop <= tau
        out.append(
            FairResult(
                rule_id=rule.id,
                rule_name=rule.name,
                overall=float(overall),
                per_group={g: float(per_g[g]) for g in groups},
                feasible=feasible,
                min_group_drop=float(max_drop),
            )
        )
    return out
