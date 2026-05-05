"""Common estimator interface.

Every estimator fits on a collection of logged records and evaluates candidate
rules.  The interface returns both a point estimate and a per-rule standard
error (used for coverage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class EstimatorResult:
    estimate: float
    stderr: float
    n_effective: float  # effective sample size, for diagnostics


class Estimator:
    name: str = "base"

    def fit(self, logs: Sequence[LoggedRecord]) -> "Estimator":
        return self

    def value(self, rule: Rule, logs: Sequence[LoggedRecord]) -> EstimatorResult:
        raise NotImplementedError

    def value_many(
        self, rules: Sequence[Rule], logs: Sequence[LoggedRecord]
    ) -> dict[str, EstimatorResult]:
        return {r.id: self.value(r, logs) for r in rules}
