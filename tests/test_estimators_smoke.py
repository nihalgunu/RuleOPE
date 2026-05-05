"""Smoke test for the released estimator panel.

Verifies that DoublyRobust, MRDR, SwitchDR, RuleOPE, and the per-rule
NonCompositionalDR baseline all fit + value() on a tiny synthetic log
without errors and produce finite estimates. Not a numerical correctness
test — only a regression guard for the estimator interface.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.estimators.doubly_robust import DoublyRobust
from src.estimators.mrdr import MRDR
from src.estimators.rule_ope import RuleOPE
from src.estimators.switch_dr import SwitchDR
from src.logs import LoggedRecord
from src.rule_dsl import ATOMS, Rule
from experiments.ablations import NonCompositionalDR


def _build_logs(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    actions = ("noop", "filter", "rerank")
    feature_names = sorted({a.feature for a in ATOMS})
    logs = []
    for i in range(n):
        ctx = {f: float(rng.uniform(0.0, 1.0)) for f in feature_names}
        a = actions[int(rng.integers(0, 3))]
        cf = {act: float(rng.beta(1, 4)) for act in ("noop", "filter", "rerank", "abstain")}
        logs.append(LoggedRecord(
            query_id=f"q{i}", ctx=ctx,
            logged_action=a, logged_propensity=1 / 3,
            logged_reward=float(cf[a]),
            correction=int(rng.integers(0, 2)),
            cf_rewards=cf,
        ))
    return logs


def _build_rule() -> Rule:
    return Rule(atoms=tuple(ATOMS[:2]), action="filter")


def test_smoke_all_estimators():
    logs = _build_logs()
    rule = _build_rule()
    for est in (
        DoublyRobust(),
        MRDR(),
        SwitchDR(tau=5.0),
        RuleOPE(),
        NonCompositionalDR(),
    ):
        est.fit(logs)
        out = est.value(rule, logs)
        v = out.estimate if hasattr(out, "estimate") else float(out)
        assert math.isfinite(v), f"{est.__class__.__name__} returned non-finite estimate"


if __name__ == "__main__":
    test_smoke_all_estimators()
    print("smoke OK")
