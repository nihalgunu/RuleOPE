"""15.N  Bandit-of-rules online deployment.

Use RuleOPE offline estimates as a *warm start* for a UCB bandit over
the rule pool.  Compare against cold-start UCB.

Hybrid offline-online formal regret: for finite horizons T, the
warm-started UCB inherits the offline confidence intervals as initial
priors and continues with standard UCB updates.  Expected: warm-start
strictly dominates cold-start UCB until the online sample size matches
the offline calibration size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class BanditResult:
    cumulative_regret_cold: list[float]
    cumulative_regret_warm: list[float]
    final_regret_cold: float
    final_regret_warm: float


def warm_start_ucb(
    rules: Sequence[Rule],
    offline_logs: Sequence[LoggedRecord],
    online_reward_fn: Callable[[Rule, int], float],
    horizon: int,
    cfg: RuleOPEConfig | None = None,
    cold_start: bool = False,
    seed: int = 0,
) -> list[float]:
    cfg = cfg or RuleOPEConfig()
    K = len(rules)
    rng = np.random.default_rng(seed)

    if not cold_start:
        est = RuleOPE(cfg).fit(offline_logs)
        means = np.array([est.value(r, offline_logs).estimate for r in rules])
        ses = np.array([est.value(r, offline_logs).stderr for r in rules])
        ses = np.maximum(ses, 1e-3)
        n_pseudo = (1.0 / ses) ** 2
        sums = means * n_pseudo
        n_arm = n_pseudo.copy()
    else:
        sums = np.zeros(K)
        n_arm = np.full(K, 1e-3)

    rewards = []
    for t in range(horizon):
        ucb = sums / n_arm + np.sqrt(2.0 * np.log(t + 2.0) / n_arm)
        a = int(np.argmax(ucb + rng.normal(0, 1e-6, size=K)))
        r = online_reward_fn(rules[a], t)
        sums[a] += r
        n_arm[a] += 1
        rewards.append(r)
    return rewards


def cumulative_regret(rewards: list[float], best_value: float) -> list[float]:
    out = []
    s = 0.0
    for t, r in enumerate(rewards, 1):
        s += best_value - r
        out.append(s)
    return out
