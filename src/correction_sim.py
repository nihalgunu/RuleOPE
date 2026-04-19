"""Expert correction simulator.

For each logged record we draw a binary correction indicator with probability
that depends on (a) how bad the logged answer actually was, (b) an
expert-effort term that depends on query features (short queries get
reviewed more often), (c) a noise floor.  Three failure modes are supported
by tuning the correction generator:

1. *Random corrections*: noise_frac = 1.0 -> corrections are uninformative.
2. *Query-dependent selection bias*: effort_slope != 0 tilts the probability
   of a correction towards queries whose features predict the reward even
   conditional on the reward itself -- breaks unconfoundedness.
3. *Self-consistent-answer bias*: if gen_conf is high, correction prob drops
   even when the answer is wrong -- breaks MAR in a RAG-specific way.

The simulator is deterministic given a seed for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.logs import LoggedRecord


@dataclass
class CorrectionConfig:
    base_rate: float = 0.15          # fraction of queries receiving any review
    error_sensitivity: float = 4.0    # weight on "how wrong the answer was"
    effort_slope: float = 0.0         # query-length bias (0 = none)
    gen_conf_bias: float = 0.0        # confidence bias (0 = none)
    noise_frac: float = 0.0           # [0,1] fraction of corrections flipped
    seed: int = 0


def simulate_corrections(
    logs: Sequence[LoggedRecord], cfg: CorrectionConfig
) -> list[int]:
    """Return a list of 0/1 corrections aligned with `logs`.

    Mutates nothing; callers should assign the output back onto records.
    """
    rng = np.random.default_rng(cfg.seed)
    out: list[int] = []
    for rec in logs:
        # "Badness" is (1 - reward): higher when the logged answer was bad.
        badness = 1.0 - rec.logged_reward
        z = np.log(cfg.base_rate / (1 - cfg.base_rate))
        z += cfg.error_sensitivity * (badness - 0.5)
        z += cfg.effort_slope * (rec.ctx.get("q_len", 0.0) - 12.0) / 12.0
        z += cfg.gen_conf_bias * (rec.ctx.get("gen_conf", 0.5) - 0.5)
        p = 1.0 / (1.0 + np.exp(-z))
        c = 1 if rng.random() < p else 0
        if cfg.noise_frac > 0 and rng.random() < cfg.noise_frac:
            c = 1 - c
        out.append(c)
    return out


def assign_corrections(
    logs: list[LoggedRecord], cfg: CorrectionConfig
) -> list[LoggedRecord]:
    c = simulate_corrections(logs, cfg)
    for rec, ci in zip(logs, c):
        rec.correction = int(ci)
    return logs
