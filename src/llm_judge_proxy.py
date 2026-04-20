"""15.L  LLM-judge proxy.

Real-data §15.L would call an LLM (GPT-4 / Claude / Llama-3) to score
answer quality on HotpotQA-style queries.  We do not run that API in
this screening pass; instead we emulate the LLM judge with a
calibrated noisy proxy whose noise model is taken from the published
LLM-as-judge calibration literature (Zheng et al. 2023 -- LLM judges
agree with human raters at rho ~= 0.8 on factuality).

Concretely the LLM-judge reward is:

    r_LLM(x, a) = clip(r_true(x, a) + N(bias(a), sigma_judge), 0, 1)

with a per-action systematic bias (LLM judges are known to favour
longer answers) and Gaussian noise with sigma_judge = 0.15 (consistent
with the ~0.8 rank correlation reported above).

The experiment substitutes r_LLM for the oracle r_true everywhere in
the RuleOPE pipeline and reports the *robustness* of the top-20
rule ranking under judge noise.

(If a Lambda Cloud API key is supplied via the LAMBDA_API_KEY env var,
a real LLM call can be plugged in via `lambda_judge` -- see protocol.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class JudgeConfig:
    sigma_judge: float = 0.15
    bias_per_action: dict[str, float] | None = None
    seed: int = 0

    def __post_init__(self):
        if self.bias_per_action is None:
            self.bias_per_action = {
                "noop": 0.0,
                "filter": -0.02,   # LLM judge slightly penalises shorter answers
                "rerank": +0.01,
                "abstain": -0.10,  # LLM judges dislike abstentions
            }


def llm_proxy_reward(
    true_reward: float, action: str, cfg: JudgeConfig, rng: np.random.Generator
) -> float:
    bias = cfg.bias_per_action.get(action, 0.0)
    noisy = true_reward + bias + rng.normal(0.0, cfg.sigma_judge)
    return float(np.clip(noisy, 0.0, 1.0))


def relabel_logs(logs, cfg: JudgeConfig | None = None):
    """Return a deep copy of `logs` with logged_reward replaced by the LLM proxy."""
    cfg = cfg or JudgeConfig()
    rng = np.random.default_rng(cfg.seed)
    out = []
    for r in logs:
        new_r = type(r)(**{**r.__dict__})
        new_r.logged_reward = llm_proxy_reward(r.logged_reward, r.logged_action, cfg, rng)
        if r.cf_rewards:
            new_r.cf_rewards = {
                a: llm_proxy_reward(v, a, cfg, rng) for a, v in r.cf_rewards.items()
            }
        out.append(new_r)
    return out
