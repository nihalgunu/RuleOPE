"""Misspecified-reward substrate for stress-testing the compositional estimator.

The default substrate (`src/rag_substrate.py`) generates rewards as a
sigmoid of a LINEAR combination of features, which our ridge regression
can recover exactly.  Reviewers will (correctly) suspect the
compositional estimator has an unfair advantage there.

This module provides a variant where the reward is a NON-ADDITIVE
function of features, specifically involving:
  * pairwise interactions between atoms that are NOT in our ridge model;
  * a non-monotone (sinusoidal) perturbation of the latent quality;
  * correlated action advantages that break the "action-parallel"
    approximation of A5.

The goal is to evaluate whether the identification+efficiency claims of
Theorems A-F continue to hold, or degrade gracefully, when the
regression is mis-specified and A5's linear structural form is violated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.logs import LoggedRecord
from src.rag_substrate import _generate_one, SubstrateConfig, ACTIONS_ENV, SOURCES


def _misspecified_rewards(ctx: dict, cfg: SubstrateConfig, rng: np.random.Generator) -> dict[str, float]:
    latent = ctx["_latent"]
    top1 = ctx["top1_score"]
    gap = ctx["score_gap"]
    red = ctx["redundancy"]
    stub = ctx["top1_src_stub"]
    forum = ctx["top1_src_forum"]
    ppl = ctx["q_ppl"]
    multihop = ctx["q_multihop"]
    entities = (
        ctx["q_has_person"] + ctx["q_has_place"] + ctx["q_has_org"]
        + ctx["q_has_time"] + ctx["q_has_num"]
    )

    # Non-additive structure: pairwise interactions + sinusoidal perturbation.
    interaction = 0.7 * multihop * (1 - stub) - 0.5 * gap * (entities > 1)
    sinus = 0.3 * np.sin(2.5 * latent) + 0.2 * np.cos(1.8 * ppl / 10)

    def sig(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    r_noop = sig(1.5 * latent + interaction + sinus - 0.8 * stub - 0.6 * forum - 0.6 * red + 0.1)
    r_filter = sig(1.5 * latent + 0.6 * stub + 0.5 * forum - 0.2 * red + 0.3 * interaction + sinus)
    r_rerank = sig(1.5 * latent - 0.4 * gap + 0.4 * stub + 0.2 + 0.4 * interaction + 0.8 * sinus)
    r_abstain = cfg.r_abstain

    return {
        "noop":    float(np.clip(r_noop, 0.0, 1.0)),
        "filter":  float(np.clip(r_filter, 0.0, 1.0)),
        "rerank":  float(np.clip(r_rerank, 0.0, 1.0)),
        "abstain": float(r_abstain),
    }


def generate_logs_misspecified(cfg: SubstrateConfig) -> list[LoggedRecord]:
    rng = np.random.default_rng(cfg.seed)
    records: list[LoggedRecord] = []
    for i in range(cfg.n_queries):
        ctx = _generate_one(rng)
        cf = _misspecified_rewards(ctx, cfg, rng)
        if cfg.logging == "deterministic":
            logged_action = "noop"
            logged_prop = 1.0
        else:
            logged_action = ACTIONS_ENV[int(rng.choice(4, p=np.array(cfg.logging_probs)))]
            logged_prop = float(cfg.logging_probs[ACTIONS_ENV.index(logged_action)])

        r_obs = cf[logged_action] + 0.03 * rng.normal()
        r_obs = float(np.clip(r_obs, 0.0, 1.0))
        public_ctx = {k: v for k, v in ctx.items() if not k.startswith("_")}
        records.append(
            LoggedRecord(
                query_id=f"q{i:07d}",
                ctx=public_ctx,
                logged_action=logged_action,
                logged_propensity=logged_prop,
                logged_reward=r_obs,
                correction=0,
                cf_rewards=cf,
            )
        )
    return records


def ground_truth_many_misspecified(rules, records) -> dict[str, float]:
    vals: dict[str, float] = {}
    for rule in rules:
        agg = []
        for rec in records:
            if rule.fires(rec.ctx):
                agg.append(rec.cf_rewards[rule.action])
            else:
                agg.append(rec.cf_rewards["noop"])
        vals[rule.id] = float(np.mean(agg))
    return vals
