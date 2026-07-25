"""E2 — Estimator comparison under DETERMINISTIC logging with an active
post-hoc correction channel (the regime that motivates the paper).

Headline sweeps use uniform-stochastic logging (propensity 1/3, corrections
off), under which RuleOPE's correction term vanishes and it collapses to
compositional-DR. Here we log the way production RAG actually logs:

  logged_action = noop, propensity = 1.0 (deterministic)
  logged_reward = true noop reward + Gaussian judge noise
  correction C ~ Bernoulli((1 - r_noop) / beta_logged)   [A5 linear form,
                 beta_logged = 4.0 — sparse corrections, rate <= 25%]

Estimators: the 5-estimator panel plus RuleOPE in mode="eif"
(beta_target=1.0, beta_logged=4.0, matched to the generative model), i.e.
the Thm-D bridge implementation whose machinery is untested in-grid.

Grid: 36 in-grid cells x N in {150, 300, 600, 1200}. Ground truth is replay,
as in the headline sweep. Reports per-cell MSEs so conditional ranking and
the sigma_R^2 mechanism can be re-tested in this regime.

Run:  python3 experiments/rebuttal/e2_deterministic.py [--n_trials 50]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from joblib import Parallel, delayed

from experiments.multi_estimator_n_sweep import load_cell
from experiments.ablations import NonCompositionalDR
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.switch_dr import SwitchDR
from src.estimators.mrdr import MRDR
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord

BETA_LOGGED = 4.0
BETA_TARGET = 1.0
JUDGE_NOISE = {"hotpot": 0.05, "trivia": 0.02, "nq": 0.02}
LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35", "zephyr7b", "mistral",
        "qwen", "qwencoder7b", "internlm7b", "olmo7b", "granite8b", "yi15"]


def run_one_trial_det(trial, N, samples_all, oracle, rules, _score_passages,
                      _atom_features, bench, seed_base):
    from src.estimators._regression import fires_mask
    rng = np.random.default_rng(seed_base * N + trial + 31337)
    idx = rng.choice(len(samples_all), size=min(N, len(samples_all)), replace=False)
    samples_tr = [samples_all[int(i)] for i in idx]
    noise = JUDGE_NOISE[bench]
    logs = []
    for s in samples_tr:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        r_noop = float(oracle[s.qid]["noop"])
        logged_reward = float(np.clip(r_noop + rng.normal(0, noise), 0.0, 1.0))
        p_corr = np.clip((1.0 - r_noop) / BETA_LOGGED, 0.0, 1.0)
        logs.append(LoggedRecord(
            query_id=s.qid, ctx=ctx, logged_action="noop", logged_propensity=1.0,
            logged_reward=logged_reward,
            correction=int(rng.random() < p_corr),
            cf_rewards=dict(oracle[s.qid]),
        ))
    firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules}
    tr_rules = [r for r in rules if 0.05 <= firing[r.id] <= 0.95]
    if len(tr_rules) < 10:
        return None
    gt = np.array([
        float(np.mean([rec.cf_rewards[r.action] if r.fires(rec.ctx)
                       else rec.cf_rewards["noop"] for rec in logs]))
        for r in tr_rules
    ])
    out = {}
    for name, est in [
        ("NonCompDR",    NonCompositionalDR()),
        ("DR",           DoublyRobust()),
        ("SwitchDR",     SwitchDR(tau=5.0)),
        ("MRDR",         MRDR()),
        ("RuleOPE",      RuleOPE()),   # heuristic correction fusion (paper default)
        ("RuleOPE_eif",  RuleOPE(RuleOPEConfig(mode="eif",
                                               beta_target=BETA_TARGET,
                                               beta_logged=BETA_LOGGED))),
    ]:
        est.fit(logs)
        res = est.value_many(tr_rules, logs)
        vals = np.array([res[r.id].estimate for r in tr_rules])
        out[name] = float(np.mean((vals - gt) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_path = root / "experiments/results/rebuttal/e2_deterministic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [(b, l) for l in LLMS for b in ["hotpot", "trivia", "nq"]]
    Ns = [150, 300, 600, 1200]
    seed_bases = {"hotpot": 8000, "trivia": 5000, "nq": 9000}

    out = {"regime": "deterministic_noop_logging",
           "beta_logged": BETA_LOGGED, "beta_target": BETA_TARGET,
           "n_trials": args.n_trials, "cells": {}}
    t0 = time.time()
    for i, (bench, llm) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {bench}/{llm} ({time.time()-t0:.0f}s)", flush=True)
        samples_all, oracle, rules, sp, af = load_cell(bench, llm)
        cell_out = {"benchmark": bench, "generator": llm, "scaling": {}}
        for N in Ns:
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(run_one_trial_det)(t, N, samples_all, oracle, rules, sp, af,
                                           bench, seed_bases[bench])
                for t in range(args.n_trials)
            )
            results = [r for r in results if r is not None]
            if not results:
                continue
            cell = {}
            for name in results[0]:
                vals = np.array([r[name] for r in results])
                cell[name] = {"MSE_mean": float(vals.mean()),
                              "MSE_std": float(vals.std()),
                              "n_trials": int(len(vals))}
            cell_out["scaling"][str(N)] = cell
            ro = cell["RuleOPE"]["MSE_mean"]; roe = cell["RuleOPE_eif"]["MSE_mean"]
            mr = cell["MRDR"]["MSE_mean"]; dr = cell["DR"]["MSE_mean"]
            print(f"    N={N}: RuleOPE {ro:.5f}  RuleOPE_eif {roe:.5f}  "
                  f"MRDR {mr:.5f}  DR {dr:.5f}", flush=True)
        out["cells"][f"{bench}__{llm}"] = cell_out
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
