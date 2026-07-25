"""E4 — A5 (correction-linearity) noise-injection ablation, re-run with 10 seeds.

The submitted Table 8 reports RuleOPE MAE IMPROVING as the A5 violation grows
(0.0669 -> 0.0546), contradicting the accompanying text. No ablation script
ships with the repo, so this reconstructs the protocol explicitly and tests
the mechanical explanation:

  As the correction channel is corrupted, the learnt gate approaches a
  constant, the correction-fusion term shrinks toward zero, and RuleOPE
  collapses onto its correction-free DR backbone. If the correction term is
  net-harmful in MAE at zero violation (aggressive pseudo-reward imputation),
  injected violation "improves" MAE — a regularization-by-degradation
  artefact, not a genuine robustness win.

Setup: deterministic noop logging with corrections C ~ Bernoulli((1-r)/4)
(the E2 regime). A5 violation via symmetric correction flips with prob
eps in {0, 0.05, 0.1, 0.2, 0.4}. Metric: MAE of rule-value estimates vs
replay ground truth. Estimators: RuleOPE (heuristic), RuleOPE (eif), and the
correction-free DR backbone as reference. 12 cells (4 LLMs x 3 benchmarks),
N = 300, 10 seeds per (cell, eps).

Output: e4_a5_noise.csv
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.multi_estimator_n_sweep import load_cell
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord

BETA_LOGGED = 4.0
JUDGE_NOISE = {"hotpot": 0.05, "trivia": 0.02, "nq": 0.02}
EPS_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4]
N = 300
N_SEEDS = 10
CELLS = [(b, l) for l in ("mistral", "qwen", "phi35", "yi15")
         for b in ("hotpot", "trivia", "nq")]


def one_trial(seed, eps, N, samples_all, oracle, rules, sp, af, bench):
    from src.estimators._regression import fires_mask
    rng = np.random.default_rng(90000 + seed)
    idx = rng.choice(len(samples_all), size=min(N, len(samples_all)), replace=False)
    logs = []
    for i in idx:
        s = samples_all[int(i)]
        scores = sp(s)
        ctx = af(s, scores)
        r_noop = float(oracle[s.qid]["noop"])
        p_corr = np.clip((1.0 - r_noop) / BETA_LOGGED, 0.0, 1.0)
        c = int(rng.random() < p_corr)
        if rng.random() < eps:          # A5 violation: symmetric flip
            c = 1 - c
        logs.append(LoggedRecord(
            query_id=s.qid, ctx=ctx, logged_action="noop", logged_propensity=1.0,
            logged_reward=float(np.clip(r_noop + rng.normal(0, JUDGE_NOISE[bench]), 0, 1)),
            correction=c, cf_rewards=dict(oracle[s.qid]),
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
        ("DR",          DoublyRobust()),
        ("RuleOPE",     RuleOPE()),
        ("RuleOPE_eif", RuleOPE(RuleOPEConfig(mode="eif", beta_target=1.0,
                                              beta_logged=BETA_LOGGED))),
    ]:
        est.fit(logs)
        res = est.value_many(tr_rules, logs)
        vals = np.array([res[r.id].estimate for r in tr_rules])
        out[name] = float(np.mean(np.abs(vals - gt)))
    return out


def main():
    root = Path(__file__).resolve().parents[2]
    out_csv = root / "experiments/results/rebuttal/e4_a5_noise.csv"
    rows = []
    t0 = time.time()
    for ci, (bench, llm) in enumerate(CELLS, 1):
        print(f"[{ci}/{len(CELLS)}] {bench}/{llm} ({time.time()-t0:.0f}s)", flush=True)
        samples_all, oracle, rules, sp, af = load_cell(bench, llm)
        for eps in EPS_LEVELS:
            results = Parallel(n_jobs=-1)(
                delayed(one_trial)(s, eps, N, samples_all, oracle, rules, sp, af, bench)
                for s in range(N_SEEDS)
            )
            results = [r for r in results if r is not None]
            for est in ("DR", "RuleOPE", "RuleOPE_eif"):
                vals = np.array([r[est] for r in results])
                rows.append({"bench": bench, "llm": llm, "eps_flip": eps,
                             "estimator": est, "MAE_mean": float(vals.mean()),
                             "MAE_std": float(vals.std()), "n_seeds": len(vals)})
            m = {est: np.mean([r[est] for r in results]) for est in results[0]}
            print(f"    eps={eps:0.2f}: RuleOPE {m['RuleOPE']:.4f}  "
                  f"eif {m['RuleOPE_eif']:.4f}  DR {m['DR']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print("\nPooled MAE by eps (RuleOPE heuristic):")
    pooled = df[df.estimator == "RuleOPE"].groupby("eps_flip")["MAE_mean"].mean()
    print(pooled.round(4))
    mono = pooled.is_monotonic_decreasing
    print(f"\nMAE decreases monotonically with violation: {mono} "
          f"({'reproduces' if mono else 'does NOT reproduce'} the Table 8 direction)")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
