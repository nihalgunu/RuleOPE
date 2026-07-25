"""E7 — Frozen selector on a NEW rule pool (w7Aq Q3).

Constructs rules_v2: 500 conjunctive rules from the same 48-atom DSL and the
same action set, enumerated with a different random seed and disjoint (by
name) from rules_v1. Composition matched to v1 (approx. 135/185/180 rules at
depth 1/2/3; ~1/3 per action).

Key structural point: sigma_R^2 is a property of the (LLM, benchmark) reward
channel and does not depend on the rule pool at all, so the frozen
thresholds transfer verbatim; only the pilot tiebreaker and the ground
truth are pool-dependent. We re-run the estimator panel with rules_v2 on
12 cells (4 LLMs x 3 benchmarks) at N in {150, 1200} and evaluate the
FROZEN selector (paper (0.05, 0.10) and the in-grid fit (0.00, 0.06))
without refitting anything.

Outputs: eval/rules_v2.jsonl, e7_rules_v2.csv
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.rule_dsl import enumerate_rules, save_rules, load_rules
from experiments.multi_estimator_n_sweep import load_cell, run_one_trial
from experiments.rebuttal.common import ROOT, OUT_DIR, load_cell_table

DEPTH_QUOTA = {1: 135, 2: 185, 3: 180}
N_TRIALS = 30
NS = [150, 1200]
CELLS = [(b, l) for l in ("mistral", "qwen", "phi35", "yi15")
         for b in ("hotpot", "trivia", "nq")]
SELECTORS = {"paper_fixed": (0.05, 0.10), "fit_in_grid": (0.00, 0.06)}


def build_rules_v2(path):
    if path.exists():
        return load_rules(str(path))
    v1_names = {json.loads(l)["name"] for l in open(ROOT / "eval/rules_v1.jsonl")}
    pool = [r for r in enumerate_rules(max_depth=3, cap_per_depth=4000, rng_seed=1)
            if r.name not in v1_names]
    rng = np.random.default_rng(2)
    chosen = []
    for depth, quota in DEPTH_QUOTA.items():
        cands = [r for r in pool if r.depth() == depth]
        # balance actions within each depth
        per_action = quota // 3
        for action in ("filter", "rerank", "abstain"):
            ac = [r for r in cands if r.action == action]
            idx = rng.choice(len(ac), size=min(per_action, len(ac)), replace=False)
            chosen.extend(ac[int(i)] for i in idx)
        # top up to quota with random leftovers of this depth
        rest = [r for r in cands if r not in chosen]
        need = quota - sum(1 for r in chosen if r.depth() == depth)
        if need > 0:
            idx = rng.choice(len(rest), size=need, replace=False)
            chosen.extend(rest[int(i)] for i in idx)
    save_rules(chosen, str(path))
    print(f"rules_v2: {len(chosen)} rules  "
          f"actions={Counter(r.action for r in chosen)}  "
          f"depths={Counter(r.depth() for r in chosen)}")
    return chosen


def main():
    rules_v2 = build_rules_v2(ROOT / "eval/rules_v2.jsonl")
    table = load_cell_table().set_index(["bench", "llm"])
    seed_bases = {"hotpot": 8000, "trivia": 5000, "nq": 9000}

    rows = []
    t0 = time.time()
    for ci, (bench, llm) in enumerate(CELLS, 1):
        print(f"[{ci}/{len(CELLS)}] {bench}/{llm} ({time.time()-t0:.0f}s)", flush=True)
        samples_all, oracle, _rules_v1, sp, af = load_cell(bench, llm)
        sig = float(table.loc[(bench, llm), "sigma_R2"])
        rec = {"bench": bench, "llm": llm, "sigma_R2": sig}
        for N in NS:
            results = Parallel(n_jobs=-1)(
                delayed(run_one_trial)(t, N, samples_all, oracle, rules_v2,
                                       sp, af, bench, seed_bases[bench] + 555)
                for t in range(N_TRIALS)
            )
            results = [r for r in results if r is not None]
            for est in ("RuleOPE", "MRDR", "DR", "NonCompDR"):
                rec[f"mse_{est}_{N}"] = float(np.mean([r[est] for r in results]))
        rec["truth_v2"] = ("RuleOPE" if rec["mse_RuleOPE_1200"] < rec["mse_MRDR_1200"]
                           else "MRDR")
        for name, (lo, hi) in SELECTORS.items():
            if sig < lo:
                pred = "RuleOPE"
            elif sig > hi:
                pred = "MRDR"
            else:
                pred = ("RuleOPE" if rec["mse_RuleOPE_150"] < rec["mse_MRDR_150"]
                        else "MRDR")
            rec[f"pred_{name}"] = pred
            rec[f"ok_{name}"] = pred == rec["truth_v2"]
        rows.append(rec)
        print(f"    truth_v2={rec['truth_v2']}  "
              + "  ".join(f"{n}:{'ok' if rec['ok_' + n] else 'X'}" for n in SELECTORS),
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "e7_rules_v2.csv", index=False)

    df = pd.DataFrame(rows)
    print()
    for name in SELECTORS:
        k = int(df[f"ok_{name}"].sum())
        print(f"frozen selector [{name}] on rules_v2: {k}/{len(df)} = {100*k/len(df):.0f}%")
    k = int((df["truth_v2"] == "RuleOPE").sum())
    print(f"always-RuleOPE baseline: {k}/{len(df)}")
    print(f"wrote {OUT_DIR}/e7_rules_v2.csv")


if __name__ == "__main__":
    main()
