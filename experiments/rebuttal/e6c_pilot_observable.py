"""E6c — The pilot tiebreaker uses only OBSERVABLE quantities (w7Aq Q2).

The middle-band pilot compares RuleOPE vs MRDR MSE on a 150-query pilot.
Everything in that comparison is observable in deployment:
  - 150 pilot queries with gold labels (a one-off annotation),
  - all-action replay on those queries: 150 x 3 = 450 generator calls
    (vs ~7,939 calls for full per-rule replay at N=150 per the shipped
    cost_panel.json),
  - per-rule "pilot truth" = replay value of each rule ON THE PILOT SAMPLE
    (no oracle policy values, no full-population quantities).

This script runs SINGLE-DRAW pilots (what a practitioner actually gets, not
the 100-trial average the headline tables use): for each middle-band cell,
20 independent 150-query pilots; each pilot picks argmin-MSE between RuleOPE
and MRDR against its own replay truth. We report per-cell single-draw pilot
accuracy vs the N=1200 truth, and majority-vote accuracy.

Output: e6c_pilot_observable.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.multi_estimator_n_sweep import load_cell, run_one_trial
from experiments.rebuttal.common import ROOT, OUT_DIR, load_cell_table, PAPER_LO, PAPER_HI

N_PILOTS = 20
N_PILOT = 150


def main():
    table = load_cell_table()
    band = table[(table["sigma_R2"] >= PAPER_LO) & (table["sigma_R2"] <= PAPER_HI)]
    print(f"{len(band)} middle-band cells (sigma_R2 in [{PAPER_LO}, {PAPER_HI}])")

    seed_bases = {"hotpot": 8000, "trivia": 5000, "nq": 9000,
                  "musique": 7000, "2wiki": 6000}
    rows = []
    t0 = time.time()
    for ci, (_, cell) in enumerate(band.iterrows(), 1):
        bench, llm = cell["bench"], cell["llm"]
        print(f"[{ci}/{len(band)}] {bench}/{llm} ({time.time()-t0:.0f}s)", flush=True)
        samples_all, oracle, rules, sp, af = load_cell(bench, llm)
        results = Parallel(n_jobs=-1)(
            delayed(run_one_trial)(t, N_PILOT, samples_all, oracle, rules, sp, af,
                                   bench, seed_bases[bench] + 777)
            for t in range(N_PILOTS)
        )
        results = [r for r in results if r is not None]
        picks = ["RuleOPE" if r["RuleOPE"] < r["MRDR"] else "MRDR" for r in results]
        truth = cell["truth"]
        single_acc = float(np.mean([p == truth for p in picks]))
        majority = max(set(picks), key=picks.count)
        rows.append({"bench": bench, "llm": llm, "sigma_R2": cell["sigma_R2"],
                     "truth": truth, "n_pilots": len(picks),
                     "picks_ruleope": picks.count("RuleOPE"),
                     "picks_mrdr": picks.count("MRDR"),
                     "single_draw_accuracy": single_acc,
                     "majority_pick": majority,
                     "majority_correct": majority == truth,
                     "gen_calls_per_pilot": N_PILOT * 3})
        print(f"    truth={truth}  single-draw acc={single_acc:.2f}  "
              f"majority={majority} ({'ok' if majority == truth else 'WRONG'})", flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "e6c_pilot_observable.csv", index=False)

    df = pd.DataFrame(rows)
    print(f"\nmean single-draw pilot accuracy: {df['single_draw_accuracy'].mean():.3f}")
    print(f"majority-vote correct: {int(df['majority_correct'].sum())}/{len(df)} cells")
    print(f"cost per pilot: {N_PILOT * 3} generator calls + {N_PILOT} labels "
          f"(full per-rule replay: ~7939 calls, cost_panel.json)")
    print(f"wrote {OUT_DIR}/e6c_pilot_observable.csv")


if __name__ == "__main__":
    main()
