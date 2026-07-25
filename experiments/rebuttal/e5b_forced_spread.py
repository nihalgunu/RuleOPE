"""E5b — Populating sigma_R^2 > 0.10 with forced-spread semi-synthetic cells.

Contrast amplification (e5_high_sigma.py) saturates around sigma_R^2 ~ 0.09
even at k=16: queries whose three replay rewards coincide contribute zero
within-query variance no matter how much the deviations are scaled, and on
HotpotQA those queries dominate the average. To actually reach the selector's
upper branch we FORCE action differentiation on a fraction p of queries:

  with prob p:  the query's three action rewards become a random permutation
                of {clip(m - d), m, clip(m + d)},  m = natural mean, d = 0.55
  with prob 1-p: natural rewards kept

p in {0.4, 0.6, 0.8, 1.0} sweeps sigma_R^2 across the 0.10 threshold
(p = 1, m ~ 0.5 gives sigma_R^2 ~ 0.167). Everything else (atoms, rules,
logging, estimators) is the headline protocol. 6 HotpotQA cells,
N in {150, 1200}, 30 trials.

Prop 7 test at last: does MRDR overtake RuleOPE once sigma_R^2 > 0.10?

Output: e5b_forced_spread.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.multi_estimator_n_sweep import load_cell, run_one_trial

P_LEVELS = [0.5, 0.75, 1.0]
NS = [150, 1200]
N_TRIALS = 30
LLMS = sys.argv[1:] or ["smollm17b", "phi35", "zephyr7b", "mistral", "qwen", "yi15"]
REPLAY_ACTIONS = ("noop", "filter", "rerank")


def forced_spread(oracle, p, seed=123):
    """Maximal-spread triple {0, m, 1}: delta-based spreads clip on the
    low-reward queries that dominate hotpot and stall below sigma ~ 0.10."""
    rng = np.random.default_rng(seed)
    out = {}
    for qid, rs in oracle.items():
        vals = np.array([rs[a] for a in REPLAY_ACTIONS], dtype=float)
        if rng.random() < p:
            m = vals.mean()
            vals = rng.permutation(np.array([0.0, m, 1.0]))
        d = {a: float(v) for a, v in zip(REPLAY_ACTIONS, vals)}
        d["abstain"] = rs.get("abstain", 0.5)
        out[qid] = d
    return out


def sigma_from_oracle(oracle):
    return float(np.mean([np.var([rs[a] for a in REPLAY_ACTIONS])
                          for rs in oracle.values()]))


def main():
    root = Path(__file__).resolve().parents[2]
    out_csv = root / "experiments/results/rebuttal/e5b_max_spread.csv"
    rows = []
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
        done = {(r["llm"], r["p_spread"]) for r in rows}
    else:
        done = set()
    t0 = time.time()
    for ci, llm in enumerate(LLMS, 1):
        bench = "hotpot"
        print(f"[{ci}/{len(LLMS)}] {bench}/{llm} ({time.time()-t0:.0f}s)", flush=True)
        samples_all, oracle, rules, sp, af = load_cell(bench, llm)
        for p in P_LEVELS:
            if (llm, p) in done:
                continue
            oracle_p = forced_spread(oracle, p)
            sig = sigma_from_oracle(oracle_p)
            rec = {"bench": bench, "llm": llm, "p_spread": p, "sigma_R2": sig}
            for N in NS:
                results = Parallel(n_jobs=-1)(
                    delayed(run_one_trial)(t, N, samples_all, oracle_p, rules,
                                           sp, af, bench, 8000)
                    for t in range(N_TRIALS)
                )
                results = [r for r in results if r is not None]
                for est in ("RuleOPE", "MRDR", "DR"):
                    rec[f"mse_{est}_{N}"] = float(np.mean([r[est] for r in results]))
            rec["mrdr_gap_pct_1200"] = 100.0 * (rec["mse_RuleOPE_1200"] /
                                                rec["mse_MRDR_1200"] - 1.0)
            rec["truth_1200"] = ("MRDR" if rec["mse_MRDR_1200"] < rec["mse_RuleOPE_1200"]
                                 else "RuleOPE")
            rows.append(rec)
            print(f"    p={p:.1f}: sigma={sig:.3f}  gap={rec['mrdr_gap_pct_1200']:+.1f}%  "
                  f"truth={rec['truth_1200']}", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    hi = df[df["sigma_R2"] > 0.10]
    print(f"\nsigma_R2 > 0.10: {len(hi)} cells, MRDR wins {int((hi['truth_1200']=='MRDR').sum())}")
    from scipy.stats import spearmanr
    rho, pv = spearmanr(df["sigma_R2"], df["mrdr_gap_pct_1200"])
    print(f"Spearman(sigma, MRDR advantage): rho={rho:.3f} p={pv:.2g}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
