"""E5 — Populating the untested high-sigma_R^2 branch (sigma_R^2 > 0.10) with
semi-synthetic contrast amplification.

sigma_R^2 > 0.10 never occurs in the natural 67-cell distribution, so the
selector's upper branch (always-MRDR) and Prop 7's widening-gap prediction
are untested there. We amplify the per-query action contrast of REAL cached
reward channels:

    r'(x, a) = clip( rbar(x) + k * (r(x, a) - rbar(x)), 0, 1 ),
    rbar(x) = mean over the three replay actions, k in {1, 2, 3, 4}

k = 1 is the natural cell; larger k inflates within-query action
differentiation (sigma_R^2 scales ~ k^2 before clipping) while preserving
the query difficulty profile, atom features, and rule structure.

Base cells: the 12 HotpotQA cells (sweep reward protocol == audit sigma
protocol on this benchmark, so amplified sigma_R^2 is directly comparable
to the paper's thresholds). Uniform-stochastic logging as in the headline
sweep. N in {150, 1200}, 30 trials.

Prop 7 test: does the MRDR-over-RuleOPE MSE advantage at N=1200 widen as
sigma_R^2 crosses 0.10?

Output: e5_high_sigma.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.multi_estimator_n_sweep import load_cell, run_one_trial

K_LEVELS = [float(k) for k in sys.argv[1:]] or [1.0, 2.0, 3.0, 4.0]
NS = [150, 1200]
N_TRIALS = 30
LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35", "zephyr7b", "mistral",
        "qwen", "qwencoder7b", "internlm7b", "olmo7b", "granite8b", "yi15"]
REPLAY_ACTIONS = ("noop", "filter", "rerank")


def amplify(oracle, k):
    out = {}
    for qid, rs in oracle.items():
        vals = np.array([rs[a] for a in REPLAY_ACTIONS])
        m = vals.mean()
        amp = np.clip(m + k * (vals - m), 0.0, 1.0)
        d = {a: float(v) for a, v in zip(REPLAY_ACTIONS, amp)}
        d["abstain"] = rs.get("abstain", 0.5)
        out[qid] = d
    return out


def sigma_from_oracle(oracle):
    return float(np.mean([np.var([rs[a] for a in REPLAY_ACTIONS])
                          for rs in oracle.values()]))


def main():
    root = Path(__file__).resolve().parents[2]
    suffix = ("_k" + "_".join(f"{k:g}" for k in K_LEVELS)) if len(sys.argv) > 1 else ""
    out_csv = root / f"experiments/results/rebuttal/e5_high_sigma{suffix}.csv"
    rows = []
    t0 = time.time()
    for ci, llm in enumerate(LLMS, 1):
        bench = "hotpot"
        print(f"[{ci}/{len(LLMS)}] {bench}/{llm} ({time.time()-t0:.0f}s)", flush=True)
        samples_all, oracle, rules, sp, af = load_cell(bench, llm)
        for k in K_LEVELS:
            oracle_k = amplify(oracle, k)
            sig = sigma_from_oracle(oracle_k)
            rec = {"bench": bench, "llm": llm, "k": k, "sigma_R2_amp": sig}
            for N in NS:
                results = Parallel(n_jobs=-1)(
                    delayed(run_one_trial)(t, N, samples_all, oracle_k, rules,
                                           sp, af, bench, 8000)
                    for t in range(N_TRIALS)
                )
                results = [r for r in results if r is not None]
                if not results:
                    continue
                for est in ("RuleOPE", "MRDR", "DR", "NonCompDR"):
                    rec[f"mse_{est}_{N}"] = float(np.mean([r[est] for r in results]))
            if "mse_RuleOPE_1200" in rec and "mse_MRDR_1200" in rec:
                rec["mrdr_gap_pct_1200"] = 100.0 * (
                    rec["mse_RuleOPE_1200"] / rec["mse_MRDR_1200"] - 1.0)
                rec["truth_1200"] = ("MRDR" if rec["mse_MRDR_1200"] < rec["mse_RuleOPE_1200"]
                                     else "RuleOPE")
            rows.append(rec)
            print(f"    k={k:.0f}: sigma={sig:.3f}  gap(MRDR adv %)="
                  f"{rec.get('mrdr_gap_pct_1200', float('nan')):+.1f}  "
                  f"truth={rec.get('truth_1200', '?')}", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    hi = df[df["sigma_R2_amp"] > 0.10]
    lo = df[df["sigma_R2_amp"] <= 0.10]
    print(f"\ncells with sigma_R2 > 0.10: {len(hi)} "
          f"(MRDR truth in {int((hi['truth_1200'] == 'MRDR').sum())}/{len(hi)})")
    print(f"cells with sigma_R2 <= 0.10: {len(lo)} "
          f"(MRDR truth in {int((lo['truth_1200'] == 'MRDR').sum())}/{len(lo)})")
    from scipy.stats import spearmanr
    rho, p = spearmanr(df["sigma_R2_amp"], df["mrdr_gap_pct_1200"])
    print(f"Spearman(sigma_R2, MRDR-advantage gap): rho={rho:.3f}, p={p:.2g}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
