"""E6a — Table 9 atom-sharing ablation, extended to NQ.

The atom-sharing ablation contrasts RuleOPE (single joint ridge over the
whole pool, atom-action coefficients shared across rules) with
NonCompositionalDR (fresh per-rule ridge on only the logs where the rule
fires). The submitted Table 9 covers HotpotQA + TriviaQA; the underlying
sweep (full_36cell_4N_5estimator.json) already contains both estimators on
the 12 NQ cells, so this is an extraction in the same format, all three
benchmarks side by side for context.

Output: e6a_table9_nq.csv (per-cell MSEs + pct improvement of RuleOPE over
NonCompDR at each N, paired-bootstrap significance from the sweep JSON).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/rebuttal"

data = json.loads((ROOT / "experiments/results/full_36cell_4N_5estimator.json").read_text())

rows = []
for cell_key, cell in data["cells"].items():
    bench, llm = cell_key.split("__")
    for N, sc in cell["scaling"].items():
        if "RuleOPE" not in sc or "NonCompDR" not in sc:
            continue
        rows.append({
            "bench": bench, "llm": llm, "N": int(N),
            "mse_NonCompDR": sc["NonCompDR"]["MSE_mean"],
            "mse_RuleOPE": sc["RuleOPE"]["MSE_mean"],
            "ruleope_pct_vs_noncomp": sc["RuleOPE"].get("pct_mean_logratio", np.nan),
            "ci90_lo": sc["RuleOPE"].get("pct_CI90_bootstrap", [np.nan, np.nan])[0],
            "ci90_hi": sc["RuleOPE"].get("pct_CI90_bootstrap", [np.nan, np.nan])[1],
            "significant": sc["RuleOPE"].get("significance_bootstrap", None),
        })

df = pd.DataFrame(rows).sort_values(["bench", "llm", "N"])
df.to_csv(OUT / "e6a_table9_nq.csv", index=False)

print("Atom-sharing ablation (RuleOPE vs NonCompDR), pct MSE improvement:")
for bench in ("hotpot", "trivia", "nq"):
    sub = df[df["bench"] == bench]
    for N in (150, 300, 600, 1200):
        s = sub[sub["N"] == N]
        n_sig = int(s["significant"].fillna(False).sum())
        print(f"  {bench:8s} N={N:5d}: median {s['ruleope_pct_vs_noncomp'].median():+7.1f}%  "
              f"mean {s['ruleope_pct_vs_noncomp'].mean():+7.1f}%  "
              f"significant {n_sig}/{len(s)} cells")

nq = df[(df["bench"] == "nq")]
print(f"\nNQ headline (N=150): median {nq[nq.N == 150]['ruleope_pct_vs_noncomp'].median():+.1f}%, "
      f"{int(nq[nq.N == 150]['significant'].fillna(False).sum())}/{len(nq[nq.N == 150])} cells significant")
print(f"wrote {OUT}/e6a_table9_nq.csv")
