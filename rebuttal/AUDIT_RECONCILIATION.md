# Definitive audit accounting (resolves N3HW + PAT count objections)

Source of truth: `experiments/rebuttal/e10_audit_reconciliation.py`
(`e10_cell_accounting.csv`, `e10_summary.json`). All numbers below use the
audit-protocol sigma_R^2 (token-F1 against the primary gold answer) and the
paper thresholds LO = 0.05, HI = 0.10.

## Canonical cell sets

| set | composition | cells | middle band | max sigma_R^2 |
|---|---|---|---|---|
| **paper audit set** | in-grid 36 + MuSiQue 12 + 14B anchors 3 + 32B anchors 3 | **54** | **17** | 0.0941 |
| extended audit set | + MuSiQue anchors 2 (E6b) + 2Wiki 14 | 70 | 19 | 0.0941 |

Middle-band membership (paper set): HotpotQA 7 (granite8b, internlm7b,
phi3mini, qwen, yi15, qwen14b, qwen32b), NQ 10 (8 in-grid: granite8b,
internlm7b, mistral, phi35, phi3mini, qwen, qwen3b, yi15; + the two NQ
anchors), TriviaQA 0.

## Each flagged inconsistency, resolved

1. **17 vs 19 middle-band cells** (N3HW; PAT "arithmetic inconsistencies"):
   17 is correct for the 54-cell paper set; 19 belongs to the 70-cell
   extended set (the two 2Wiki anchor cells are middle-band). The App-B
   audit-summary arithmetic (7 rescues + 0 harms + 3 wrong + 9 agreements
   = 19) was computed on the extended set while the surrounding text and
   table described the 54-cell set. Fix: present both sets explicitly; the
   54-cell table keeps 17; the extended-audit paragraph gets its own table.
2. **"All 12 in-grid NQ cells have sigma_R^2 >= 0.05" (App B.2 text)**: false
   — it is 8/12 (range 0.0079-0.0941; smollm17b, zephyr7b, qwencoder7b and
   olmo7b sit below 0.05). The App-B *table* showing 10 NQ middle-band rows
   was always correct (8 in-grid + 2 anchors). Fix: correct the B.2 sentence.
3. **"54 cells" vs 67 rows in `middle_band_audit.csv` vs "51-cell" docstring**:
   the CSV includes 13 usable 2Wiki cells on top of the paper set (54 + 13 =
   67); the "51" docstring predates the anchor cells. The docstring is fixed
   in this commit; Fig 3's "54" label matches the paper set and stays.
4. **Two sigma_R^2 protocols** (root cause of "sigma_R^2 > 0.10 never
   observed" vs regression-CSV values up to 0.113): audit scripts score F1
   against the primary gold; `full36_phenomenon_pairs.csv` uses alias-max F1
   on trivia/NQ. Spearman between protocols = 0.991; M3 adj-R^2 is 0.898
   (alias-max) vs 0.897 (audit) — every conclusion is protocol-robust, but
   6 NQ cells exceed 0.10 only under alias-max. Fix: standardise on the
   audit protocol everywhere (selector, regressions, abstract), with
   alias-max reported once as a robustness footnote.
5. **Figure 5 legend vs Table 2** (N3HW "mislabeling"): the plotted 0.80 /
   0.97 / 0.88 adj-R^2 are the noop_F1-model values; the sigma_R^2 models
   give 0.803 / 0.974 / 0.898 (see `full_loocv.stdout.txt`). Since the two
   sets are nearly identical this was a label slip, not a wrong result;
   fix: plot and quote the sigma_R^2-model numbers everywhere
   (0.80 / 0.97 / 0.90) and correct the legend.
6. **45 vs 48 atoms**: the DSL has 48 atoms (`src/rule_dsl.py`); "45" in
   Sec. 2 is stale. Note 13-23 atoms are constant per benchmark, so the
   *effective* design rank is 20-22 (e9) — worth one sentence in Sec. 2.
7. **Abstract's "8/36 NQ cells"**: the deepening happens in 8 of the 12 NQ
   cells (e8); "36" referred to the full in-grid count. Fix: say "8/12 NQ
   cells" .

## Paper-side edit list (Overleaf)
- Sec. 2: V = 48; uniform logging gamma/(K_a+1); B2 coverage 1/K_a; define
  p_rho; drop the dead rule-tuple element; rename overloads (K_a/K_f,
  lambda_ridge, sigma2_eps, kernel Sigma/Q) per THEORY_FIXES.md.
- Sec. 5: quote sigma-model adj-R^2 (0.80/0.97/0.90); Fig 5 legend fix;
  define proxy reward = alias-max F1; A3 t-stats: describe the held-out
  (cross-fitted) residual protocol, not in-sample OLS residuals.
- Sec. 6 + App. B: single sigma_R^2 protocol; 54/17 as canonical with the
  70/19 extended audit separately; fix B.2 sentence; "at deployment N" ->
  "at the pilot size N_pilot = 150" (both reviewers asked).
- Table 8: replace with e4's 10-seed version + the regularization-by-
  degradation explanation (text and table now agree; direction reproduces
  per-cell where the correction term is MAE-harmful, pooled curve is
  U-shaped).
- Table 9: add the NQ column from e6a.
