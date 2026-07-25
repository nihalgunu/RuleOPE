# Rebuttal experiments (NeurIPS 2026 submission 33107)

All experiments run from cached generator outputs in `eval/` except E6b,
which generated the two missing MuSiQue anchor cells on a Lambda 1x H100
PCIe (vLLM 0.10.2, greedy decoding, qwen chat template, max 64 tokens;
instance terminated after use, verified via API). Scripts live in
`experiments/rebuttal/`; every command below runs from the repo root.
Trial seeds are deterministic functions of (benchmark seed base, N, trial
index) exactly as in the headline sweep.

Reviewer keys: N3HW, w7Aq, PAT = the three referee reports.

---

## E1 — Out-of-sample selector validation  [N3HW Q4+lim3; w7Aq Q3]

```
python3 experiments/rebuttal/e1_selector_cv.py
```
Files: `e1_lolo.csv`, `e1_lobo.csv`, `e1_frozen_transfer.csv`.
Selector = σ²_R thresholds + pilot tiebreaker in the middle band
(protocol of `experiments/analysis_middle_band_audit.py`); thresholds are
re-fit per fold by grid search (LO ∈ [0, 0.10], HI ∈ [0.02, 0.20], step
0.01, ties → narrower band, then proximity to the paper's (0.05, 0.10)).
Truth = argmin MSE at N=1200; evaluation grid = 36 in-grid + 34 held-out
cells (MuSiQue 12, 2Wiki 14, anchors 3+3, MuSiQue anchors 2).

| protocol | full procedure | single threshold | always-RuleOPE |
|---|---|---|---|
| leave-one-LLM-out (12 folds) | **29/36 = 80.6%** | 77.8% | 55.6% |
| leave-one-benchmark-out (3 folds) | 21/36 = 58.3% | 58.3% | 55.6% |
| frozen transfer (34 held-out cells, in-grid-fit thresholds) | **23/34 = 67.6%** | 67.6% | 61.8% |
| frozen transfer (paper's fixed 0.05/0.10) | 22/34 = 64.7% | 61.8% | 61.8% |

Per-group frozen transfer (fit thresholds): qwen14b anchors 3/3, qwen32b
anchors 3/3, MuSiQue 9/12, MuSiQue anchors 1/2, 2Wiki 7/14. The selector
generalises across LLMs; benchmark-level transfer is the honest weak spot
(2Wiki: σ²_R compresses to near zero, cf. E3).

## E2 — Deterministic logging + active correction channel  [N3HW Q5+lim1]

```
python3 experiments/rebuttal/e2_deterministic.py --n_trials 50
```
Files: `e2_deterministic.json`, `e2_summary.csv`. Logging: noop,
propensity 1.0, judge noise per substrate defaults; corrections
C ~ Bernoulli((1−r_noop)/β) with β_logged = 4 (A5 linear form);
RuleOPE run in both heuristic and Thm-D EIF modes (β_target=1, β_logged=4).

- RuleOPE ≠ CompDR at last: median |RuleOPE−DR| = 6.6% MSE; the correction
  fusion beats its own DR backbone in 20/36 cells (median +2.7%).
- EIF mode is deliberately conservative: its mean-centred correction term
  leaves it within ±0.01% of DR everywhere (matches the config docstring).
- Conditional ranking persists (trivia → RuleOPE 11/12; hotpot 8/12 and
  NQ 7/12 → MRDR at N=1200).
- σ²_R mechanism persists directionally: Spearman ρ(σ²_R, gap) = 0.374,
  p = 0.025 — but the uniform-regime thresholds transfer poorly (56%);
  a regime-refit single threshold (HI = 0.04) reaches 83.3% in-sample
  vs 55.6% always-RuleOPE. Mechanism survives; calibration is
  regime-specific.

## E3 — A3 validation, ĉ, A3-aware M3  [N3HW lim2, Q3]

```
python3 experiments/rebuttal/e3_a3_validation.py
```
Files: `e3_a3_per_cell.csv`, `e3_a3_summary.json`. Protocol matches the
shipped `a3_validation_nq_mistral.json` (within-query OLS, 48 atom
indicators × 3 actions; reproduces its R² to 0.449 vs 0.460 with reward
means matching exactly).

- R²(A3) by benchmark (mean over 12 LLMs): **NQ 0.449** (max 0.603),
  hotpot 0.097, trivia 0.048, musique 0.023, 2wiki 0.038. The benchmark
  the reviewer flagged is where A3 holds BEST.
- ĉ (Δ²/σ²_R kernel ratio, Fig-6 protocol, 36 cells): median 0.702,
  5–95th pct [0.065, 69.9] (tail driven by near-zero-σ² cells).
- A3-violation-aware M3: adding σ²_R × R²(A3) to the M3 interaction model
  lifts adj-R² 0.898 → 0.964; interaction coefficient p = 3.2e-08.
  σ²_R's slope is amplified precisely where A3 holds — the mechanism is
  A3-dependent, and NQ is its best-case regime.

## E4 — A5 noise-injection ablation, 10 seeds  [N3HW/PAT contradiction]

```
python3 experiments/rebuttal/e4_a5_noise.py
```
File: `e4_a5_noise.csv`. Deterministic-logging regime; A5 violation =
symmetric correction flips, ε ∈ {0, .05, .1, .2, .4}; MAE vs replay truth;
12 cells × 10 seeds.

- The submitted Table 8 direction REPRODUCES per-cell where the correction
  term is MAE-harmful: hotpot/mistral 0.0709 → 0.0613 as ε 0 → 0.4
  (submitted: 0.0669 → 0.0546), with the DR backbone flat at 0.0536.
- Mechanism: as corrections degrade to noise the learnt gate flattens and
  the correction term shrinks toward zero, so RuleOPE collapses onto its
  correction-free DR backbone; when that backbone has lower MAE, violation
  "helps". It is regularization-by-degradation, not robustness.
- Pooled over 12 cells the curve is U-shaped (0.0839 → 0.0826 → 0.0849),
  so the text should describe the per-cell direction, not a global one.

## E5 — High-σ²_R branch  [N3HW weakness 5]

```
python3 experiments/rebuttal/e5_high_sigma.py            # k = 1..4
python3 experiments/rebuttal/e5_high_sigma.py 8 16       # saturation check
python3 experiments/rebuttal/e5b_forced_spread.py        # crosses 0.10
```
Files: `e5_high_sigma.csv`, `e5_high_sigma_k8_16.csv`,
`e5b_forced_spread.csv`.

- Pure contrast amplification saturates: clipping + zero-swing queries cap
  average σ²_R near 0.09 even at k = 16, so no natural-log transformation
  reaches the 0.10 branch — itself an explanation for why the branch is
  empty in the wild.
- Within amplified HotpotQA (σ²_R ≤ 0.09), the MRDR-advantage gap
  correlates NEGATIVELY with σ²_R (Spearman ρ = −0.691, p = 5.5e-08):
  within-benchmark, higher σ²_R favours RuleOPE. The paper's threshold
  logic is a cross-benchmark pattern (cf. M3's benchmark interactions),
  and this should be stated explicitly.
- Forced-spread cells (E5b, maximal {0, m, 1} triples;
  `e5b_max_spread.csv`) populate σ²_R ∈ [0.10, 0.21] — 18 cells above the
  0.10 threshold across 6 LLMs. **MRDR wins 0/18**; the RuleOPE advantage
  WIDENS with σ²_R (Spearman ρ = −0.759, p = 2.6e-04, reaching −88% gap
  at σ²_R ≈ 0.21). Prop 7's widening-gap prediction fails on this
  semi-synthetic axis: high within-query action variance alone does not
  hand the win to MRDR. Caveat: forced spreads randomise the
  action-reward assignment (no atom→action signal), so these cells probe
  σ²_R in isolation; the natural high-σ²_R cells (NQ) couple it with high
  R²(A3) (see E3), and that coupling — not σ²_R per se — is where MRDR
  wins. The σ²_R > 0.10 → MRDR branch should be described as an empirical
  cross-benchmark decision rule, not a causal mechanism.

## E6a — Table 9 atom-sharing ablation on NQ

```
python3 experiments/rebuttal/e6a_table9_nq.py
```
File: `e6a_table9_nq.csv` (extraction from the shipped
`full_36cell_4N_5estimator.json`; RuleOPE vs NonCompDR, paired-bootstrap
CIs from the sweep). NQ median gain +22.3% at N=150 (11/12 cells
significant), decaying to −12.2% at N=1200 (3/12) — versus +143–249%
medians on hotpot/trivia. Atom sharing helps everywhere at pilot scale;
NQ is where it fades, consistent with the rank-flip narrative.

## E6b — MuSiQue frontier anchors (the two missing cells)

```
# on Lambda 1x H100 PCIe (us-west-3), vllm==0.10.2, transformers==4.55.2:
python3 remote_generate.py Qwen/Qwen2.5-14B-Instruct prompts_qwen_1500.jsonl outputs_qwen14b_1500.jsonl
python3 remote_generate.py Qwen/Qwen2.5-32B-Instruct prompts_qwen_1500.jsonl outputs_qwen32b_1500.jsonl
# locally:
python3 experiments/rebuttal/e6b_musique_anchors.py
```
Files: `eval/musique/outputs_qwen{14b,32b}_1500.jsonl` (4500 lines each,
UNKNOWN rates 86.8%/88.9%; in-family — cached Qwen2.5-7B: 82.0%),
`e6b_musique_anchors.json`. Both anchors show the paper's pattern:
RuleOPE +22–30% at N=150, MRDR ahead by N=1200 (rank flip at scale).

## E6c — Observable pilot  [w7Aq Q2]

```
python3 experiments/rebuttal/e6c_pilot_observable.py
```
File: `e6c_pilot_observable.csv`. Single-draw 150-query pilots using only
observables: gold labels for the pilot queries + all-action replay
(150 × 3 = 450 generator calls; full per-rule replay would be ~7,939,
`cost_panel.json`). Across the 19 middle-band cells: mean single-draw
accuracy 70.3%; majority-of-20 correct in 16/19 cells (misses: nq/mistral,
2wiki/qwen14b, 2wiki/qwen32b). No oracle quantities anywhere in the loop.

## E7 — Frozen selector on a new rule pool  [w7Aq Q3]

```
python3 experiments/rebuttal/e7_rules_v2.py
```
Files: `eval/rules_v2.jsonl`, `e7_rules_v2.csv`. rules_v2 = 500 rules,
name-disjoint from v1, same 48-atom DSL and action set. (v1 exhausts
135/144 of the depth-1 space, so v2 is necessarily deeper: 9 depth-1 +
~245 each depth-2/3.) σ²_R is pool-independent, so frozen thresholds
transfer verbatim; only pilot and truth are recomputed on the new pool.
12 cells (4 LLMs × 3 benchmarks), 30 trials.

- Frozen selector (paper 0.05/0.10): **11/12 = 92%**.
- Frozen selector (in-grid fit 0.00/0.06): 11/12 = 92%.
- Always-RuleOPE baseline: 7/12.

---

### Reproducibility notes
- The five substrate modules had a hard-coded `/opt/homebrew` sys.path
  injection that shadowed scipy inside joblib workers; removed in commit
  dd253fe (first commit on this branch) — parallel sweeps crash without it.
- σ²_R values per cell are cached in `sigma_r2_cells.csv` (audit-protocol
  F1 against the primary gold answer, as in
  `analysis_middle_band_audit.py`).
- E3 uses the sweep-protocol rewards (alias-max on trivia/NQ) to match the
  shipped a3_validation JSONs; the two protocols agree on hotpot/musique/
  2wiki.
