# RuleOPE — Exhaustive Experiment Protocol

This document specifies **every** experiment that should be run to
defend the paper's claims, the baselines each experiment compares
against, the data regime, the metrics, and the expected outcome tied to
a specific theorem. Run these in order; each later experiment assumes
the earlier benchmark artifacts are built.

Notation: $N$ = log size; $M$ = rule set size; seeds indexed over
multiple trials to give standard-deviation bars.

---

## 0  Pre-requisite: build the frozen benchmark

```
python3 eval/build_benchmark.py --out eval --n_queries 4000 --target_rules 500 --seed 0
```

Produces:
- `eval/benchmark_v1.jsonl` (public logs, cf_rewards stripped)
- `eval/benchmark_v1_with_cf.jsonl` (private, with ground-truth cf rewards)
- `eval/rules_v1.jsonl` (500 rules, depth 1–3)
- `eval/ground_truth_rule_values.json` (exact $V(\rho)$ per rule)
- `eval/correction_logs_noise_{00, 10, 30}.jsonl` (three noise regimes)
- `eval/MANIFEST.json` (SHA-256 checksums)

Verify integrity: `python3 tests/test_freeze.py` must print `freeze ok`.

---

## 1  Smoke test (every commit)

Purpose: sanity-check the pipeline end-to-end.

```
python3 scripts/smoke_test.py
```

- $N = 500$, $M = 30$, 4 estimators (DM, IPS, SNIPS, RuleOPE).
- Pass criterion: all four finite MSE values, DM MSE < 0.001,
  RuleOPE MSE $\leq$ DM MSE.

---

## 2  Main head-to-head comparison (Claim: Thm B + Thm E in practice)

```
python3 scripts/run_all_baselines.py                     # benchmark-v1 snapshot
python3 experiments/synthetic_controlled.py              # 3 regimes × 3 trials
python3 experiments/small_n_comparison.py                # deterministic N∈{300,600,1200,2400}
```

### Estimators

| Estimator | Role |
|---|---|
| DM (Direct Method) | regression-only baseline |
| IPS | classical importance sampling |
| SNIPS | self-normalised IPS |
| DR | Robins et al. 1994 |
| CIPS (clip=20) | Clipped IPS |
| CIPS-DR | Clipped-IPS Doubly Robust |
| CascadeDR | Kiyohara et al. 2022 (position-factorised DR) |
| **RuleOPE** | our estimator |
| **DualShrinkOPE** | our between-estimator shrinkage |
| **JointRuleOPE** | our cross-rule random-effects shrinkage |

### Regimes

- **R1** stochastic logging ($\pi_0 = (0.70, 0.15, 0.10, 0.05)$), correction noise 10%.
- **R2** deterministic logging ($\pi_0 \equiv$ noop), correction noise 10%.  **Primary regime.**
- **R3** stochastic logging, correction noise 30%.

### Metrics (per cell)

- MSE against ground truth, mean $\pm$ std over trials.
- Bias (signed).
- 95% CI coverage.
- Top-20 Kendall's $\tau$.
- Wall-clock time.

### Expected outcome

- Under R1/R3: every DR-family estimator ties within Monte Carlo noise.
- Under R2: DR, CIPS-DR, CascadeDR collapse to the same regression-
  driven estimate. RuleOPE reduces MSE by 10–23% for $N = 300$–$2400$
  (confirming Thm B). DualShrinkOPE trades 6–14% MSE reduction for
  lower variance.

---

## 3  Identification-interval diagnostic (Claim: Thm A)

```
python3 experiments/identification_gap.py
```

- Compute sharp bounds $[V_L, V_U]$ per rule (`src.identification.partial_id_bounds`).
- Measure where DR, RuleOPE, and ground truth sit inside each interval.
- Report: avg interval width, estimator positions, % outside interval (should be $\sim 0$).

### Expected outcome

- Mean interval width $\approx 0.15$–$0.20$, 8–10× typical estimator error.
- Both estimators inside the interval (Thm A sanity).
- Ground truth position $\approx$ RuleOPE position, further from DR position
  (confirming RuleOPE tracks truth more tightly, as Thms C–E predict).

---

## 4  Rigorous efficiency validation (Claims: Thms B, E, F simultaneously)

```
python3 experiments/efficiency_validation.py
```

Two substrates:
1. **Compositional** (BEIR-calibrated, linear atom rewards) — A5 plausibly holds.
2. **Misspecified** (`src/rag_substrate_misspec.py`, atom interactions + sinusoid) — A5 violated.

- $N = 1500$, 60 bootstrap resamples, 25 variance-subset rules.
- For each substrate, compute:
  - Thm B test: DR abs-bias vs RuleOPE abs-bias, and sign-consistency of DR bias.
  - Thm E test: empirical variance reduction of RuleOPE over DR (%).
  - Thm F test: correlation between the closed-form formula
    $E[p^2 b^2 g(1-g)]$ and the bootstrap-empirical $\mathrm{Var}(V_{DR}) - \mathrm{Var}(V_{ROPE})$.

### Pass/Fail

- Compositional: all three tests PASS.
- Misspecified: Thm B & E still PASS (graceful degradation); Thm F
  correlation weakens but stays $> 0$.

---

## 5  Ablations

```
python3 experiments/ablations.py
```

### 5.A Compositional factorisation vs per-rule regression

- Estimators: RuleOPE, classical DR (compositional), `NonCompositionalDR`
  (refits a ridge per rule).
- Claim: NonCompDR has $\sim$ 4× higher MSE and dramatically worse coverage.

### 5.B Correction-noise sensitivity

- Noise $\in \{0, 0.10, 0.20, 0.30, 0.50\}$.
- Claim: RuleOPE does not degrade with noise; the gate shrinks toward
  zero when corrections are uninformative.

### 5.C Rule-depth stratification

- Depths $\{1, 2, 3\}$.
- Claim: MSE decreases with depth; tau decreases with depth.

### 5.D Sample efficiency

- $N \in \{250, 500, 1000, 2000, 4000\}$.
- Claim: RuleOPE and DR both reach the noise floor by $N = 1000$; DM
  slightly better at $N = 250$ because it's all-regression.

---

## 6  Scaling in |R| (Claim: Theorem 2, variance sublinear in M)

```
python3 experiments/public_rag.py
```

- $|\mathcal{R}| \in \{50, 500, 5000\}$.
- Measure: per-rule MSE, total evaluation time.
- Expected: RuleOPE time scales sublinearly with $|\mathcal{R}|$ (one
  regression fit, linear per-rule evaluation). MSE per rule is flat.

---

## 7  Failure-mode stress tests (Claims of §5.3 in paper)

```
python3 experiments/failure_modes.py
```

Four settings:

- **benign** — no violation of A4/A5.
- **F1 effort bias** — $P(C \mid X)$ depends on query length (A4 violated via a confounder not in $\phi$).
- **F2 self-consistent bias** — high gen-conf suppresses corrections (A4 violated).
- **F3 corpus drift** — train/eval distribution mismatch.

Metrics per setting: MSE, tau@20.

### Expected outcome

- F1: RuleOPE tau improves over DR (adding the query-length atom to $\mathcal{V}$ recovers A4).
- F2: DM edges out DR and RuleOPE — the correction signal is
  systematically wrong in this regime. Documented as a fundamental
  limitation (§10 of paper).
- F3: RuleOPE tau > DR > DM. RuleOPE's DR correction is partially
  rescued by overlapping records.

---

## 8  Case study on top-20 rules (qualitative)

```
python3 experiments/case_study.py
```

- Fit RuleOPE on frozen benchmark.
- Print top-20 rules by estimated value.
- Inspect manually: each rule should correspond to a sensible failure
  mode of the retrieval pipeline.

### Expected outcome

- Top-20 is a mix of depth-1 and depth-2 rules, dominated by
  low-trust-source atoms and missing-entity atoms.
- All top-20 have absolute error $< 0.008$ vs ground truth.

---

## 9  Shrinkage estimators (Claim: Propositions 1 + Thm 3 in proofs)

```
python3 experiments/shrinkage_experiment.py
```

- Deterministic logging, $N \in \{300, 600, 1200, 2400\}$.
- Compare: RuleOPE, DR, `JointRuleOPE(per_rule_eb)`, `JointRuleOPE(james_stein)`, `DualShrinkOPE`.
- Claim: shrinkage variants reduce MSE and/or variance in regimes where
  the primary estimator has residual bias.

### Expected outcome

- JointRuleOPE does not dominate on our benchmark because the
  compositional regression already captures most of the rule-level
  signal; we report this honestly.
- DualShrinkOPE provides 6–14% MSE reduction with lower variance.

---

## 10  Pessimistic rule selection (Claim: Thm 4 compositional LCB)

```
python3 experiments/pessimistic_selection.py
```

- Deterministic logging, $N \in \{300, 600, 1200, 2400\}$, 5 seeds.
- Compare three selectors:
  - **naive**: $\arg\max \widehat V(\rho)$.
  - **std LCB**: $\arg\max (\widehat V - \sqrt{2 \log(M/\delta)} \widehat\sigma)$.
  - **compositional LCB**: $\arg\max (\widehat V - \sqrt{2((s+1)\log(d+1) + \log(1/\delta))} \widehat\sigma)$ with LASSO sparsity $s$.
- Metric: regret $V(\rho^\dagger) - V(\widehat\rho)$ averaged over seeds.

### Expected outcome

- All three selectors tie on our benchmark (per-rule SEs homogeneous).
- Framework contribution; differentiation requires heterogeneous per-
  rule SE. We document this.

---

## 11  CRRM rule learning (Claim: Thm 5)

```
python3 experiments/crrm_experiment.py
```

- Deterministic logging, 5 seeds per $N \in \{300, 600, 1200, 2400\}$.
- Compare ERM (naive argmax), LCB, CRRM (compositional + depth
  penalty).
- Metric: regret.

### Expected outcome

- On the default benchmark ERM = LCB = CRRM (same rule argmax).
  Honestly documented. The CRRM framework contribution stands on the
  Rademacher regret bound (Thm 5); empirical differentiation requires
  a richer rule space with sparser best rules.

---

## 12  Standard OPE methodology regression (Claim: matches OBP protocol)

This is a *protocol* check rather than a new experiment. We confirm
our evaluation follows the Dudík-Langford-Li / Saito et al. 2021 OBP
standard synthetic-eval protocol:

- Context-dependent reward function with known ground truth.
- Logging policy with known propensities.
- Ground truth computed directly from the reward function.
- Estimators evaluated on MSE vs ground truth across bootstrap resamples.

Everything in §§2–11 follows this protocol. The reframe of the paper
(§7 of the paper) cites the methodology explicitly rather than claiming
a new benchmark.

---

## 13  Reproducibility matrix

| Experiment                       | Seeds | Wall time on 1 CPU |
|----------------------------------|-------|--------------------|
| build_benchmark                  | 1     | $\approx 10$s      |
| smoke_test                       | 1     | $\approx 2$s       |
| run_all_baselines                | 1     | $\approx 4$min     |
| synthetic_controlled             | 3     | $\approx 10$min    |
| small_n_comparison               | 5     | $\approx 8$min     |
| identification_gap               | 1     | $\approx 1$min     |
| efficiency_validation            | 60 bootstraps | $\approx 25$min |
| ablations                        | 1     | $\approx 6$min     |
| public_rag (scaling)             | 1     | $\approx 4$min     |
| failure_modes                    | 1     | $\approx 3$min     |
| case_study                       | 1     | $\approx 30$s      |
| shrinkage_experiment             | 3     | $\approx 4$min     |
| pessimistic_selection            | 5     | $\approx 3$min     |
| crrm_experiment                  | 5     | $\approx 3$min     |
| **Total**                        |       | $\approx 70$min    |

All experiments are seeded; running the full suite with a single
`make reproduce` command is a camera-ready deliverable.

---

## 14  What is NOT in this protocol (and why)

- **Real-data experiment on BEIR/KILT with Llama-3**: deferred due to
  multi-GPU cost; tracked as a camera-ready item.
- **Live-deployment A/B test**: out of scope for an OPE paper.
- **Rule auto-mining**: the CRRM framework supports this but an
  end-to-end mining experiment is a separate project.

---

## 15  Summary of claims $\to$ experiments mapping

| Claim | Tested in |
|-------|-----------|
| Rule-OPE formal problem is well-posed | §2 (smoke), §3 (identification) |
| DR is biased under deterministic logging (Thm B) | §4 (efficiency_validation) |
| RuleOPE attains the SEB (Thm E) | §4 (efficiency_validation) |
| Variance gap $= E[p^2 b^2 g(1-g)]$ (Thm F) | §4 (efficiency_validation) |
| Variance scales sublinearly in M (Thm 2) | §6 (public_rag) |
| Correction-linearity sufficient for A5 | §5 (ablations noise) |
| Shrinkage estimators dominate in specific regimes | §9 |
| Compositional LCB regret (Thm 4) | §10 |
| CRRM regret (Thm 5) | §11 |
| Estimator robust to A4/A5 violations | §7 (failure modes) |
| Top-20 rules are interpretable | §8 (case study) |
