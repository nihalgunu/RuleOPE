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

## 15  High-risk, high-novelty exploratory experiments

These are experiments that, if successful, constitute their own
main-conference contribution. Each carries non-trivial risk of
failing; each would meaningfully distinguish the paper if it works.
Implement sequentially or in parallel; each is self-contained.

### 15.A  Corrections-as-instrumental-variables (IV-RuleOPE)

- **Claim**: Under the assumption that corrections are an *instrument*
  for the unobserved reward (satisfying exclusion-restriction and
  relevance conditions), $V(\rho)$ is point-identified *without*
  requiring A5's linear structural form. Miao–Geng–Tchetgen-Tchetgen
  (2018) proximal ID applied to the correction signal.
- **Implementation**: `src/iv_ruleope.py` — two-stage estimator that
  uses $C$ as a proximal outcome, $R$ as the outcome of interest,
  with a shared hidden confounder $U$. Plug into the bridge-function
  machinery.
- **Novelty**: first OPE paper (to our knowledge) to use the correction
  signal as a proximal instrument. Replaces A5 with strictly different
  (and arguably weaker for RAG) conditions.
- **Risk**: HIGH. The exclusion restriction may fail in practice —
  corrections do depend directly on rewards. Needs a careful
  characterisation of when it holds.
- **Success criterion**: on a synthetic substrate satisfying the IV
  conditions, RuleOPE-IV achieves zero bias while DR / standard
  RuleOPE have non-zero bias.

### 15.B  Compositional rule-ensemble evaluation

- **Claim**: A *set* of rules $\mathcal{S} = \{\rho_1, \ldots, \rho_k\}$
  induces a composed policy $\pi_{\mathcal{S}}$ whose value is not the
  sum or product of individual values — rule interactions matter.
  Estimate $V(\pi_{\mathcal{S}})$ from logs.
- **Implementation**: `src/rule_ensemble.py` — handle overlapping
  firing domains, action precedence rules (e.g., abstain beats
  filter beats noop), and a combinatorial variance-bound that scales
  sublinearly in $|\mathcal{S}|$ via inclusion-exclusion on atoms.
- **Novelty**: combinatorial OPE has been studied for slate
  recommendation but not for rule SETS. To our knowledge the first
  OPE paper on rule-set interactions.
- **Risk**: MEDIUM. The combinatorial structure is challenging but
  tractable; the main risk is that without strong assumptions the
  bounds are loose.
- **Experiment**: evaluate the top-20 rules from the benchmark case
  study, first individually, then as an ensemble. Show the ensemble
  value differs materially from the sum of individual values.

### 15.C  Conformal rule-OPE (distribution-free CIs)

- **Claim**: Provide valid $1 - \delta$ confidence intervals for
  $V(\rho)$ without any parametric assumption, using conformal
  prediction (Vovk 2005; Romano et al. 2020).
- **Implementation**: `src/conformal_ruleope.py` — split-conformal
  calibration using out-of-fold RuleOPE residuals; per-rule CI via
  the conformal quantile.
- **Novelty**: no existing OPE paper uses conformal prediction for
  counterfactual policy values. A clean translation of conformal
  inference to OPE that would be cited broadly.
- **Risk**: MEDIUM. Conformal needs exchangeability; rule evaluation
  is i.i.d. over contexts, so the assumption is natural. The
  *sharpness* of conformal intervals may be loose.
- **Success criterion**: empirical coverage at nominal $1 - \delta$
  across rule classes, sharper than Wald intervals in misspecified
  regimes.

### 15.D  Adversarial minimax rule-OPE

- **Claim**: Compute $\underline{V}(\rho) = \inf_{P' \in \mathcal{C}} V_{P'}(\rho)$
  where $\mathcal{C}$ is the set of distributions compatible with the
  observed $(X, R, C)$ marginals plus a user-specified family of
  structural assumptions (e.g., A4, A5 with unknown $\beta$ in a
  range).
- **Implementation**: `src/minimax_ruleope.py` — convex optimisation
  over moment conditions, using ECOS or CVXPY.
- **Novelty**: distributionally-robust OPE extended to rule-specific
  settings; connection to Namkoong–Duchi, Zhan et al. 2024 DRO-OPE.
- **Risk**: MEDIUM-HIGH. The optimisation can be large; convergence
  behaviour in the rule-ensemble case is open.
- **Success criterion**: $\underline{V}(\rho)$ coverage of true $V(\rho)$
  at nominal rate $1-\delta$, across rule classes.

### 15.E  Active-query rule-OPE

- **Claim**: Given a budget to *collect* additional correction labels
  on specific queries, which queries should we label to maximally
  reduce the variance of $\widehat V(\rho^*)$ for the top rule?
  Formulate as an active-learning problem with a minimax-optimal
  query policy.
- **Implementation**: `src/active_ruleope.py` — EIF-gradient-based
  query scoring, sequential halving on the rule set.
- **Novelty**: active-learning for OPE has not been studied under the
  rule framework. Active-OPE more broadly is only in a handful of
  recent papers (Li et al. 2024).
- **Risk**: LOW-MEDIUM. The query-policy design is a clean
  optimisation; the main risk is that in our substrate the
  variance reduction is dominated by identification, not
  estimation, so active labelling doesn't help as much as in
  a pure-estimation regime.

### 15.F  Transductive rule-OPE with conformal calibration

- **Claim**: Provide *per-query* counterfactual predictions
  $\widehat R(x, \pi_\rho(x))$ with conformal prediction intervals,
  rather than expected-value estimates $V(\rho)$.
- **Implementation**: `src/transductive_ruleope.py` — per-query EIF
  plug-in + split conformal.
- **Novelty**: rule-OPE typically estimates expectations; per-query
  counterfactual intervals are a strictly finer object that
  practitioners can act on query-by-query. Connection to jackknife+,
  CQR for counterfactuals (Lei & Candès 2021).
- **Risk**: MEDIUM. Per-query calibration needs careful handling of
  firing vs. non-firing queries.

### 15.G  Scientific-method FDR-controlled rule evaluation

- **Claim**: Treat each rule as a hypothesis ("this rule increases
  reward"). Use Benjamini–Hochberg or knockoffs (Barber–Candès 2015)
  to identify *discoveries* with controlled false discovery rate.
  Ship rules with FDR guarantees.
- **Implementation**: `src/fdr_ruleope.py` — compute per-rule p-values
  from the RuleOPE influence function, apply BH.
- **Novelty**: no OPE paper uses FDR for rule deployment. Direct
  translation of FDR machinery to OPE.
- **Risk**: LOW. BH is well-understood and easy to apply. Main risk
  is that the p-values are conservative if RuleOPE CIs are
  misspecified.
- **Experiment**: on the benchmark, identify rules with
  BH-controlled FDR $\le 0.05$; compare to a naive top-k selection.
  Expected: FDR selection is slightly more conservative but has a
  controlled error rate, which is what a practitioner wants.

### 15.H  Meta-learned bridge functions via transformers

- **Claim**: The bridge function $b_\rho(x)$ depends on the rule $\rho$.
  Train a transformer that takes the rule's atom composition as
  input and outputs the bridge. Amortizes the bridge estimation
  across rules.
- **Implementation**: `src/meta_bridge.py` — transformer with rule-
  atom tokens and context features, trained on synthetic
  substrates with known bridges.
- **Novelty**: meta-learned identification components are a new
  direction; amortized DR estimators of any kind are rare.
- **Risk**: HIGH. Transformer training on small benchmarks may
  overfit; generalisation to new rules is the open question.

### 15.I  Differentiable rule discovery

- **Claim**: Replace the atom-indicator $\phi_\alpha(x) \in \{0, 1\}$
  with a smoothed soft-indicator $\sigma_\tau(\phi_\alpha(x))$
  parameterised by a learnable threshold, and gradient-descend
  through RuleOPE to find the rule that maximises $\widehat V_{\rm LCB}(\rho)$.
  Non-convex but empirically effective.
- **Implementation**: `src/diffrule.py` — JAX or PyTorch
  implementation of the full pipeline.
- **Novelty**: differentiable rule discovery is a distinct
  sub-field; doing it inside an OPE objective is new.
- **Risk**: HIGH. Non-convex optimisation, saddle points, and
  getting the gradient flow correct through the cross-fit
  regression require care.

### 15.J  Causal-mechanism RuleOPE

- **Claim**: Model the RAG pipeline as a structural causal model
  (Pearl 2009) with explicit retrieval-generation-correction
  mechanisms. Rules are interventions on specific nodes. Use
  do-calculus to identify $V(\rho)$ from logs under testable
  graph assumptions.
- **Implementation**: `src/scm_ruleope.py` — DoWhy-based
  identification and estimation.
- **Novelty**: OPE papers rarely use SCMs explicitly; we would
  bridge OPE and causal-inference literatures.
- **Risk**: HIGH. Graph assumptions are strong and hard to verify.
  May not yield a better estimator than our existing framework.

### 15.K  Rule-OPE under non-stationarity (temporal drift)

- **Claim**: The target distribution drifts over time (corpus
  updates). Estimate $V(\rho)$ at time $t_1$ from logs at time $t_0$
  under a parametric drift model.
- **Implementation**: `src/temporal_ruleope.py` — weighted RuleOPE
  with importance weights from a drift-model density ratio.
- **Novelty**: time-varying OPE is a small literature; rule-specific
  drift is unaddressed.
- **Risk**: LOW-MEDIUM. Well-scoped; main risk is the drift model
  is misspecified.

### 15.L  Rule-OPE with LLM-as-judge reward as ground truth

- **Claim**: Use an LLM judge (Claude / GPT-4) to score answer
  quality; treat the score as the reward. Run the full RuleOPE
  pipeline on real RAG logs with LLM-judge rewards. Compare
  estimator MSE against a gold-labeled test split.
- **Implementation**: `experiments/real_data_llm_judge.py` —
  uses a small public RAG benchmark (e.g., HotpotQA 500-query
  subset), a standard RAG pipeline (BM25 + MiniLM retrieval +
  a small LM), and a judge LLM for rewards.
- **Novelty**: first application of rule-OPE to a real RAG
  pipeline with LLM-judge rewards.
- **Risk**: LOW-MEDIUM. Requires API access or a local judge LLM.
- **Success criterion**: the rankings and biases observed on our
  synthetic benchmark qualitatively transfer to the real-data
  setting.

### 15.M  Fairness-constrained rule-OPE

- **Claim**: A rule may improve average reward but hurt a protected
  subgroup. Estimate both $V(\rho)$ and its subgroup values
  $V_G(\rho)$, and select rules subject to a fairness
  constraint.
- **Implementation**: `src/fair_ruleope.py` — subgroup-stratified
  RuleOPE + Pareto-optimal selection.
- **Novelty**: fairness in OPE is an emerging area; fairness for
  rule selection is unaddressed.
- **Risk**: MEDIUM. Subgroup sample sizes may be small; need
  careful variance bounds.

### 15.N  Bandit-of-rules online deployment

- **Claim**: Deploy the top RuleOPE-ranked rules online in a
  bandit; use the online reward to re-rank offline estimates.
  Gives a hybrid offline-online system with formal regret
  guarantees.
- **Implementation**: `src/bandit_deployment.py` — UCB over the
  rule set using RuleOPE offline estimates as warm starts.
- **Novelty**: bandit warm-start from OPE is a natural marriage
  but has not been done for rule-based interventions.
- **Risk**: LOW. Well-scoped. The main open question is
  quantifying the benefit of the warm start vs. cold UCB.

### Ranking for implementation priority

| Experiment | Novelty | Risk | Effort | Priority |
|------------|---------|------|--------|----------|
| 15.C Conformal rule-OPE | High | Med | 1-2 days | **P1** |
| 15.G FDR-controlled selection | Med-High | Low | 1 day | **P1** |
| 15.L Real-data LLM judge | Med | Med | 3-5 days | **P1** |
| 15.A Corrections as IV | Very High | High | 4-6 days | **P2** |
| 15.B Rule ensemble OPE | High | Med | 3-5 days | **P2** |
| 15.D Adversarial minimax | High | Med-High | 5-7 days | **P2** |
| 15.E Active-query rule-OPE | High | Low-Med | 3 days | **P2** |
| 15.F Transductive rule-OPE | Med | Med | 3 days | **P3** |
| 15.H Meta-learned bridges | Very High | High | 1-2 weeks | **P3** |
| 15.I Differentiable rules | High | High | 1-2 weeks | **P3** |
| 15.J SCM rule-OPE | Med | High | 1-2 weeks | **P3** |
| 15.K Temporal drift | Med | Low-Med | 3-5 days | **P3** |
| 15.M Fairness-constrained | Med | Med | 4-6 days | **P3** |
| 15.N Bandit deployment | Low-Med | Low | 2-3 days | **P3** |

P1 (low risk, high value): pursue immediately as camera-ready
additions. P2 (medium risk, high value): scope for a follow-up
paper. P3 (high risk or high effort): multi-paper program.

## 16  Summary of claims $\to$ experiments mapping

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
