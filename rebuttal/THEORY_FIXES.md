# Theory corrections for submission 33107

Corrected statements, what was wrong, and the numerical validation backing each
fix. Written to be ported into the paper (Overleaf); every claim marked
**[verified]** has a reproducing script in `experiments/rebuttal/` and a result
file in `experiments/results/rebuttal/`.

Notation used throughout (fixing the overloads flagged by all three reports):

| symbol | meaning | replaces |
|---|---|---|
| `K_a` | number of non-noop actions (K_a = 2; action space size K_a + 1 = 3) | overloaded `K` |
| `K_f` | cross-fitting folds | overloaded `K` |
| `gamma` | uniform-logging shuffling rate | — |
| `lambda_ridge` | ridge penalty | overloaded `lambda` / `gamma` in App. G |
| `c` | post-hoc correction flag (scalar r.v.) | — |
| `Sigma = E[phi phi^T]` | second moment of standardised atom features | previously undefined kernel "C" |
| `Q = q q^T`, `q = E[phi]` (standardised coords) | mean-feature kernel | previously undefined "A" / "mu" |
| `V_beta = E_a[(beta_a - beta_bar)(beta_a - beta_bar)^T]` | across-action coefficient covariance | — |
| `sigma2_eps` | reward observation-noise variance | overloaded `sigma^2_R` in Thm 8 |

---

## 1. Theorem 4 (commensurability "sandwich") — corrected statement

### What was wrong
1. **w7Aq Q1 is correct.** `V_beta` is the covariance of 3 centred coefficient
   vectors, so rank(V_beta) <= 2 in d = 48 dimensions. It is never positive
   definite, and the submitted "vanish together exactly when the LLM does not
   differentiate actions" is not available from PD of `V_beta`.
   **[verified]** e9: rank(V_beta) <= 2 in 36/36 cells.
2. **PAT's algebra objection is correct.** The proof's step
   `E_a[beta_a^T C beta_a] - mu^T C mu = tr(V_beta C)` only holds when
   `mu = beta_bar`, which would make the kernel LLM-dependent and contradict
   the theorem's own premise. The correct centred identity is
   `E_a[beta_a^T C beta_a] - beta_bar^T C beta_bar = tr(V_beta C)`.
3. The empirical statistic `sigma-hat^2_R` (variance of the three *realised*
   rewards) is not `tr(V_beta Sigma)`: it carries an additive observation-noise
   floor. The submitted statement omitted the noise term.
4. A two-sided sandwich `c1 * sigma^2_R <= Delta^2 <= c2 * sigma^2_R` with
   universal (LLM-independent) constants is impossible: `Delta^2` is the trace
   against the rank-one kernel `Q`, so its lower bound against `sigma^2` is 0.
   **[verified]** e3/e9: the per-cell ratio `Delta^2 / sigma^2` spans
   [0.03, 108] across the 36 cells — no universal constants exist.

### Corrected Theorem 4'
Assume A1-A3 with standardised atom features, and let
`sigma2_sig := tr(V_beta Sigma)`, `Delta^2 := tr(V_beta Q)`. Then:

(i) *(shared structure — survives as stated)* Both `sigma2_sig` and `Delta^2`
are linear trace functionals of the same LLM-dependent PSD matrix `V_beta`
against LLM-independent PSD kernels (`Sigma`, `Q` are functionals of the query
distribution and retrieval pipeline only).

(ii) *(one-directional vanishing — replaces "vanish together iff")*
`V_beta = 0  =>  sigma2_sig = Delta^2 = 0`. Conversely `sigma2_sig = 0 =>
V_beta = 0 => Delta^2 = 0`, *restricted to range(Sigma)* — and the
restriction is free rather than an assumption: ridge coefficients satisfy
`beta_hat_a in range(Sigma)` by construction (the normal equations map into
the design row space), so `V_beta` is supported on range(Sigma), where
`Sigma` is PD. Atoms that never vary on a benchmark cannot enter any
estimate, so nothing is lost. The reverse implication from `Delta^2` fails
in general: because rank(Q) = 1, `Delta^2 = 0` is possible with
`sigma2_sig > 0` (actions differing only in X-varying components of equal
means) — exactly w7Aq's counterexample, which we now state as a remark
instead of contradicting.
**[verified]** e9: `Sigma` is singular in the ambient d = 48 (13-23 atoms
are constant per benchmark; rank(Sigma) = 20-22) but PD on its range in
36/36 cells, with smallest nonzero eigenvalue 5.8e-3 to 1.8e-1 — the
on-range direction holds on our data with a healthy margin.

(iii) *(observable statistic)* The deployment statistic satisfies
`sigma-hat^2_R = sigma2_sig + E_X[within-query residual variance] +
O(shrinkage)`: signal functional plus the action variance the linear atom
model does not capture. On our data the second term dominates in magnitude
(median 96% of the statistic) yet the two quantities are almost perfectly
rank-aligned: **[verified]** e9, Spearman(sigma-hat^2_R, tr(V_beta Sigma))
= 0.971 (p = 8e-23) over the 36 cells. The revised text therefore claims
rank alignment (which is all the thresholded selector uses), not equality —
and explicitly notes that sigma-hat^2_R measures *total* within-query
action differentiation, of which the A3-linear part is one component.

(iv) *(no universal sandwich — honest replacement)* For any PSD `V_beta`,
`Delta^2 <= lambda_max(Sigma^{-1/2} Q Sigma^{-1/2}) * sigma2_sig` when
`Sigma` is PD; no matching lower bound with an LLM-independent constant
exists. We therefore report the empirical kernel-ratio band (median 0.70,
5-95th pct [0.065, 69.9]) and *withdraw* the calibration claim: `sigma^2_R`
is a vanishing-linked, monotonically-associated proxy for `Delta^2`, not a
calibrated estimate of it. This directly answers N3HW Q3 ("sign-related, not
calibrated") and the missing-`c-hat` complaint: c-hat is now reported.

### Reviewer answers this supports
- w7Aq Q1: conceded and fixed; the corrected theorem never needs PD of
  `V_beta`, and the "nonzero sigma^2 with Delta = 0" scenario is acknowledged
  as real and stated as a remark.
- N3HW Q3: c-hat reported (median 0.702, wide band); claim downgraded from
  calibration to vanishing/monotone association.
- PAT App C.4: centred identity fixed; kernel definitions (`Sigma`, `Q`)
  now explicit; noise floor added to the `sigma-hat^2_R` equation.

---

## 2. Theorem 3 (gap magnitude) and the NQ deepening — corrected statement

### What was wrong
- Theorem 3's displayed decomposition keeps only the positive O(K_a/N)
  variance-geometry term, which predicts NonCompDR worse at every N — yet the
  paper *attributes the NQ reversal to this theorem*. N3HW Q1 and PAT's
  "sign contradiction" are both correct: as displayed, the theorem cannot
  produce a negative gap.
- The formal proof (derivation of the matrices, constants, and residual
  bounds) was missing (App. C.3 is prose).

### Corrected Theorem 3'
Under A1-A3 with cross-fitting, for the pooled-vs-per-rule ridge pair,

    E_rho[ MSE_NC(N) - MSE_RO(N) ] = a + b/N + r(N),

where

- `b = c_geom * K_a * sigma2_eps * tr(G)` > 0 collects the variance-geometry
  advantage of the atom-shared regression (`G` the design-dependent PSD
  matrix, written out explicitly in the revised App. C.3),
- `a = bias^2_NC - bias^2_RO` is the N-independent squared-bias difference:
  `bias_RO` grows with the A3-misspecification of the *pooled* fit, while
  `bias_NC` is the per-rule refit bias, so `a` can take either sign,
- `r(N) = O(N^{-3/2})` is the discarded remainder, now bounded.

The sign structure is the content: at pilot scale the b/N term dominates
(RuleOPE wins); whenever `a < 0` the gap crosses zero at `N* = -b/a` and
converges to `a < 0` from above — the gap *deepens* with N rather than
recovering. The submitted paper's claim that the theorem "predicts" the NQ
failure via the variance term was wrong; it is predicted by the *bias
asymptote*, i.e. by the terms the submitted display discarded.

### Numerical validation **[verified — e8_gap_crossover.py]**
Fitting `gap(N) = a + b/N` to the five cached sample sizes per cell
(N in {150, 300, 600, 1200, 2400}):
- median fit R^2 = 0.886 with two parameters;
- `b > 0` in **36/36** cells (the theorem's leading term, everywhere);
- `sign(a)` predicts `sign(gap at N=2400)` in **36/36** cells;
- `a < 0` in exactly the **8/12 NQ cells** where RuleOPE falls below
  NonCompDR, and in 1/12 hotpot, 0/12 trivia — the reported NQ deepening is
  the bias asymptote, now derived and validated rather than asserted;
- fitted crossovers N* range 247-5,907 on those NQ cells, consistent with
  flips visible between N=600 and N=2400 in the raw profiles.

This also resolves N3HW Q1 (sign and N-dependence of the residual: the
residual is the N-independent `a`, negative exactly on the deepening cells)
and E6a's independent observation (NQ atom-sharing gain +22.3% at N=150
decaying to -12.2% at N=1200).

---

## 3. Theorems 2/8/9 (aggregate variance scaling) — corrected claims

### What was wrong
PAT's algebra is right: summing per-rule prediction variances over a pool of
R rules gives `sum_rho x_rho^T Cov(beta_hat) x_rho >= lambda_min(Cov) * R`
(each rule fires at least one atom), so *aggregate* variance is Omega(R).
Theorem 8's own lower bound says the same. Theorem 9's attempted
reconciliation via the conditioning constant `kappa` fails because `kappa`
itself scales linearly in R — the sublinear-in-R *aggregate* claim is
algebraically impossible and is withdrawn.

### Corrected Theorems 2'/9'
What is true, and what the experiments actually use:
- **Per-rule variance is O(d/N), independent of R**: every rule's estimate
  reuses the same 48x3-parameter shared fit, so adding rules to the pool does
  not degrade any individual estimate. In contrast, per-rule refits (MRDR,
  NonCompDR) concentrate on the firing subset, with effective sample
  `N * p_rho` per rule.
- **Aggregate variance is Theta(R * d/N)** for the shared fit — matching
  Theorem 8's lower bound, with the tightness corollary restated for the
  *average* (aggregate/R), which is where the constant `kappa` is genuinely
  R-independent under pool assumptions B1-B2.
- **The generator-call complexity claim is untouched**: O(A + V) vs
  O(R * A) replay calls is a statement about data collection, not estimator
  variance, and survives as the scalability contribution.
- Notation fix: Theorem 8's noise variance is `sigma2_eps`, not `sigma^2_R`
  (the abstract's within-query action variance) — the conflation is repaired
  everywhere.

---

## 4. Proposition 7 — withdrawn as a theorem, restated as an empirical rule

### What was wrong
- N3HW Q2 (circularity): the sign of the leading coefficient was fixed by
  the very sign reversal (Test 2) it claimed to predict, via a hardcoded
  indicator with an NQ-fitted threshold inside a "derived" expansion. A
  population-level Taylor expansion cannot contain a dataset-fitted step
  function; conceded.
- PAT's sign error in the MRDR-RuleOPE subtraction: the K_a/N coefficient
  is `(c'_2 - c_2)`, not `(c_2 - c'_2)`; fixed.
- E5b (forced-spread cells, sigma^2_R in [0.10, 0.21]): MRDR wins 0/18 —
  the "widening gap in the high-sigma^2_R regime" prediction *fails* on the
  one axis where the branch is populated. Within-benchmark, higher sigma^2_R
  in fact favours RuleOPE (amplified-HotpotQA Spearman rho = -0.691).

### Corrected Proposition 7' (empirical decision rule, not a theorem)
The `sigma^2_R > HI -> MRDR` branch is a *cross-benchmark empirical
regularity* of the calibration distribution (it separates NQ-like cells,
where high sigma^2_R co-occurs with high R^2(A3), from the rest), not a
causal consequence of sigma^2_R. The revised text:
- states the selector as an empirical decision rule calibrated on the audit
  grid, with the E1 out-of-sample validation (LOLO 80.6% vs 55.6% baseline)
  as its evidence,
- reports E5b as a negative result bounding the mechanism's scope: sigma^2_R
  *in isolation* does not hand the win to MRDR; the coupling with reward
  compositionality (R^2(A3), cf. E3's M3a interaction, p = 3.2e-08) does,
- keeps the honest limitation that the high branch never fires on the
  natural distribution (max observed sigma^2_R = 0.094), with E5's
  saturation analysis explaining *why* (clipping + zero-swing queries cap
  natural sigma^2_R near 0.09).

---

## 5. Smaller formal repairs (PAT minor list, all conceded and fixed)

- Uniform logging normalisation: `pi_0(a|x) = gamma / (K_a + 1)` summed over
  the full action space (not `gamma / K_a`); B2's balanced coverage is
  `1/K_a` over non-noop actions.
- B1's firing-rate variable `p_rho` (marginal rule firing rate) now defined
  where first used.
- Rule tuple: dead third element removed (rules are (predicate, action)
  pairs; `rule_dsl.py` ground truth).
- Pseudo-reward imputation `r-tilde` in Eq. 1 now defined: the fitted
  atom-model reward `g(X, a_target)` evaluated under the correction gate —
  see `rule_ope.py:173-189` (heuristic mode) and the closed-form bridge term
  (EIF mode, `rule_ope.py:162-172`); both printed in the revised App. H with
  the informativeness-gate clipping bounds (`clip(w, 0.25, 4.0)`) that answer
  the division-by-zero question.
- Atom vocabulary: V = 48 everywhere (the "45" in Sec. 2 was stale; the DSL
  has 48 atoms; Bonferroni denominators updated).
- Theorem-4 kernel `C-tilde = C / lambda_max(C)` naming collision resolved as
  in App. C.4.
- Proxy reward (Test 4) now defined: alias-max token-F1 (the sweep protocol
  reward, `experiments/rebuttal/common.py::_f1`).
- Checklist Q7: Table 3 reference corrected (LOLO-CV/RMSE, no F-statistics).
