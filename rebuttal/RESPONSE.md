# Point-by-point response — NeurIPS 2026 submission 33107

Every empirical claim below has a reproducing script in
`experiments/rebuttal/` and a result artifact in
`experiments/results/rebuttal/` on this branch. Theory corrections are
stated in full in `rebuttal/THEORY_FIXES.md`; count reconciliations in
`rebuttal/AUDIT_RECONCILIATION.md`. New experiments are labelled E1-E10.

---

## Meta review — the three major issues

**1. "Theoretical issues: potential errors, circular claims, gaps between
theorems and high-level claims."** All four substantive objections were
correct, and all four are now repaired with validated replacements:

- *Theorem 4 (commensurability)*: the PD premise was impossible
  (rank(V_beta) <= 2; confirmed in 36/36 cells, E9) and the proof's
  centring step was wrong. Corrected Theorem 4' keeps what was true — both
  sigma^2_R and Delta^2 are trace functionals of the same LLM-dependent
  V_beta against LLM-independent kernels — states vanishing as
  one-directional (provably free of extra assumptions for ridge fits, and
  holding on-range in 36/36 cells with margin), and *withdraws* the
  universal two-sided sandwich: the empirical kernel ratio spans
  [0.03, 108], so sigma^2_R is a vanishing-linked, rank-aligned proxy
  (Spearman 0.971 with the signal functional, E9), not a calibrated one.
- *Theorem 3 and the NQ deepening*: the sign contradiction was real. The
  corrected Theorem 3' keeps the positive O(1/N) variance-geometry term
  and derives the previously-discarded N-independent bias asymptote
  a = bias^2_NC - bias^2_RO. Fitting gap(N) = a + b/N to the five cached
  sample sizes: median R^2 0.886, b > 0 in 36/36 cells, and sign(a)
  predicts sign(gap at N=2400) in 36/36 cells — a < 0 in exactly the 8 NQ
  cells that deepen (E8). The mechanism is now derived and validated, not
  asserted.
- *Theorems 2/8/9 (aggregate variance)*: the sublinear-in-R aggregate claim
  was algebraically impossible and is withdrawn; the correct statements are
  per-rule variance independent of R and aggregate Theta(R d/N), matching
  our own lower bound. The O(A+V) generator-call complexity claim is
  unaffected.
- *Proposition 7*: circular as charged; withdrawn as a theorem and restated
  as an empirical cross-benchmark decision rule — with a new negative
  result (E5b) honestly bounding it: on forced-spread cells with
  sigma^2_R in [0.10, 0.21], MRDR wins 0/18, so high sigma^2_R alone does
  not flip the ranking; the coupling with reward compositionality does
  (M3a interaction p = 3.2e-08, E3).

**2. "How is the auxiliary estimator used in deployment — what information
is required and how is it obtained?"** The revised Sec. 6 recipe, now fully
explicit and demonstrated end-to-end on observables only (E6c):

1. Collect the deployment logs (no replay needed for this step).
2. Draw a pilot of N_pilot = 150 logged queries; obtain gold labels for
   those queries only; replay all 3 actions on the pilot
   (150 x 3 = 450 generator calls — vs ~7,939 for full per-rule replay).
3. Compute sigma-hat^2_R from the pilot replay (mean over queries of the
   variance of the three per-action rewards).
4. sigma-hat^2_R < 0.05 -> RuleOPE; > 0.10 -> MRDR; otherwise run both
   estimators *on the pilot at N = 150* and keep the lower-MSE one
   ("at deployment N" was a typo both reviewers caught — it is the pilot
   size).
5. Run the chosen estimator on the full logs.

No oracle quantities appear anywhere: E6c executes exactly this loop on
single 150-query draws and gets mean 70.3% correct selection across the 19
extended middle-band cells (majority-of-20 draws: 16/19), vs 55.6% for
always-RuleOPE.

**3. Presentation.** Sec. 1-2 are rewritten around a concrete problem
setup (drafted below for 39jG), every coined term is either defined at
first use or renamed to a standard one, and the notation overloads are
eliminated (rename table in THEORY_FIXES.md).

---

## Reviewer w7Aq (borderline reject)

**Q1 — "Why is V_beta positive definite in 45 dimensions? Nonzero sigma^2_R
with Delta = 0 seems possible."** You are right on both counts, and we have
corrected the theorem rather than defended it. V_beta has rank <= 2
(verified 36/36 cells, E9), so the PD premise was unusable; and because
Delta^2 traces V_beta against a rank-one kernel, Delta^2 = 0 with
sigma^2_R > 0 is genuinely possible (your counterexample is now a remark).
Corrected Theorem 4' claims only: shared PSD-trace structure;
V_beta = 0 => both vanish; sigma^2_sig = 0 => both vanish (this direction
is assumption-free for ridge fits, since coefficients live in
range(Sigma), where Sigma is PD — rank 20-22/48 with smallest nonzero
eigenvalue 5.8e-3-0.18). The two-sided commensurability with universal
constants is withdrawn. Details: THEORY_FIXES.md §1, e9_kernel_check.csv.

**Q2 — "How is pilot MSE computed without oracle values or all-action
replay?"** It does use all-action replay — but only on the 150-query pilot
(450 generator calls, ~2% of the cost of full per-rule replay), plus gold
labels for those 150 queries. Nothing else. The revised Sec. 6 states this
cost explicitly, and E6c demonstrates the full loop on observables only
(70.3% single-draw, 16/19 majority-of-20). We agree the submitted text
obscured this and have rewritten step 4 (the "at deployment N" phrasing
was an error — the pilot is evaluated at N = 150).

**Q3 — "Can the selector be frozen and tested on new generators,
benchmarks, rule pools, and action sets?"** We ran exactly this (E1, E7):

| transfer axis | frozen selector | always-RuleOPE |
|---|---|---|
| leave-one-LLM-out (12 folds, threshold refit per fold) | 29/36 = 80.6% | 55.6% |
| 34 held-out cells (MuSiQue, 2Wiki, 14B/32B anchors), thresholds fit on in-grid only | 23/34 = 67.6% | 61.8% |
| new 500-rule pool (rules_v2, depth 2-3, name-disjoint), paper thresholds verbatim | 11/12 = 92% | 7/12 |
| leave-one-benchmark-out | 21/36 = 58.3% | 55.6% |

Generator-, pool-, and scale-transfer hold; benchmark-level transfer is the
honest weak spot (2Wiki compresses sigma^2_R toward zero) and is now stated
as a limitation rather than discovered by the reader. A new action set
requires new logs by definition; we scope the claim accordingly.

**Limitations (K=0 case, pilot information).** Both added: K_f-fold
cross-fitting requires N >= K_f per fold (trivially met at N_pilot = 150);
the pilot's information requirements are itemised in the recipe above.

---

## Reviewer N3HW (reject)

**W1/Q1 — Theorem 3 vs the NQ deepening.** Conceded and repaired; see meta
item 1 and THEORY_FIXES.md §2. The residual you asked us to derive is the
N-independent bias-difference asymptote; its sign is negative in exactly
the 8 deepening NQ cells and its N-dependence is constant (the 1/N term is
positive everywhere), validated at median R^2 0.886 (E8).

**W2/Q2 — Proposition 7 circularity.** Conceded; withdrawn as a theorem
(meta item 1, fourth bullet). We add E5b as an adversarial test of our own
mechanism: it fails on the axis where sigma^2_R is manipulated in
isolation — the revised paper says so.

**Q3 — value of c-hat.** Reported: median 0.702, 5-95th percentile
[0.065, 69.9] across the 36 cells (E3). The spread is why the calibration
claim is withdrawn (sigma^2_R is rank-aligned with Delta^2's driver, not
calibrated to it) — your "only sign-related" reading was the correct one.

**Q4 — out-of-sample selector validation.** Done, with threshold refit per
fold (grid search LO in [0, 0.10], HI in [0.02, 0.20]): LOLO 80.6%, LOBO
58.3%, frozen-threshold transfer to 34 held-out cells 67.6% (all vs 55.6%
/ 61.8% always-RuleOPE); plus a new rule pool at 92% (E1, E7). The
in-sample framing of the submitted audit is corrected throughout.

**Q5/L1 — "Under uniform-stochastic logging RULEOPE collapses to
compositional-DR, so the motivating deterministic regime is untested."**
Correct, and this was the most important gap you identified. E2 runs the
deterministic-logging regime with the correction channel active
(propensity-1 noop logging, C ~ Bernoulli((1-r_noop)/beta)): RuleOPE
separates from its DR backbone (median |gap| 6.6% MSE, beats it in 20/36
cells), conditional ranking persists (trivia -> RuleOPE 11/12; hotpot 8/12
and NQ 7/12 -> MRDR at N=1200), and the sigma^2_R association persists
(rho = 0.374, p = 0.025) — but the uniform-regime thresholds transfer
poorly (56%; regime-refit threshold reaches 83.3%). The revision presents
uniform logging as the identification-clean benchmark regime, the
deterministic regime as the deployment regime with regime-specific
calibration, and the collapse property as an explicit design note in §3.

**Q6/W-clarity — undefined coined terms.** "Commensurability sandwich" is
gone (the theorem it named was wrong; its replacement is stated in
standard trace-functional language). "Bridge-fusion correction" is now
"correction-fusion term", defined by its equation (the gated pseudo-reward
`c * gate * (r-tilde - m_rho)`, App. H, with the gate's clipping bounds).
Every remaining term is defined at first use; the Sec. 1 rewrite (below)
opens with a two-paragraph plain-language problem setup.

**L2 — "A3 has near-zero explanatory power on NQ."** This premise is
inverted on our data: the within-query R^2 of the A3 working model is
*highest* on NQ — mean 0.449, max 0.603 across the 12 NQ cells, vs 0.097
(HotpotQA) and 0.048 (TriviaQA) (E3, protocol identical to the shipped
a3_validation JSONs, which it reproduces at 0.449 vs 0.460). We suspect
the reading arose from the incremental-R^2 panel in Fig. 8; the revision
reports R^2(A3) per benchmark in the main text so the number is
unambiguous. Where A3 *is* weak (TriviaQA/2Wiki), that weakness now enters
the theory explicitly: adding a sigma^2_R x R^2(A3) interaction to M3
lifts adj-R^2 from 0.898 to 0.964 (p = 3.2e-08) — the mechanism is
A3-gated, and NQ is its best case, not its weakest.

**L3 — in-sample selector validation presented as a deployment recipe.**
Conceded and fixed (Q4 + meta item 2).

**Significance — "reduces to a known N-dependent bias-variance tradeoff;
selector adds little over a constant baseline."** The revision claims
exactly that mechanism — the contribution is showing *which* deployment-
observable statistic indexes where the tradeoff bites (sigma^2_R,
rank-aligned with the structural functional at rho = 0.971), that the
resulting selector transfers out-of-sample (80.6% LOLO / 92% new-pool vs
55.6% / 58.3% for the constant baseline), and that single-generator
single-benchmark reporting would have hidden the 8 NQ reversals entirely.
We believe the corrected framing — an empirical phenomenon with a
validated two-term mechanism, not a new estimator — is the honest and
useful contribution.

---

## Reviewer 39jG (strong reject — clarity)

We accept the core criticism: the submission assumed OPE background it
never provided. The revised Sec. 1 now opens with (verbatim draft):

> *A RAG system answers a query by retrieving documents and letting a
> language model generate from them. Teams routinely propose rules that
> change retrieval — "if the top passages disagree, re-rank them", "if
> retrieval confidence is low, filter the context". Shipping a rule to
> find out if it helps is slow and risky, so we want to estimate each
> rule's value from logs the system already produced. This estimate-from-
> logs problem is off-policy evaluation (OPE): the logs were produced by
> one policy (the current system) and we must value another (the rule)
> without deploying it. Many OPE estimators exist; this paper is about
> choosing between them. Our finding is that the choice genuinely depends
> on which language model generates, which benchmark the queries come
> from, and how many logged queries you have — the same two estimators can
> swap places when any of the three changes. We call this conditional
> ranking, we exhibit a cheap statistic that predicts most of the
> swapping, and we give a procedure that uses it.*

All coined terms are defined or renamed (N3HW Q6 above), "linear PSD-trace
functionals" is now accompanied by its one-line definition, and every
symbol overload is removed (rename table in THEORY_FIXES.md). We would
welcome a re-read against the revised manuscript.

---

## PAT (automated) — erratum status

All confirmed items are folded into the fixes above; the remainder:
normalisation gamma/(K_a+1) and B2 = 1/K_a (fixed); p_rho defined; dead
rule-tuple element removed; r-tilde defined with gate clipping bounds
(division-by-zero answered); V = 48 atoms everywhere; sigma2_eps vs
sigma^2_R disambiguated (Thm 8); C-tilde scaling collision resolved;
Prop 7 subtraction sign (c'_2 - c_2) fixed; proxy reward defined
(alias-max F1); Fig 5 legend corrected to the sigma-model numbers
(0.80/0.97/0.90); Table 8 replaced by the 10-seed e4 version with the
regularization-by-degradation explanation (text now matches data; the
"never inverts" sentence is gone); Table 9 gains the NQ column (e6a);
A3 t-statistics described as cross-fitted (held-out) residual tests;
checklist Q7's Table 3 reference corrected; middle-band counts reconciled
(AUDIT_RECONCILIATION.md).

---

## Summary of new evidence on this branch

| experiment | answers | headline |
|---|---|---|
| E1 selector CV | N3HW Q4, w7Aq Q3 | LOLO 80.6%, frozen transfer 67.6%, vs 55.6%/61.8% baseline |
| E2 deterministic regime | N3HW Q5/L1 | correction channel active; ranking + mechanism persist; calibration regime-specific |
| E3 A3 + c-hat + M3a | N3HW L2, Q3 | R^2(A3) on NQ = 0.449 (highest); c-hat median 0.702; M3a adj-R^2 0.964 |
| E4 noise ablation (10 seeds) | Table 8 contradiction | direction reproduces per-cell; mechanism = correction-gate collapse; text corrected |
| E5/E5b high-sigma branch | Prop 7 | branch unreachable naturally; forced-spread MRDR 0/18 -> claim withdrawn |
| E6a Table 9 NQ | missing column | +22.3% at N=150 decaying to -12.2% at N=1200 |
| E6b MuSiQue anchors | audit coverage | rank flip at scale on both frontier anchors |
| E6c observable pilot | w7Aq Q2, meta 2 | 70.3% single-draw, 16/19 majority-of-20, zero oracle access |
| E7 new rule pool | w7Aq Q3 | frozen thresholds 92% vs 58% baseline |
| E8 gap crossover | N3HW Q1, meta 1 | gap = a + b/N; sign(a) predicts deepening 36/36 |
| E9 kernel checks | w7Aq Q1, meta 1 | rank(V_beta)=2 everywhere; Sigma PD on-range; audit stat rank-aligned (0.971) |
| E10 accounting | all count objections | 54/17 canonical; 70/19 extended; protocols rho=0.991 |
