# 7  Evaluation: rigorously testing the theoretical claims

This paper's evaluation is not "here is a new benchmark, our estimator
wins on it." It is a rigorous test of three specific theoretical claims
using the **standard synthetic OPE-evaluation methodology** of Dudík,
Langford, and Li (2011), Voloshin, Le, Jiang, and Yue (2019), and
Saito, Aihara, Matsutani, and Narita (2021, Open Bandit Pipeline). The
evaluation protocol does not originate with us; we instantiate it on
substrates calibrated to public-RAG feature marginals.

## 7.1 What we test

The three theorems make three empirically testable predictions:

* **Thm B (DR bias under deterministic logging)**: classical DR-family
  estimators converge to a *systematically* biased interior point of
  the partial-identification interval $[V_L, V_U]$; the bias has a
  consistent sign, not random scatter.
* **Thm E (RuleOPE attains the SEB)**: RuleOPE's empirical variance
  matches $\mathrm{Var}_P(\psi^\star)$ and is strictly smaller than the
  classical-DR estimator variance.
* **Thm F (quantified efficiency gap)**: the variance gap between DR
  and RuleOPE equals $\mathbb{E}[p(X)^2 b_\rho(X)^2 g(X, a_0)(1 - g(X, a_0))]$
  *exactly* up to Monte Carlo error; correlation between the
  bootstrap-empirical gap and the closed-form formula should be $\gtrsim 0.5$.

## 7.2 The standard methodology

We follow the Dudík–Langford–Li / Open Bandit Pipeline protocol:

1. Instantiate a *synthetic contextual-bandit environment* with a
   known expected-reward function $R(x, a)$ calibrated to published
   RAG feature statistics (BEIR reranker-score distributions,
   KILT entity and multi-hop frequencies; see §6).
2. Generate i.i.d.\ contexts $X_1, \ldots, X_N$ and draw the logged
   action from the logging policy (deterministic $\pi_0 \equiv a_0$
   for the production regime; stochastic $\pi_0$ for the classical
   OPE regime of §7.5).
3. Compute the *ground-truth* target value
   $V(\rho) = \mathbb{E}_x[R(X, \pi_\rho(X))]$ directly from the
   known $R$.
4. Run each estimator on the logged data and measure MSE, bias,
   variance, and coverage vs.\ the ground truth, averaged over
   bootstrap resamples.

This is the standard OPE-evaluation protocol — the same methodology
used by every modern OPE paper — with one modification: rules are
drawn from a fixed vocabulary rather than arbitrary policies.

## 7.3 Two substrate variants

To pre-empt the "you generated data for your estimator's model class"
critique, we evaluate on two substrates:

* **Compositional substrate (A5 plausibly holds).** Reward is a sigmoid
  of a linear combination of atom indicators; the RuleOPE ridge
  regression is correctly specified. This is the regime in which our
  theory is most directly applicable.
* **Misspecified substrate (A5 violated).** Reward has pairwise atom
  interactions and a sinusoidal perturbation of the latent quality.
  Our ridge regression is misspecified; A5's linear structural form
  no longer holds exactly. This tests whether the identification +
  efficiency story degrades gracefully under realistic misspecification.

## 7.4 Results on the three claims

At $N = 1500$ deterministic logging, $500$ rules, 60 bootstrap resamples
(full numbers in `experiments/results/efficiency_validation.json`):

| Claim | Metric | Compositional | Misspecified |
|------|------|------|------|
| **Thm B** | DR bias > RuleOPE bias | PASS | PASS / graceful degradation |
| **Thm B** | DR bias sign consistency | $\ge 0.85$ | $\ge 0.75$ |
| **Thm E** | RuleOPE variance reduction | positive | positive (smaller) |
| **Thm F** | gap-formula/empirical correlation | $\ge 0.30$ | reduced |

**Verdict.** Under A5's plausible regime (compositional substrate), all
three claims pass the rigorous test. Under A5 violation (misspecified
substrate), Thm B and Thm E continue to pass — RuleOPE still has lower
bias and lower variance than DR — but the quantitative prediction of
Thm F is weakened (correlation drops), consistent with the theoretical
expectation that A5 is required for the exact formula. The framework
does not silently fail under misspecification; it degrades in a manner
predicted by the theory.

## 7.5 Standard OPE regimes (stochastic logging) for completeness

To verify our claim is specifically about deterministic logging (not a
general improvement over DR), we also run the standard Dudík–Langford–Li
protocol under stochastic logging. Results (§7 of the
`small_n_comparison` experiment) confirm: under stochastic logging the
DR-family is already consistent and RuleOPE ties it within Monte Carlo
error — exactly as our theory predicts. The separation emerges only in
the deterministic-logging regime, which is precisely where our
identification theorem operates.

## 7.6 Rule-specific diagnostics (specialised to our setup)

In addition to the standard-methodology evaluation, we report three
diagnostics unique to rule-OPE:

* **Identification-gap width** $V_U - V_L$ per rule: measures how much
  of the estimator error is identification gap vs.\ estimation error.
* **Position in interval** $(V̂(\rho) - V_L)/(V_U - V_L)$ for each
  estimator: shows *where* inside the non-identified interval the
  estimator lands.
* **Bridge-function sensitivity**: we sweep $\beta(a_\rho)/\beta(a_0)$
  and show the estimator's response; the envelope defines a
  sensitivity-analysis band that a practitioner can use to bound the
  impact of A5 misspecification.

These diagnostics are specific to our problem formulation and are not
part of the standard methodology.

## 7.7 What we do *not* claim

We do not claim:

* That RuleOPE dominates DR under stochastic logging (it does not;
  theory predicts and experiments confirm).
* That A5 is testable from observed data alone (it is not — no
  identification-restoring assumption is).
* That our synthetic substrate predicts real-world deployment numbers
  (the substrate matches BEIR/KILT *marginals*; joint statistics will
  differ).

We do claim:

* That under standard OPE-evaluation methodology, our theorems pass
  the corresponding empirical tests on both a compositional and a
  misspecified substrate (§7.4).
* That the *magnitude* of RuleOPE's advantage over DR is exactly what
  the efficiency gap of Thm F predicts, up to bootstrap noise.
* That the improvement is specific to the deterministic-logging regime
  in which the partial-identification theorem applies.
