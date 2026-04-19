# Appendix B  Decisions log

This appendix collects every material design decision taken in the
project, with the reasoning behind it. It is intended as a record for
reviewers who want to understand why we did *not* take apparently
natural alternatives.

## B.1 Rule DSL

**Decision.** Rules are *conjunctive* boolean clauses over a fixed atom
vocabulary of size 48. Depth at most 3.

*Alternative considered:* general CNF/DNF, decision lists, small
decision trees. *Reason:* conjunctive clauses match the form in which
production engineers actually write rules, and their compositional
structure is the direct target of Theorem 2's variance bound. Depth 3
covers 99% of rules we have seen in practice; going higher blows up
the rule space without adding interpretable rules.

**Decision.** The atom vocabulary is fixed and shared between
estimators and rules.

*Alternative considered:* learning atoms from features (e.g., rule
extraction à la RIPPER). *Reason:* a fixed vocabulary lets the
compositional regression parameters be exactly the objects the theory
bounds. Learned atoms would require a separate complexity analysis.

**Decision.** Actions are $\{\texttt{noop}, \texttt{filter},
\texttt{rerank}, \texttt{abstain}\}$.

*Alternative considered:* parameterised actions (e.g., "filter with
threshold $\tau$"). *Reason:* each rule action must be deterministic
and atomic to keep the estimator well-defined. Parameterised actions
can be recovered by discretising the parameter and adding one atom per
threshold.

## B.2 Estimator

**Decision.** Compositional regression with ridge + cross-fitting.

*Alternative considered:* boosted trees, kernel ridge, or a deeper
neural network. *Reason:* ridge in a fixed-dimensional feature space is
root-$N$-consistent, satisfies the double-ML regime conditions
automatically, and gives closed-form variance bounds. The benchmark's
reward function is close to linear in atom indicators, so a richer
regressor adds variance without improving bias materially.

**Decision.** Correction-gate is a logistic regression on joint
(atom, action) features.

*Alternative considered:* a single global correction rate (no
conditioning). *Reason:* the gate needs to distinguish informative from
uninformative corrections; conditioning on $(X, A)$ is the minimum
viable structure for doing so while staying within the double-ML
regime.

**Decision.** Clip the gate at 5.

*Alternative considered:* self-normalisation, no clipping, adaptive
clipping. *Reason:* the gate is an *extrapolated* ratio under
deterministic logging and can blow up on corner cases. A fixed clip
is transparent and leaves the estimator's consistency properties
intact; adaptive clips would require additional theory.

**Decision.** Cross-fit with $K = 5$ folds.

*Reason:* Standard choice in the double-ML literature balancing
overfitting mitigation against held-out sample noise.

## B.3 Benchmark

**Decision.** Fully synthetic substrate.

*Alternative considered:* KILT+BGE+Llama, hybrid synthetic rewards on
real retrieval. *Reason:* exact counterfactual ground truth, cheap
reproducibility, clean separation between benchmark data-generating
process and estimator-family. See §6.4 for full discussion.

**Decision.** $N = 4000$ queries, 500 rules, 3 noise regimes.

*Reason:* $N = 4000$ gives enough minimum-support coverage for every
depth-3 rule in the benchmark at the $20$-firing threshold; 500 rules
is large enough to make scaling claims credible and small enough that
every Phase-3 experiment runs in under a minute on a laptop; three
noise regimes span the production range.

**Decision.** Include counterfactual rewards in a private `_with_cf`
log file but strip them from the public release.

*Reason:* estimators must not have access to ground truth, or the
benchmark is contaminated. The evaluation harness uses the private
file; estimator code reads the public one. Enforced by convention
and by a CI check.

## B.4 Experimental protocol

**Decision.** 3 seeds for the main comparison.

*Alternative considered:* 10. *Reason:* pilot runs showed inter-seed
std was small relative to inter-estimator gaps; additional seeds
would not change the ordering. We will increase to 10 for the
camera-ready version.

**Decision.** Top-$20$ tau as the ranking metric.

*Alternative considered:* full-rank Kendall tau, NDCG@$k$, precision@$k$.
*Reason:* the decision-theoretic operation a practitioner performs is
"pick the top-$k$ rules to ship", which is exactly what top-$k$ tau
measures. Full-rank tau puts equal weight on uninteresting rules at
the bottom.

**Decision.** Report bias, MSE, coverage, tau@$k$ together.

*Reason:* no single scalar captures rule-evaluation quality. Bias and
MSE speak to point-estimate quality; coverage to uncertainty
quantification; tau@$k$ to the downstream decision.

## B.5 Non-decisions (deferred to future work)

* Decision-list rules (see L2).
* Multiple correction streams (see L3).
* Vector-valued rewards (see L4).
* On-policy validation on a deployed RAG pipeline
  (camera-ready appendix).
* Counterfactual rule *learning* (the obvious CRM extension; out of
  scope for the evaluation-focused paper).
