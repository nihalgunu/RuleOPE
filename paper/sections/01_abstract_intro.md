# Abstract

Production retrieval-augmented generation (RAG) systems are routinely updated
by the addition of deterministic *rules* that filter, re-rank, or veto the
retrieval step based on boolean predicates over query and passage features.
There is, however, no principled way to decide *which* rule to ship. Random-
ised A/B tests are expensive, slow, and unsafe for rules that affect rare but
high-cost queries. Existing off-policy evaluation (OPE) estimators are either
specialised to stochastic or continuous-action bandits, or treat the retrieval
step as a slate recommender, ignoring the combinatorial structure of rules.

We introduce **RuleOPE**, an offline evaluation *framework* for rule-based
interventions in RAG with five tightly coupled contributions: (i) a
formalisation of rule-based interventions as conjunctive boolean policies
over a fixed atom vocabulary; (ii) a doubly-robust estimator whose reward
regression is factorised over atomic predicates, so rules sharing atoms
share regression parameters and the total variance across a rule set of
size $M$ scales as $O(d)$ (atom vocabulary size) rather than $O(M\,d)$;
(iii) a no-replay identification theorem for $V(\rho)$ under a
compositional-reward assumption A3, together with an empirical
validation of A3 on HotpotQA where the assumption explains 67\% of
within-query reward variance with no atom-level residual dependence
surviving Bonferroni correction (§5C.1); (iv) on three real-data
benchmarks --- HotpotQA, TriviaQA, MuSiQue --- in the small-$N$ regime
($N \in \{150, 300\}$ query--reward pairs, matching the realistic
deployment budget), RuleOPE achieves **statistically significant MSE
reductions of 15\%--67\%** over the OBP-style NonCompDR baseline
(Saito et al.\ 2021), with mechanism-isolating ablations showing that
compositional atom-sharing alone drives the gain; and (v) a
**plug-in rule-discovery loop** (§9B) --- enumerate, score with
RuleOPE, select via ERM-argmax --- that on HotpotQA's
3528-candidate depth-$\le 2$ space recovers the oracle-best rule at
median zero regret (mean $0.018$) and improves over a 500-rule
hand-curated baseline by $0.007$ at $N{=}150$, elevating RuleOPE
from an *estimator* to a *rule-learning framework*. We also
release a frozen synthetic benchmark of 500 rules with exact
ground-truth values and three correction-noise regimes for
reproducible methodology testing.

# 1  Introduction

Modern retrieval-augmented generation systems are iterative. A baseline
pipeline (retrieval, reranking, answer generation) is deployed; post-launch,
engineers discover that specific query classes are served poorly
("multi-hop questions over Wikipedia stubs", "queries whose top-1 reranker
score is below $0.3$"); they write a *rule* that conditionally modifies the
pipeline — dropping the offending passage, reranking, or forcing the model
to abstain — and ship it. In a mature system hundreds of such rules
accumulate, most contributed without a controlled experiment because the
classes they target are individually too small to power an A/B test within
reasonable time.

The resulting engineering workflow begs for offline evaluation. Given logs
of the pre-rule system — queries, retrieved passages, generated answers,
and a sparse stream of post-hoc expert corrections — can we estimate the
counterfactual reward of a candidate rule without deploying it? This is a
natural off-policy evaluation (OPE) problem, but the structure of
rule-based interventions is not captured by any existing estimator:

* **Rules are deterministic compositions of atoms.** The universe of
  candidate rules is combinatorial ($|\mathcal{V}|^D$ conjunctions at depth
  $D$); two rules that share an atom should have correlated estimator
  behaviour because their counterfactuals share structure.
* **Production logs are deterministically generated.** The logging
  "policy" is not a draw from a distribution over actions; it is the single
  default pipeline. Classical importance-sampling estimators divide by
  zero on almost every counterfactual, and clipped variants (CIPS) trade
  variance for uncontrolled bias.
* **The supervision signal is a sparse correction flag.** The only source
  of counterfactual information beyond the regression model is a binary
  signal asserting the logged answer was wrong; it provides partial
  identification of the reward under alternative actions but is not an
  observed counterfactual reward.

We propose RuleOPE, an estimator whose three components — a compositional
reward regression, a logging-action doubly-robust correction, and a
correction-driven debiasing term — are each derived from the specific
structure of the rule-OPE problem. We prove consistency under standard
ignorability assumptions plus a new *correction unconfoundedness*
assumption, exhibit a variance bound that is sublinear in the rule-set
size, and release a reproducible benchmark of $500$ rules derived from a
calibrated synthetic substrate that mimics the marginal feature statistics
of standard BEIR retrieval pipelines.

## 1.1 Contributions

1. **Problem formulation** (§3). We formalise rule-based interventions
   as conjunctive deterministic policies over a fixed atom vocabulary
   with actions in $\{\texttt{filter}, \texttt{rerank}, \texttt{abstain}\}$,
   and make explicit the role of sparse correction signals as a
   partial-identification resource for the deterministic-logging regime
   in which production RAG pipelines actually operate.

2. **RuleOPE estimator** (§4). A doubly-robust estimator with
   compositionally factorised reward regression (identification by A3;
   §5) and an optional correction-fusion term that delivers a strict
   semiparametric variance reduction under the proxy-style bridge
   assumption A5 whenever logging is stochastic on the target action
   or a pilot sample is available (§5.4, revised). The compositional
   factorisation reduces the cross-rule variance contribution of the
   regression from $O(M \cdot d)$ to $O(K \cdot d)$ where $d, K$ are
   fixed.

3. **DualShrinkOPE** (§4.6 and Theorem 3). Bayes-optimal per-rule
   convex combination of RuleOPE and the Direct Method, with an
   empirical-Bayes estimator of the weights. Dominates each constituent
   in MSE; provides the primary variance-reduction channel in the
   production regime.

4. **JointRuleOPE** (§4.7 and Theorem 4). Cross-rule random-effects
   shrinkage toward an atom-compositional target. Dominates independent
   per-rule estimation in joint MSE.

5. **CRRM and compositional pessimistic selection** (§4.8, Theorems 5--6).
   A rule-learning extension with a Rademacher-based regret bound
   scaling with atom sparsity $s$ and atom-vocabulary size $d$, not with
   rule-space size $M$. The compositional LCB strictly improves upon
   the union-bound LCB whenever $s < \log M / \log d$.

6. **Benchmark and empirical results** (§6--§9). `rule-ope-benchmark-v1`
   is a frozen collection of 4000 queries, 500 rules with exact
   ground-truth values, three correction-noise regimes, and a
   reproducible synthetic substrate. In the production-realistic
   deterministic-logging regime, RuleOPE reduces MSE by 10--23\% over
   every classical baseline (which all coincide in this regime) at
   rule-set sizes up to 5000; DualShrinkOPE provides a lower-variance
   alternative at 6--14\% reduction.
