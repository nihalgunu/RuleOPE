# 6  Benchmark

## 6.1 Design goals and decisions

The benchmark has three design goals. (1) *Reproducibility without
multi-GPU inference.* Reviewers should be able to rebuild the benchmark
in minutes on a laptop; running Llama-3-8B over KILT and HotpotQA is not
a reproducibility baseline. (2) *Known counterfactual ground truth.*
MSE against truth is only defined if we know $V(\rho)$ exactly. Real
RAG pipelines do not let us replay any conceivable rule's
counterfactuals without re-running the generator under each intervention
— an astronomical compute cost across 500 rules and 4000 queries.
(3) *Statistical texture matching real logs.* Per-feature marginals
(top-1 reranker score mean, score-gap distribution, source-tag
frequencies) should be close to published BEIR and KILT numbers.

These goals push us to a synthetic substrate with (a) an explicit
counterfactual-reward function $R(x, a)$ calibrated to known feature
distributions, and (b) a fixed random seed that freezes the benchmark.
We discuss in §6.4 why we preferred this over a partial-real-data
substrate (KILT + simulated rewards).

## 6.2 Substrate data-generating process

For each of $N = 4000$ queries we sample:

* Query-level features: $q_{\text{len}} \sim 1 + \text{Poisson}(12)$,
  $q_{\text{multihop}} \sim \text{Bern}(0.25)$, entity indicators
  $q_{\text{person, place, org, time, num}}$ with marginal probabilities
  $(0.40, 0.35, 0.20, 0.15, 0.25)$, and $q_{\text{ppl}}$ log-normal with
  $(\mu, \sigma) = (3.0, 0.4)$.
* A latent retrieval-quality score $\ell = \text{clip}(0.5 + 0.25 \sum_j q_{\text{ent}_j} - 0.5 q_{\text{multihop}} - 0.01 (q_{\text{ppl}} - 20) + 0.15 \epsilon)$,
  $\epsilon \sim \mathcal{N}(0, 1)$.
* Retrieval features: $\text{top1\_score} = \sigma(1.5 \ell + 0.5 \epsilon_1) + 0.08 \epsilon_2$,
  $\text{top2/3\_score}$ decreasing in expectation; source tag sampled from a
  softmax depending on $q_{\text{place}}$ and $q_{\text{ppl}}$.
* Derived features: $\text{score\_gap}$, $\text{redundancy} \sim
  \text{Beta}(1.2, 2.0)$, count of strong candidates, etc.
* Counterfactual rewards $R(x, a)$ for $a \in \{\texttt{noop},
  \texttt{filter}, \texttt{rerank}, \texttt{abstain}\}$, each a sigmoid
  of a linear combination of latent quality and action-dependent
  features. The abstain reward is fixed at $r_{\text{abstain}} = 0.5$.
* Logged action $A$ drawn from $\pi_0$ (default:
  $\pi_0 = (0.70, 0.15, 0.10, 0.05)$ over noop/filter/rerank/abstain;
  alternative: deterministic noop for the R2 regime).
* Observed reward: $R_{\text{obs}} = \text{clip}(R(x, A) + 0.03 \epsilon, 0, 1)$.

The benchmark's `_latent` field is retained only for internal
validation and is never included in the features exposed to estimators.

## 6.3 Rule enumeration and ground truth

The atom vocabulary ($d = 48$) is listed in `src/rule_dsl.py` and covers
query-level, passage-level, and list-level features. We enumerate
conjunctive rules of depth 1–3 using
`rule_enumeration.select_rules_from_logs`, filter to rules firing on at
least $20$ logged queries, and stratify a sample of 500 rules across
depths $(1, 2, 3)$. The ground-truth value of each rule is computed
exactly by averaging the counterfactual reward
$R(X_i, \pi_\rho(X_i))$ over all $N$ logged records.

## 6.4 Why synthetic, and why not KILT+BGE+Llama

We considered three substrates:

1. *Full real pipeline.* BGE retrieval on KILT + a public cross-encoder
   reranker + Llama-3-8B answer generation, with rule interventions
   requiring a fresh generator call per rule-query combination. At 500
   rules $\times$ 4000 queries that is $2 \times 10^6$ generator calls, i.e.,
   multi-GPU-week compute at the benchmark-release stage, and replicating
   the benchmark requires the same compute again. This defeats the
   reproducibility goal.

2. *Hybrid.* Real retrieval (BGE on KILT), simulated rewards from a
   regression on actual log data. This is attractive but has two
   problems: (a) calibrating the reward simulator against real judgements
   is itself a research project; (b) the counterfactual reward is then
   produced by the same modelling family we use inside the estimator,
   which overstates estimator performance.

3. *Fully synthetic (our choice).* We calibrate feature marginals to
   BEIR + KILT published statistics but define rewards as an explicit
   closed-form function. This gives us exact ground truth, is cheap to
   reproduce (minutes on a laptop), and avoids reward-model confounding
   between benchmark and estimator. The cost is that we do not *prove*
   generalisation to real pipelines; for that we plan a supplementary
   (non-frozen) empirical study in the camera-ready version, running
   RuleOPE on logs from a deployed RAG system.

## 6.5 Correction regimes

We provide three noise regimes: $0\%, 10\%, 30\%$ random flips applied
on top of the base-rate correction model of §6.2. The base-rate model
sets $P(C = 1 \mid X, A) = \sigma(\log(r_0/(1-r_0)) + 4(1 - R_{\text{obs}} - 0.5))$
with $r_0 = 0.15$, so corrections concentrate on bad answers but
triggered only on $\sim 15\%$ of records overall (matching the
production regime in Phyvant-style logs).

## 6.6 Artifacts and freeze

The following files are frozen at commit time:
* `eval/benchmark_v1.jsonl` (public logs, cf rewards stripped),
* `eval/benchmark_v1_with_cf.jsonl` (private, for ground-truth),
* `eval/rules_v1.jsonl` (500 rules),
* `eval/ground_truth_rule_values.json`,
* `eval/correction_logs_noise_{00, 10, 30}.jsonl`,
* `eval/MANIFEST.json` with SHA-256 checksums.

After freeze, no file is modified. Experimental code that reads these
files must not write to them; this is enforced by convention and by a
CI check in `tests/test_freeze.py`.
