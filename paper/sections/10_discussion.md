# 10  Discussion

## 10.1 When to use RuleOPE

RuleOPE is a conservative drop-in replacement for DR in rule-evaluation
workflows. On benchmarks where DR is already consistent and
low-variance (stochastic logging, well-specified regression), RuleOPE
does not underperform. In settings with *deterministic logging*,
*miscoverage of important query classes*, or *unstable retrieval*,
where classical DR collapses into pure DM, RuleOPE's correction-fusion
term provides strictly more information at the cost of a learnt gate.
Practitioners should deploy RuleOPE whenever a correction signal is
collected anyway (as in Phyvant-style production logs) — the marginal
cost is a single logistic regression on the same feature space as the
reward model.

## 10.2 Limitations

**L1: Calibration to real pipelines.** Our benchmark is synthetic.
The substrate's feature marginals are calibrated to published BEIR
and KILT statistics (§6.2), but the reward function is closed-form
rather than a real generator. We expect real pipelines to have
heavier-tailed reward distributions and more feature correlations.
The camera-ready version will add an empirical appendix on production
logs from a public RAG deployment (we will not use Phyvant data — the
paper is framed entirely around public substrates; see §11 for the
rationale and the separate production-adaptation roadmap).

**L2: Conjunctive-rule restriction.** The rule class is conjunctive
(CNF with one clause). Decision-list rules ("if $\phi_1$ then $a_1$
elif $\phi_2$ then $a_2$ else default") and general DNF rules are
natural extensions; the consistency argument generalises without
modification, but the variance bound of Theorem 2 needs tightening.

**L3: Single-correction model.** A4 is a strong assumption and F2
(§9.1) demonstrates that it can fail systematically without a clean
diagnostic. In production the correction distribution drifts with
reviewer population and policy changes; robust extensions using
multiple correction streams or independent audits are future work.

**L4: Scalar reward.** We assume one-dimensional $R \in [0, 1]$.
Multi-dimensional rewards (faithfulness, latency, cost) are common in
RAG; treating them jointly requires either a scalar aggregator (which
we defer to the practitioner) or a vector-valued DR extension.

## 10.3 Open problems

**Rule mining.** RuleOPE evaluates a given rule set. Learning the rule
set itself from logs — off-policy rule discovery — is the natural
extension. A counterfactual risk-minimisation variant (Swaminathan and
Joachims 2015) over the conjunctive rule space is immediate and gives
CRM-style generalisation bounds.

**Rule interaction.** When multiple rules are deployed simultaneously
their interactions matter. Our framework evaluates each rule against
the default policy; evaluating rule sets composed of multiple rules
that may overlap on the same query is an open problem, potentially
addressable via a chain-of-rules DR analogue.

**Bandits of rules.** A deployment that updates its rule set over time
is a contextual-bandit problem where the arms are rules; RuleOPE's
offline estimator is the natural warm-start policy value for such a
bandit. The connection to Bandit Superlearner algorithms (Kuusela et
al.\ 2024) is tight.

## 10.4 Ethics and release

The benchmark uses no human data. All query text is synthetic; no
personally identifying information exists in the released logs. The
substrate's source-tag distributions are calibrated to published
aggregate statistics, not to any specific user interactions.

Release: `rule-ope-benchmark-v1` on HuggingFace (CC-BY-4.0). Code
under the same licence. Reproduction: `python3 eval/build_benchmark.py`
reconstructs the benchmark from a single seed; checksum verification
is enforced by `tests/test_freeze.py`. A Dockerfile will be provided
for the camera-ready.

## 10.5 Wider impact

Offline evaluation of rule-based interventions lets practitioners ship
rules that demonstrably help, and refrain from rules that don't. The
downside of better offline evaluation is that it can be misused to
ship more interventions, fragmenting a pipeline. We recommend a
deployment protocol where rules are evaluated offline, shipped
online, *and continuously re-evaluated* as the correction distribution
drifts; the paper's methods support the first and third of these.
