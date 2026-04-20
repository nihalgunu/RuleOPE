# 10  Discussion

## 10.1 When to use RuleOPE

RuleOPE is a conservative drop-in replacement for DR in rule-evaluation
workflows. On benchmarks where DR is already consistent and
low-variance (stochastic logging, well-specified regression), RuleOPE
does not underperform. The empirical regime in which our method
materially helps is the **small-$N$ regime** ($N \approx 100$--$500$
query--reward pairs, which matches the realistic deployment budget
when rules are evaluated from an internal review queue or a partial
online experiment). On three real-data benchmarks (HotpotQA, TriviaQA,
MuSiQue) in this regime, we report statistically significant MSE
reductions of 15\%--67\% over the OBP-style NonCompDR baseline
(§7C.3); at $N \ge 600$ both estimators saturate toward the shared
noise floor and the advantage shrinks. Practitioners with logs in the
thousand-query range should expect diminishing returns; practitioners
operating at $N \le 300$ get the largest benefit.

In settings where classical DR collapses into pure DM — *stochastic
logging with very small $\pi_0(a_\rho \mid x)$*, *miscoverage of
important query classes*, or *unstable retrieval* — RuleOPE's
correction-fusion term provides additional variance reduction at the
cost of a learnt gate, provided A5's proxy-style bridge is
non-vacuous (scenario (i) stochastic logging, or scenario (ii)
deterministic logging plus a pilot on $a_\rho$; see §5.4 revised).
Under strictly deterministic logging with no pilot and no second
proxy, A5 generically fails and no estimator using only
$(X, A = a_0, R, C)$ can close Thm A's $\mathbb{E}[p(X)]$-wide
identification gap. On empirical data where the correction-linearity
sufficient condition holds approximately (§5C.2), adding the bridge
term reduces held-out MAE by roughly 17\% over CompDR.

## 10.2 Limitations

**L1: Large-$N$ saturation.** The real-data advantage is largest at
small $N$ ($\le 300$) and shrinks into CI noise at $N \ge 600$ on
HotpotQA and TriviaQA (§7C.3). Practitioners operating at
$N \gg 1000$ should expect classical DR to perform comparably.
MuSiQue retains a large advantage at all tested $N$; we attribute
this to its higher rule-pool effective dimension (more rules with
sparse firing patterns, larger compositional headroom).

**L2: Real-data breadth.** We evaluate on three QA benchmarks
(HotpotQA, TriviaQA, MuSiQue) with a single LLM (Mistral-7B). Further
benchmarks --- Natural Questions, HybridQA, MultiHop-RAG, and
non-QA tasks (summarisation, open-ended generation) --- are deferred
to the camera-ready appendix.

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
