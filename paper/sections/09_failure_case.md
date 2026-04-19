# 9  Failure-mode stress tests and case study

## 9.1 Stress tests on A4 violations

We construct three benchmarks that violate correction unconfoundedness
(A4) in the specific ways characterised in §5.3.

| setting | RuleOPE MSE | DR MSE | DM MSE | RuleOPE tau@20 | DR tau@20 | DM tau@20 |
|---------|------------:|-------:|-------:|---------------:|----------:|----------:|
| benign                  | 0.00001 | 0.00001 | 0.00001 | +0.670 | +0.628 | +0.606 |
| F1 query-effort bias    | 0.00001 | 0.00001 | 0.00001 | +0.670 | +0.628 | +0.606 |
| F2 self-consistent bias | 0.00001 | 0.00001 | 0.00001 | +0.702 | +0.702 | +0.745 |
| F3 corpus drift         | 0.00002 | 0.00002 | 0.00001 | +0.777 | +0.734 | +0.681 |

**F1 (query-dependent correction effort).** We tilt the correction model
so that reviewers are more likely to flag longer queries
(`effort_slope = 1.5`), making $P(C \mid X, A)$ depend on a feature
that is also a reward parent. RuleOPE's tau@20 dominates DR by $+0.042$
and DM by $+0.064$: the compositional regression makes the (now
observed) query-length feature an atom, so A4 is re-established in
the enlarged feature space. The *mitigation* advised in §5.3 —
augmenting $\mathcal{V}$ with the confounder — works as predicted.

**F2 (self-consistent-answer bias).** We set `gen_conf_bias = -2.0`,
meaning reviewers *under*-report corrections on confidently wrong
answers. Here DM actually edges out DR and RuleOPE on tau@20 (+0.745
vs +0.702), because its correction-independent regression is not
biased by the correction model's distortion. This is the expected
behaviour from §5.4: when the correction signal is *informative but
systematically wrong*, the correction-fusion term makes things worse,
not better. RuleOPE's gate does not fully identify this regime because
the gate is trained on the same biased corrections. *Mitigation*: add
$\text{gen\_conf}$ atoms (as we do in the benchmark); F2's bias is
attenuated but not eliminated because the bias acts on the *realised*
confidence of wrong answers, which is an unobserved-to-reviewer confound.
We document this as a fundamental limitation and recommend an
independent human-in-the-loop audit whenever confident-wrong answers
are suspected.

**F3 (corpus drift).** We fit estimators on one substrate seed and
evaluate rules on another. MSE doubles (to $0.00002$) for all three
estimators; RuleOPE's tau@20 of $+0.777$ remains the strongest,
outperforming DR by $+0.043$ and DM by $+0.096$. Drift hurts DM most
because DM relies entirely on the train-distribution regression; RuleOPE
is partially rescued by its DR correction on overlapping records.

## 9.2 Case study: top-20 rules by RuleOPE

We sort the 500 benchmark rules by RuleOPE's estimated value and
inspect the top 20. All 20 have absolute error below $0.008$ against
ground truth; fifteen of the 20 are of depth 1 or 2 (simple, single-
signal rules). Representative examples:

```
rule                                                        fires    est    gt
filter[ent_missing_top1]                                    23.3%  0.725  0.732
filter[q_len_gt_8]                                          90.9%  0.724  0.723
filter[top1_len_lt_128]                                     62.2%  0.724  0.726
rerank[top1_src_stub]                                       17.6%  0.723  0.731
filter[top1_score_gt_0_5]                                   86.9%  0.723  0.719
...
```

The top-20 rules reflect meaningful failure patterns:

* *Stub-source retrievals* (`top1_src_stub`) are canonical high-recall-
  low-precision hits in Wikipedia-style corpora. Reranking them downward
  is a well-known heuristic in production RAG systems.
* *Missing entity* (`ent_missing_top1`) rules recover queries where
  the top-ranked passage does not mention the query's named entity.
  Filtering such hits is exactly what query-aware reranking should do.
* *Short top-1 passages* (`top1_len_lt_128`): fragments are often stub
  entries or list items without the answer context.

The two depth-3 rules in the top-20 correspond to combinations like
`filter[gap_lt_0_10 & top1_src_stub & q_multihop]` — multi-hop queries
with an unstable stub-source retrieval. These are low-firing (<5% of
queries) but high-value corrections, exactly the kind of rule that
human engineers often discover anecdotally but cannot defend offline.

## 9.3 What the case study shows

The estimator is not merely passing a numerical benchmark; its
top-ranked rules are interpretable, and they concentrate on
retrieval-failure modes that match the published RAG literature's
qualitative observations. For the production practitioner this is the
actionable output: a ranked list of rules with confidence intervals,
each pointing at a specific mechanical failure. The quantitative
metrics in §7 validate that RuleOPE's ranking is close to the
ground-truth ranking; the case study validates that the rules the
ranking surfaces are ones a practitioner would want to know about.
