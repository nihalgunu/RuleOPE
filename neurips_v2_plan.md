# NeurIPS 2026 Paper Plan — Theorem-Forward with No-Replay Framing

Date: 2026-04-20

## The one-sentence pitch

*We prove that rule-based off-policy evaluation of RAG retrieval interventions is identifiable from logged tuples alone, without re-running the generator at counterfactual retrievals, under a compositional decomposition of the reward — and characterise the efficient no-replay estimator.*

## Why this is a genuine NeurIPS contribution

Every published OPE method assumes you can either (a) evaluate the reward at counterfactual (context, action) pairs (DR/DM/IPS, CSPI-MT, Waudby-Smith) or (b) replay the generator offline. In RAG the generator is a black-box LLM whose reward surface depends on a non-linear function of (query, retrieval); replay is prohibitively expensive at scale.

The **no-replay** constraint is the actual bottleneck in deploying OPE for real RAG systems. Nobody has formally solved it. Our contribution:

1. **Identification theorem** (`theory/noreplay_theorem.md`): under compositional reward decomposition A3, $V(\rho)$ is identified from $(q_i, r_0(q_i), R_i)$ alone.
2. **Efficient estimator**: RuleOPE (the existing estimator from this paper's backbone) attains the semiparametric efficiency bound in the no-replay class.
3. **Compositional variance reduction** (the existing compositional argument) is what makes the no-replay identification tight: atom-sharing in the reward regression borrows strength across rules.
4. **Empirical validation** on a real RAG pipeline (HotpotQA + Mistral-7B) where the oracle replays the generator on every counterfactual retrieval and no-replay methods are the candidates.

## The paper's structure

1. **Problem**: rule-OPE for RAG with expensive black-box generator. Formalize the no-replay constraint.
2. **Identification** (Theorem 1): no-replay V(ρ) identification under A3.
3. **Efficiency** (Theorem 2): RuleOPE attains the semiparametric bound; DR does not.
4. **Compositional variance reduction** (Theorem 3 — existing in theory/proofs.tex): cross-rule MSE bound.
5. **Empirical validation**:
   - Synthetic (§2–§6 of existing work): 11–23 % MSE reduction, theorem predictions hold tightly.
   - Real: HotpotQA + Mistral-7B, oracle-vs-no-replay gap.
6. **Discussion**: limitations of the compositional assumption, practical guidance.

## What to cut from the current draft

- ADAPT-v1/v2/v3/v4 FDR machinery → appendix as "extensions."
- Waudby-Smith / CSPI-MT head-to-heads → framed as "classical OPE baselines assuming replay" in related work.
- The drift/active/FDR triple → not the spine; mention as future work.
- All §15 P1/P2/P3 explorations except the no-replay one.

## What stays central

- RuleOPE estimator (existing).
- Compositional atom regression (existing).
- The theorem proofs (need sharpening for the no-replay claim).
- The efficiency validation (§4 of existing).
- The real-data HotpotQA experiment.

## Honest positioning

The empirical wins on HotpotQA are modest (RuleOPE < DR on MSE by 1–3 %, directionally matching the theorem but not statistically significant at N ≤ 1500). The paper is carried by:
- The theorem's novelty (no-replay identifiability is genuinely new in OPE literature).
- The synthetic validation (clean 11–23 % win under the regime the theorem applies to).
- The real-data demonstration that the framework works end-to-end on a standard public benchmark.

This is a *theory paper with clean empirical validation*, not a *benchmark-breaking empirical paper*. NeurIPS theory-track reviewers reward this profile.

## Status check

| item | status |
|---|---|
| Theorem statement | drafted (`theory/noreplay_theorem.md`) |
| Theorem proof | needs sharpening (currently proof sketch) |
| Synthetic §2–§6 | DONE (existing results) |
| HotpotQA prompts + generator | DONE (1800 + 4479 prompts, Mistral-7B on A10) |
| No-replay harness | DONE (`experiments/noreplay_ope_retq.py`) |
| Real-data results at N=600 | IN (RuleOPE 0.00386, DR 0.00390) |
| Real-data results at N=1500 | IN PROGRESS |
| Paper draft | TODO |

## Fallback if N=1500 doesn't strengthen

If the real-data effect size stays <5 % and within bootstrap CIs:
- Option A: submit as-is to NeurIPS with honest framing; expect 35–45 % odds.
- Option B: switch to AISTATS (Oct 2026 deadline); expect 70 % odds.
- Option C: add a second real-data dataset (TriviaQA or NQ) to strengthen the empirical section.

My current recommendation: commit to Option A with the honest framing. The theorem alone with clean synthetic validation is a defensible NeurIPS submission.
