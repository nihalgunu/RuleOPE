# Conditional Ranking of Off-Policy Evaluators in Retrieval-Augmented Generation

Supplementary code, data, and reproduction scripts for the NeurIPS 2026
submission *Conditional Ranking of Off-Policy Evaluators in
Retrieval-Augmented Generation*.

The paper documents **OPE conditional ranking**: on production-RAG logs the
relative quality of two OPE estimators is jointly conditional on the
language model, the benchmark, and the sample size. We identify
$\sigma_R^2 := \mathbb{E}_X[\mathrm{Var}_a R(X, a)]$ — the within-query
variance of per-action reward — as the LLM-side predictor and propose a
$\sigma_R^2$-thresholded pilot tiebreaker for selecting between RuleOPE and
MRDR at deployment time. This repository contains everything required to
reproduce the figures and tables in the paper from cached generator
outputs.

## What is in here

```
src/                         Library code: estimators + RAG substrates + atom DSL
experiments/                 Experiment drivers (grid sweeps, σ_R² selector audits)
experiments/results/         Cached JSON / CSV outputs for every reported number
eval/                        Cached generator outputs per (benchmark, LLM)
                               and the 500-rule pool (rules_v1.jsonl)
scripts/                     Figure builder, LOOCV, three-pair analysis
final_figures/               Built figures (PDF + PNG, ready for LaTeX inclusion)
tests/                       Estimator interface smoke test
```

## Estimator panel

The five estimators in the paper's headline panel (`src/estimators/` plus
`experiments/ablations.py`):

| Class                         | Module                                | Role                |
|-------------------------------|---------------------------------------|---------------------|
| `NonCompositionalDR`          | `experiments/ablations.py`            | per-rule DR baseline |
| `DoublyRobust`                | `src/estimators/doubly_robust.py`     | classical DR        |
| `SwitchDR`                    | `src/estimators/switch_dr.py`         | Wang et al. 2017    |
| `MRDR`                        | `src/estimators/mrdr.py`              | Farajtabar et al. 2018 |
| `RuleOPE`                     | `src/estimators/rule_ope.py`          | this paper          |

All five share the `Estimator` interface in `src/estimators/base.py`. RuleOPE
factorises a single ridge regression over the entire rule pool through atom
features (cf. Theorem 2's $\Theta(d/N)$ aggregate variance).

## Reproduce the figures

The figure script reads only cached JSON / CSV artefacts in
`experiments/results/` plus the generator outputs in `eval/`. No GPU or
Lambda access is needed.

```bash
pip install -e .
python scripts/build_figures.py
```

Outputs land in `final_figures/` (8 figures × {PDF, PNG}). The build runs
in ~90s on a single CPU.

| Figure | Filename                              | Paper reference                                         |
|--------|---------------------------------------|---------------------------------------------------------|
| 1      | `fig_headline_sigmar2_pairgap`        | $\sigma_R^2$ vs MRDR–RuleOPE pair gap, 36 cells         |
| 2      | `fig_nq_rankflip_n_trajectory`        | NQ pct vs $N$, RuleOPE goes negative                    |
| 3      | `fig_selector_decomp`                 | Selector accuracy by regime, 54-cell pooled audit       |
| 4      | `fig_rankflip_heatmap`                | 36-cell heatmap, NQ rank-flip cells outlined            |
| 5      | `fig_variance_attribution`            | Where $\sigma_R^2$ lives (3 estimator pairs)            |
| 6      | `fig_commensurability_sandwich`       | $\sigma_R^2 \leftrightarrow \Delta^2$ (Theorem 4)       |
| 7      | `fig_a3_validation`                   | A3 atom-level residual independence                     |
| 8      | `fig_cost_panel`                      | Replay vs no-replay generator-call cost                 |

## Reproduce the OPE sweeps

The five grid sweeps that produced the cached results in
`experiments/results/`:

```bash
# 12 LLMs × 3 benchmarks × 4 N values (the headline 36 × 4-N catalogue)
python experiments/multi_estimator_n_sweep.py --grid full_36_4N

# N=2400 extrapolation on the same 36 cells
python experiments/multi_estimator_n_sweep.py --grid full_36_n2400

# Cross-substrate validation on MuSiQue (12 LLMs)
python experiments/multi_estimator_n_sweep.py --grid musique_12LLM_4N

# Cross-substrate validation on 2WikiMultiHopQA (12 base LLMs + 14B + 32B)
python experiments/multi_estimator_n_sweep.py --grid 2wiki_12LLM

# Frontier-scale anchors at 14B and 32B (HotpotQA, TriviaQA, NQ)
python experiments/multi_estimator_n_sweep.py --grid qwen14b_anchor
python experiments/multi_estimator_n_sweep.py --grid qwen32b_anchor
```

Each sweep is joblib-parallelised across CPU cores (`--n_jobs -1`); the full
36 × 4-N grid takes ~30 minutes on an 8-core laptop. Results are written
incrementally to the corresponding JSON in `experiments/results/`.

## Reproduce the selector analyses

```bash
# σ_R² + selector accuracy on the 12 MuSiQue cells
python experiments/analysis_2wiki_sigma_R2.py
python experiments/analysis_2wiki_selector.py

# 14B / 32B anchor selector accuracy (per cell)
python experiments/analysis_qwen14b_selector.py
python experiments/analysis_qwen32b_selector.py

# Pooled middle-band audit across the full 67-cell evaluation
python experiments/analysis_middle_band_audit.py
```

## Three-pair characterisation and LOOCV

```bash
# Adj-R² for the three estimator pairs (full 36-cell grid)
python scripts/analyze_full36_pairs.py

# Leave-one-out / leave-one-LLM-out / leave-one-benchmark-out CV
python scripts/full_loocv.py
```

## Generator outputs

`eval/{benchmark}/outputs_{llm}_1500.jsonl` contains the cached generator
outputs used throughout. Each line is `{"id": "<qid>__<action>", "text":
"<answer>"}` for action $\in \{\texttt{noop}, \texttt{filter},
\texttt{rerank}\}$ on 1500 dev queries. The released models are 12 open-weights
generators in the 1.7B–9B range plus Qwen2.5-14B-Instruct and
Qwen2.5-32B-Instruct frontier-scale anchors:

```
SmolLM2-1.7B    Qwen2.5-3B       Phi-3-mini       Phi-3.5-mini
Zephyr-7B-β     Mistral-7B       Qwen2.5-7B       Qwen2.5-Coder-7B
InternLM2.5-7B  OLMo-7B          Granite-3.0-8B   Yi-1.5-9B
                 Qwen2.5-14B (anchor)        Qwen2.5-32B (anchor)
```

The 5th held-out benchmark, 2WikiMultiHopQA, has 14 LLM cells (the 12
base + both anchors).

**Prompt files are not shipped** (they are deterministic and large: ~213 MB
across the five benchmarks × six chat templates). Regenerate them in one
pass:

```bash
for tpl in mistral qwen phi35 llama3 olmo granite; do
  python experiments/build_hotpot_prompts_multi.py --template $tpl
  python experiments/build_trivia_prompts.py        --template $tpl
  python experiments/build_nq_prompts.py            --template $tpl
  python experiments/build_musique_prompts.py       --template $tpl
  python experiments/build_2wiki_prompts.py         --template $tpl
done
```

This rebuilds `eval/{benchmark}/prompts_{template}_1500.jsonl` from the
shipped `dev.parquet` files, in the exact `{"id", "prompt"}` schema the
generators consumed. Six chat templates are supported (`mistral`, `qwen`,
`phi35`, `llama3`, `olmo`, `granite`).

To regenerate from scratch (i.e., re-run an LLM generator from the prompts)
you need a generator endpoint of your choice — the cached
`outputs_*_1500.jsonl` files cover all the LLMs reported in the paper, so
generation is not required to reproduce any number.

## Smoke test

```bash
python tests/test_estimators_smoke.py
```

Exercises all five estimators on a tiny synthetic log and verifies that
each produces a finite estimate.

## License & attribution

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Copyright © 2026 Nihal Gunukula and Phyvant. Research conducted at Phyvant.

## Citation

```
@inproceedings{ruleope2026,
  title        = {Conditional Ranking of Off-Policy Evaluators in Retrieval-Augmented Generation},
  author       = {Nihal Gunukula},
  booktitle    = {Advances in Neural Information Processing Systems},
  year         = {2026}
}
```

## Acknowledgements

This research was supported by [Phyvant](https://phyvant.com).
