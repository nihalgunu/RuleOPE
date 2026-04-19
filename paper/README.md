# Paper

The NeurIPS 2026 submission is drafted as one Markdown file per section
under `paper/sections/`; `paper/main.tex` is the NeurIPS-style LaTeX
skeleton that `\input`s converted-to-LaTeX section bodies.

To produce the PDF:

```bash
./paper/build.sh        # pandoc: paper/sections/*.md -> *_body.tex
cd paper && pdflatex main.tex
```

Each section file is self-contained and was written independently to
keep individual drafting work bounded (no single monolithic file). The
decisions log at `sections/11_decisions_log.md` records every material
design decision taken during the project, with its rationale.

## Section map

| #  | file                          | content                              |
|----|-------------------------------|--------------------------------------|
| 1  | `01_abstract_intro.md`        | abstract, introduction, contributions|
| 2  | `02_related_work.md`          | related work                         |
| 3  | `03_formulation.md`           | problem statement, assumptions       |
| 4  | `04_estimator.md`             | RuleOPE estimator and algorithm      |
| 5  | `05_theory.md`                | consistency, variance, failure modes |
| 6  | `06_benchmark.md`             | benchmark v1 design and artifacts    |
| 7  | `07_experiments.md`           | main comparison across 3 regimes     |
| 8  | `08_ablations_scaling.md`     | ablations, rule-depth, scaling       |
| 9  | `09_failure_case.md`          | failure-mode stress, case study      |
| 10 | `10_discussion.md`            | limitations, open problems, release  |
| 11 | `11_decisions_log.md`         | appendix: decisions with rationale   |

The full consistency proof is kept separately in `theory/proofs.tex`.
