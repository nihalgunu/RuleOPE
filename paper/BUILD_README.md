# Building the PDF

This repo's paper/main.tex expects the section bodies as
`.tex` files (produced from the `sections/*.md` markdown by
`paper/build.sh`). Two toolchain requirements are needed that
are **not** present on this workstation:

1. **pandoc** (for `.md` -> `.tex` conversion)
2. **pdflatex** (or `tectonic` / `xelatex`) for the final `.pdf`

## One-shot install (macOS)

```
# Using Homebrew + MacTeX (no sudo; installs ~4 GB of TeX)
brew install pandoc
brew install --cask mactex
```

or a lighter alternative:

```
brew install pandoc
brew install tectonic   # single-binary LaTeX engine, no full texlive
```

## Build

```
cd paper
./build.sh                    # md -> tex (calls pandoc)
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex    # second pass for refs
# or, with tectonic:
# tectonic main.tex
```

Expected output: `paper/main.pdf`.

## Sections

The build consumes these markdown sources (in order of appearance in
`main.tex`):

| Source markdown | `\input{}` name | Role |
|---|---|---|
| `01_abstract_intro.md` | `intro_body.tex` | Abstract + intro |
| `02_related_work.md` | `related_body.tex` | Related work |
| `03_formulation.md` | `formulation_body.tex` | Problem formulation |
| `04_estimator.md` | `estimator_body.tex` | Estimator |
| `05_theory.md` | `theory_body.tex` | Theory |
| `05c_assumption_validation.md` | `assumption_validation_body.tex` | **NEW** A3/A5 empirical validation |
| `06_benchmark.md` | `benchmark_body.tex` | Synthetic benchmark |
| `07_experiments.md` | `experiments_body.tex` | Synthetic experiments |
| `07c_real_data.md` | `real_data_body.tex` | **NEW** Real-data eval (HotpotQA/TriviaQA/MuSiQue) |
| `08_ablations_scaling.md` | `ablations_body.tex` | Ablations |
| `09_failure_case.md` | `failure_body.tex` | Failure modes |
| `10_discussion.md` | `discussion_body.tex` | Discussion |
| `11_decisions_log.md` | `decisions_body.tex` | Decisions log (appendix) |

Figures are in `paper/figs/` (both `.pdf` and `.png`).

## If pandoc is unavailable

Each `sections/*.md` file is mostly plain Markdown with LaTeX math.
You can compile by:

1. Concatenating the markdown sources into a single document via
   `cat sections/{01,02,03,04,05,05c,06,07,07c,08,09,10,11}*.md > all.md`.
2. Using a Markdown-aware LaTeX-rendering tool (e.g. an overleaf
   import, or a GitHub web render with MathJax) for review.

For the camera-ready submission, pandoc + MacTeX is the recommended
path.
