#!/usr/bin/env bash
# Convert paper/sections/*.md to paper/sections/*_body.tex (used by main.tex \input).
# Requires pandoc (the camera-ready build will use a NeurIPS style file).
set -e
cd "$(dirname "$0")"
declare -A MAP=(
  [01_abstract_intro]=intro_body
  [02_related_work]=related_body
  [03_formulation]=formulation_body
  [04_estimator]=estimator_body
  [05_theory]=theory_body
  [06_benchmark]=benchmark_body
  [07_experiments]=experiments_body
  [08_ablations_scaling]=ablations_body
  [09_failure_case]=failure_body
  [10_discussion]=discussion_body
  [11_decisions_log]=decisions_body
)
for src in "${!MAP[@]}"; do
  pandoc -f markdown -t latex "sections/${src}.md" -o "sections/${MAP[$src]}.tex"
done
