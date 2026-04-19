# RuleOPE: Offline Evaluation of Rule-Based Interventions in RAG

Research code accompanying the NeurIPS 2026 submission
*RuleOPE: Doubly-Robust Offline Evaluation of Compositional Rule-Based Interventions in Retrieval-Augmented Generation*.

## Quickstart

```bash
python3 -m pip install -e .
python3 scripts/smoke_test.py
python3 eval/build_benchmark.py --out eval/benchmark_v1.jsonl --n_queries 4000 --seed 0
python3 scripts/run_all_baselines.py
python3 experiments/synthetic_controlled.py
python3 experiments/ablations.py
```

## Layout

See `paper/main.tex` for the full method description and experimental protocol.
All ground-truth rule values and frozen benchmark artifacts live under `eval/`.
Do not modify `eval/benchmark_v1.jsonl` or `eval/ground_truth_rule_values.json`
after Phase 1 has been frozen (see `eval/FROZEN.md`).
