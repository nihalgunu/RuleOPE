# Frozen benchmark

This directory contains the frozen `rule-ope-benchmark-v1` artifacts.
Do not modify any file after Phase 1 has been committed.
Rules: 500.  Queries: 4000.  Noise regimes: [0, 10, 30].
Ground-truth values are computed from the substrate counterfactuals;
the `_with_cf` variant of the logs is kept private and never used by
estimators, only by the evaluation harness.
