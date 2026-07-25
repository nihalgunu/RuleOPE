"""E6b — MuSiQue frontier-anchor cells (qwen14b, qwen32b), filling the last
two missing cells of the 69-cell evaluation.

Requires eval/musique/outputs_qwen{14b,32b}_1500.jsonl (generated with
vLLM 0.10.2, greedy, qwen chat template — see rebuttal README).

Runs the standard 5-estimator sweep protocol (uniform-stochastic logging,
N in {150, 300, 600, 1200}, 50 trials) via run_cell, and writes a
sweep-compatible JSON so the selector analyses can consume it.

Output: e6b_musique_anchors.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.multi_estimator_n_sweep import run_cell

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/rebuttal/e6b_musique_anchors.json"


def main():
    out = {"grid": "musique_anchors", "n_trials": 50, "cells": {}}
    for llm in ("qwen14b", "qwen32b"):
        cell = run_cell("musique", llm, [150, 300, 600, 1200], 50, -1)
        out["cells"][f"musique__{llm}"] = cell
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
