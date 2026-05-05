"""Multi-estimator, multi-N sweep on a configurable cell grid.

Runs NonCompDR / DR / SwitchDR / MRDR / RuleOPE on each (LLM, benchmark)
cell at multiple N values, with joblib trial-level parallelism for
~N_CORES× speedup.

Two modes via --grid:

    --grid mrdr_sweep   — 4 LLMs × 3 benchmarks × N ∈ {150, 300, 600, 1200}
                          for MRDR-vs-RuleOPE crossover characterization
                          (12 cells × 4 N values = 48 cell-N combos)

    --grid full_36      — 12 LLMs × 3 benchmarks × N=150
                          for the second-estimator-pair phenomenon claim
                          (36 cells × 1 N = 36 cell-N combos, all 5 estimators)

Run:
    python3 experiments/multi_estimator_n_sweep.py --grid mrdr_sweep --n_trials 50
    python3 experiments/multi_estimator_n_sweep.py --grid full_36   --n_trials 100
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.estimators.doubly_robust import DoublyRobust
from src.estimators.switch_dr import SwitchDR
from src.estimators.mrdr import MRDR
from src.estimators.rule_ope import RuleOPE
from src.logs import LoggedRecord
from src.rule_dsl import load_rules
from experiments.ablations import NonCompositionalDR


_WS = re.compile(r"[^\w\s]")


def _norm(s):
    s = s.lower().strip(); s = _WS.sub(" ", s); s = " ".join(s.split())
    for lead in ("answer:", "a:", "the answer is"):
        if s.startswith(lead): s = s[len(lead):].strip()
    return s


def _f1(p, g):
    p = _norm(p).split(); g = _norm(g).split()
    if not p or not g: return 0.0
    c = set(p) & set(g)
    if not c: return 0.0
    return 2 * len(c) / len(p) * len(c) / len(g) / (len(c) / len(p) + len(c) / len(g))


def _reward_hotpot(g, sample):
    if _norm(g) in ("unknown", ""): return 0.0
    return _f1(g, sample.answer)


def _reward_alias(g, aliases):
    if _norm(g) in ("unknown", ""): return 0.0
    return max((_f1(g, a) for a in aliases), default=0.0)


def paired_bootstrap_ci(log_ratio, n_boot=5000, seed=17):
    rng = np.random.default_rng(seed)
    boots = np.array([
        log_ratio[rng.integers(0, len(log_ratio), size=len(log_ratio))].mean()
        for _ in range(n_boot)
    ])
    return float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))


def load_cell(bench, llm):
    """Returns (samples_all, oracle, rules) for a (benchmark, LLM) cell."""
    if bench == "hotpot":
        from src.rag_substrate_hotpot import _load_hotpot, _score_passages, _atom_features
        if llm == "mistral":
            outputs_path = "eval/hotpot/outputs_1500.jsonl"
        else:
            outputs_path = f"eval/hotpot/outputs_{llm}_1500.jsonl"
        samples_all = _load_hotpot("eval/hotpot/dev.parquet", 1500, 0)
        answers = {s.qid: {} for s in samples_all}
        with open(outputs_path) as f:
            for line in f:
                d = json.loads(line); qid, action = d["id"].rsplit("__", 1)
                if qid in answers: answers[qid][action] = d["text"]
        samples_all = [s for s in samples_all if len(answers[s.qid]) == 3]
        oracle = {s.qid: {a: _reward_hotpot(answers[s.qid].get(a, ""), s)
                           for a in ("noop", "filter", "rerank")} for s in samples_all}
    elif bench == "trivia":
        from src.rag_substrate_trivia import _load_trivia, _score_passages, _atom_features
        if llm == "mistral":
            outputs_path = "eval/trivia/outputs_1500.jsonl"
        else:
            outputs_path = f"eval/trivia/outputs_{llm}_1500.jsonl"
        samples_all = _load_trivia("eval/trivia/dev.parquet", 1500, 0)
        answers = {s.qid: {} for s in samples_all}
        with open(outputs_path) as f:
            for line in f:
                d = json.loads(line); qid, action = d["id"].rsplit("__", 1)
                if qid in answers: answers[qid][action] = d["text"]
        samples_all = [s for s in samples_all if len(answers[s.qid]) == 3]
        oracle = {s.qid: {a: _reward_alias(answers[s.qid].get(a, ""), s.answer_aliases)
                           for a in ("noop", "filter", "rerank")} for s in samples_all}
    elif bench == "nq":
        from src.rag_substrate_nq import _load_nq, _score_passages, _atom_features
        outputs_path = f"eval/nq/outputs_{llm}_1500.jsonl"
        samples_all = _load_nq("eval/nq/dev.parquet", 1500, 0)
        answers = {s.qid: {} for s in samples_all}
        with open(outputs_path) as f:
            for line in f:
                d = json.loads(line); qid, action = d["id"].rsplit("__", 1)
                if qid in answers: answers[qid][action] = d["text"]
        samples_all = [s for s in samples_all if len(answers[s.qid]) == 3]
        oracle = {s.qid: {a: _reward_alias(answers[s.qid].get(a, ""), s.answer_aliases)
                           for a in ("noop", "filter", "rerank")} for s in samples_all}
    elif bench == "musique":
        from src.rag_substrate_musique import _load_musique, _score_passages, _atom_features
        if llm == "mistral":
            outputs_path = "eval/musique/outputs_1500.jsonl"
        else:
            outputs_path = f"eval/musique/outputs_{llm}_1500.jsonl"
        samples_all = _load_musique("eval/musique/dev.parquet", 1500, 0)
        answers = {s.qid: {} for s in samples_all}
        with open(outputs_path) as f:
            for line in f:
                d = json.loads(line); qid, action = d["id"].rsplit("__", 1)
                if qid in answers: answers[qid][action] = d["text"]
        samples_all = [s for s in samples_all if len(answers[s.qid]) == 3]
        oracle = {s.qid: {a: _reward_hotpot(answers[s.qid].get(a, ""), s)
                           for a in ("noop", "filter", "rerank")} for s in samples_all}
    elif bench == "2wiki":
        from src.rag_substrate_2wiki import _load_2wiki, _score_passages, _atom_features
        if llm == "mistral":
            outputs_path = "eval/2wiki/outputs_1500.jsonl"
        else:
            outputs_path = f"eval/2wiki/outputs_{llm}_1500.jsonl"
        samples_all = _load_2wiki("eval/2wiki/dev.parquet", 1500, 0)
        answers = {s.qid: {} for s in samples_all}
        with open(outputs_path) as f:
            for line in f:
                d = json.loads(line); qid, action = d["id"].rsplit("__", 1)
                if qid in answers: answers[qid][action] = d["text"]
        samples_all = [s for s in samples_all if len(answers[s.qid]) == 3]
        oracle = {s.qid: {a: _reward_hotpot(answers[s.qid].get(a, ""), s)
                           for a in ("noop", "filter", "rerank")} for s in samples_all}
    else:
        raise ValueError(bench)
    for s in samples_all:
        oracle[s.qid]["abstain"] = 0.5
    rules = load_rules("eval/rules_v1.jsonl")
    rules = [r for r in rules if r.action in ("filter", "rerank", "abstain")]
    return samples_all, oracle, rules, _score_passages, _atom_features


def run_one_trial(trial, N, samples_all, oracle, rules, _score_passages, _atom_features,
                  bench, seed_base):
    """Self-contained: one trial across 5 estimators. Used in joblib.Parallel."""
    from src.estimators._regression import fires_mask
    rng = np.random.default_rng(seed_base * N + trial)
    idx = rng.choice(len(samples_all), size=min(N, len(samples_all)), replace=False)
    samples_tr = [samples_all[int(i)] for i in idx]
    logs = []
    ACTIONS = ("noop", "filter", "rerank")
    for s in samples_tr:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        a = ACTIONS[int(rng.integers(0, 3))]
        logs.append(LoggedRecord(
            query_id=s.qid, ctx=ctx, logged_action=a, logged_propensity=1/3,
            logged_reward=float(oracle[s.qid][a]),
            correction=0, cf_rewards=dict(oracle[s.qid]),
        ))
    firing = {r.id: float(fires_mask(logs, r).mean()) for r in rules}
    tr_rules = [r for r in rules if 0.05 <= firing[r.id] <= 0.95]
    if len(tr_rules) < 10:
        return None
    gt = np.array([
        float(np.mean([rec.cf_rewards[r.action] if r.fires(rec.ctx) else rec.cf_rewards["noop"] for rec in logs]))
        for r in tr_rules
    ])
    out = {}
    for name, est in [
        ("NonCompDR", NonCompositionalDR()),
        ("DR",        DoublyRobust()),
        ("SwitchDR",  SwitchDR(tau=5.0)),
        ("MRDR",      MRDR()),
        ("RuleOPE",   RuleOPE()),
    ]:
        est.fit(logs)
        res = est.value_many(tr_rules, logs)
        vals = np.array([res[r.id].estimate for r in tr_rules])
        out[name] = float(np.mean((vals - gt) ** 2))
    return out


def run_cell(bench, llm, Ns, n_trials, n_jobs):
    samples_all, oracle, rules, score_fn, atom_fn = load_cell(bench, llm)
    seed_base = {"hotpot": 8000, "trivia": 5000, "nq": 9000, "musique": 7000, "2wiki": 6000}[bench]
    print(f"  cell {bench}/{llm}: {len(samples_all)} samples, {len(rules)} rules", flush=True)
    cell_out = {"benchmark": bench, "generator": llm, "scaling": {}}
    for N in Ns:
        t0 = time.time()
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_one_trial)(t, N, samples_all, oracle, rules, score_fn, atom_fn, bench, seed_base)
            for t in range(n_trials)
        )
        results = [r for r in results if r is not None]
        if not results: continue
        mse = {name: np.array([r[name] for r in results]) for name in results[0]}
        eps = 1e-9
        cell = {}
        for name, vals in mse.items():
            cell[name] = {"MSE_mean": float(vals.mean()), "MSE_std": float(vals.std()), "n_trials": int(len(vals))}
        # Compute paired-bootstrap pct vs NonCompDR for every other estimator
        nc = mse["NonCompDR"]
        for name in ("DR", "SwitchDR", "MRDR", "RuleOPE"):
            v = mse[name]
            log_ratio = np.log(nc + eps) - np.log(v + eps)
            ci_lo, ci_hi = paired_bootstrap_ci(log_ratio)
            cell[name]["pct_mean_logratio"] = 100.0 * (np.exp(log_ratio.mean()) - 1)
            cell[name]["pct_CI90_bootstrap"] = [100.0 * (np.exp(ci_lo) - 1), 100.0 * (np.exp(ci_hi) - 1)]
            cell[name]["paired_t_pvalue"] = float(ttest_rel(nc, v).pvalue)
            cell[name]["significance_bootstrap"] = bool(ci_lo > 0)
        # Also pairwise: MRDR-vs-DR and RuleOPE-vs-DR (the second pairs we care about)
        for left, right in [("MRDR", "DR"), ("RuleOPE", "DR"), ("MRDR", "RuleOPE")]:
            l = mse[left]; r = mse[right]
            log_ratio = np.log(r + eps) - np.log(l + eps)
            ci_lo, ci_hi = paired_bootstrap_ci(log_ratio)
            cell[f"{left}_vs_{right}"] = {
                "pct_mean_logratio": 100.0 * (np.exp(log_ratio.mean()) - 1),
                "pct_CI90_bootstrap": [100.0 * (np.exp(ci_lo) - 1), 100.0 * (np.exp(ci_hi) - 1)],
                "significance_bootstrap": bool(ci_lo > 0),
            }
        cell_out["scaling"][str(N)] = cell
        elapsed = time.time() - t0
        ro = cell["RuleOPE"]; mr = cell["MRDR"]; dr = cell["DR"]
        print(f"    N={N}: RuleOPE +{ro['pct_mean_logratio']:6.1f}%  MRDR +{mr['pct_mean_logratio']:6.1f}%  DR +{dr['pct_mean_logratio']:6.1f}%  ({elapsed:.0f}s)", flush=True)
    return cell_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["mrdr_sweep", "full_36", "full_36_4N", "full_36_n2400", "musique_12LLM_4N", "qwen14b_anchor", "qwen32b_anchor", "2wiki_subset", "2wiki_12LLM"], required=True)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--n_jobs", type=int, default=-1, help="joblib parallel jobs (-1 = all cores)")
    ap.add_argument("--results_dir", default="experiments/results")
    args = ap.parse_args()

    if args.grid == "mrdr_sweep":
        # 4 LLMs spanning HotpotQA noop F1 spectrum × 3 benchmarks × 4 N values
        cells = [(b, l) for l in ["mistral", "qwen", "yi15", "phi35"]
                        for b in ["hotpot", "trivia", "nq"]]
        Ns = [150, 300, 600, 1200]
        out_path = Path(args.results_dir) / "mrdr_sweep_4LLM_4N.json"
    elif args.grid == "full_36":
        # 12 LLMs × 3 benchmarks × N=150 (5 estimators each, all pairwise pcts)
        LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35",
                "zephyr7b", "mistral", "qwen", "qwencoder7b",
                "internlm7b", "olmo7b", "granite8b", "yi15"]
        cells = [(b, l) for l in LLMS for b in ["hotpot", "trivia", "nq"]]
        Ns = [150]
        out_path = Path(args.results_dir) / "full_36cell_5estimator.json"
    elif args.grid == "full_36_4N":
        # 12 LLMs × 3 benchmarks × N ∈ {150, 300, 600, 1200}: full crossover catalogue
        LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35",
                "zephyr7b", "mistral", "qwen", "qwencoder7b",
                "internlm7b", "olmo7b", "granite8b", "yi15"]
        cells = [(b, l) for l in LLMS for b in ["hotpot", "trivia", "nq"]]
        Ns = [150, 300, 600, 1200]
        out_path = Path(args.results_dir) / "full_36cell_4N_5estimator.json"
    elif args.grid == "full_36_n2400":
        # 12 LLMs × 3 benchmarks × N=2400 (extrapolation beyond the 4N range)
        LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35",
                "zephyr7b", "mistral", "qwen", "qwencoder7b",
                "internlm7b", "olmo7b", "granite8b", "yi15"]
        cells = [(b, l) for l in LLMS for b in ["hotpot", "trivia", "nq"]]
        Ns = [2400]
        out_path = Path(args.results_dir) / "full_36cell_n2400_5estimator.json"
    elif args.grid == "musique_12LLM_4N":
        # 4th benchmark: MuSiQue, 12 LLMs × N ∈ {150, 300, 600, 1200}, 5 estimators each
        LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35",
                "zephyr7b", "mistral", "qwen", "qwencoder7b",
                "internlm7b", "olmo7b", "granite8b", "yi15"]
        cells = [("musique", l) for l in LLMS]
        Ns = [150, 300, 600, 1200]
        out_path = Path(args.results_dir) / "musique_12LLM_4N_5estimator.json"
    elif args.grid == "qwen14b_anchor":
        # Frontier-scale anchor: Qwen2.5-14B-Instruct × 3 benchmarks × N ∈ {150,300,600,1200}
        cells = [(b, "qwen14b") for b in ["hotpot", "trivia", "nq"]]
        Ns = [150, 300, 600, 1200]
        out_path = Path(args.results_dir) / "qwen14b_anchor_4N_5estimator.json"
    elif args.grid == "qwen32b_anchor":
        # Frontier-scale anchor (≥32B): Qwen2.5-32B-Instruct × 3 in-grid benchmarks × 4N
        # (2wiki/qwen32b already covered by the 2wiki_12LLM grid).
        all_cells = [(b, "qwen32b") for b in ["hotpot", "trivia", "nq"]]
        cells = []
        for b, l in all_cells:
            path = Path(f"eval/{b}/outputs_{l}_1500.jsonl")
            if path.exists():
                cells.append((b, l))
            else:
                print(f"  skip {b}/{l}: outputs missing ({path})", flush=True)
        Ns = [150, 300, 600, 1200]
        out_path = Path(args.results_dir) / "qwen32b_anchor_4N_5estimator.json"
    elif args.grid == "2wiki_subset":
        # 2WikiMultiHopQA held-out cross-substrate: subset of LLMs × N ∈ {150,300,600,1200}
        # Used to probe σ_R² distribution + selector accuracy on the high-σ_R² branch.
        # Subset chosen to be the LLMs most likely to produce high σ_R² (per the NQ pattern).
        LLMS = ["mistral", "qwen", "phi35", "yi15"]
        cells = [("2wiki", l) for l in LLMS]
        Ns = [150, 300, 600, 1200]
        out_path = Path(args.results_dir) / "2wiki_subset_4N_5estimator.json"
    elif args.grid == "2wiki_12LLM":
        # Full 2WikiMultiHopQA × 12 LLMs × 4N held-out cross-substrate, plus qwen14b/qwen32b
        # frontier-scale anchors. Cells with missing outputs are skipped at runtime.
        LLMS = ["smollm17b", "qwen3b", "phi3mini", "phi35",
                "zephyr7b", "mistral", "qwen", "qwencoder7b",
                "internlm7b", "olmo7b", "granite8b", "yi15",
                "qwen14b", "qwen32b"]
        cells = []
        for l in LLMS:
            path = Path(f"eval/2wiki/outputs_1500.jsonl" if l == "mistral"
                        else f"eval/2wiki/outputs_{l}_1500.jsonl")
            if path.exists():
                cells.append(("2wiki", l))
            else:
                print(f"  skip {l}: outputs missing ({path})", flush=True)
        Ns = [150, 300, 600, 1200]
        out_path = Path(args.results_dir) / "2wiki_12LLM_4N_5estimator.json"

    print(f"=== {args.grid}: {len(cells)} cells × {len(Ns)} N values ===", flush=True)
    print(f"    n_trials={args.n_trials}, n_jobs={args.n_jobs}", flush=True)
    out = {"grid": args.grid, "n_trials": args.n_trials, "cells": {}}
    t_start = time.time()
    for i, (bench, llm) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {bench}/{llm}  ({time.time()-t_start:.0f}s elapsed)", flush=True)
        cell_out = run_cell(bench, llm, Ns, args.n_trials, args.n_jobs)
        out["cells"][f"{bench}__{llm}"] = cell_out
        # Save incrementally so partial progress isn't lost
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f: json.dump(out, f, indent=2)

    print(f"\n=== done in {(time.time()-t_start)/60:.1f} min — wrote {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
