"""Evaluation metrics for rule-OPE estimators.

Three target metrics from the project spec:
    * MSE of estimated vs ground-truth value, averaged over rules.
    * Coverage of 95% Wald CIs at nominal rate.
    * Kendall's tau between estimator ranking and ground-truth ranking on top-k.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, norm


def mse(
    estimates: Mapping[str, float], ground_truth: Mapping[str, float]
) -> float:
    keys = sorted(set(estimates) & set(ground_truth))
    if not keys:
        return float("nan")
    e = np.array([estimates[k] for k in keys], dtype=np.float64)
    g = np.array([ground_truth[k] for k in keys], dtype=np.float64)
    return float(((e - g) ** 2).mean())


def bias(estimates: Mapping[str, float], ground_truth: Mapping[str, float]) -> float:
    keys = sorted(set(estimates) & set(ground_truth))
    if not keys:
        return float("nan")
    e = np.array([estimates[k] for k in keys], dtype=np.float64)
    g = np.array([ground_truth[k] for k in keys], dtype=np.float64)
    return float((e - g).mean())


def coverage_95(
    estimates: Mapping[str, float],
    stderrs: Mapping[str, float],
    ground_truth: Mapping[str, float],
) -> float:
    """Fraction of rules whose true value lies in estimate +/- 1.96 * stderr."""
    z = norm.ppf(0.975)
    keys = sorted(set(estimates) & set(ground_truth) & set(stderrs))
    if not keys:
        return float("nan")
    hits = 0
    for k in keys:
        lo = estimates[k] - z * stderrs[k]
        hi = estimates[k] + z * stderrs[k]
        if lo <= ground_truth[k] <= hi:
            hits += 1
    return hits / len(keys)


def topk_tau(
    estimates: Mapping[str, float],
    ground_truth: Mapping[str, float],
    k: int = 20,
) -> float:
    """Kendall's tau on the top-k rules selected by the estimator.

    The ranking is induced by the estimator's ordering of the top-k rules; the
    comparator is that ordering against the *true* ordering of those same
    rules.  If the estimator picks a different top-k than the oracle, the tau
    still measures whether the ordering within the estimator's top-k is
    consistent with reality.
    """
    keys = sorted(set(estimates) & set(ground_truth))
    if len(keys) < 2:
        return float("nan")
    e = np.array([estimates[k] for k in keys], dtype=np.float64)
    g = np.array([ground_truth[k] for k in keys], dtype=np.float64)
    order_e = np.argsort(-e)[: min(k, len(keys))]
    tau, _ = kendalltau(e[order_e], g[order_e])
    return float(tau) if np.isfinite(tau) else 0.0


def all_metrics(
    estimates: Mapping[str, float],
    stderrs: Mapping[str, float],
    ground_truth: Mapping[str, float],
    topk: int = 20,
) -> dict[str, float]:
    return {
        "mse": mse(estimates, ground_truth),
        "bias": bias(estimates, ground_truth),
        "coverage_95": coverage_95(estimates, stderrs, ground_truth),
        "topk_tau": topk_tau(estimates, ground_truth, topk),
    }
