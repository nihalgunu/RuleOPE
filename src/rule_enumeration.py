"""Rule enumeration helpers.

The `rule_dsl.enumerate_rules` function produces the combinatorial universe of
candidate rules.  For the frozen benchmark we further subselect rules that
(a) fire at least `min_fires` times on the logged data to keep ground-truth
estimates meaningful, and (b) are a mix of depths 1, 2, 3.  This module wraps
that selection.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

from src.rule_dsl import Rule, enumerate_rules


def select_rules_from_logs(
    contexts: Sequence[Mapping[str, float]],
    max_depth: int = 3,
    cap_per_depth: int = 400,
    min_fires: int = 50,
    target_count: int = 500,
    rng_seed: int = 0,
) -> list[Rule]:
    """Return ~`target_count` rules with reasonable coverage on `contexts`.

    We enumerate candidate rules, drop those that fire fewer than `min_fires`
    times (insufficient support for ground-truth value estimation), and
    stratify by depth so that all three depths are represented.
    """
    import random

    candidates = enumerate_rules(
        max_depth=max_depth, cap_per_depth=cap_per_depth, rng_seed=rng_seed
    )

    # Compute firing counts for every candidate.
    counts: Counter[str] = Counter()
    for r in candidates:
        c = 0
        for ctx in contexts:
            if r.fires(ctx):
                c += 1
        counts[r.name] = c

    by_depth: dict[int, list[Rule]] = {1: [], 2: [], 3: []}
    for r in candidates:
        if counts[r.name] >= min_fires:
            by_depth.setdefault(r.depth(), []).append(r)

    rng = random.Random(rng_seed)
    # stratify so no depth dominates
    per_depth = max(1, target_count // max_depth)
    chosen: list[Rule] = []
    for d, pool in by_depth.items():
        rng.shuffle(pool)
        chosen.extend(pool[:per_depth])

    # If we are short, top up with remaining candidates (deterministically).
    if len(chosen) < target_count:
        remaining = [
            r
            for r in candidates
            if counts[r.name] >= min_fires and r not in chosen
        ]
        rng.shuffle(remaining)
        chosen.extend(remaining[: target_count - len(chosen)])

    return chosen[:target_count]
