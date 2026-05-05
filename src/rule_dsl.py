"""Rule DSL.

A rule is a deterministic intervention on the retrieval step of a RAG pipeline.
Each rule is represented as a boolean predicate over the per-query *context*
(query features and per-passage features of the retrieved list), together with
an *action* that describes how the retrieval list is modified when the predicate
fires.  Rules in this work are conjunctive (CNF with a single clause); the
estimator's compositional variance reduction exploits the fact that two rules
that share an atomic predicate induce overlapping regression targets.

Feature vocabulary
------------------
The benchmark uses a finite atom vocabulary shared between substrates:

Query-level atoms
    q_len_gt_k             query token count > k
    q_has_entity_X         query contains entity of type X (person, place, org, time, num)
    q_is_multihop          query requires multi-hop reasoning

Passage-level atoms (applied to top-1 retrieval)
    top1_score_gt_t        reranker score on top-1 passage > t
    top1_score_lt_t        reranker score on top-1 passage < t
    top1_source_is_S       top-1 passage source tag equals S (wiki, stub, blog, forum)
    top1_len_lt_k          top-1 passage length (tokens) < k

List-level atoms
    score_gap_lt_g         score(top1) - score(top2) < g      (retrieval unstable)
    n_above_t_lt_k         fewer than k passages with score > t
    redundancy_gt_r        Jaccard overlap of top-2 passages > r

We fix the atom vocabulary in `ATOMS` below.  Any rule in the benchmark is a
conjunction of a subset of `ATOMS` of size 1-3.

Actions
-------
`filter`  -- drop the top-1 passage and slide the list up
`rerank`  -- swap top-1 with the passage that maximises a secondary score
`abstain` -- force an "I don't know" answer (modelled as reward = r_abstain)

The reward of a query under a rule is defined counterfactually by replaying the
downstream answer model on the modified retrieval list; see
`src/rag_substrate.py`.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Atomic predicates
# ---------------------------------------------------------------------------
# Each atom is a (name, feature_key, threshold, comparator) tuple.  We keep the
# vocabulary small and enumerable so that compositional structure is explicit.

_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "gt": lambda x, t: x > t,
    "lt": lambda x, t: x < t,
    "ge": lambda x, t: x >= t,
    "le": lambda x, t: x <= t,
    "eq": lambda x, t: x == t,
}


@dataclass(frozen=True)
class Atom:
    name: str
    feature: str
    threshold: float
    comparator: str  # one of _COMPARATORS

    def eval(self, ctx: Mapping[str, float]) -> bool:
        return _COMPARATORS[self.comparator](ctx[self.feature], self.threshold)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


def _mk(name: str, feature: str, comparator: str, threshold: float) -> Atom:
    return Atom(name=name, feature=feature, threshold=threshold, comparator=comparator)


# Fixed atom vocabulary: 48 atoms across query, passage, and list scopes.
ATOMS: tuple[Atom, ...] = (
    # query length buckets
    _mk("q_len_gt_8",  "q_len", "gt",  8),
    _mk("q_len_gt_16", "q_len", "gt", 16),
    _mk("q_len_gt_24", "q_len", "gt", 24),
    # query entity presence (one-hot features in [0,1])
    _mk("q_ent_person", "q_has_person", "gt", 0.5),
    _mk("q_ent_place",  "q_has_place",  "gt", 0.5),
    _mk("q_ent_org",    "q_has_org",    "gt", 0.5),
    _mk("q_ent_time",   "q_has_time",   "gt", 0.5),
    _mk("q_ent_num",    "q_has_num",    "gt", 0.5),
    _mk("q_multihop",   "q_multihop",   "gt", 0.5),
    # top-1 score buckets
    _mk("top1_score_gt_0_5", "top1_score", "gt", 0.5),
    _mk("top1_score_gt_0_7", "top1_score", "gt", 0.7),
    _mk("top1_score_gt_0_9", "top1_score", "gt", 0.9),
    _mk("top1_score_lt_0_3", "top1_score", "lt", 0.3),
    _mk("top1_score_lt_0_5", "top1_score", "lt", 0.5),
    # top-1 source tag (one-hot features in [0,1])
    _mk("top1_src_wiki",  "top1_src_wiki",  "gt", 0.5),
    _mk("top1_src_stub",  "top1_src_stub",  "gt", 0.5),
    _mk("top1_src_blog",  "top1_src_blog",  "gt", 0.5),
    _mk("top1_src_forum", "top1_src_forum", "gt", 0.5),
    # top-1 length buckets
    _mk("top1_len_lt_64",  "top1_len", "lt", 64),
    _mk("top1_len_lt_128", "top1_len", "lt", 128),
    _mk("top1_len_gt_256", "top1_len", "gt", 256),
    # score gap atoms (retrieval stability)
    _mk("gap_lt_0_05", "score_gap", "lt", 0.05),
    _mk("gap_lt_0_10", "score_gap", "lt", 0.10),
    _mk("gap_lt_0_20", "score_gap", "lt", 0.20),
    # count of strong candidates
    _mk("n_above_0_5_lt_2", "n_above_0_5", "lt", 2),
    _mk("n_above_0_7_lt_2", "n_above_0_7", "lt", 2),
    _mk("n_above_0_5_lt_3", "n_above_0_5", "lt", 3),
    # redundancy
    _mk("red_gt_0_5", "redundancy", "gt", 0.5),
    _mk("red_gt_0_7", "redundancy", "gt", 0.7),
    _mk("red_gt_0_9", "redundancy", "gt", 0.9),
    # combined "weak retrieval" proxies used by policy families
    _mk("top2_score_gt_0_5", "top2_score", "gt", 0.5),
    _mk("top2_score_gt_0_7", "top2_score", "gt", 0.7),
    _mk("top3_score_gt_0_5", "top3_score", "gt", 0.5),
    _mk("mean_score_gt_0_5", "mean_score", "gt", 0.5),
    _mk("mean_score_lt_0_3", "mean_score", "lt", 0.3),
    # query perplexity / difficulty proxy
    _mk("q_ppl_gt_20", "q_ppl", "gt", 20),
    _mk("q_ppl_gt_40", "q_ppl", "gt", 40),
    _mk("q_ppl_lt_10", "q_ppl", "lt", 10),
    # answer-side proxies (available in logs)
    _mk("gen_conf_lt_0_5", "gen_conf", "lt", 0.5),
    _mk("gen_conf_lt_0_3", "gen_conf", "lt", 0.3),
    _mk("gen_len_gt_32",   "gen_len",  "gt", 32),
    # source-type combinations
    _mk("src_low_trust", "src_low_trust_frac", "gt", 0.33),
    _mk("src_mixed",     "src_entropy",        "gt", 0.8),
    # anchor-entity mismatch
    _mk("ent_missing_top1", "ent_missing_top1", "gt", 0.5),
    _mk("ent_missing_top3", "ent_missing_top3", "gt", 0.5),
    # retrieval position of ground-truth entity mention (available as proxy)
    _mk("first_ent_pos_gt_3", "first_ent_pos", "gt", 3),
    _mk("first_ent_pos_gt_5", "first_ent_pos", "gt", 5),
    _mk("first_ent_pos_lt_1", "first_ent_pos", "lt", 1),
)

ATOM_BY_NAME: dict[str, Atom] = {a.name: a for a in ATOMS}
assert len(ATOM_BY_NAME) == len(ATOMS), "atom names must be unique"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

ACTIONS = ("filter", "rerank", "abstain")


@dataclass(frozen=True)
class Rule:
    """A conjunctive rule.

    The rule `fires` on a context if every atom in `atoms` evaluates True.
    When the rule fires, the retrieval list is modified according to `action`.
    """

    atoms: tuple[Atom, ...]
    action: str  # one of ACTIONS
    name: str = field(default="")

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action: {self.action}")
        if not self.atoms:
            raise ValueError("rule must have at least one atom")
        # canonicalise the atom order for deterministic hashing
        object.__setattr__(self, "atoms", tuple(sorted(self.atoms, key=lambda a: a.name)))
        if not self.name:
            object.__setattr__(self, "name", self._build_name())

    def _build_name(self) -> str:
        return f"{self.action}[{'&'.join(a.name for a in self.atoms)}]"

    def fires(self, ctx: Mapping[str, float]) -> bool:
        return all(a.eval(ctx) for a in self.atoms)

    def atom_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.atoms)

    def depth(self) -> int:
        return len(self.atoms)

    @property
    def id(self) -> str:
        payload = self.name.encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    # Serialisation ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "atoms": list(self.atom_names()),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Rule":
        atoms = tuple(ATOM_BY_NAME[n] for n in d["atoms"])
        return cls(atoms=atoms, action=d["action"], name=d.get("name", ""))


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def enumerate_rules(
    max_depth: int = 3,
    actions: Sequence[str] = ACTIONS,
    atoms: Sequence[Atom] = ATOMS,
    cap_per_depth: int | None = None,
    rng_seed: int = 0,
) -> list[Rule]:
    """Enumerate conjunctive rules up to `max_depth`.

    For depth >= 2 we enumerate all combinations but cap the count via a
    deterministic pseudo-random subsample to keep the benchmark tractable.
    """
    import random

    rng = random.Random(rng_seed)
    rules: list[Rule] = []
    for d in range(1, max_depth + 1):
        combos = list(itertools.combinations(atoms, d))
        if cap_per_depth is not None and len(combos) > cap_per_depth:
            combos = rng.sample(combos, cap_per_depth)
        for combo in combos:
            for action in actions:
                rules.append(Rule(atoms=combo, action=action))
    # Deduplicate by name (different atom orderings collapse via canonicalisation).
    seen: dict[str, Rule] = {}
    for r in rules:
        seen.setdefault(r.name, r)
    return list(seen.values())


def save_rules(rules: Iterable[Rule], path: str) -> None:
    with open(path, "w") as f:
        for r in rules:
            f.write(json.dumps(r.to_dict()) + "\n")


def load_rules(path: str) -> list[Rule]:
    rules = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rules.append(Rule.from_dict(json.loads(line)))
    return rules
