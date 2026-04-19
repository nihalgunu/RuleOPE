"""Log record schema and helpers.

A logged record describes one query-response episode of the RAG system,
together with a sparse post-hoc correction signal provided by an expert
annotator or automatic proxy.

Fields
------
query_id          : str
ctx               : dict[str, float]    feature vector used by atoms
logged_action     : str                 action taken by the logging policy
logged_propensity : float               probability of `logged_action` under pi_0
logged_reward     : float               observed reward (answer correctness in [0, 1])
correction        : int                 0/1, sparse (often 0)
cf_rewards        : dict[str, float]    GROUND-TRUTH counterfactual rewards per action;
                                         REMOVED from released logs, kept only for
                                         benchmark ground-truth computation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Iterable, Mapping, Sequence


@dataclass
class LoggedRecord:
    query_id: str
    ctx: dict[str, float]
    logged_action: str
    logged_propensity: float
    logged_reward: float
    correction: int
    cf_rewards: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_cf: bool = False) -> dict:
        d = asdict(self)
        if not include_cf:
            d.pop("cf_rewards", None)
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "LoggedRecord":
        return cls(
            query_id=d["query_id"],
            ctx=dict(d["ctx"]),
            logged_action=d["logged_action"],
            logged_propensity=float(d["logged_propensity"]),
            logged_reward=float(d["logged_reward"]),
            correction=int(d["correction"]),
            cf_rewards=dict(d.get("cf_rewards", {})),
        )


def save_logs(records: Iterable[LoggedRecord], path: str, include_cf: bool = False) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(include_cf=include_cf)) + "\n")


def load_logs(path: str) -> list[LoggedRecord]:
    recs: list[LoggedRecord] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            recs.append(LoggedRecord.from_dict(json.loads(line)))
    return recs
