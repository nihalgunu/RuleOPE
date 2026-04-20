"""15.F  Transductive Rule-OPE — per-query counterfactual prediction
intervals.

Rather than estimate E[R | rule, X], we estimate per-query
R(x_j, pi_rho(x_j)) with a conformal prediction interval.  Practical
value: a practitioner can decide query-by-query whether to apply
a rule.

Implementation: split-conformal calibration on counterfactual
residuals from the cross-fit reward regression at the rule's action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.estimators._regression import (
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
)
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord
from src.rule_dsl import Rule


@dataclass
class TransductiveInterval:
    query_id: str
    point: float
    lower: float
    upper: float
    fires: bool


def per_query_intervals(
    rule: Rule,
    train_logs: Sequence[LoggedRecord],
    eval_logs: Sequence[LoggedRecord],
    delta: float = 0.1,
    cfg: RuleOPEConfig | None = None,
) -> list[TransductiveInterval]:
    cfg = cfg or RuleOPEConfig()
    est = RuleOPE(cfg).fit(train_logs)

    # Calibration residuals: predict logged action, compare to observed reward.
    m_logged_train = est.reg.predict_logged(train_logs).astype(np.float64)
    r_train = np.array([rec.logged_reward for rec in train_logs], dtype=np.float64)
    cal_residuals = np.abs(r_train - m_logged_train)
    n_cal = len(cal_residuals)
    q_idx = int(np.ceil((1.0 - delta) * (n_cal + 1))) - 1
    q_idx = max(0, min(q_idx, n_cal - 1))
    half = float(np.sort(cal_residuals)[q_idx])

    point_eval = est.reg.predict_for_rule(eval_logs, rule).astype(np.float64)
    out = []
    for j, rec in enumerate(eval_logs):
        out.append(
            TransductiveInterval(
                query_id=rec.query_id,
                point=float(point_eval[j]),
                lower=float(point_eval[j] - half),
                upper=float(point_eval[j] + half),
                fires=rule.fires(rec.ctx),
            )
        )
    return out


def coverage(intervals: list[TransductiveInterval], gt: dict[str, float]) -> float:
    inside = 0
    n = 0
    for ti in intervals:
        if ti.query_id not in gt:
            continue
        n += 1
        if ti.lower <= gt[ti.query_id] <= ti.upper:
            inside += 1
    return inside / max(n, 1)
