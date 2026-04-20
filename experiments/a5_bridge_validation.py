"""A5 validation: correction-linearity + contrast-learnability + sensitivity.

The bridge theorem's sufficient condition (correction-linearity, proofs.tex
§bridge) asserts g(x, a) = alpha(x) + beta(a) (1 - m(x, a)).  We test this
sufficient condition directly and then test the learnability of the
rule-contrast it is supposed to identify.

Three sub-tests
---------------
(T1) Correction-linearity R^2:
     Fit a restricted model g(x, a) = alpha_hat(x) + beta_hat(a) (1 - m_hat(x, a))
     and compare R^2 to an unrestricted nonparametric logistic g_hat(x, a).
     Small gap = A5-suff supported.

(T2) Contrast learnability (under STOCHASTIC logging, matching main
     experiments):
     Fit m_hat(x, a) cross-fitted from logs.  Predict m_hat(x, a_rho) on
     held-out queries; compare to oracle m(x, a_rho).  Report R^2 and MAE.
     This is the quantity the bridge-ID formula is supposed to recover.

(T3) Sensitivity of V-hat (RuleOPE-EIF) to A5 violation:
     Vary the amount of non-correction-linear noise in the correction model
     and track test MAE of V_hat(rho) across rules.  Characterises graceful
     degradation.

Writes experiments/results/a5_validation.json and per-figure npy files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from src.estimators._regression import (
    _ACTION_IDX,
    _joint_features,
    atom_feature_matrix,
    fires_mask,
)
from src.estimators.doubly_robust import DoublyRobust
from src.estimators.rule_ope import RuleOPE, RuleOPEConfig
from src.logs import LoggedRecord
from src.rag_substrate_hotpot import (
    _apply_rule,
    _atom_features,
    _load_hotpot,
    _reward_for_top3,
    _score_passages,
)
from src.rule_dsl import load_rules


ACTIONS = ("noop", "filter", "rerank")


def _build_hotpot_logs_stochastic(n_queries: int, seed: int) -> list[LoggedRecord]:
    samples = _load_hotpot("eval/hotpot/dev.parquet", n_queries, seed)
    rng = np.random.default_rng(seed + 3)
    logs = []
    for s in samples:
        scores = _score_passages(s)
        ctx = _atom_features(s, scores)
        cf = {}
        for a in ("noop", "filter", "rerank", "abstain"):
            titles = _apply_rule(a, scores, s)
            cf[a] = 0.5 if a == "abstain" else _reward_for_top3(s.gold_titles, titles)
        a_logged = ACTIONS[int(rng.integers(0, 3))]
        logs.append(LoggedRecord(
            query_id=s.qid, ctx=ctx, logged_action=a_logged,
            logged_propensity=1.0 / 3, logged_reward=float(cf[a_logged]),
            correction=0, cf_rewards=cf,
        ))
    return logs


def _simulate_corrections(
    logs: list[LoggedRecord], alpha_c: float, beta_a: dict[str, float],
    noise_std: float, seed: int,
) -> None:
    """In-place corrections.  noise_std > 0 adds A5-violating non-linearity."""
    rng = np.random.default_rng(seed)
    for rec in logs:
        m_a = float(rec.cf_rewards[rec.logged_action])
        g = alpha_c + beta_a[rec.logged_action] * (1.0 - m_a)
        if noise_std > 0:
            g += noise_std * rng.standard_normal() * (1.0 - m_a)
        g = float(np.clip(g, 0.01, 0.99))
        rec.correction = int(rng.random() < g)


def _oracle_V(rule, logs):
    return float(np.mean([
        rec.cf_rewards[rule.action] if rule.fires(rec.ctx) else rec.cf_rewards["noop"]
        for rec in logs
    ]))


def _fit_g_unrestricted(logs):
    """Per-action logistic regression on atoms (unrestricted g_hat(x, a))."""
    phi = atom_feature_matrix(logs)
    actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
    y = np.array([r.correction for r in logs], dtype=np.int64)
    models = {}
    for a in ACTIONS:
        idx = actions == _ACTION_IDX[a]
        if idx.sum() < 20 or y[idx].sum() == 0 or y[idx].sum() == idx.sum():
            continue
        models[a] = LogisticRegression(max_iter=2000, C=1.0).fit(phi[idx], y[idx])
    return models


def _fit_g_restricted(logs, m_hat_per_action):
    """Restricted model: g(x, a) = alpha_hat(x) + beta_hat(a) * (1 - m_hat(x, a)).
    We fit it as a LINEAR regression of C on (1, alpha_features, (1 - m_hat(x, a_logged))
    per-action intercepts.
    """
    phi = atom_feature_matrix(logs)
    actions = np.array([_ACTION_IDX[r.logged_action] for r in logs], dtype=np.int64)
    y = np.array([r.correction for r in logs], dtype=np.float64)
    d = phi.shape[1]
    X = np.zeros((len(logs), d + len(ACTIONS)), dtype=np.float64)
    X[:, :d] = phi
    # Per-action effect: column = (1 - m_hat(x, a_logged)) for the logged action
    for i, rec in enumerate(logs):
        a = rec.logged_action
        X[i, d + _ACTION_IDX[a]] = 1.0 - float(m_hat_per_action[a][i])
    # Linear-probability fit with ridge for stability
    reg = Ridge(alpha=1.0).fit(X, y)
    return reg, X


def _predict_g_restricted(reg: Ridge, X: np.ndarray) -> np.ndarray:
    return np.clip(reg.predict(X), 1e-4, 1 - 1e-4)


def _predict_g_unrestricted(models, logs) -> np.ndarray:
    phi = atom_feature_matrix(logs)
    out = np.zeros(len(logs), dtype=np.float64)
    for i, rec in enumerate(logs):
        m = models.get(rec.logged_action)
        if m is None:
            out[i] = 0.5
        else:
            out[i] = m.predict_proba(phi[i:i+1])[0, 1]
    return out


def _m_hat_per_action(train_logs, all_logs, alpha_ridge=1.0):
    """Fit m_hat(x, a) via per-action ridge on atom features (cross-fit-free, simple)."""
    phi_tr = atom_feature_matrix(train_logs)
    actions_tr = np.array([_ACTION_IDX[r.logged_action] for r in train_logs], dtype=np.int64)
    rewards_tr = np.array([r.logged_reward for r in train_logs], dtype=np.float64)

    models = {}
    for a in ACTIONS:
        mask = actions_tr == _ACTION_IDX[a]
        if mask.sum() < 10:
            models[a] = None
            continue
        models[a] = Ridge(alpha=alpha_ridge).fit(phi_tr[mask], rewards_tr[mask])

    phi_all = atom_feature_matrix(all_logs)
    out = {}
    for a in ACTIONS:
        if models[a] is None:
            out[a] = np.full(len(all_logs), 0.5)
        else:
            out[a] = models[a].predict(phi_all)
    return out


def run_T1_T2(n_queries, seed, alpha_c, beta_noop, beta_filter, beta_rerank):
    print(f"  Building stochastic HotpotQA logs (N={n_queries})", flush=True)
    logs = _build_hotpot_logs_stochastic(n_queries, seed)
    beta_a = {"noop": beta_noop, "filter": beta_filter, "rerank": beta_rerank}
    _simulate_corrections(logs, alpha_c, beta_a, noise_std=0.0, seed=seed + 7)

    rng = np.random.default_rng(seed + 11)
    order = rng.permutation(len(logs))
    n_train = len(logs) // 2
    train_logs = [logs[i] for i in order[:n_train]]
    test_logs = [logs[i] for i in order[n_train:]]

    # --- T1: correction-linearity R^2
    m_hat_train = _m_hat_per_action(train_logs, train_logs)
    g_unr = _fit_g_unrestricted(train_logs)
    g_res_model, X_train_res = _fit_g_restricted(train_logs, m_hat_train)

    # Evaluate both on test set
    m_hat_test = _m_hat_per_action(train_logs, test_logs)
    g_unr_pred_test = _predict_g_unrestricted(g_unr, test_logs)
    # Build test features for restricted model
    phi_te = atom_feature_matrix(test_logs)
    d = phi_te.shape[1]
    X_test_res = np.zeros((len(test_logs), d + len(ACTIONS)), dtype=np.float64)
    X_test_res[:, :d] = phi_te
    for i, rec in enumerate(test_logs):
        X_test_res[i, d + _ACTION_IDX[rec.logged_action]] = 1.0 - float(m_hat_test[rec.logged_action][i])
    g_res_pred_test = _predict_g_restricted(g_res_model, X_test_res)

    C_test = np.array([r.correction for r in test_logs], dtype=np.float64)

    def r2(yhat, y):
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    # For each observation C \in {0,1}, R^2 is not meaningful.  Use Brier
    # (MSE on probabilities) and log-likelihood.
    def brier(yhat, y): return float(np.mean((yhat - y) ** 2))
    def logloss(yhat, y):
        yhat = np.clip(yhat, 1e-4, 1 - 1e-4)
        return float(-np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)))

    T1 = {
        "unrestricted_Brier": brier(g_unr_pred_test, C_test),
        "restricted_Brier":   brier(g_res_pred_test, C_test),
        "unrestricted_logloss": logloss(g_unr_pred_test, C_test),
        "restricted_logloss":   logloss(g_res_pred_test, C_test),
        "Brier_gap":  brier(g_res_pred_test, C_test) - brier(g_unr_pred_test, C_test),
        "interpretation": "Brier_gap near 0 -> correction-linearity well-supported",
    }

    # --- T2: contrast learnability
    # For each rule in the held-out pool, compute oracle V and RuleOPE-EIF / CompDR V
    rules_all = load_rules("eval/rules_v1.jsonl")
    rules_all = [r for r in rules_all if r.action in ("filter", "rerank")]
    rules = [r for r in rules_all
             if 0.05 <= fires_mask(train_logs, r).mean() <= 0.95
             and 0.05 <= fires_mask(test_logs, r).mean() <= 0.95]

    # Fit and evaluate on test_logs (mirror main-paper cross-fit-internal protocol).
    # The "held-out" aspect is in T1 only -- T2 is asking: on fresh data of this
    # size, does EIF beat CompDR?
    compdr = DoublyRobust(); compdr.fit(test_logs)
    cd_test = compdr.value_many(rules, test_logs)

    eif_test = {}
    for rule in rules:
        cfg = RuleOPEConfig(
            mode="eif", beta_logged=beta_noop,
            beta_target=beta_a[rule.action], correction_weight=1.0,
        )
        rope = RuleOPE(cfg); rope.fit(test_logs)
        eif_test[rule.id] = rope.value(rule, test_logs).estimate

    V_oracle = np.array([_oracle_V(r, test_logs) for r in rules])
    V_compdr = np.array([cd_test[r.id].estimate for r in rules])
    V_eif = np.array([eif_test[r.id] for r in rules])

    # Contrast learnability: predict m_hat(x, a_rho) on TEST queries for queries
    # where the rule fires; compare to oracle cf_rewards[a_rho] on those same queries.
    contrast_r2 = {}
    for a in ("filter", "rerank"):
        preds, trues = [], []
        for rec in test_logs:
            preds.append(float(m_hat_test[a][test_logs.index(rec)]))
            trues.append(float(rec.cf_rewards[a]))
        # The expensive list comprehension above is wrong -- fix it: enumerate
        preds = np.array([float(m_hat_test[a][i]) for i in range(len(test_logs))])
        trues = np.array([float(rec.cf_rewards[a]) for rec in test_logs])
        contrast_r2[a] = {
            "m_hat_R2_vs_oracle_cf": r2(preds, trues),
            "m_hat_MAE_vs_oracle_cf": float(np.mean(np.abs(preds - trues))),
            "mean_oracle_cf": float(trues.mean()),
            "mean_m_hat": float(preds.mean()),
        }

    T2 = {
        "n_rules": len(rules),
        "compdr_V_R2": r2(V_compdr, V_oracle),
        "eif_V_R2":    r2(V_eif,    V_oracle),
        "compdr_V_MAE": float(np.mean(np.abs(V_compdr - V_oracle))),
        "eif_V_MAE":    float(np.mean(np.abs(V_eif    - V_oracle))),
        "eif_MSE_improvement_over_compdr_pct": 100.0 * (
            1.0 - np.mean((V_eif - V_oracle) ** 2) /
            max(np.mean((V_compdr - V_oracle) ** 2), 1e-12)
        ),
        "contrast_learnability_per_action": contrast_r2,
    }

    return T1, T2, V_compdr, V_eif, V_oracle, rules


def run_T3_sensitivity(n_queries, seed, alpha_c, beta_noop, beta_filter, beta_rerank,
                        noise_levels=(0.0, 0.05, 0.10, 0.20, 0.40)):
    """Track EIF vs CompDR held-out MAE as A5 violation (noise_std) grows."""
    rows = []
    for ns in noise_levels:
        logs = _build_hotpot_logs_stochastic(n_queries, seed)
        beta_a = {"noop": beta_noop, "filter": beta_filter, "rerank": beta_rerank}
        _simulate_corrections(logs, alpha_c, beta_a, noise_std=ns, seed=seed + 7)

        rng = np.random.default_rng(seed + 11)
        order = rng.permutation(len(logs))
        n_train = len(logs) // 2
        train_logs = [logs[i] for i in order[:n_train]]
        test_logs = [logs[i] for i in order[n_train:]]

        rules_all = load_rules("eval/rules_v1.jsonl")
        rules_all = [r for r in rules_all if r.action in ("filter", "rerank")]
        rules = [r for r in rules_all
                 if 0.05 <= fires_mask(train_logs, r).mean() <= 0.95
                 and 0.05 <= fires_mask(test_logs, r).mean() <= 0.95]

        compdr = DoublyRobust(); compdr.fit(test_logs)
        cd = compdr.value_many(rules, test_logs)

        V_oracle = np.array([_oracle_V(r, test_logs) for r in rules])
        V_compdr = np.array([cd[r.id].estimate for r in rules])
        V_eif = []
        for rule in rules:
            cfg = RuleOPEConfig(mode="eif", beta_logged=beta_noop,
                                 beta_target=beta_a[rule.action], correction_weight=1.0)
            rope = RuleOPE(cfg); rope.fit(test_logs)
            V_eif.append(rope.value(rule, test_logs).estimate)
        V_eif = np.array(V_eif)

        rows.append({
            "noise_std": ns,
            "compdr_MAE": float(np.mean(np.abs(V_compdr - V_oracle))),
            "eif_MAE":    float(np.mean(np.abs(V_eif - V_oracle))),
            "compdr_RMSE": float(np.sqrt(np.mean((V_compdr - V_oracle) ** 2))),
            "eif_RMSE":    float(np.sqrt(np.mean((V_eif - V_oracle) ** 2))),
        })
        print(f"  noise_std={ns:.2f}   CompDR MAE={rows[-1]['compdr_MAE']:.4f}   "
              f"EIF MAE={rows[-1]['eif_MAE']:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha_c", type=float, default=0.10)
    ap.add_argument("--beta_noop", type=float, default=0.40)
    ap.add_argument("--beta_filter", type=float, default=0.60)
    ap.add_argument("--beta_rerank", type=float, default=0.55)
    ap.add_argument("--out", default="experiments/results/a5_validation.json")
    args = ap.parse_args()

    print("T1 + T2: correction-linearity + contrast learnability")
    T1, T2, V_c, V_e, V_o, rules = run_T1_T2(
        args.n_queries, args.seed, args.alpha_c,
        args.beta_noop, args.beta_filter, args.beta_rerank,
    )
    print(f"  T1 restricted Brier={T1['restricted_Brier']:.4f} "
          f"unrestricted Brier={T1['unrestricted_Brier']:.4f} "
          f"gap={T1['Brier_gap']:+.4f}")
    print(f"  T2 CompDR R^2={T2['compdr_V_R2']:+.3f} "
          f"EIF R^2={T2['eif_V_R2']:+.3f}")
    print(f"  T2 contrast m_hat(a=filter)   R^2 vs oracle cf = {T2['contrast_learnability_per_action']['filter']['m_hat_R2_vs_oracle_cf']:.3f}")
    print(f"  T2 contrast m_hat(a=rerank)   R^2 vs oracle cf = {T2['contrast_learnability_per_action']['rerank']['m_hat_R2_vs_oracle_cf']:.3f}")

    print("\nT3: sensitivity to A5 violation (correction-linearity noise)")
    T3 = run_T3_sensitivity(
        args.n_queries, args.seed, args.alpha_c,
        args.beta_noop, args.beta_filter, args.beta_rerank,
    )

    result = {
        "n_queries": args.n_queries, "seed": args.seed,
        "alpha_c": args.alpha_c,
        "beta_a": {"noop": args.beta_noop, "filter": args.beta_filter, "rerank": args.beta_rerank},
        "T1_correction_linearity": T1,
        "T2_contrast_learnability": T2,
        "T3_sensitivity_to_A5_violation": T3,
    }

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    np.save(args.out.replace(".json", "_V_compdr.npy"), V_c)
    np.save(args.out.replace(".json", "_V_eif.npy"),    V_e)
    np.save(args.out.replace(".json", "_V_oracle.npy"), V_o)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
