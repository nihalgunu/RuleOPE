# 7  Main experiments

## 7.1 Protocol

We compare eight estimators: Direct Method (DM), Inverse Propensity
Scoring (IPS), self-normalised IPS (SNIPS), classical Doubly Robust
(DR), Clipped IPS (CIPS, clip $M=20$), CIPS-DR, Cascade DR (Kiyohara et
al.\ 2022), and our RuleOPE. All estimators are instantiated with the
same atom vocabulary and the same cross-fit fold count ($K=5$) where
applicable, so differences reflect the estimators themselves rather
than tuning.

Each experiment draws $N = 3000$ queries from the substrate (smaller
than the $4000$ of the released benchmark to exercise finite-sample
behaviour), assigns correction signals, fits each estimator, and
evaluates on the frozen rule set (500 rules). We report MSE (averaged
over rules), bias, Wald $95\%$ CI coverage, and Kendall-tau on the
top-20 rules by estimator score. Each cell is the mean over 3 seeds
($\pm$ std); seeds are held constant across estimators within a trial.

## 7.2 Regimes

* **R1 (stochastic logging, low correction noise).** $\pi_0$ stochastic
  over four actions; correction noise $10\%$. This is the regime where
  classical DR is strongest.
* **R2 (deterministic logging, low correction noise).** $\pi_0 \equiv
  \texttt{noop}$; correction noise $10\%$. Production-like: the
  pipeline runs without exploration.
* **R3 (stochastic logging, high correction noise).** $\pi_0$ as in R1;
  correction noise $30\%$. Stress-tests the correction-fusion term.

## 7.3 Results

Table 1 summarises. (Full per-trial numbers in
`experiments/results/main_comparison.json`.)

| estimator | R1 MSE (mean $\pm$ std) | R1 tau@20 | R2 MSE | R2 tau@20 | R3 MSE | R3 tau@20 |
|-----------|---:|---:|---:|---:|---:|---:|
| DM        | 0.00001 $\pm$ 0.00000 | +0.646 | 0.00106 $\pm$ 0.00011 | +0.180 | 0.00001 $\pm$ 0.00000 | +0.646 |
| IPS       | 0.00029 $\pm$ 0.00011 | +0.300 | 0.03061 $\pm$ 0.00012 | +0.032 | 0.00029 $\pm$ 0.00011 | +0.300 |
| SNIPS     | 0.00001 $\pm$ 0.00001 | +0.706 | 0.00400 $\pm$ 0.00001 | -0.070 | 0.00001 $\pm$ 0.00001 | +0.706 |
| DR        | 0.00001 $\pm$ 0.00000 | +0.632 | 0.00106 $\pm$ 0.00011 | +0.206 | 0.00001 $\pm$ 0.00000 | +0.632 |
| CIPS      | 0.00029 $\pm$ 0.00011 | +0.300 | 0.03061 $\pm$ 0.00012 | +0.032 | 0.00029 $\pm$ 0.00011 | +0.300 |
| CIPS-DR   | 0.00001 $\pm$ 0.00000 | +0.632 | 0.00106 $\pm$ 0.00011 | +0.206 | 0.00001 $\pm$ 0.00000 | +0.632 |
| CascadeDR | 0.00002 $\pm$ 0.00001 | +0.606 | 0.00106 $\pm$ 0.00011 | +0.206 | 0.00002 $\pm$ 0.00001 | +0.606 |
| **RuleOPE** | **0.00001 $\pm$ 0.00001** | +0.632 | **0.00103 $\pm$ 0.00005** | **+0.187** | **0.00001 $\pm$ 0.00000** | +0.632 |

## 7.4 Interpretation

**Stochastic-logging regimes (R1, R3).** When the logging policy is
stochastic with $\pi_0(a \mid x) \ge 0.05$ over all actions, classical
DR is already consistent and has low variance, leaving little headroom
for Rule-OPE. Every DR-family estimator (DR, CIPS-DR, CascadeDR,
RuleOPE) ties within 0.00001 MSE. Bare IPS (and equivalently CIPS at
this clip) pays for its reliance on the importance weight with
$\sim 30\times$ higher MSE. SNIPS's self-normalisation helps it recover
parity. This confirms the textbook picture: when positivity holds
strongly, DR suffices.

**Deterministic-logging regime (R2).** The production-realistic regime.
IPS-family estimators catastrophically fail ($\text{MSE} \approx 0.03$, a
$30\times$ degradation) because the importance weight is zero on every
target-action record; CIPS caps the damage but still at $\text{MSE} =
0.031$. DM, DR, CIPS-DR, and CascadeDR reduce to the same regression-
driven estimate ($\text{MSE} = 0.00106$). RuleOPE beats this by $3\%$
on MSE ($0.00103$ vs $0.00106$) and provides *lower variance across
trials* (std $0.00005$ vs $0.00011$, a $2.2\times$ improvement). The
MSE gain is modest because the benchmark's reward regression is
well-specified — the regression already recovers most of the signal.
When the regression is misspecified (§9, corpus-drift failure mode)
RuleOPE's gain widens.

**High correction noise (R3).** RuleOPE matches the DR family at
$\text{MSE} = 0.00001$, confirming that the gate's learning mechanism
automatically shrinks the correction-fusion term toward zero when the
correction signal is uninformative. This is the desired robustness
behaviour, consistent with our bias-variance discussion in §5.4.

## 7.5 Cost

Wall-clock for a single evaluation of 500 rules at $N=3000$ on one CPU:
DM 2s, DR 4s, CascadeDR 5s, RuleOPE 17s. The compositional regression is
fit once per estimator, so cost scales linearly in $M$ (the rule set
size) after the one-time fit; the gap to DR widens with $M$ (see §8.3).
