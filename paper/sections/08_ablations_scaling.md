# 8  Ablations and scaling

## 8.1 Compositional factorisation vs. per-rule regression

We compare three DR variants: (1) RuleOPE with its compositionally
factorised regression; (2) classical DR fit on the same joint feature
space (which is equivalent to the compositional model when used as a
black box); (3) `NonCompositionalDR`, which refits a fresh ridge
regression *only on records where the rule fires* for each rule.

`NonCompositionalDR` is a strawman: it is what one gets by naively
applying DR "per rule" without exploiting shared structure. It is
biased on rules that fire rarely and high-variance everywhere.

Results at $N = 3000$, $500$ rules, stochastic logging, noise $10\%$:

| estimator          | MSE      | bias     | cov95  | tau@20 |
|--------------------|---------:|---------:|-------:|-------:|
| RuleOPE            | 0.00001 | +0.002 | 0.91 | +0.632 |
| DR (compositional) | 0.00001 | +0.002 | 0.91 | +0.632 |
| NonCompDR          | 0.00004 | +0.005 | 0.66 | +0.512 |

The compositional factorisation — shared across rules via a single
ridge fit — yields $4\times$ lower MSE and dramatically better coverage
than the per-rule regression, confirming Theorem 2's prediction in
finite samples. RuleOPE and the classical compositional DR tie at $M =
500$ because the correction-fusion term contributes minimally under
stochastic logging (see §7.4 and §8.2).

## 8.2 Correction-noise sensitivity

We sweep noise $\in \{0, 0.10, 0.20, 0.30, 0.50\}$ and re-run RuleOPE,
DR, and DM at $N = 3000$, 500 rules. MSE stays within $10^{-5}$ across
the sweep for all three estimators, confirming that the correction
signal does not bias the estimator even when corrections are nearly
pure noise. The gate mechanism shrinks $\widehat h^\star$ toward zero
when the correction model has no signal, preserving the DR-family
point estimate.

## 8.3 Rule-depth sensitivity

Stratifying the 500 rules by depth:

| depth | # rules | RuleOPE MSE | DR MSE | DM MSE | RuleOPE tau@10 | DR tau@10 |
|------:|--------:|------------:|-------:|-------:|---------------:|----------:|
| 1 | 166 | 0.00002 | 0.00002 | 0.00002 | +0.911 | +0.911 |
| 2 | 167 | 0.00001 | 0.00001 | 0.00001 | +0.733 | +0.733 |
| 3 | 167 | 0.00000 | 0.00000 | 0.00000 | +0.511 | +0.511 |

Depth-3 rules have the smallest MSE because they fire on narrow
subpopulations with more concentrated reward distributions. The top-$k$
tau degrades with depth because deeper rules have more variance in
ground-truth value, making the top-10 harder to disambiguate. All
DR-family estimators tie at each depth; the compositional factorisation
is working uniformly.

## 8.4 Sample efficiency

Varying $N \in \{250, 500, 1000, 2000, 4000\}$ at 500 rules, stochastic
logging, noise $10\%$:

| N    | RuleOPE MSE | DR MSE | DM MSE |
|-----:|------------:|-------:|-------:|
| 250  | 0.00007 | 0.00007 | 0.00005 |
| 500  | 0.00002 | 0.00002 | 0.00002 |
| 1000 | 0.00001 | 0.00001 | 0.00001 |
| 2000 | 0.00001 | 0.00001 | 0.00001 |
| 4000 | 0.00001 | 0.00001 | 0.00001 |

At $N = 250$ the regression's variance dominates and DM — which is
*fully* regression-driven — edges out the DR family because the DR
correction's importance weights inflate variance with too little data
to compensate. By $N = 500$ DR closes the gap; by $N = 1000$ all three
are at noise floor. RuleOPE does not dominate DM in the very low-data
regime but does not underperform it significantly either.

## 8.5 Scaling in $|\mathcal{R}|$

Rule-set sizes $|\mathcal{R}| \in \{50, 500, 5000\}$ at $N = 3000$,
stochastic logging, noise $10\%$. Reported: MSE, top-$k$ tau (with $k =
\min(20, |\mathcal{R}|/2)$), and wall time.

| $|\mathcal{R}|$ | RuleOPE MSE / t | DR MSE / t | DM MSE / t |
|----------------:|----------------:|-----------:|-----------:|
| 50  | 0.00001 / 8.2s | 0.00001 / 0.9s | 0.00001 / 0.4s |
| 500 | 0.00001 / 17.5s | 0.00001 / 3.2s | 0.00001 / 1.0s |
| 5000 | 0.00000 / 49.1s | 0.00000 / 19.4s | 0.00000 / 7.8s |

RuleOPE's time scales sub-linearly in $|\mathcal{R}|$ (from 8.2s at 50
rules to 49.1s at 5000), reflecting the one-time cost of fitting the
compositional regression and correction gate, plus a linear per-rule
evaluation cost. MSE is preserved across sizes: the compositional
estimator does not accumulate error as the rule set grows, even at the
$100\times$ larger $|\mathcal{R}| = 5000$ setting. A naively per-rule
DR estimator (not shown, from §8.1) would scale linearly in both MSE
and time.
