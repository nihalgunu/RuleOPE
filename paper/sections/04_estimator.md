# 4  The RuleOPE estimator

## 4.1 Overview

RuleOPE is a doubly-robust estimator with three additive components:
$$
\widehat V^{\text{ROPE}}(\rho) = \underbrace{\widehat V^{\text{DM}}_C(\rho)}_{\text{compositional DM}} + \underbrace{\widehat \Delta^{\text{logged}}(\rho)}_{\text{DR correction}} + \underbrace{\widehat \Delta^{\text{corr}}(\rho)}_{\text{correction fusion}}.
$$
The first term is a Direct Method estimate that uses a *compositionally
factorised* reward regression. The second is the standard DR correction
at records where the logged action coincides with the target action.
The third is new: it uses the correction signal $C_i$ to partially
identify the reward under target actions that were never logged, so the
estimator is not reduced to pure DM when logging is deterministic.

## 4.2 Compositional reward regression

We parameterise the conditional-mean reward as
$$
m_\theta(x, a) = \beta_{0, a} + \phi(x)^\top \beta_a, \qquad \beta_a \in \mathbb{R}^d.
$$
Two rules $\rho, \rho'$ that share atom $\alpha \in S_\rho \cap S_{\rho'}$
share the coefficient $\beta_{\alpha, a}$ whenever $a_\rho = a_{\rho'}$.
We fit $\theta$ by ridge regression on the logged data with
regularisation $\lambda$, using $K$-fold cross-fitting (Chernozhukov et
al.\ 2018) to avoid overfitting bias in the downstream DR correction.

### Design decision: why atom indicators rather than raw features?

An alternative is to use raw continuous features (scores, lengths, perplexity)
directly in the regression. We chose atom indicators for three reasons:
(1) the atom vocabulary is the same object the rules are defined over, so
the regression's effective parameters are exactly the objects the theory
needs to bound; (2) binarised indicators make the regression's effective
degrees of freedom $O(d)$ independent of feature scaling; (3) the resulting
regression is easy to debug — each $\beta_{\alpha, a}$ is directly
interpretable as "the reward bonus from this atom firing under this
action." The cost is a small amount of approximation error whenever the
true reward is non-monotone in a feature; we treat this as a modelling
assumption and check its impact in the sample-efficiency ablation (§8.4).

## 4.3 The compositional Direct Method term

$$
\widehat V^{\text{DM}}_C(\rho) = \frac{1}{N}\sum_{i=1}^N \widehat m(X_i, \pi_\rho(X_i)),
$$
where $\pi_\rho(X_i) = a_\rho$ if $F_\rho(X_i) = 1$ else $a_0$. Cross-
fitted predictions are used to avoid leakage.

## 4.4 The logged-action DR correction

$$
\widehat \Delta^{\text{logged}}(\rho) = \frac{1}{N}\sum_{i=1}^N \frac{\mathbb{1}[A_i = \pi_\rho(X_i)]}{\pi_0(A_i \mid X_i)}\bigl(R_i - \widehat m(X_i, A_i)\bigr).
$$
This is the standard DR correction (Robins et al.\ 1994). Under
A1–A3 it has zero population mean, so it does not bias the estimator;
its purpose is variance reduction.

## 4.5 The correction-fusion term

Under deterministic logging, $\widehat \Delta^{\text{logged}}(\rho)$
vanishes for every record where $F_\rho(X_i) = 1$ and $a_\rho \ne a_0$,
because $A_i = a_0 \ne \pi_\rho(X_i)$. The estimator collapses to
$\widehat V^{\text{DM}}_C$, inheriting all of its bias. The correction
signal $C_i$ is the only source of partial counterfactual information
available to rescue us.

We define
$$
\widehat \Delta^{\text{corr}}(\rho) = \frac{1}{N}\sum_{i=1}^N C_i \cdot h_i(\rho) \cdot \bigl(\tilde r_i(\rho) - \widehat m(X_i, \pi_\rho(X_i))\bigr)
$$
where
* $h_i(\rho) = \mathbb{1}[\pi_\rho(X_i) \ne A_i] \cdot \widehat h^\star(X_i, \rho)$
  is a gate that is active only when the target action differs from the
  logged one, weighted by a learnt *correction-informativeness* factor
  $\widehat h^\star \in [0, \text{clip}]$;
* $\widehat h^\star(x, \rho) = \max\!\left(0, 1 - \widehat g(x, a_\rho)/\widehat g(x, a_0)\right) / \max(\widehat g(x, a_0), \epsilon)$
  is the model's estimate of how much the correction probability *decreases*
  under $a_\rho$ relative to $a_0$; large $\widehat h^\star$ indicates the
  correction is highly informative that $a_\rho$ would have helped;
* $\widehat g(x, a) = \widehat P(C = 1 \mid X = x, A = a)$ is a learnt
  logistic regression on the same joint feature space as $\widehat m$;
* $\tilde r_i(\rho)$ is a pseudo-reward imputation: $r_{\text{abs}}$ for
  $a_\rho = \texttt{abstain}$; $\max(\widehat m(X_i, a_\rho), r_{\text{abs}})$
  for $a_\rho \in \{\texttt{filter}, \texttt{rerank}\}$. The abstain
  baseline reflects the minimum reward attainable by refusing to answer.

### Design decision: why gate on $\widehat g(x, a_\rho)/\widehat g(x, a_0)$?

The ratio captures "the correction is less likely under $a_\rho$ than
under $a_0$" — the natural formalisation of "the expert would not have
corrected $a_\rho$." Under A4, $g(x, a)$ is identified (from logged data
at action $a$ when $a$ is in the support of $\pi_0$) and the ratio is a
well-defined function. In the deterministic-logging regime where $a_\rho$
is not in the logging support, $\widehat g(\cdot, a_\rho)$ is identified
only through the regression's *extrapolation*; we clip $\widehat h^\star$
to $[0, 5]$ to protect against pathological extrapolation. Sensitivity
to the clip is examined in §8.

## 4.6 Cross-fitting, standard errors, and confidence intervals

We use $K = 5$ folds. Within each fold, the regression and correction-
gate models are trained on the other $K-1$ folds and used to produce
out-of-fold predictions for records in the held-out fold; the entire
estimator is then evaluated on those out-of-fold predictions. Standard
errors are the empirical standard deviation of the per-record influence
function $\psi_i(\rho)$ divided by $\sqrt{N}$; 95% confidence intervals
are $\widehat V(\rho) \pm 1.96\, \widehat{\mathrm{SE}}(\rho)$.

## 4.7 Algorithm

```
Input: logs {(X_i, A_i, R_i, C_i)}_{i=1}^N, rule rho, K folds
1. Fit reward regression m_hat by cross-fitted ridge over joint atom*action features.
2. Fit correction gate g_hat by cross-fitted logistic regression.
3. For each i:
   a. Compute pi_rho(X_i).
   b. Compute m_hat(X_i, pi_rho(X_i)) and m_hat(X_i, A_i).
   c. Compute w_i = 1[A_i = pi_rho(X_i)] / pi_0(A_i | X_i).
   d. Compute h_i = 1[pi_rho(X_i) != A_i] * h_star(X_i, rho).
   e. Compute tilde r_i per the pseudo-reward rule.
   f. psi_i = m_hat(X_i, pi_rho(X_i))
            + w_i * (R_i - m_hat(X_i, A_i))
            + C_i * h_i * (tilde r_i - m_hat(X_i, pi_rho(X_i))).
4. Return V_hat = mean_i psi_i, SE = std_i(psi_i) / sqrt(N).
```
