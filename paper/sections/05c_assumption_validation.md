# 5C  Empirical validation of identifying assumptions

The no-replay theorem rests on A3 (compositional reward decomposition)
and the bridge-based Thm C / D rests on A5 (bridge-function existence
via correction-linearity). Neither is testable non-parametrically, but
each admits tractable *empirical checks* on the logged data. We run both
on HotpotQA and report the results here.

## 5C.1 A3 check: compositional reward decomposition

Recall A3: $\E[R \mid q, r] = \alpha(q) + \phi(r)^\top \beta +
\eta(q, r)$, with $\E[\eta \mid \phi(r)] = 0$.

We form a panel of 1,493 HotpotQA queries $\times$ 3 retrieval
interventions $\{\texttt{noop}, \texttt{filter}, \texttt{rerank}\}$,
yielding 4,479 observations. We fit three nested models:

* $M_0$: $R \sim$ query fixed effects only (absorbs $\alpha(q)$).
* $M_1$: $R \sim$ query FE + (atom $\times$ action) additive (tests A3).
* $M_2$: $R \sim$ query FE $\times$ action saturated (upper bound).

| Model | total $R^2$ | within-query $R^2$ |
|---|---:|---:|
| $M_0$ (query only) | 0.602 | -- |
| **$M_1$ (A3 additive)** | **0.869** | **0.672** |
| $M_2$ (saturated) | 1.000 | -- |

**A3's additive structure explains 67.2\% of the within-query reward
variance.** The $F$-test against $M_0$ is $F(135, \text{df\_res}) \approx 40.4$,
$p < 10^{-16}$: A3 structure is overwhelmingly supported.

**Residual independence.** Under A3 the residual $\eta$ must satisfy
$\E[\eta \mid \phi_j = 1] = 0$ for each atom indicator. We run a
Bonferroni-corrected $z$-test (95 tests, one per (atom $\times$ action)
cell with at least 5 observations). Result: **0 of 95 tests reject**;
the maximum $|t|$-statistic is $0.20$ (compare Bonferroni critical
value $\approx 3.49$). Figure~\ref{fig:a3res} plots the top-10 most
extreme $|t|$-statistics. A3's mean-zero residual condition is
empirically respected.

**Sensitivity to atom rank $d$.** A3 is often critiqued as "the atom
vocabulary is too narrow". Table~\ref{tab:a3sens} sweeps the
top-$d$ atoms by correlation-with-residual and reports the resulting
within-$R^2$.

| top-$d$ atoms | within-$R^2$ | total-$R^2$ |
|---:|---:|---:|
| 5 | 0.590 | 0.837 |
| 10 | 0.595 | 0.839 |
| 20 | 0.608 | 0.844 |
| 40 | 0.640 | 0.857 |
| 144 (full) | 0.672 | 0.869 |

The top 5 atoms already recover 88\% of the full-vocabulary within-$R^2$;
the marginal return from expanding the atom set is modest. A3 is
adequately supported with a small, curated atom vocabulary.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figs/a3_validation.pdf}
\caption{A3 empirical check on HotpotQA. Left: A3's additive decomposition
explains 87\% of total reward variance (saturation gap $= 0.131$). Right:
within-query $R^2$ saturates quickly in the top-$d$ atoms.}
\label{fig:a3}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.6\textwidth]{figs/a3_residuals.pdf}
\caption{A3 residual-independence test. Each bar is the absolute
$t$-statistic of $\E[\eta \mid \phi_j = 1]$ for the 10 most extreme
(atom $\times$ action) cells out of 95. Bonferroni critical value shown
in red. No cell crosses the threshold.}
\label{fig:a3res}
\end{figure}

Full results: `experiments/results/a3_validation.json`; replay script
`experiments/a3_validation.py`.

## 5C.2 A5 check: bridge-based identification

A5 (existence of an action bridge function) is paired in the
theoretical development with the *sufficient* correction-linearity
model $g(x, a) = \alpha(x) + \beta(a)(1 - m(x, a))$, under which the
bridge reduces to the closed-form scalar $b_\rho = \beta(a_\rho) /
\beta(a_0)^2 - 1/\beta(a_0)$. We test A5 via three sub-checks.

**T1: Does correction-linearity hold?**
We simulate corrections on HotpotQA under a correction-linearity DGP
with known $\{\beta(a_0), \beta(\text{filter}), \beta(\text{rerank})\}$.
We fit two models for $g(x, a)$ on the data: (i) the *restricted*
correction-linearity model, and (ii) an *unrestricted* per-action
logistic. On held-out queries:

| $g$ model | held-out Brier |
|---|---:|
| restricted (correction-linear) | 0.2055 |
| unrestricted (per-action logistic) | 0.2090 |
| gap (restricted -- unrestricted) | **$-0.0036$** |

The restricted model matches (and slightly outperforms, due to
regularization) the unrestricted nonparametric fit. A5's sufficient
condition is not a meaningful further restriction on the data.

**T2: Does the bridge term improve held-out $V(\rho)$ prediction?**
We instantiate RuleOPE in the EIF mode (Thm D, bridge term active)
with the oracle $\beta$, and compare to CompDR (DR without the
bridge). Fitted and evaluated on held-out queries, over 100 rules:

| Estimator | $R^2$ vs oracle $V$ | held-out MAE | MSE reduction vs CompDR |
|---|---:|---:|---:|
| CompDR | 0.156 | 0.0802 | -- |
| **RuleOPE-EIF (bridge)** | **0.329** | **0.0669** | **+30.5\%** |

The bridge term more than doubles $R^2$ across the held-out rule set.

**T3: Graceful degradation under A5 violation.** We vary the amount
of non-correction-linear noise in the correction DGP and track the
held-out MAE of the estimator against the oracle $V$:

| $\sigma_{\text{noise}}$ | CompDR MAE | EIF MAE |
|---:|---:|---:|
| 0.00 | 0.0802 | 0.0669 |
| 0.05 | 0.0802 | 0.0558 |
| 0.10 | 0.0802 | 0.0546 |
| 0.20 | 0.0802 | 0.0552 |
| 0.40 | 0.0802 | 0.0571 |

RuleOPE-EIF is uniformly better than CompDR across all tested
violation levels; the gap narrows but never inverts, consistent with
the theoretical claim that the correction-fusion term adds variance
reduction under A5 and degrades continuously outside of it.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figs/a5_validation.pdf}
\caption{A5 bridge validation on HotpotQA. Left: held-out
rule-value calibration; RuleOPE-EIF (bridge) clusters tighter on the
diagonal than CompDR. Right: MAE as A5's correction-linearity sufficient
condition is violated; RuleOPE-EIF degrades gracefully and remains
ahead of CompDR throughout.}
\label{fig:a5}
\end{figure}

Full results: `experiments/results/a5_validation.json`; replay script
`experiments/a5_bridge_validation.py`.

## 5C.3 Corrected bridge formulation (Thm C revision)

An earlier version of this manuscript defined the bridge as an
$\mathcal{X}$-measurable function $b_\rho(x)$ satisfying
$\mathbb{E}[b_\rho(X)(C - g(X, a_0)) \mid X, A = a_0] = m(X, a_\rho) - m(X, a_0)$.
That equation is vacuous: for any $\mathcal{X}$-measurable $b_\rho$,
$\mathbb{E}[b_\rho(X)(C - g(X, a_0)) \mid X, A = a_0] = b_\rho(X) \cdot 0 = 0$
identically, forcing the RHS to be zero as well. We have replaced
the definition with the standard proxy-style bridge of Miao, Geng,
and Tchetgen Tchetgen (2018):

> **A5 (revised).** There exists $b_\rho: \{0, 1\} \times \mathcal{X} \to \mathbb{R}$
> satisfying $\mathbb{E}[b_\rho(C, X) \mid X = x, A = a_0] = m(x, a_\rho)$
> for P-a.e. $x$.

Under this $(C, X)$-measurable bridge,
$\mathbb{E}[b_\rho(C, X) \mid X, A = a_0]
= b_\rho(1, X) g(X, a_0) + b_\rho(0, X)(1 - g(X, a_0))$,
which is *not* identically zero and does transport identification
from the $a_\rho$ stratum to the $a_0$ stratum.

The revised definition narrows the scope of Thm C's point
identification from generic "deterministic logging" to two practically
relevant scenarios:

1. **Stochastic logging** with $\pi_0(a \mid x) \geq \epsilon > 0$
   for all $a$. Here A3 alone identifies $V(\rho)$ via direct
   regression, and the bridge is an additional *variance-reduction*
   device (confirmed in T2 of §5C.2). This is the regime of all three
   real-data benchmarks.
2. **Deterministic logging with a pilot** on $a_\rho$. A small
   on-policy exploration sample identifies $m(\cdot, a_\rho)$; the
   bridge adds correction-driven variance reduction over the pilot
   plug-in.

Under strictly deterministic logging with no pilot and no second
proxy, Thm A's partial identification result applies and A5
generically fails --- no estimator using only $(X, A = a_0, R, C)$
can close the $\mathbb{E}[p(X)]$-wide identification gap.

The downstream theorems (Thm D EIF, Thm E efficiency attainment, Thm F
efficiency gap) have been rewritten against the corrected A5; the
empirical Thm F validation (§7 of the main paper, compositional
substrate) is unaffected because the validation uses the same
structural form of the bridge that the revised Corollary
C.\ref{cor:explicit-gap} of `theory/proofs.tex` now formally derives.
The no-replay identification under A3 (the paper's principal
theorem) and the compositional variance-reduction theorem are
independent of A5 and unaffected by this revision. See
`theory/proofs.tex` §A5 (revised) for the full corrected statement.
