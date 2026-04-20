# 5  Identification and semiparametric efficiency under deterministic logging

This section is the theoretical centerpiece of the paper. We prove that
under **deterministic logging** — the regime in which production RAG
pipelines actually operate — the target $V(\rho)$ is *not point-
identified* from $(X, A, R)$ alone under the standard OPE assumptions
(A1–A4). We show that classical DR-family estimators converge to a
biased limit within a partial-identification interval. We then
introduce a **bridge-function assumption** (A5) analogous to the
proximal-identification framework (Miao, Geng, and Tchetgen Tchetgen,
2018; Kallus, Mao, and Uehara, 2021) and prove that under A1–A5 the
target becomes point-identified via the correction signal, with a new
efficient influence function that **includes** $C$ as part of the
efficient score. RuleOPE attains the semiparametric efficiency bound;
any estimator ignoring $C$ has strictly larger asymptotic variance,
with a gap we compute explicitly.

## 5.1 The partitioned target

Under deterministic logging $A_i \equiv a_0$. Let $p(x) = \mathbb{1}[\phi_\rho(x) = 1]$
and $q(x) = 1 - p(x)$. Decompose
$$
V(\rho) = \underbrace{\mathbb{E}[q(X) R(X, a_0)]}_{V_0(\rho)\ \text{identified}} + \underbrace{\mathbb{E}[p(X) R(X, a_\rho)]}_{V_1(\rho)\ \text{hard}}.
$$

$V_0(\rho)$ is trivially identified from logged $(X, R)$; the difficulty
lies entirely in $V_1(\rho)$, which requires a counterfactual reward
at an action never taken by the logging policy.

## 5.2 Theorem A — sharp partial-identification bounds under A1–A4

> **Theorem A.** *Assume A1–A4, $R \in [0, 1]$, and deterministic
> logging. $V(\rho)$ is not point-identified from $P$; the sharp identified
> interval is*
> $$
> V(\rho) \in [V_L(\rho), V_U(\rho)], \qquad V_L = \mathbb{E}[q R],\ V_U = V_L + \mathbb{E}[p].
> $$
> *The width of the interval is $\mathbb{E}[p(X)]$, the rule's firing
> probability.*

*Sketch.* Set $R(x, a_\rho) \equiv 0$ or $\equiv 1$; neither choice
conflicts with any identified marginal of $P$ under A4. (See `theory/proofs.tex`
for full proof.)

The practical consequence is severe: for a rule firing on 30\% of
queries, the identification gap is $0.3$ on the reward scale — larger
than any estimator's sampling variance on typical log sizes. No
estimator using only $(X, A, R)$ can close this gap.

## 5.3 Theorem B — DR-family estimators converge to a biased interior point

Under deterministic logging the importance-weight indicator is zero on
every firing record, so DR, CIPS-DR, and Cascade DR reduce to the same
regression-driven formula. Let $m_\star(x, a) = \lim_N \widehat m(x, a)$.

> **Theorem B.** *Under A1–A4 and deterministic logging,*
> $$
> \widehat V^{\mathrm{DR}}(\rho) \xrightarrow{P} V_0(\rho) + \mathbb{E}[p(X)\, m_\star(X, a_\rho)],
> $$
> *with bias $\mathbb{E}[p(X)(m_\star(X, a_\rho) - m(X, a_\rho))]$.*

The DR-family bias is bounded only by the identification gap
$\mathbb{E}[p]$, not by any data-driven quantity: the regression at
$A = a_\rho$ is an out-of-support extrapolation.

## 5.4 A5 — a bridge-function assumption

We now introduce a structural assumption on the correction mechanism
that restores point identification. The form mirrors *bridge functions*
in proximal/negative-control identification (Miao–Geng–Tchetgen-Tchetgen,
2018). Crucially, A5 is strictly stronger than A4 and strictly weaker
than "corrections are counterfactual outcomes."

> **Definition (Action bridge).** *A measurable $b_\rho: \mathcal{X} \to \mathbb{R}$
> is an **action bridge function** if, for P-a.e.\ $x$,*
> $$
> \mathbb{E}\bigl[b_\rho(X)(C - g(X, a_0)) \mid X = x, A = a_0\bigr] = m(x, a_\rho) - m(x, a_0).
> $$

**A5** (bridge existence): an action bridge $b_\rho$ as above exists.

**Sufficient concrete form.** Under the *correction-linearity* model
$$
g(x, a) = \alpha(x) + \beta(a)(1 - m(x, a)), \qquad \beta(a_0) \ne 0,
$$
the bridge is a constant $b_\rho = (\beta(a_\rho) - \beta(a_0))/\beta(a_0)^2$.
$\beta$ can be estimated from a small amount of on-policy exploration
data, or taken as a prior from the correction-calibration literature.

## 5.5 Theorem C — point identification under A1–A5

> **Theorem C.** *Under A1–A5 and deterministic logging,*
> $$
> V(\rho) = \mathbb{E}[q(X) R] + \mathbb{E}[p(X)\, m(X, a_0)] + \mathbb{E}\bigl[p(X)\cdot \mathbb{E}[b_\rho(X)(C - g(X, a_0)) \mid X, A = a_0]\bigr],
> $$
> *and each term is identified from $P_{(X, A=a_0, R, C)}$.*

A5 transforms the unobserved counterfactual $m(X, a_\rho)$ into an
observable functional of $(X, R, C)$.

## 5.6 Theorem D — the efficient influence function

> **Theorem D.** *Under A1–A5 and deterministic logging, the efficient
> influence function of $V(\rho)$ at $P$ is*
> $$
> \psi^\star(O) = m(X, \pi_\rho(X)) - V(\rho) + q(X)(R - m(X, a_0)) + p(X) b_\rho(X)(C - g(X, a_0)),
> $$
> *and the semiparametric efficiency bound is $\mathrm{Var}_P(\psi^\star)$.*

The new term $p(X) b_\rho(X)(C - g(X, a_0))$ is specific to our
formulation. Under A4 alone, $C \perp R \mid X, A$ so $C$ is
conditionally uninformative and the EIF reduces to the classical DR EIF.
Under A5, $C$ enters the efficient score and the efficiency bound
*strictly decreases*.

## 5.7 Theorem E — RuleOPE attains the bound

> **Theorem E.** *Assume A1–A5, deterministic logging, and cross-fitted
> nuisance estimators satisfying the double-ML conditions (product-rate
> on $\widehat m$–$\widehat g$, $o_P(1)$ on $\widehat b_\rho$). Then the
> cross-fitted RuleOPE estimator satisfies*
> $$
> \sqrt{N}(\widehat V^{\mathrm{ROPE}}(\rho) - V(\rho)) \xrightarrow{d} \mathcal{N}(0, \mathrm{Var}_P(\psi^\star)).
> $$

RuleOPE is the unique estimator in the DR family (to our knowledge)
whose influence function coincides with $\psi^\star$ asymptotically;
hence it is semiparametrically efficient under A1–A5.

## 5.8 Theorem F — strict inefficiency of DR-family baselines

> **Theorem F (quantified gap).** *Under A1–A5, deterministic logging,
> and regression convergence $m_\star = m$, any classical DR-family
> estimator has asymptotic variance $\mathrm{Var}_P(\psi^{\mathrm{DR}})$ with*
> $$
> \mathrm{Var}_P(\psi^{\mathrm{DR}}) - \mathrm{Var}_P(\psi^\star) = \mathbb{E}\bigl[p(X)^2 b_\rho(X)^2 g(X, a_0)(1 - g(X, a_0))\bigr] > 0,
> $$
> *whenever $P(p(X) = 1) > 0$ and $b_\rho \not\equiv 0$.*

The gap is non-negligible exactly when (i) the rule fires frequently,
(ii) the correction is informative about the counterfactual contrast,
(iii) the correction probability is bounded away from 0 and 1. These
three conditions obtain in typical RAG deployments.

## 5.9 Summary of the argument

1. Under A1–A4 + deterministic logging, $V(\rho)$ is *not* point-identified;
   the sharp interval has width $\mathbb{E}[p(X)]$ (Thm A).
2. DR / CIPS-DR / CascadeDR converge to a point in this interval
   determined by out-of-support regression extrapolation, with no
   data-driven bias guarantee (Thm B).
3. The bridge assumption A5 restores point identification (Thm C) by
   turning the correction signal into a component of the identified
   functional.
4. The EIF includes a new correction term specific to A5 (Thm D).
5. RuleOPE attains the bound (Thm E); any estimator ignoring $C$ has
   strictly larger variance with a gap we compute explicitly (Thm F).

Items 1–2 explain *why* classical estimators cannot close the
identification gap; items 3–5 explain *how* corrections, under a
structural assumption the practitioner can test, restore identification
and make RuleOPE the unique efficient estimator.

## 5.10 What's new, precisely

The classical DR theorem (Robins et al., 1994) and the standard OPE
efficiency results (Hahn, 1998) assume stochastic logging; under
deterministic logging they give a trivial efficiency bound because
$V(\rho)$ is not identifiable. We extend the semiparametric theory to
the deterministic-logging setting using the correction signal as a
bridge, characterise the partial-identification interval exactly, derive
the new EIF, and quantify the efficiency gap between RuleOPE and every
estimator that ignores $C$. To our knowledge, no existing OPE paper
addresses identification and efficiency in the deterministic-logging
regime via structural assumptions on the correction process.
