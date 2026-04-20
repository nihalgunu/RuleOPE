# 5  Identification and semiparametric efficiency

This section is the theoretical centerpiece of the paper. We first
prove that under **strictly deterministic logging** with no auxiliary
signal, the target $V(\rho)$ is *not point-identified* from
$(X, A, R)$ alone under the standard OPE assumptions (A1–A4), and
classical DR-family estimators converge to a biased limit within a
partial-identification interval. We then introduce a proxy-style
**bridge-function assumption** A5 (Miao, Geng, and Tchetgen Tchetgen,
2018; Kallus, Mao, and Uehara, 2021) on the $(C, X)$-joint law, and
identify two regimes where A5 becomes non-vacuous and $V(\rho)$ is
point-identified:

1. **Stochastic logging** with $\pi_0(a \mid x) \ge \epsilon > 0$ for
   all $a$ (the regime of our three real-data benchmarks), in which A3
   alone already identifies $V(\rho)$ and A5 delivers a **strict
   semiparametric variance reduction** on top.
2. **Deterministic logging + pilot** on $a_\rho$, in which A5 combined
   with a small on-policy sample identifies $V(\rho)$ and reduces
   variance relative to the pilot-only plug-in.

Under strictly deterministic logging with no pilot and no second
proxy, A5 generically fails and Thm A (partial identification)
applies: no estimator using only $(X, A = a_0, R, C)$ can close the
$\mathbb{E}[p(X)]$-wide gap. The earlier version of this section
stated Thm C under generic deterministic logging; the scope has been
tightened, and the downstream EIF, efficiency-attainment, and
efficiency-gap theorems (Thm D, E, F) have been re-derived against
the corrected A5. See §5C.3 and `theory/proofs.tex` for the full
revision notes.

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

## 5.4 A5 — a proxy-style bridge-function assumption (revised)

We now introduce a structural assumption on the correction mechanism.
The form mirrors *bridge functions* in proximal/negative-control
identification (Miao, Geng, and Tchetgen Tchetgen, 2018). Crucially,
A5 is strictly stronger than A4 (which treats $C$ as a nuisance
orthogonal to $R$) and strictly weaker than "corrections are
counterfactual outcomes."

> **Definition (Action bridge, revised).** *A measurable
> $b_\rho: \{0, 1\} \times \mathcal{X} \to \mathbb{R}$
> is an **action bridge function** if, for P-a.e.\ $x$,*
> $$
> \mathbb{E}\bigl[b_\rho(C, X) \mid X = x, A = a_0\bigr] \;=\; m(x, a_\rho).
> $$

**A5** (bridge existence): a $(C, X)$-measurable action bridge
$b_\rho$ as above exists.

The earlier version of this paper used an $\mathcal{X}$-measurable
$b_\rho(X)$ multiplied by the residual $(C - g(X, a_0))$. That
conditional expectation is identically zero, so the left-hand side
was vacuous. The corrected form above, taken from Miao–Geng–Tchetgen-
Tchetgen (2018), does transport identification from the $a_\rho$
stratum to the $a_0$ stratum whenever $b_\rho$ depends *both* on $C$
and on $X$. See §5C.3 and `theory/proofs.tex` §Thm C (revised).

**Sufficient concrete form.** Under the *correction-linearity* model
$$
g(x, a) = \alpha(x) + \beta(a)(1 - m(x, a)), \qquad \beta(a_0) \ne 0,
$$
one explicit $(C, X)$-measurable solution is
$$
b_\rho(C, X) \;=\; m(X, \pi_\rho(X)) \;+\; \tfrac{\beta(a_\rho) - \beta(a_0)}{\beta(a_0)^2}\,(C - g(X, a_0)),
$$
whose conditional mean at $A = a_0$ equals $m(X, a_\rho)$. The scalar
$(\beta(a_\rho) - \beta(a_0))/\beta(a_0)^2$ is the *optimal weight on
the correction residual* in the efficient score; $\beta$ is
identifiable from stochastic logging (scenario (i)) or from the pilot
sample (scenario (ii)).

**When is A5 non-vacuous?** A $(C, X)$-measurable bridge can only
identify $m(X, a_\rho)$ when $(X, C)$ carries enough information about
the $A = a_\rho$ stratum. We isolate two practically relevant regimes:

* **Scenario (i): stochastic logging** with $\pi_0(a \mid x) \ge \epsilon > 0$.
  Here A3 already identifies $V(\rho)$ via direct regression on
  $(X, A, R)$; A5 adds a *variance-reduction* device through $C$.
  *(This is the regime of §7C: HotpotQA, TriviaQA, MuSiQue all use
  uniform-stochastic logging over* $\{\texttt{noop, filter, rerank}\}$.*)*
* **Scenario (ii): deterministic logging + pilot.** A small on-policy
  sample on $a_\rho$ identifies $m(\cdot, a_\rho)$ directly; the
  bridge adds correction-driven variance reduction over the pilot
  plug-in.

Under strictly deterministic logging with *no* pilot and no second
proxy, A5 generically fails and Thm A applies.

## 5.5 Theorem C — point identification under A1, A3, A5 (revised)

> **Theorem C (revised).** *Under A1, A3, and A5 (revised bridge) in
> either scenario (i) or (ii) of §5.4,*
> $$
> V(\rho) \;=\; \mathbb{E}[q(X)\, R] \;+\; \mathbb{E}\bigl[p(X) \cdot \mathbb{E}[b_\rho(C, X) \mid X, A = a_0]\bigr],
> $$
> *and each term is identified from $P_{(X, A=a_0, R, C)}$ together
> with whichever auxiliary signal (stochastic logging at $a_\rho$, or
> pilot data on $a_\rho$) makes A5 non-vacuous.*

*Scope change from earlier draft.* The earlier Thm C claimed point
identification under generic deterministic logging with an
$\mathcal{X}$-measurable bridge; that claim was vacuous because the
earlier bridge residual had conditional mean zero identically.
Theorem C is now stated under scenarios (i)–(ii) only, and is proved
in `theory/proofs.tex` §Thm C (revised) against the corrected
$(C, X)$-measurable A5.

## 5.6 Theorem D — the efficient influence function (revised)

> **Theorem D (revised).** *Under A1, A3, A5 and stochastic logging
> (scenario (i)), the efficient influence function of $V(\rho)$ at $P$
> is*
> $$
> \psi^\star(O) \;=\; \underbrace{m(X, \pi_\rho(X)) - V(\rho)}_{\text{DM plug-in}}
> + \underbrace{\tfrac{\mathbb{1}[A = \pi_\rho(X)]}{\pi_0(A \mid X)}\,(R - m(X, A))}_{\text{DR at logged action}}
> + \underbrace{p(X)\bigl(b_\rho(C, X) - \mathbb{E}[b_\rho(C, X) \mid X, A = a_0]\bigr)}_{\text{bridge variance reduction}},
> $$
> *and the semiparametric efficiency bound is $\mathrm{Var}_P(\psi^\star)$.*

Under A4 alone, $C \perp R \mid X, A$ and the bridge-variance-reduction
term has zero contribution, recovering the classical DR EIF (Robins,
Rotnitzky, and Zhao, 1994). Under A5 the bridge term has strictly
positive $L^2$-mass whenever $b_\rho(1, X) \ne b_\rho(0, X)$ with
positive probability, and the efficiency bound strictly decreases.
The EIF under scenario (ii) is analogous with the pilot sample
entering the DR-logged term.

## 5.7 Theorem E — RuleOPE attains the bound (revised)

> **Theorem E (revised).** *Assume A1, A3, A5, stochastic logging with
> $\pi_0(a \mid x) \ge \epsilon$, and cross-fitted nuisance estimators
> satisfying the double-ML conditions (product-rate on
> $\widehat m \!\cdot\! \widehat \pi_0$; $o_P(1)$ on $\widehat b_\rho$
> and $\widehat g$). Then the cross-fitted RuleOPE estimator satisfies*
> $$
> \sqrt{N}(\widehat V^{\mathrm{ROPE}}(\rho) - V(\rho)) \xrightarrow{d} \mathcal{N}(0, \mathrm{Var}_P(\psi^\star)).
> $$

RuleOPE is the unique estimator in the DR family (to our knowledge)
whose influence function coincides with $\psi^\star$ asymptotically;
hence it is semiparametrically efficient in scenarios (i)–(ii).

## 5.8 Theorem F — strict inefficiency of DR-family baselines (revised)

> **Theorem F (revised, quantified gap).** *Under A1, A3, A5 and
> scenario (i), any DR-family estimator $\widehat V^{\mathrm{DR}}$
> that uses only $(X, A, R)$ has asymptotic-variance gap*
> $$
> \mathrm{Var}_P(\psi^{\mathrm{DR}}) - \mathrm{Var}_P(\psi^\star) \;=\; \mathrm{Var}_P\!\Bigl(p(X)\bigl(b_\rho(C, X) - \mathbb{E}[b_\rho(C, X) \mid X, A = a_0]\bigr)\Bigr) \;\ge\; 0,
> $$
> *strictly positive whenever $P(p(X) = 1) > 0$ and $b_\rho(1, X) \ne b_\rho(0, X)$ with positive probability.*

**Corollary (explicit gap under correction-linearity).** Substituting
the correction-linear bridge of §5.4 yields
$$
\mathrm{Var}_P(\psi^{\mathrm{DR}}) - \mathrm{Var}_P(\psi^\star)
\;=\; \bigl(\tfrac{\beta(a_\rho) - \beta(a_0)}{\beta(a_0)^2}\bigr)^{\!2}\;\mathbb{E}\bigl[p(X)^2\, g(X, a_0)(1 - g(X, a_0))\bigr] \;>\; 0.
$$

The gap is non-negligible exactly when (i) the rule fires frequently,
(ii) corrections discriminate across actions
($\beta(a_\rho)/\beta(a_0) \ne 1$), (iii) the correction probability
is bounded away from 0 and 1. These three conditions obtain in
typical RAG deployments.

## 5.9 Summary of the argument (revised)

1. Under A1–A4 and *strictly* deterministic logging with no auxiliary
   signal, $V(\rho)$ is *not* point-identified; the sharp interval
   has width $\mathbb{E}[p(X)]$ (Thm A).
2. DR / CIPS-DR / CascadeDR converge to a point in this interval
   determined by out-of-support regression extrapolation, with no
   data-driven bias guarantee (Thm B).
3. Under A3 and A5 (corrected bridge), in either scenario (i)
   stochastic logging or (ii) deterministic logging + pilot,
   $V(\rho)$ is point-identified (Thm C, revised). Under strictly
   deterministic logging with no pilot and no second proxy, A5
   generically fails and item 1 applies.
4. The EIF includes a bridge variance-reduction term
   $p(X)\bigl(b_\rho(C, X) - \mathbb{E}[b_\rho(C, X) \mid X, A = a_0]\bigr)$
   with conditional mean zero (Thm D, revised).
5. RuleOPE attains the semiparametric efficiency bound in scenario
   (i) / (ii) (Thm E, revised); any estimator ignoring $C$ has
   strictly larger variance by $\mathrm{Var}(\xi) > 0$ (Thm F,
   revised).

Items 1–2 explain *why* estimators that ignore $C$ cannot beat the
partial-identification gap. Items 3–5 explain *when* and *how*
corrections, under a structural assumption the practitioner can
empirically check (§5C.2), restore identification (scenario (ii)) or
reduce variance (scenario (i)).

## 5.10 What's new, precisely

The classical DR theorem (Robins et al., 1994) and the standard OPE
efficiency results (Hahn, 1998) assume stochastic logging over the
target action; under strictly deterministic logging they give a
trivial efficiency bound because $V(\rho)$ is not identifiable. We
contribute:

(a) an explicit partial-identification interval under
A1–A4 + strict deterministic logging (Thm A), with the sharp bias of
DR-family baselines (Thm B);

(b) a $(C, X)$-measurable proxy-style bridge assumption A5
(§5.4, revised) that is non-vacuous in two named scenarios —
stochastic logging and deterministic logging with pilot — and whose
sufficient condition (correction-linearity) is empirically checkable
(§5C.2);

(c) the revised EIF (Thm D) whose bridge-variance-reduction term is
an $L^2$-orthogonal projection onto the $C$-tangent subspace, RuleOPE's
attainment of the resulting bound (Thm E), and an explicit-form
efficiency gap against any $C$-ignoring DR-family baseline (Thm F,
with correction-linearity corollary).

The earlier draft stated the identification claim under generic
deterministic logging via an $\mathcal{X}$-measurable bridge; that
formulation was vacuous. The present version restricts scope to the
two scenarios above and uses the standard Miao–Geng–Tchetgen-Tchetgen
$(C, X)$-measurable bridge. To our knowledge, this is still the first
formal treatment of OPE identification and efficiency in RAG settings
with a structured correction signal.
