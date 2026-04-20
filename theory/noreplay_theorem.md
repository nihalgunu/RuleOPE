# No-Replay Identification Theorem for RAG Rule-OPE

## Setting

A RAG pipeline consists of:
- A query distribution $q \sim P$.
- A retrieval function $r: \mathcal{Q} \to \mathcal{R}$ mapping queries to retrieved passages.
- A *black-box* generator $G: \mathcal{Q} \times \mathcal{R} \to \mathcal{Y}$ mapping (query, retrieval) to an answer.
- A reward function $R: \mathcal{Q} \times \mathcal{Y} \to [0, 1]$ scoring answer quality.

Logging policy $r_0 = \text{noop}$: applies the stock retriever.

We have a log $\mathcal{D} = \{(q_i, x_i, y_i, R_i)\}_{i=1}^n$ where $x_i$ is a vector of retrieval features, $y_i = G(q_i, r_0(q_i))$ is the logged generator output, and $R_i = R(q_i, y_i)$.

A *rule* $\rho$ is a deterministic intervention on retrieval: when $\rho$ fires on $x_i$ it maps $r_0(q_i) \mapsto r_\rho(q_i)$, where $r_\rho$ is a known function (filter top-1, rerank, etc.).

The target: $V(\rho) = \mathbb{E}_q[R(q, G(q, r_\rho(q)))]$.

## The no-replay constraint

Classical OPE assumes we can compute or estimate $R(q, G(q, r_\rho(q)))$ for any query. In RAG this requires **re-running the generator** on each counterfactual retrieval $r_\rho(q)$ — a cost that makes offline evaluation with large rule pools or large query logs infeasible.

**Problem**: Estimate $V(\rho)$ from $\mathcal{D}$ alone, without evaluating $G$ at any counterfactual $(q_i, r_\rho(q_i))$.

## Assumptions

- **A1 (positivity)**: for each $\rho$ there exists $\epsilon > 0$ such that the marginal density of $r_\rho(q)$ is bounded below on its support under $P$.
- **A2 (observable rule features)**: the rule-firing mask $\mathbb{1}[\rho(x) = 1]$ is a measurable function of the logged retrieval features.
- **A3 (compositional reward)**: there exists an atom feature map $\phi: \mathcal{R} \to \{0,1\}^d$ and coefficients $\beta \in \mathbb{R}^d$ such that
  $$
  \mathbb{E}_y[R(q, y) \mid q, r] = \alpha(q) + \phi(r)^\top \beta + \eta(q, r),
  $$
  where $\eta(q, r)$ is mean-zero given the atom indicators: $\mathbb{E}[\eta(q, r) \mid \phi(r)] = 0$.

A3 is the **no-replay identifying assumption**: it posits that the conditional reward decomposes into a query-intrinsic component $\alpha(q)$ (observable as the logged reward averaged over queries sharing $\phi(r_0(q))$) and a retrieval-quality component $\phi(r)^\top \beta$ (estimable from the log via ridge regression on atom indicators).

Note A3 is *strictly weaker* than standard "rewards are fully determined by $\phi$" — we allow non-linear query-specific noise $\eta$, which is unavoidable in practice because the generator is non-linear.

## Theorem (No-Replay Identification)

Under A1–A3, the rule value $V(\rho)$ is identified by
$$
V(\rho) = \mathbb{E}_q\left[ \alpha(q) + \phi(r_\rho(q))^\top \beta \right],
$$
where $\alpha$ is the query-intrinsic reward bias and $\beta$ is the atom coefficient vector, both **estimable from the logged tuples $(q_i, r_0(q_i), R_i)$ alone**.

*Proof sketch.* By A3, for any retrieval $r$ and query $q$,
$$
\mathbb{E}_y[R(q, y) \mid q, r] = \alpha(q) + \phi(r)^\top \beta + \eta(q, r).
$$
Averaging over $q$,
$$
V(\rho) = \mathbb{E}_q\big[\alpha(q) + \phi(r_\rho(q))^\top \beta + \mathbb{E}_y[\eta(q, r_\rho(q))]\big]
        = \mathbb{E}_q[\alpha(q)] + \mathbb{E}_q[\phi(r_\rho(q))^\top \beta].
$$
The second equality uses the zero-mean condition on $\eta$. The first term is identified by the empirical mean of $R_i$; the second by plugging the known rule-modified retrieval features $\phi(r_\rho(q_i))$ into the ridge estimate of $\beta$ fit on the log. Neither step requires evaluating $G$ at any counterfactual retrieval.

## RuleOPE as the efficient no-replay estimator

The RuleOPE estimator of the current paper is
$$
\widehat V^{\text{RuleOPE}}(\rho) = \frac{1}{n}\sum_{i=1}^n \Big[ \widehat m_\rho(x_i) + \frac{\mathbb{1}[A_i = \pi_\rho(x_i)]}{\widehat \pi_0(A_i \mid x_i)} (R_i - \widehat m(x_i, A_i)) + p(x_i)(\widehat b_\rho(c_i, x_i) - \widehat{\mathbb{E}}[\widehat b_\rho(C, x_i) \mid x_i, A = a_0]) \Big],
$$
where $\widehat m_\rho(x_i) = \widehat m(x_i, \pi_\rho(x_i))$ is the atom-additive ridge estimate, and the correction-fusion term is the *variance-reduction projection* of the correction residual onto the counterfactual contrast (see `theory/proofs.tex` §A5 revised for the Miao–Geng–Tchetgen-style proxy-bridge derivation). The bridge $\widehat b_\rho$ is $(C, X)$-measurable in the corrected formulation.

**Proposition (Efficiency under A3 + stochastic logging).** Under A1–A3 and stochastic logging with positivity on every action, the influence function of $\widehat V^{\text{RuleOPE}}$ is the semiparametric efficient influence function for $V(\rho)$: no estimator using only $(q_i, r_0(q_i), A_i, R_i, C_i)$ can achieve strictly lower asymptotic variance. Under strictly deterministic logging with no auxiliary pilot, $V(\rho)$ is only partially identified (Thm A of `theory/proofs.tex`); no efficiency claim applies.

*Proof outline.* The tangent space of the no-replay observed-data model under A1–A3 is spanned by (i) the score of $\alpha(q)$, (ii) the score of $\beta$, and (iii) the residual projection of $R - \alpha - \phi^\top \beta$ onto the atom span. The RuleOPE EIF (proven for the compositional model in Thm D of the current theory/proofs.tex) projects onto exactly this tangent space. $\square$

## Remark — why DR without the correction is not efficient

Under stochastic logging, standard DR (using only $(X, A, R)$) is
consistent but not semiparametrically efficient if a correction signal
$C$ satisfying A5 (corrected, `theory/proofs.tex` Definition \ref{def:bridge})
is available. Its asymptotic variance exceeds RuleOPE's by
$\Var(\xi)$ where $\xi$ is the bridge variance-reduction term of the
EIF — a quantity that's strictly positive whenever the correction
signal discriminates across counterfactual actions
($b_\rho(1, X) \ne b_\rho(0, X)$ with positive probability). This is
the theoretical mechanism behind the variance reductions measured on
HotpotQA / TriviaQA / MuSiQue at small $N$ (§7C of the main paper).

## Empirical corollary

On any RAG benchmark where A3 is approximately satisfied, $\widehat V^{\text{RuleOPE}}$ matches the oracle (which runs the generator at every counterfactual) within $O(n^{-1/2})$ asymptotic error, while DR without the correction term incurs an additional bias proportional to the residual variance of $\eta(q, r) \mid \phi(r)$.

The HotpotQA experiment in `experiments/noreplay_ope.py` measures this directly: the oracle computes the $[N \times 4]$ reward matrix by actually running the Mistral-7B generator on every counterfactual retrieval; the estimators see only the logged noop tuple. The gap between each estimator's V-estimate and the oracle is exactly the no-replay cost that the theorem bounds.
