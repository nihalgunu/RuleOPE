# 5  Theoretical guarantees

## 5.1 Consistency

**Theorem 1 (Consistency, informal).** *Under A1–A4 and positivity
(A2), if either (i) the compositional reward regression
$\widehat m \xrightarrow{P} m$ in $L^2$-norm, or (ii) the product of the
logging propensity and the correction-informativeness gate converges to
the true inverse-reweighting factor, then*
$$
\widehat V^{\text{ROPE}}(\rho) \xrightarrow{P} V(\rho).
$$

**Proof sketch.** The logged-action DR correction
$\widehat \Delta^{\text{logged}}$ has zero population mean under A2–A3,
so its consistency is classical (Robins et al.\ 1994). The correction-
fusion term is a zero-mean influence function under A4 whenever
$\widehat h^\star$ is chosen so that the population moment
$\mathbb{E}[C_i \, h^\star_i \, (R(X_i, \pi_\rho(X_i)) - m(X_i, \pi_\rho(X_i)))]
= 0$. With $h^\star(x, \rho) = \mathbb{1}[\pi_\rho(x) \ne a_0] \cdot
(1 - g(x, a_\rho)/g(x, a_0)) / g(x, a_0)$ this moment condition is
satisfied under A4 (detailed in theory/proofs.tex, Appendix A). Standard
double-ML arguments (Chernozhukov et al.\ 2018) then give consistency
under either (i) or (ii). $\Box$

## 5.2 Variance reduction from compositional regression

**Theorem 2 (Sublinear variance scaling).** *Let $\mathcal{R}$ be a
collection of $M$ rules each of depth at most $D$, using $K$ target
actions, with atom vocabulary of size $d$. Suppose the compositional
reward regression is trained by ridge with regularisation $\lambda > 0$
on $N$ samples, and $\|\phi(X)\|_2 \le B$ almost surely. Then*
$$
\sum_{\rho \in \mathcal{R}} \operatorname{Var}(\widehat V^{\text{ROPE}}(\rho)) \le \frac{1}{N}\left( \sigma_R^2 \cdot w_{\max}^2 \cdot M + K(d+1) \cdot \kappa(\lambda) \right),
$$
*where $\sigma_R^2 = \max_{i,a} \operatorname{Var}(R \mid X_i, A = a)$,
$w_{\max} = \max_{i, \rho} w_i(\rho)$, and $\kappa(\lambda) = O(B^2 / \lambda)$
is the regression's effective complexity.*

The first term is the unavoidable IPS contribution, which scales linearly
with $M$. The second term — the regression's total contribution across
all rules — scales as $O(K d)$, independent of $M$. For rule sets with
$M \gg K d$ this is a strict improvement over any estimator that refits
a separate regression per rule (the `NonCompositionalDR` baseline in
§8.1), whose regression-variance term scales as $O(M \cdot d)$.

*Proof sketch.* The RuleOPE estimator is linear in the regression
parameters $\beta \in \mathbb{R}^{K(d+1)}$. Under ridge with $\lambda > 0$
the parameter covariance is bounded by $\kappa(\lambda)/N$ uniformly.
The total variance contribution from $\beta$ across all rules is the
trace of the covariance matrix times the sum of squared linear
combinations, which is bounded by $K(d+1)\kappa(\lambda)/N$ because the
linear combinations are indicator weights. The IPS contribution is
per-rule and unavoidable. $\Box$

## 5.3 Failure modes of A4

Assumption A4 can fail in three specific, recurring ways in RAG
evaluation. For each we give a mitigation.

**F1: Query-dependent correction effort.** If $P(C \mid X, A)$ depends
on a query feature $U$ that is also a direct parent of $R$ but is not
included in $\phi(X)$ (e.g.\ the expert only reviews short queries
because they are fast, and short queries happen to be easier), the gate
$h^\star$ is miscalibrated. *Diagnostic:* residual correlation of
$\widehat g$ with held-out reward after conditioning on $\phi(X), A$.
*Mitigation:* add $U$ to the atom vocabulary, turning the hidden
confounder into an observed one.

**F2: Self-consistent-answer bias.** RAG generators often emit
confident wrong answers (the "hallucination" regime). Experts then
under-review confidently wrong queries, producing $P(C=1 \mid R=\text{bad},
\text{gen\_conf high}) \ll P(C=1 \mid R=\text{bad}, \text{gen\_conf low})$,
which violates A4 conditional on $X$ because $R$ enters the correction
probability through $\text{gen\_conf}$ not captured by $X$. *Mitigation:*
add $\text{gen\_conf}$ atoms to $\mathcal{V}$; we do so in our benchmark.

**F3: Corpus drift.** If the logging distribution of $X$ differs between
training (logs) and evaluation (the population we want to estimate on),
positivity (A2) effectively degrades and ESS $\ll N$. *Diagnostic:*
cross-fold ESS monitoring. *Mitigation:* covariate-shift reweighting at
the cost of extra variance.

## 5.4 When is RuleOPE's correction term worth it?

Theorem 1 says the correction term does not bias the estimator under
A4, but it does add variance. The correction-fusion term is *beneficial*
when its bias-reduction effect on $\widehat V^{\text{DM}}_C$ outweighs its
variance contribution. A sufficient condition is
$\operatorname{Var}(\psi^{\text{corr}}_i) < 2 \cdot |\text{Bias}(\widehat V^{\text{DM}}_C)| \cdot \mathbb{E}|\psi^{\text{corr}}_i|$,
which holds whenever corrections are informative (large $|\psi^{\text{corr}}_i|$
on records where the regression is most biased) and sparse (low
frequency of correction, keeping variance contained). The config flag
`correction_weight` in our implementation lets a practitioner shrink
the correction term; we show in §8.2 that the estimator is robust to
misspecification of this shrinkage parameter.
