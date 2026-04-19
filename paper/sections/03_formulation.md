# 3  Problem formulation

## 3.1 RAG pipeline and logs

A retrieval-augmented generation pipeline is a composition
$f = f_{\text{gen}} \circ f_{\text{rerank}} \circ f_{\text{retr}}$
where $f_{\text{retr}}: \mathcal{Q} \to \mathcal{P}^K$ returns the top-$K$
passages from a corpus, $f_{\text{rerank}}: \mathcal{P}^K \to \mathcal{P}^K$
re-orders them, and $f_{\text{gen}}: \mathcal{Q} \times \mathcal{P}^K \to
\mathcal{A}_{\text{text}}$ generates an answer. A user or automated judge
assigns a reward $R \in [0, 1]$ (we assume scalar reward without loss of
generality).

A *log record* is a tuple $(X, A, R, C)$ where
$X \in \mathcal{X}$ is a feature vector summarising the query, the ranked
passage list, and the generated answer; $A \in \mathcal{A} = \{a_0, a_1,
\ldots, a_K\}$ is the *intervention action* taken by the pipeline (in a
typical production pipeline the baseline action $a_0 = \texttt{noop}$ is
taken exclusively); $R$ is the observed reward; and $C \in \{0, 1\}$ is a
sparse post-hoc correction flag provided by an expert or automated proxy.
We observe an i.i.d.\ sample of $N$ such tuples.

## 3.2 Rules

Let $\mathcal{V} = \{\phi_\alpha\}_{\alpha \in \mathcal{I}}$ be a finite
atom vocabulary — a set of boolean predicates over $\mathcal{X}$. In our
benchmark $|\mathcal{V}| = 48$; atoms include e.g.\ $\phi_{\text{top1}<0.3}(x) =
\mathbb{1}[\text{top-1 reranker score} < 0.3]$ and $\phi_{\text{multihop}}(x)
= \mathbb{1}[\text{query is multi-hop}]$.

A *rule* is a triple $\rho = (S_\rho, a_\rho, \texttt{type}_\rho)$ where
$S_\rho \subseteq \mathcal{V}$ is a non-empty *conjunctive clause*, $a_\rho
\in \mathcal{A} \setminus \{a_0\}$ is the action to take when the clause
fires, and $\texttt{type}_\rho \in \{\texttt{filter}, \texttt{rerank},
\texttt{abstain}\}$ is the rule category. The *policy induced by* $\rho$
is
$$
\pi_\rho(x) = \begin{cases} a_\rho & \text{if } \prod_{\alpha \in S_\rho} \phi_\alpha(x) = 1 \\ a_0 & \text{otherwise.} \end{cases}
$$
We restrict to conjunctive clauses of depth $|S_\rho| \le D$ in this
paper; $D = 3$ is our default. Extensions to more general boolean
structures (arbitrary CNF, decision lists) are natural and discussed in
§10.

## 3.3 Target of estimation

The value of a rule is the expected reward under its induced policy:
$$
V(\rho) = \mathbb{E}_x[R(X, \pi_\rho(X))],
$$
where $R(x, a)$ is the potential outcome under action $a$ (Rubin 1974).
Note that $V(\rho)$ is defined exactly even when $\rho$ is never
deployed: it is a counterfactual quantity.

The primary estimand of the paper is the $M$-tuple
$(V(\rho_1), \ldots, V(\rho_M))$ for a fixed candidate rule set
$\mathcal{R} = \{\rho_m\}_{m=1}^M$. Downstream decisions (pick top-$k$,
filter by a value threshold) are derived from this.

## 3.4 Assumptions

The estimator we propose in §4 is consistent under the following
standard-plus-one assumptions:

**A1 (Consistency of potential outcomes, SUTVA):** $R = R(X, A)$.

**A2 (Positivity):** There exists $\epsilon > 0$ with
$\pi_0(a \mid x) \ge \epsilon$ for all $a, x$.

**A3 (Action unconfoundedness):** $\{R(x, a)\}_{a \in \mathcal{A}}
\perp A \mid X$.

**A4 (Correction unconfoundedness):** There exists $g: \mathcal{X} \times
\mathcal{A} \to [0, 1]$ such that $P(C = 1 \mid X, A, R) = g(X, A)$,
i.e.\ the correction decision is independent of the reward realisation
given $(X, A)$.

A1–A3 are the standard ignorability conditions for off-policy
evaluation. A4 is the new assumption our correction-fusion term
relies on. It says the expert's decision to correct depends only on the
features $X$ and the intervention action $A$, not on the specific reward
realisation $R$ beyond what $(X, A)$ already reveals. We characterise in
§5.3 three concrete RAG settings in which A4 fails.

## 3.5 Notation

We write $\phi(x) := (\phi_\alpha(x))_{\alpha \in \mathcal{V}} \in
\{0, 1\}^d$ for the atom indicator vector, with $d = |\mathcal{V}|$.
The *firing mask* for rule $\rho$ is $F_\rho(x) = \prod_{\alpha \in S_\rho}
\phi_\alpha(x)$. The conditional mean reward is $m(x, a) = \mathbb{E}[R(X,
a) \mid X = x]$.
