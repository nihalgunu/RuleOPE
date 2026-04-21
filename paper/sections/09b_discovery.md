# 9B  Rule discovery with RuleOPE: from evaluator to learner

The preceding sections evaluate RuleOPE on a *given* rule set. A
production team, however, does not know the optimal intervention in
advance: it *enumerates* candidate rules, *scores* them offline, and
*ships* those that clear an offline bar. This section shows that
RuleOPE can be plugged directly into this loop, re-positioning it
from an *estimator* to a *rule-learning framework*.

## 9B.1 The discovery loop

```
enumerate depth-<=2 rules over the atom vocabulary  (|C| = 3528)
    -> sample N_train logs under uniform-stochastic logging over
       {noop, filter, rerank}  (scenario (i) of Thm C, revised)
    -> fit RuleOPE on logs; compute (V_hat(rho), SE_hat(rho))
       for every candidate
    -> select top-k rules by V_hat (ERM-argmax).
```

**Metric.** Simple oracle regret at top-$k$:
$$
\mathrm{Regret}_k(S) = V(\rho^\star) - \max_{\rho \in S_k} V(\rho),
$$
where $\rho^\star = \arg\max_{\rho \in C} V(\rho)$ is the oracle-best
*enumerated* rule and $V$ is computed exactly on a held-out eval
split via HotpotQA counterfactual replay (§6.2). Regret is averaged
over $n_\text{trials}=25$ resamples from a 1500-query pool.

**Baselines.** We compare ERM-argmax on the enumerated candidate
space $C$ against (i) random selection and (ii) the 500-rule
hand-curated set `eval/rules_v1.jsonl` scored either by $\hat V$
(ERM on the hand set) or by the oracle $V$ (the *skyline* achievable
by a human with perfect hindsight on the hand set). A pessimistic
LCB selector (CRRM, §9B.4) is reported as an ablation.

Script: `experiments/rule_discovery.py`.
Data: `experiments/results/rule_discovery.json` (main, $N{=}400$)
and `experiments/results/rule_discovery_n150.json` (small-$N$).

## 9B.2 Main result: RuleOPE + ERM-argmax enables auto-discovery

At $N_\text{train}=400$ on HotpotQA, ERM-argmax on RuleOPE point
estimates over a 3528-candidate enumerated space achieves mean
simple regret $0.017$ (median $0$) — matching ERM on the
hand-authored set and within $0.017$ of the oracle-best hand rule:

| selector | $\mathrm{Regret}_1$ (mean) | median | 90\% CI | top-1 oracle $V$ |
|---|---:|---:|---|---:|
| Random | $0.045$ | $0.024$ | $[0.001, 0.173]$ | $0.656$ |
| **RuleOPE + ERM-argmax on $C$ ($|C|{=}3528$)** | $\mathbf{0.017}$ | $\mathbf{0.000}$ | $[0.000, 0.163]$ | $\mathbf{0.684}$ |
| RuleOPE + ERM-argmax on the 500-rule hand set | $0.017$ | $0.000$ | $[0.000, 0.163]$ | $0.684$ |
| Hand-curated set, oracle-best (skyline) | $0.000$ | $0.000$ | $[0, 0]$ | $0.701$ |

(Numbers from `experiments/results/rule_discovery.json` →
`summary`, $n_\text{trials}=25$.)

ERM-argmax on $C$ recovers the oracle-best enumerated rule at
$\mathrm{Regret}_1 = 0$ on $16/25$ trials (median zero), trailing
the hand-curated skyline by only $0.017$ in mean top-1 oracle
value. The enumerated space and the hand-curated set behave
identically at $k=1$ because the hand set was systematically
constructed and happens to contain the oracle-best depth-$\le 2$
rule at every tested $N$; the interesting regime is therefore
small-$N$ (§9B.3) where extra candidates matter more.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figs/discovery_regret.pdf}
\caption{Rule discovery on HotpotQA at $N_\text{train}{=}400$,
$n_\text{trials}{=}25$. Left: simple oracle regret at top-$k$ across
selectors, with 90\% bootstrap CIs. RuleOPE + ERM-argmax (green)
tracks the ERM-on-hand baseline (orange) and both approach the
hand-authored oracle-best skyline (purple). Right: head-to-head
top-1 oracle $V$ of the CRRM-LCB ablation (§9B.4) vs the
hand-authored oracle-best — included to motivate the
data-adaptive-LCB discussion.}
\label{fig:discovery}
\end{figure}

## 9B.3 Small-$N$ sensitivity: where auto-discovery helps

At $N_\text{train} = 150$ the RuleOPE standard errors roughly
double. This is the regime where the enumerated space's extra
coverage should matter most:

| selector | $\mathrm{Regret}_1$ (mean) | median | 90\% CI | top-1 oracle $V$ |
|---|---:|---:|---|---:|
| Random | $0.080$ | $0.040$ | $[0.001, 0.271]$ | $0.621$ |
| **RuleOPE + ERM-argmax on $C$** | $\mathbf{0.018}$ | $\mathbf{0.002}$ | $[0.000, 0.159]$ | $\mathbf{0.683}$ |
| RuleOPE + ERM-argmax on the 500-rule hand set | $0.025$ | $0.002$ | $[0.000, 0.198]$ | $0.676$ |
| Hand-curated set, oracle-best (skyline) | $0.000$ | $0.000$ | $[0, 0]$ | $0.701$ |

(Numbers from `experiments/results/rule_discovery_n150.json`.)

At $N=150$ ERM on $C$ improves over ERM on the hand set by
$0.007$ in mean regret: the first genuine win for enumerated
auto-discovery over a human-curated candidate set, visible where
the hand set's finite coverage becomes the binding constraint.

## 9B.4 Ablation: pessimistic selection (CRRM-LCB)

Theorem 5 in `theory/proofs.tex` gives a pessimistic rule selector
CRRM-LCB whose uniform-coverage regret bound scales with the
*effective atom sparsity* $s$ rather than $\log |C|$. Empirically,
the LCB constant
$$
c = \sqrt{2((s{+}1)\log(d{+}1) + \log(1/\delta))} \approx 12
\quad (d{=}53,\ s{\approx}20,\ \delta{=}0.05)
$$
dominates the $\sim 10^{-2}$ differences between top-performing
rules' $\hat V$ at these $N$, so the LCB picks confidently-mediocre
rules (small $\widehat{\mathrm{SE}}$, moderate $\hat V$) over
probably-high ones. The atom-aware variant therefore underperforms
ERM-argmax on HotpotQA:

| selector | $\mathrm{Regret}_1$ @ N=400 | @ N=150 |
|---|---:|---:|
| RuleOPE + ERM-argmax on $C$ | **0.017** | **0.018** |
| CRRM-LCB, union-bound constant | 0.105 | 0.161 |
| CRRM-LCB, atom-aware constant (Thm 5) | 0.137 | 0.177 |

This is **not** a refutation of Theorem 5 — the uniform-coverage
claim holds, as verified by the fact that CRRM-LCB's selected rule
has $V \ge \mathrm{LCB}$ on every trial. It is instead evidence
that the *worst-case* Rademacher constant is loose for the
empirical rule-value gaps on this benchmark. A data-adaptive
LCB — for example, calibrating $c$ on a held-out fold so the
predicted coverage matches the empirical coverage — would preserve
the safety guarantee without the empirical over-penalisation, and
is the natural next step. We flag this as future work in §10.

## 9B.5 What this section claims and does not claim

We claim:

1. **RuleOPE supports plug-in rule discovery.** Given the atom
   vocabulary of §3 and any substrate with counterfactual rewards
   (§6), the enumerate → score → ERM-argmax loop runs end-to-end
   on standard hardware in minutes on a 3528-candidate, $N{=}400$
   HotpotQA setup.
2. **RuleOPE + ERM-argmax is competitive with a hand-curated set.**
   At $N=400$ the two tie at $0.017$ mean regret; at $N=150$ the
   enumerated-space approach improves by $0.007$, the expected
   direction when hand-curation coverage is the binding constraint.
3. **The pessimistic-selection theory (Thm 5) is sound but
   practically loose** in this regime; we report the gap openly
   and propose data-adaptive LCB calibration as the fix.

We do *not* claim:

1. Uniform dominance over hand-curated rule sets at all $N$. The
   hand set contains the oracle-best depth-$\le 2$ rule on this
   benchmark.
2. That CRRM-LCB outperforms ERM-argmax empirically; it does not,
   and §9B.4 shows exactly why.
3. End-to-end deployment. The loop produces offline-scored
   candidates; a human still promotes to production.

## 9B.6 Takeaway

This extends RuleOPE's claim from "offline evaluation of a fixed
rule set" to "offline rule discovery over a large combinatorial
candidate space," re-using the existing compositional variance-
reduction machinery plus a simple ERM-argmax selector. The
pessimistic CRRM-LCB extension is reported as an ablation: the
theory's uniform-coverage guarantee holds but its constant is
looser than the empirical value gaps on HotpotQA, giving a clean
separation between *what the theory guarantees* and *when its
constants are tight enough to bite* — a practically useful
boundary for practitioners.
