# 11  Novel contributions (upgrades beyond classical DR)

Beyond the correction-fusion RuleOPE estimator itself, we introduce
three conceptual upgrades that collectively position this work as a
framework for rule-OPE, not a single estimator.

## 11.1 DualShrinkOPE: between-estimator shrinkage

Under deterministic logging, all DR-family estimators (DR, CIPS-DR,
CascadeDR) collapse to the Direct Method's regression prediction plus
a zero-weighted IPS correction, inheriting DM's bias.  The RuleOPE
correction-fusion term de-biases part of this, but at the cost of
added variance from the correction-gate model.

**DualShrinkOPE** takes the Bayes-optimal convex combination
$$
\widetilde V(\rho) = w(\rho) \widehat V_{\rm DM}(\rho) + (1 - w(\rho)) \widehat V_{\rm RuleOPE}(\rho)
$$
with per-rule weight $w(\rho) = (\sigma_{\rm DR}^2 + b_{\rm DR}^2) / (\sigma_{\rm DR}^2 + b_{\rm DR}^2 + \sigma_{\rm DM}^2 + b_{\rm DM}^2)$
estimated by empirical Bayes (Proposition 1 in `theory/proofs.tex`).
This is a rule-OPE analogue of Wang et al.'s switch estimator but with
a soft, per-rule data-driven weight, which to our knowledge is new.
Experiments (§7 below) show it achieves the bulk of RuleOPE's gain at
a fraction of the variance.

## 11.2 JointRuleOPE: cross-rule empirical-Bayes shrinkage

Per-rule estimates $\{\widehat V(\rho_m)\}$ are correlated through the
shared compositional regression.  Treating them as a single
vector-valued estimation target lets us apply a random-effects shrinkage
toward an atom-compositional target
$$
\widehat\mu_m = F(\rho_m)^\top \widehat\beta,
$$
where $F$ maps rules to an atom-plus-action feature vector and
$\widehat\beta$ is fit by weighted ridge on $(\widehat V(\rho_m),
\widehat\sigma_m^{-2})$.  The shrinkage weight is
$w_m = \widehat\sigma_m^2 / (\widehat\sigma_m^2 + \widehat\tau^2)$ with
$\widehat\tau^2$ estimated by DerSimonian--Laird.  Theorem 3 in
`theory/proofs.tex` establishes joint-MSE dominance of this estimator
over the independent per-rule baseline.  Because JointRuleOPE's
shrinkage target is the \emph{same} compositional structure the
primary RuleOPE estimator already uses, most of its benefit is
absorbed when the primary estimator has low bias; its main contribution
here is to give the benchmark's \emph{joint} MSE a principled lower
bound.

## 11.3 Compositional pessimistic rule selection

Selecting the highest-value rule is a \emph{decision-theoretic}
operation: the downstream practitioner ships the selected rule and
suffers regret $V(\rho^\dagger) - V(\widehat\rho)$.  The naive
$\argmax$ is optimistic; standard LCB-based selection pays a union-bound
factor $\sqrt{\log M}$.  We derive a \emph{compositional} LCB whose
complexity term scales with the atom sparsity $s$ rather than the rule
count $M$:
$$
c_M = \sqrt{2 \bigl((s + 1)\log(d + 1) + \log(1/\delta)\bigr)}.
$$
Theorem 4 shows this is strictly tighter than the union-bound LCB
whenever $s < \log M / \log d$.  For our benchmark ($d = 48$, $M =
500$), the threshold is $s < 1.6$, i.e.\ the compositional LCB wins
when the best rule is at most a depth-2 conjunction whose value is
well-explained by one informative atom.

## 11.4 CRRM: counterfactual rule risk minimisation

Extending evaluation to \emph{learning}: given logs and a candidate
rule set, solve
$$
\widehat\rho = \argmax_{\rho \in \mathcal{R}} \bigl\{\widehat V_{\rm LCB}(\rho) - \lambda |S_\rho|\bigr\}.
$$
Theorem 5 gives a regret bound that scales with the compositional
Rademacher complexity of the rule class, not with $|\mathcal{R}|$.  This
is the rule-OPE analogue of counterfactual risk minimisation
(Swaminathan and Joachims 2015) for the conjunctive-rule hypothesis
class.  In the experiments (§8) we benchmark CRRM against ERM (naive
argmax) and standard LCB on the rule-learning task and report
regret.

## 11.5 Empirical summary

The headline result: in the production-realistic regime (deterministic
logging, 300--2400 queries, 500 rules), RuleOPE achieves 10--23\% MSE
reduction over the strongest classical baseline (DR / CIPS-DR /
CascadeDR, which all coincide in this regime):

| $N$   | DR MSE   | RuleOPE MSE | reduction | DualShrink MSE | reduction |
|-------|----------|-------------|-----------|----------------|-----------|
| 300   | 0.00104  | 0.00082     | 21.1\%    | 0.00092        | 11.5\%    |
| 600   | 0.00104  | 0.00091     | 12.9\%    | 0.00096        | 7.2\%     |
| 1200  | 0.00096  | 0.00085     | 10.9\%    | 0.00091        | 5.7\%     |
| 2400  | 0.00101  | 0.00077     | 23.4\%    | 0.00087        | 13.7\%    |

Kendall's tau on the top-20 rules is also consistently higher for
RuleOPE (by 0.05--0.15) than for any baseline in this regime.
DualShrinkOPE provides a safer, lower-variance alternative at modest
cost in point MSE.

The cross-rule shrinkage (JointRuleOPE) and pessimistic selection
upgrades are framework contributions: they do not dominate in raw MSE
on the default benchmark (where SEs are roughly homogeneous) but are
the principled right-shape tools for (a) reporting joint error bars
across a rule set and (b) operationalising pessimistic shipping
decisions in production.
