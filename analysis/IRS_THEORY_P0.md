# P0 Theory: Restoration Is Zero-Order Evidence

Status: formal draft for implementation and falsification, not manuscript-ready.

## 1. Scope and target claim

This document studies **local functional mechanism agreement** after a neural intervention. It does not claim that a local Jacobian uniquely identifies a symbolic circuit or a global algorithm. The target is narrower and testable:

> If a patched computation has restored the same local functional mechanism as a target computation, then the two computations should respond similarly to the same admissible perturbations around their respective intervention states.

The paper-level novelty must therefore come from the complete result---zero-order non-identifiability, a positive local response bound, an estimable response signature, reference-relative admissibility, and ground-truth/real-model validation---rather than from Taylor expansion or conformal calibration alone.

## 2. Formal setting

Let a network be cut at an intervention site. Write

\[
F(h,b)\in\mathbb{R}^{k}
\]

for the downstream output functional, where \(h\in\mathbb{R}^{d}\) is the activation at the patched site and \(b\) collects every downstream-relevant state not replaced by that patch. In a transformer, \(b\) includes residual bypass information, other token positions, and unpatched components.

Let \((h_\star,b_\star)\) be the clean target computation and \((h_p,b_p)\) the patched computation. Ordinary behavioral restoration measures the zero-order error

\[
R_0 = \left\|F(h_p,b_p)-F(h_\star,b_\star)\right\|.
\]

For a perturbation \(\delta\), define the local interventional response

\[
\Delta_F(h,b;\delta)
=F(h+\delta,b)-F(h,b).
\]

Given an explicitly specified probe law \(Q_\Delta\), define the population response discrepancy

\[
D_{Q_\Delta}^{2}
=\mathbb{E}_{\delta\sim Q_\Delta}
\left[
\left\|
\Delta_F(h_p,b_p;\delta)
-\Delta_F(h_\star,b_\star;\delta)
\right\|^{2}
\right].
\]

The pair \((Q_\Delta,D_{Q_\Delta})\), not \(D\) alone, is the estimand. Changing the probe law changes the aspect of the local mechanism being tested.

For derivative-normalized comparisons, write \(\delta=ru\), where \(\|u\|=1\),
and define

\[
G_{Q_{U,R}}^2
=\mathbb{E}_{(u,r)\sim Q_{U,R}}
\left[
\left\|
\frac{\Delta_F(h_p,b_p;ru)-\Delta_F(h_\star,b_\star;ru)}{r}
\right\|^2
\right].
\]

For a fixed radius \(r\), the corresponding response discrepancy satisfies
\(D_{Q_\Delta}^2=r^2G_{Q_{U,R}}^2\).  The GPT-2 implementation estimates
\(G\), because clean-reference chord lengths vary across prompts and layers.

## 3. Theorem 1: zero-order restoration is non-identifying

### Statement

For every \(K>0\), there exists a smooth downstream map \(F\), a clean state \((h_\star,b_\star)\), and a patched state \((h_p,b_p)\) such that

\[
F(h_p,b_p)=F(h_\star,b_\star),
\]

but

\[
\left\|
J_hF(h_p,b_p)-J_hF(h_\star,b_\star)
\right\|_{\mathrm{op}}=K.
\]

Consequently, no function of zero-order restoration error alone can upper-bound local response discrepancy over a probe law that assigns positive variance to the discrepant direction.

### Proof

Take \(d=2\), scalar output, \(h_p=h_\star=(0,0)\), \(b_\star=0\), \(b_p=1\), and

\[
F(h,b)=h_1+bKh_2.
\]

Both computations output zero at the center. Their activation Jacobians are \((1,0)\) and \((1,K)\), whose difference has operator norm \(K\). For any zero-mean probe law with \(\mathbb{E}[\delta_2^2]>0\),

\[
D_{Q_\Delta}^{2}=K^2\mathbb{E}[\delta_2^2],
\]

which is unbounded as \(K\to\infty\) while restoration remains exact. \(\square\)

### Interpretation

The construction uses the same downstream map and changes only bypass state. It therefore matches the actual activation-patching problem more closely than comparing two unrelated functions: the patched activation may equal the clean activation while the unpatched context changes how downstream nonlinearities use it.

## 4. Theorem 2: local response agreement gives a transport bound

### Assumptions

For \(s\in\{p,\star\}\), let \(F_s(h)=F(h,b_s)\). Assume each \(F_s\) is twice continuously differentiable on a radius-\(r\) ball around \(h_s\), and

\[
\sup_{\|v\|=1}\|D^2F_s(h)[v,v]\|\leq H_s
\]

throughout the corresponding ball. Assume

\[
\|J F_p(h_p)-J F_\star(h_\star)\|_{\mathrm{op}}\leq\epsilon_J.
\]

### Statement

For every \(\|\delta\|\leq r\),

\[
\left\|
\Delta_F(h_p,b_p;\delta)
-\Delta_F(h_\star,b_\star;\delta)
\right\|
\leq
\epsilon_J\|\delta\|
+\frac{H_p+H_\star}{2}\|\delta\|^2.
\]

### Proof

Apply the second-order Taylor formula with integral remainder to each response:

\[
F_s(h_s+\delta)-F_s(h_s)
=J F_s(h_s)\delta+r_s(\delta),
\qquad
\|r_s(\delta)\|\leq\frac{H_s}{2}\|\delta\|^2.
\]

Subtract the two expansions and apply the triangle inequality and the operator-norm assumption. \(\square\)

### What this theorem does and does not establish

It establishes transport of **local intervention effects** over the tested neighborhood. It does not identify a unique global circuit, does not prove equality outside that neighborhood, and does not make a claim about directions excluded by \(Q_\Delta\).

## 5. Interventional Response Signature (IRS)

### Estimators

For unit probes \(u_j\) and radius \(r_j>0\), define the forward response signature

\[
\widehat g_s^+(u_j)
=\frac{F(h_s+r_ju_j,b_s)-F(h_s,b_s)}{r_j}.
\]

When both endpoints are admissible, the symmetric response signature is

\[
\widehat g_s^{\pm}(u_j)
=\frac{F(h_s+r_ju_j,b_s)-F(h_s-r_ju_j,b_s)}{2r_j}.
\]

Using either the forward or symmetric estimator consistently, the empirical raw
IRS discrepancy is

\[
\widehat G_{\mathrm{IRS}}^2
=\frac{1}{m}\sum_{j=1}^m
\left\|\widehat g_p(u_j)-\widehat g_\star(u_j)\right\|^2.
\]

This is an estimator of \(G_{Q_{U,R}}^2\), not of the unnormalized
\(D_{Q_\Delta}^2\) when radii vary.  The implementation reports both raw RMSE
and a dimensionless per-item normalization.  For item \(i\),

\[
\widehat G_{i,\mathrm{rel}}
=\frac{
\sqrt{m^{-1}\sum_j\|\widehat g_{p,ij}-\widehat g_{\star,ij}\|^2}
}{
\max\!\left(
\sqrt{m^{-1}\sum_j\|\widehat g_{\star,ij}\|^2},\epsilon
\right)
},
\]

and the reported `irs_normalized_rmse` is the mean of
\(\widehat G_{i,\mathrm{rel}}\) across items.  Raw RMSE, normalized RMSE, and
response cosine remain separate outputs.

The same probes and radii must be paired across patched and target computations. Unpaired probe sets introduce avoidable Monte Carlo variance and weaken the mechanistic interpretation.

### Proposition 3: first-order targets

If each scalar output coordinate has directional second derivative bounded by
\(H_s\) over the forward segment, Taylor's theorem gives

\[
\left\|
\widehat g_s^+(u)-J F_s(h_s)u
\right\|
\leq \frac{H_s r}{2}.
\]

If each scalar output coordinate has uniformly bounded third directional derivative \(T_s\) in the probed neighborhood, then

\[
\left\|
\widehat g_s^{\pm}(u)-J F_s(h_s)u
\right\|
\leq \frac{T_s r^2}{6}.
\]

Thus either estimator targets a random sketch of the Jacobian discrepancy, but
the forward estimator has \(O(r)\) bias whereas the symmetric estimator has
\(O(r^2)\) bias under the stated smoothness conditions. With isotropic unit probes,

\[
\mathbb{E}_u\left[(u^Tv)^2\right]=\frac{\|v\|^2}{d}
\]

for scalar-output Jacobian difference \(v\). Probe count controls sketch variance; probe radius controls approximation bias and admissibility.

### Practical probe law

The P0 implementation uses reference-chord probes:

1. choose a target-reference neighbor \(h^{(j)}\);
2. set \(\delta_j=\eta(h^{(j)}-h_s)\) for a frozen \(\eta\in(0,1]\);
3. form the forward endpoint \(h_s+\delta_j\);
4. retain/report that endpoint using the explicit target-reference conformal audit.

The GPT-2 P0 uses the forward estimator because the chord moves toward an
observed clean-reference state and therefore has substantially better
admissibility than the opposite endpoint.  Symmetric IRS remains implemented
for settings in which both endpoints can be audited.  Reference chords are not
automatically on-manifold because a manifold need not be convex. Conformal
endpoint scores are therefore diagnostics, not a theorem that chord
interpolation stays natural.

## 6. Theorem 4: probe-law coverage and finite-sketch concentration

Let

\[
A=J F_p(h_p)-J F_\star(h_\star)\in\mathbb{R}^{k\times d},
\]

and let \(u\) be a random unit probe with second-moment matrix
\(M_Q=\mathbb{E}[uu^T]\).  Then the population first-order IRS energy is

\[
\mathbb{E}\|Au\|^2
=\operatorname{tr}(A^T A M_Q).
\]

Consequently, if the probe law covers a subspace \(V\) in the explicit sense
that \(M_Q\succeq\lambda_V P_V\) for some \(\lambda_V>0\), then

\[
\mathbb{E}\|Au\|^2
\geq \lambda_V\|A P_V\|_F^2.
\]

For a single deterministic direction \(u_0\), \(M_Q=u_0u_0^T\) is rank one,
so every nonzero discrepancy satisfying \(Au_0=0\) is invisible.  For an
isotropic unit probe law, \(M_Q=I/d\) and

\[
\mathbb{E}\|Au\|^2=\frac{\|A\|_F^2}{d}.
\]

For \(m\) independent unit probes, let
\(\widehat E_m=m^{-1}\sum_{j=1}^m\|Au_j\|^2\).  Since each summand lies in
\([0,\|A\|_{\mathrm{op}}^2]\), Hoeffding's inequality gives, with probability
at least \(1-\alpha\),

\[
\left|
\widehat E_m-\operatorname{tr}(A^T A M_Q)
\right|
\leq
\|A\|_{\mathrm{op}}^2
\sqrt{\frac{\log(2/\alpha)}{2m}}.
\]

The trace identity follows by cyclicity of trace.  The coverage bound follows
from the minimum eigenvalue on \(V\), and the finite-sample statement follows
from Hoeffding's inequality.  Finite-radius forward or symmetric IRS adds the
bias terms in Proposition 3.

This theorem formalizes both the intended advantage and the limitation of a
probe distribution.  Multi-direction IRS can remove the unavoidable rank-one
blindness of a single clean--corrupt displacement only over directions actually
covered by \(M_Q\).  Reference-chord probes generally do not make \(M_Q\)
full-rank, so the empirical claim remains local and distribution-indexed.

## 7. Theorem 5: composite split-conformal admissibility

Let \(Z_F\), \(Z_N\), and \(Z_1,\ldots,Z_m\) be independent samples from a declared target reference law \(Q\). Use \(Z_F\) to fit representation geometry and \(Z_N\) to freeze all component normalizations and the scalar nonconformity function \(A(\cdot;Z_F,Z_N)\). For a query \(Z\), define

\[
p(Z)=\frac{1+\sum_{i=1}^{m}\mathbf{1}
\{A(Z_i;Z_F,Z_N)\geq A(Z;Z_F,Z_N)\}}
{m+1}.
\]

If \(Z,Z_1,\ldots,Z_m\) are exchangeable conditional on \((Z_F,Z_N)\), then for all \(\alpha\in[0,1]\),

\[
\Pr[p(Z)\leq\alpha]\leq\alpha.
\]

The result follows from exchangeability of the \(m+1\) scalar nonconformity scores and the conservative rank with the \(+1\) correction. It is marginal coverage relative to the declared \(Q\); it does not validate the causal appropriateness of choosing \(Q\).

### Implementation consequence

The historical geometric mean of three marginal ECDF tails is not a joint conformal p-value. The new implementation freezes robust component normalization on one held-out half and calibrates the final mean-softplus composite score on the untouched half.

## 8. Exact relationship to mediator--bypass interaction

IRS must not be presented as unrelated to the 2026 mediator-interaction result. For a fixed center \(h\), two bypass states \(b_p,b_\star\), and a displacement \(\delta\), define

\[
I(\delta;b_p,b_\star)
=\left[F(h+\delta,b_p)-F(h,b_p)\right]
-\left[F(h+\delta,b_\star)-F(h,b_\star)\right].
\]

This is exactly one directional response-field discrepancy. Standard clean/corrupt interaction uses the single displacement supplied by a particular prompt contrast. IRS instead estimates the energy and geometry of \(I(\delta;b_p,b_\star)\) over an explicit probe law. The intended added value is therefore:

1. a single prompt displacement can be orthogonal to a mechanism difference or exhibit cancellation;
2. a probe distribution can test a separating family of directions rather than one clean--corrupt vector;
3. target-reference conformal screening exposes whether those additional probes stay within the declared evidential scope;
4. the resulting object is a task-general local mechanism witness that can be compared with independent circuit readouts.

The empirical paper must include the ordinary single-direction interaction as a baseline. If multi-direction IRS does not add stable information beyond that baseline, the method-level novelty is insufficient.

## 9. Evidence hierarchy

The method reports three separate objects:

1. **Zero-order restoration** \(R_0\): did the output value match?
2. **Reference-relative admissibility** \(S_Q\): are the center and probe endpoints compatible with the declared target law under a calibrated test?
3. **Local functional mechanism discrepancy** \(D_{Q_\Delta}\): do patched and target computations respond similarly to the same probes?

They must not be multiplied into a universal scalar certificate. In particular:

- high \(R\), high \(S_Q\), high discrepancy is an on-support restoration lie;
- high \(R\), low \(S_Q\), high discrepancy is an off-support shortcut;
- high \(R\), high \(S_Q\), low discrepancy supports local functional mechanism restoration under the tested probe law;
- low \(R\), high \(S_Q\) is a natural but behaviorally ineffective intervention.

## 10. Claim--evidence map

| Claim | Required evidence | Current status |
|:--|:--|:--|
| Restoration alone does not identify local functional mechanism | Theorem 1 plus controlled counterexample | theorem drafted; analytic four-quadrant test passes 5/5 seeds |
| IRS controls nearby intervention-effect mismatch | Theorem 2 and radius sweep | theorem drafted; 0.10/0.25 radii robust, 0.50 changes fine layer ordering, 1.0 leaves support; strict frozen robustness gate fails |
| IRS estimates first-order response discrepancy | Proposition 3 plus analytic gradient check | symmetric directional MSE recovered to numerical precision in 5/5 seeds; forward \(O(r)\) bias and symmetric quadratic cancellation have exact regression tests |
| Multi-direction probes reduce single-direction blind spots only over their covered subspace | Theorem 4 plus controlled and empirical stress tests | exact orthogonal blind-direction construction passes 5/5 seeds (single RMSE 0, IRS RMSE \(\sqrt{2}\)); pretrained corruption test shows non-inferiority, not superiority |
| Composite conformal score controls marginal false alarms under target exchangeability | Theorem 5 plus simulation | implemented; regression test passes |
| On-support restoration lies occur in trained transformers | high R, acceptable conformal support, high IRS, independent mechanism witness | GPT-2 IOI L4--L7 pass in 3/3 prompt seeds using downstream NMH recovery |
| IRS generalizes beyond IOI | non-IOI known-mechanism task | pending |
| IRS adds information beyond single-direction interaction | fair INT baseline under the same sites/prompts | **not established on IOI**: single direction has stronger standard layer-rank association; corruption-shift gives IRS non-inferiority but not superiority; IRS has broader admissible coverage |
| Local functional agreement implies selected circuit-path restoration | corruption shift with an independent structural readout | **falsified** for pABC L4--L8: high restoration and low IRS coexist with strongly negative NMH recovery |

## 11. Immediate falsification tests

The P0 theory/method fails in its current form if any of the following holds:

1. analytic functions with known gradients are not recovered by IRS under a radius sweep;
2. IRS discrepancy is dominated by unpaired scale or output magnitude rather than response direction;
3. conformal endpoint acceptance collapses for all useful probe radii;
4. high-R/high-support/low-NMH GPT-2 sites show no IRS mismatch;
5. IRS adds no information beyond layer, activation distance, or ordinary gradient norm;
6. the empirical paper would need to describe IRS as structural circuit identification rather than local functional response agreement.
