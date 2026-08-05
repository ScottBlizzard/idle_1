# Executive Verdict

## Decision

**Current status: NO-GO for an ICLR oral submission, and NO-GO as a headline “IRS method” paper.**

The repository has correctly abandoned the falsified claim that activation overlap is a general causal-validity certificate. It has not yet replaced that claim with an oral-level result. The present evidence supports a narrower diagnostic statement:

> Behavioral restoration, local response agreement under a declared probe law, and restoration of a chosen circuit coordinate are different empirical properties.

That distinction is correct, but the current hierarchy is mostly a synthesis of existing non-identification, intervention-distribution, local-sensitivity, causal-abstraction, and mediator-interaction ideas. The repository does not yet prove a nontrivial implication between the levels, identify a transformer-relevant model class in which such an implication holds, or demonstrate that IRS provides information unavailable from a fair single-direction or existing multi-mediator diagnostic.

The prescribed repository route, including [GPTPRO_START_HERE.md](https://github.com/ScottBlizzard/idle_1/blob/main/GPTPRO_START_HERE.md), the full [red-team packet](https://github.com/ScottBlizzard/idle_1/blob/main/analysis/GPTPRO_REDTEAM_PACKET_20260805.md), the [theory note](https://github.com/ScottBlizzard/idle_1/blob/main/analysis/IRS_THEORY_P0.md), the [novelty gate](https://github.com/ScottBlizzard/idle_1/blob/main/analysis/P0_NOVELTY_GATE_20260805.md), implementation, tests, aggregate reports, and selected raw JSON all point to the same conclusion: the implementation establishes a coherent diagnostic pipeline, while the empirical evidence defeats a method-superiority story and does not validate the proposed local-functional/structural separation. 

## Probability assessment

These are subjective posterior probabilities **conditional on one disciplined 7–10 GPU-day rebuild**, not acceptance probabilities for the repository in its present form.

| Outcome | Probability | Interpretation |
|---|---:|---|
| Oral-worthy after a focused rebuild | **0.12** | Requires both a genuinely identifying theorem and a locked non-IOI result against strong baselines. |
| Poster-level contribution only | **0.31** | Most likely positive outcome if the project becomes an honest intervention-diagnostics or negative-results paper. |
| No-go without a new central result | **0.57** | Most likely because the hierarchy remains a predictable synthesis and IRS does not beat the fair baseline. |

For the **current snapshot**, before the theorem and experiment specified below, my probability of an oral-worthy paper is **at most 0.01**.

## Dominant uncertainty

The dominant uncertainty is not whether additional IOI seeds will stabilize the current plots. It is whether the project can establish the following new central result:

> Under a non-vacuous, transformer-relevant set of structural assumptions, a probe-complete collection of first- and mixed-second-order local interventional responses identifies a defined class of mediator/path parameters, while lower-order or rank-deficient probe schemes provably do not.

That would be materially stronger than the current “three levels of evidence” taxonomy. It would also create a reason to measure multi-directional or multi-mediator response signatures. Without it, the project is largely Taylor expansion, random-direction sketching, split conformal ranking, and familiar causal non-identification assembled under new terminology.

## Immediate factual constraints

The following are not minor limitations; they constrain the permissible main line:

1. On the standard IOI layer-stage diagnostic, the fair single clean–corrupt interaction direction is at least as effective as IRS: layer-level Spearman magnitude is stronger for the single direction, and leave-one-context-out error is slightly lower.
2. The preregistered strict robustness gate fails.
3. Under corruption shift, IRS achieves non-inferiority at best; the single-direction baseline is slightly stronger on the reported rank and relative-error diagnostics.
4. The claimed pABC local-functional/structural separation is not valid evidence because its normalized Name Mover Head recovery divides by a clean–corrupt gap of only **0.0080**, versus **0.4217** under standard duplicate-name IOI—a **52.7× smaller denominator**.
5. The endpoint conformal ranks do not have the claimed clean-reference coverage semantics because the endpoints are adaptively constructed from the same fitted reference geometry against which they are evaluated.
6. The layer- and prompt-level significance calculations use repeated measurements as though they were independent statistical units.

The fair-baseline, robustness, and corruption results are documented in the aggregate and stress reports; the pABC denominator pathology is visible in the raw output itself. 

---

# Novelty Collision Audit

## Bottom-line novelty decision

**The behavioral/local-functional/structural hierarchy is not materially new in its current form.**

My probability that a well-informed ICLR reviewer would regard the hierarchy itself as a new conceptual contribution is approximately **0.15**. The exact three labels and their juxtaposition may be new wording. The underlying distinctions are not:

- Pointwise behavioral equality does not identify derivatives or internal mechanism.
- Interventional conclusions depend on the intervention distribution and support.
- Local functional agreement is weaker than structural or representational equivalence.
- Multiple mediators create interaction terms not captured by one-at-a-time effects.
- Causal abstraction and circuit claims require explicit identification assumptions.
- Internal representations admit equivalence classes and reparameterizations.

The strongest collision is not a vague resemblance. It is the 2026 multi-mediator work that explicitly decomposes mediated effects into path-independent and interaction terms, studies when interactions disappear in locally affine regimes, and evaluates the issue in GPT-2 IOI. That substantially narrows the novelty available to a directional mediator–bypass interaction score. 

## Verified primary-source collision map

### 1. Behavioral restoration does not identify mechanism

[Addressing Divergent Representations from Causal Interventions on Neural Networks](https://arxiv.org/abs/2511.04638) distinguishes interventions that are behaviorally harmless because they lie in a null space from interventions that activate hidden or off-distribution pathways. This directly collides with any implication from restored endpoint behavior to restored causal mechanism. 

[Mechanistic Interpretability Must Disclose Identification Assumptions](https://arxiv.org/abs/2605.08012) argues that empirical validation of an interpretation is not equivalent to identification of the claimed mechanism. This is the same logical boundary the repository now places between behavioral evidence and structural evidence. 

[Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?](https://arxiv.org/abs/2502.20914) develops non-identifiability concerns for mechanistic interpretations rather than treating observed behavioral agreement as a unique circuit explanation. 

[Non-Identifiability of Steering Vectors](https://arxiv.org/abs/2602.06801) shows an analogous ambiguity for interventions: behaviorally indistinguishable steering constructions need not correspond to the same internal object. 

**Verified conclusion:** the repository’s zero-order non-identification statement is correct, but it occupies an already active literature rather than opening an unrecognized problem.

### 2. Local response agreement and response-field diagnostics

[Transformer Field Theory](https://arxiv.org/abs/2605.25225) explicitly develops first-order sensitivity or response-field descriptions for transformers and characterizes a local linear regime. IRS’s small-radius limit is therefore not a new species of mechanistic object; it is a probe-weighted discrepancy between local response operators. 

[Certified Interventional Fidelity](https://arxiv.org/abs/2607.08349) makes the input and intervention distributions explicit, studies fidelity under those distributions, and discusses sensitivity to intervention-law choices and adaptive inference. This is a close collision with the claim that IRS’s declared probe law and conformal reference distribution are themselves a new conceptual foundation. 

[Bucketing the Good Apples](https://arxiv.org/abs/2605.02234) organizes inputs by interchange behavior, reinforcing that interventional conclusions are region- or distribution-dependent rather than global properties of a representation. 

**Verified conclusion:** an explicit probe law is scientifically necessary, but declaring it is not enough for novelty. The contribution would have to establish a new identification, optimal-design, or finite-sample result about that law.

### 3. Off-manifold interventions and replacement design

[Addressing Divergent Representations from Causal Interventions on Neural Networks](https://arxiv.org/abs/2511.04638) directly treats the natural-distribution departure caused by interventions. 

[Beyond Importance: Interchange-Sobol for Mechanistic Analysis](https://arxiv.org/abs/2606.20678) contrasts matched replacement interventions with zero ablation, develops role-sensitive Sobol-style quantities, and includes off-manifold diagnostics. This is closer to the repository’s context-matched chord construction than a generic feature-attribution baseline would be. 

[Certified Interventional Fidelity](https://arxiv.org/abs/2607.08349) independently makes coverage conditional on a declared intervention distribution rather than treating support as an unqualified binary property. 

**Verified conclusion:** the project cannot claim that context-matched probes or distribution-indexed compatibility are new in isolation.

### 4. Multi-mediator interactions

[The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510) is the most damaging novelty collision. It decomposes natural indirect effects into path-independent and interaction components, explains why interaction magnitude depends on mediator distance and local nonlinearity, treats pairwise and higher-order interactions, and includes GPT-2 IOI. 

IRS’s exact mediator–bypass directional interaction is therefore not a sufficient new centerpiece. IRS currently measures a normed finite-difference disagreement under selected joint directions; the cited work already supplies a richer causal vocabulary for why single-mediator effects fail and how interaction terms arise.

**Verified conclusion:** calling IRS a new method without outperforming a fair single direction or distinguishing itself from PIE/INT-style factorial diagnostics will invite an immediate novelty rejection.

### 5. Representation and interpretation equivalence

[The Non-Linear Representation Dilemma](https://arxiv.org/abs/2507.08802) shows that sufficiently flexible alignment maps can trivialize representation-level causal-abstraction claims. The proceedings version is available through the [NeurIPS 2025 paper page](https://papers.nips.cc/paper_files/paper/2025/hash/dbb98528c9870377f3f0d133aae6050b-Abstract-Conference.html). 

[Tracking Equivalent Mechanistic Interpretations](https://arxiv.org/abs/2603.30002) directly studies equivalence among mechanistic interpretations and the relationship between representation-level and circuit-level descriptions. 

[When Are Two Networks the Same? Tensor Similarity](https://arxiv.org/abs/2605.15183) develops a global functional-equivalence perspective designed to respect internal symmetries. It is not the same estimand as IRS, but it occupies the representation-equivalence side of the claimed hierarchy. 

**Verified conclusion:** “local functional equivalence is not structural equivalence” is a predictable consequence of known symmetry, alignment, and non-identification problems.

### 6. Causal abstraction and structural restoration

[Causal Abstraction](https://arxiv.org/abs/2301.04709) provides a formal umbrella covering interchange interventions, causal scrubbing, path patching, and related abstraction claims. Any theorem upgrading local response agreement to structural restoration must state its causal model, alignment map, intervention set, and abstraction criterion relative to this framework. 

[Towards Verifiable Transformers](https://arxiv.org/abs/2605.24033) pursues bounded, solver-checkable notions of projected functional equivalence and edge necessity in modified transformer settings. This raises the standard for using terms such as “certificate” or “verified mechanism.” 

[Mechanistic Interpretability as Statistical Estimation](https://arxiv.org/abs/2510.00845) emphasizes estimator variance and circuit instability, reinforcing that a circuit claim cannot be established from a small collection of correlated layer and prompt observations. 

**Verified conclusion:** the repository’s hierarchy is scientifically sensible, but causal abstraction, equivalence, and identification already provide the conceptual scaffolding. A new theorem must do more than restate the hierarchy.

## What is verified versus inferred

**Verified from primary sources:**

- Existing work separately covers non-identification, local response fields, off-manifold interventions, explicit intervention laws, multiple-mediator interactions, causal abstraction, and representation equivalence.
- The multi-mediator paper is particularly close to the proposed interaction-based explanation.
- Recent work already warns that empirical validation or behavioral agreement is not mechanism identification.

**My inference from the combined record:**

- I did not identify one prior paper that uses exactly the repository’s three names in exactly the same order.
- Nevertheless, an ICLR reviewer is unlikely to credit a taxonomy assembled from these existing distinctions as a major conceptual contribution.
- The remaining novelty opportunity is not the hierarchy itself. It is an **identification theorem with a converse**, plus an experiment in which its additional interaction order predicts an independently measured path-specific quantity beyond fair alternatives.

---

# Formal Theory Audit

## 1. Zero-order non-identification

The repository’s zero-order statement is correctly scoped when interpreted as follows:

> Equality or near-equality of the patched and target outputs at one intervention point does not bound their local derivatives, absent derivative regularity or additional interventional observations.

That claim is mathematically correct. Two smooth functions can agree at a point and have arbitrarily different Jacobians there. More generally, a mediator intervention can restore a scalar behavioral readout while other local directions or bypass pathways remain different. The theory note does not prove that every behavioral restoration is causally misleading; it proves that the zero-order observation alone cannot exclude that possibility. 

The scope must remain explicit:

- It is pointwise or finite-sample non-identification, not a theorem that behavior never identifies mechanism.
- It concerns the selected output map and intervention coordinates.
- It does not distinguish reparameterization-equivalent mechanisms from substantively different mechanisms.
- It does not by itself establish that any observed IOI restoration is structurally wrong.

**Verdict:** correct, useful as a boundary lemma, but standard and not independently oral-worthy.

## 2. Local transport statement

The local transport argument is also basically correct. Let \(f_p\) and \(f_\star\) denote patched and target response maps around respective centers, and suppose both are \(C^2\). For a common local displacement \(\delta\),

\[
\begin{aligned}
f_p(z_p+\delta)-f_p(z_p)
    &= J_p\delta + R_p(\delta),\\
f_\star(z_\star+\delta)-f_\star(z_\star)
    &= J_\star\delta + R_\star(\delta),
\end{aligned}
\]

with quadratic remainder bounds under bounded Hessians. Therefore,

\[
\left\|
  \big[f_p(z_p+\delta)-f_p(z_p)\big]
  -
  \big[f_\star(z_\star+\delta)-f_\star(z_\star)\big]
\right\|
\le
\|(J_p-J_\star)\delta\|
+
O(\|\delta\|^2).
\]

If the local Jacobians agree on the probed subspace, the response discrepancy is second order in the radius. Conversely, a small finite-radius discrepancy only constrains the operator on the sampled directions and only up to the Taylor remainder. 

The theorem does **not** transport:

- to unprobed directions;
- from a representation basis to an invariant circuit identity;
- from local derivatives to a unique causal graph;
- across an intervention law whose support differs from the one used in the estimand;
- from a finite-radius normalized score to exact Jacobian equality without denominator and remainder control.

**Verdict:** correct under its assumptions, but it is a local Taylor statement rather than a structural-identification theorem.

## 3. Finite-radius IRS estimands and code alignment

A clean way to state the theoretical numerator is

\[
D_Q^2
=
\mathbb E_{\delta\sim Q}
\left\|
  \Delta_p(\delta)-\Delta_\star(\delta)
\right\|_2^2,
\qquad
\Delta_s(\delta)=f_s(z_s+\delta)-f_s(z_s),
\]

with a target-response normalizer such as

\[
S_Q^2
=
\mathbb E_{\delta\sim Q}\|\Delta_\star(\delta)\|_2^2,
\qquad
\operatorname{IRS}_Q
=
\frac{D_Q}{S_Q\vee \tau}.
\]

The theory note’s forward and symmetric finite-difference motivations are reasonable, but four mismatches or omissions must be fixed. 

### 3.1 Output-axis averaging mismatch

The written energy averages \(\|\cdot\|_2^2\) over probes. The implementation in [src/interventional_response.py](https://github.com/ScottBlizzard/idle_1/blob/main/src/interventional_response.py) takes a mean over both probe and output coordinates. For output dimension \(k\), the implementation is therefore the written energy divided by \(k\). 

This has no numerical effect in the current scalar-output GPT-2 experiment, where \(k=1\). If numerator and denominator use the same convention, the factor also cancels away from the normalization floor. It nevertheless matters for the general theorem and for any vector-valued extension; near the floor, even the normalized quantity need not be invariant to that scaling.

**Required repair:** choose one convention, write it exactly, and test it for \(k>1\).

### 3.2 Vector-valued bias bound is missing a \(\sqrt{k}\) factor

The theory’s scalar finite-difference bounds have the usual form:

\[
\left\|
\frac{f(z+ru)-f(z)}{r}-J(z)u
\right\|
\le \frac{Hr}{2},
\]

for an appropriate vector-norm bound on the second directional derivative, and

\[
\left\|
\frac{f(z+ru)-f(z-ru)}{2r}-J(z)u
\right\|
\le \frac{Tr^2}{6},
\]

for a vector-norm bound on the third directional derivative.

If the assumptions instead bound **each scalar output coordinate** by \(H\) or \(T\), the vector bounds are

\[
\frac{\sqrt{k}\,Hr}{2}
\quad\text{and}\quad
\frac{\sqrt{k}\,Tr^2}{6}.
\]

The current statement conflates coordinatewise bounds with a vector-norm conclusion. This does not affect the scalar experiment but invalidates the general vector-output proposition as written. 

**Required repair:** either add \(\sqrt{k}\), or assume the norm of the vector directional derivative is bounded directly.

### 3.3 The implemented “radius” is a variable chord length

The probe construction in [src/interventional_response.py](https://github.com/ScottBlizzard/idle_1/blob/main/src/interventional_response.py) uses

\[
\delta_{ij}
=
\eta\left(z^{\mathrm{ref}}_{ij}-z_i\right),
\]

where the reference is selected from a context-matched fit pool. Thus \(\eta\) is a **chord fraction**, not an absolute perturbation radius. The actual radius is

\[
r_{ij}
=
\eta\|z^{\mathrm{ref}}_{ij}-z_i\|.
\]

The code also samples reference donors without replacement when the pool is sufficiently large. 

Any bias claim indexed only by \(\eta\) is incomplete. It must depend on the empirical distribution of \(r_{ij}\), or at minimum on \(r_{\max}\). Comparing \(\eta=0.25\) across layers does not compare the same absolute radius because representation scales and donor distances change by layer.

**Required repair:** log the full radius distribution by layer, context, and corruption; report normalized radii; state Taylor remainder bounds in terms of actual \(r_{ij}\).

### 3.4 Normalized-score bias needs a denominator condition

A forward or symmetric finite difference can approximate a directional derivative with the usual orders. It does not follow automatically that the **normalized ratio** has the same stable error order. One needs a lower bound such as

\[
S_Q\ge s_{\min}>0,
\]

and must account for the floor \(\tau\). Otherwise a small target-response energy magnifies both approximation error and measurement noise. The pABC NMH result shows exactly why denominator conditioning cannot be treated as bookkeeping.

**Required repair:** preregister and report \(S_Q\), the fraction of cells affected by the floor, and a minimum target-energy admissibility rule.

## 4. Probe-law trace identity and concentration

Let

\[
A=J_p-J_\star
\]

and let a zero-mean probe direction \(u\) have second moment

\[
M_Q=\mathbb E_Q[uu^\top].
\]

Then

\[
\mathbb E_Q\|Au\|_2^2
=
\operatorname{tr}\!\left(A^\top A M_Q\right).
\]

For an isotropic unit-sphere law in \(d\) dimensions,

\[
M_Q=\frac{I}{d},
\qquad
\mathbb E_Q\|Au\|_2^2
=
\frac{\|A\|_F^2}{d}.
\]

This identity is correct. It is useful because it makes clear that the estimand is a **probe-weighted operator seminorm**, not an invariant property of the mechanism. 

The associated Hoeffding statement is also correct for independent bounded probe energies. It is not exactly the concentration result implemented in the repository:

- probes are selected without replacement within some donor pools;
- donor pools overlap across centers;
- each center has its own context-conditioned law \(Q_i\);
- nearest or eligible donors induce non-identical direction distributions;
- probe norms are variable.

The actual population target is closer to

\[
\frac{1}{n}\sum_{i=1}^n
\operatorname{tr}\!\left(A_i^\top A_i M_{Q_i}\right),
\]

not a single \(\operatorname{tr}(A^\top A M_Q)\).

**Required repair:** state concentration conditionally on the fitted donor pools and use an appropriate finite-population or martingale bound. More importantly, estimate the spectrum of each empirical \(M_{Q_i}\), or an aggregate covered-subspace operator. Without a lower eigenvalue or effective-rank analysis, “multi-directional coverage” means only that more than one chord was sampled; it does not imply that the relevant response subspace was covered.

## 5. Composite split conformal audit

The generic split-conformal construction in [src/validity_crossfit.py](https://github.com/ScottBlizzard/idle_1/blob/main/src/validity_crossfit.py) is defensible for a fresh exchangeable query:

1. fit the geometric score using one split;
2. use an independent normalization split;
3. compute final calibration scores on another split;
4. rank a fresh query score with the standard \(+1\) correction.

A composite scalar score such as a mean of softplus-transformed component scores can be used in split conformal as long as the entire scoring rule is frozen before the final calibration scores are evaluated. The code appears to maintain that ordering for ordinary clean queries. 

The randomized, without-replacement partition of clean IO-name examples into fit, normalization, calibration, and evaluation sets can support finite-population exchangeability for ordinary held-out clean centers. Independence is not required in its strongest i.i.d. form; symmetry of the randomized partition can suffice. 

### Endpoint problem

The endpoint claim is different. In [src/exp_p0_irs_gpt2.py](https://github.com/ScottBlizzard/idle_1/blob/main/src/exp_p0_irs_gpt2.py), the endpoint is moved toward a nearby member of the same fitted context reference set used to define the geometry and donor chords, then evaluated against the fitted score. 

There is no obvious behavioral-label or NMH-label leakage. The problem is **query-law leakage**:

- calibration examples are natural clean examples;
- endpoints are adaptively manufactured using the fitted clean reference geometry;
- therefore endpoints are not exchangeable with the clean calibration examples;
- high endpoint conformal ranks do not have clean-law marginal coverage semantics.

This can also make endpoint acceptance partially mechanical: the endpoint was explicitly constructed by moving toward an element of the fitted reference support.

### Valid repair

Use four conceptually separate sets:

- \(D\): donor construction;
- \(F\): geometric score fitting;
- \(N\): normalization;
- \(C\): final calibration.

Then define the target law precisely.

If the target is the **induced endpoint law** \(Q_E\), generate calibration endpoints by applying the same center-selection, donor-selection, and interpolation mechanism to calibration centers. Split conformal can then rank a new endpoint relative to \(Q_E\).

That conclusion would be:

> The endpoint is typical under the declared endpoint-generation procedure.

It would **not** be:

> The endpoint is natural, clean-manifold, or typical under the unmodified data distribution.

If the intended target remains the natural clean distribution, endpoint ranks should be described only as descriptive outlier scores, not conformal coverage.

### Resolution and multiplicity

With a final calibration size of 16 per context, attainable conformal \(p\)-values lie on a grid of \(1/17\approx0.0588\). At \(\alpha=0.1\), only the most extreme rank is rejected. Multiple endpoints generated from the same center and donor pool are dependent, and the code provides no simultaneous guarantee.

**Verdict:** the generic conformal function is valid under its theorem’s conditions; the current endpoint interpretation does not satisfy those conditions.

## 6. Strongest plausible theorem capable of rescuing the hierarchy

The strongest plausible theorem is not “small IRS implies nearby Jacobians under isotropic probes.” That is already implied by Taylor expansion and random-projection identities. The required theorem must connect a designed local interaction estimand to a defined structural parameter and include a converse.

### Candidate theorem: probe-complete identification of fixed-basis local mediator interactions

Let \(s\in\{p,\star\}\) index patched and target systems. Let

\[
z=(z_1,\ldots,z_q)
\]

be a **fixed, semantically declared mediator cut** in a common basis, and let

\[
f_s(z)
\]

be the downstream output map.

Assume:

1. \(f_p,f_\star\in C^4\) on radius-\(r_0\) neighborhoods of their centers.
2. The mediator basis and grouping are fixed in advance; arbitrary invertible alignment maps are not allowed.
3. For each mediator block \(i\), a product probe law \(Q_i\) has second moment bounded below on a declared subspace \(S_i\):
   \[
   M_i\succeq \lambda_i P_{S_i},\qquad \lambda_i>0.
   \]
4. First- and mixed-second-order local response tensors are
   \[
   A_{i,s}=D_i f_s(z_s),
   \qquad
   B_{ij,s}=D_{ij}^2f_s(z_s).
   \]
5. A structural parameter \(\theta_s\)—for example, fixed-basis local path gains in a specified acyclic mediator graph—maps linearly or locally bi-Lipschitzly to these tensors:
   \[
   \mathcal L(\theta_s)
   =
   \{A_{i,s},B_{ij,s}\}_{i,j},
   \]
   with
   \[
   \|\mathcal L(\theta)-\mathcal L(\theta')\|
   \ge
   \kappa\|\theta-\theta'\|,
   \qquad
   \kappa>0.
   \]
6. Derivative bounds and probe moments are sufficient to control finite-radius remainders.

Use central first-order estimators

\[
\widehat A_{i,s}^{(r)}(u)
=
\frac{
f_s(z_s+rE_i u)-f_s(z_s-rE_i u)
}{2r},
\]

and mixed four-point estimators

\[
\widehat B_{ij,s}^{(r)}(u,v)
=
\frac{
f_s(z_s+rE_i u+rE_jv)
-f_s(z_s+rE_i u-rE_jv)
-f_s(z_s-rE_i u+rE_jv)
+f_s(z_s-rE_i u-rE_jv)
}{4r^2}.
\]

Define

\[
\mathcal E_2(r)
=
\sum_i
\mathbb E_{u\sim Q_i}
\left\|
\widehat A_{i,p}^{(r)}(u)
-
\widehat A_{i,\star}^{(r)}(u)
\right\|^2
+
\sum_{i<j}
\mathbb E_{u\sim Q_i,v\sim Q_j}
\left\|
\widehat B_{ij,p}^{(r)}(u,v)
-
\widehat B_{ij,\star}^{(r)}(u,v)
\right\|^2.
\]

A contribution-level theorem would establish constants \(c_Q,C_Q,C>0\) such that

\[
c_Q
\left\|
\mathcal L(\theta_p-\theta_\star)
\right\|^2
-
Cr^2
\le
\mathcal E_2(r)
\le
C_Q
\left\|
\mathcal L(\theta_p-\theta_\star)
\right\|^2
+
Cr^2,
\]

and hence

\[
\|\theta_p-\theta_\star\|^2
\le
\frac{\mathcal E_2(r)+Cr^2}
     {c_Q\kappa^2}.
\]

It should also prove a converse:

- if the probe covariance is rank-deficient on a structurally relevant subspace, or
- if \(\mathcal L\) is non-injective because of cancellation, symmetry, or reparameterization,

then there exist distinct structural parameters \(\theta_p\ne\theta_\star\) with zero limiting response energy under the declared probes.

### Why this would not be a routine Taylor/random-projection result

Taylor expansion supplies the \(O(r^2)\) truncation behavior for central differences. Random-projection identities convert expected probe energy into a weighted tensor norm. Neither supplies:

- an if-and-only-if probe-completeness condition;
- isolation of mixed mediator interaction tensors;
- an inverse bound from response tensors to path parameters;
- a transformer-relevant graph class for which \(\kappa>0\);
- a constructive counterexample when the probe or structural map is deficient.

The multi-mediator literature provides scalar effect decompositions and interaction terms, but not automatically this full tensor-identification result under a designed product probe law. 

### Non-negotiable caveat

If the proof simply **assumes** that the desired path parameters are injectively encoded in \(\{A_i,B_{ij}\}\) and calls the assumed minimum singular value \(\kappa\), the theorem is tautological. The hard part is deriving injectivity and a usable \(\kappa\) for a substantive mediator graph or transformer computation class.

**Current verdict:** no theorem of this strength exists in the repository. A plausible theorem direction exists, but unless the structural inverse is derived rather than assumed, there is no oral-level theorem to claim.

---

# Empirical and Statistical Audit

## 1. Fair-baseline result defeats the IRS-method story

The relevant comparison is not IRS versus behavioral restoration alone. It is IRS versus a fair single clean–corrupt interaction direction evaluated with the same downstream response machinery.

The repository reports approximately:

| Diagnostic | IRS | Fair single direction | Result |
|---|---:|---:|---|
| Layer-stage Spearman correlation | \(-0.900\) | **\(-0.967\)** | Single direction has stronger magnitude. |
| Leave-one-context-out error | \(0.1673\) | **\(0.1620\)** | Single direction is slightly better. |
| Corruption-shift rank agreement | \(0.783\) | **\(0.817\)** | Single direction is better. |
| Corruption-shift relative-\(L_2\) agreement | \(0.804\) | **\(0.858\)** | Single direction is better. |

The differences are not large enough to establish that IRS is worse in every meaningful sense. They are sufficient to reject the claim that multi-directional IRS has demonstrated additional diagnostic value. The repository’s own frozen novelty gate correctly treats evidence beyond the fair direction as mandatory. 

A reviewer’s conclusion will be:

> The elaborate method recovers the same coarse layer-stage signal as the obvious clean–corrupt direction and does not improve held-out prediction.

That is fatal to a named-method oral pitch.

## 2. Robustness gate fails

At the preregistered stricter probe setting, the required robustness result does not hold. At \(\eta=0.5\), the strict gate fails; at \(\eta=1\), the probes leave the intended support regime. The repository therefore cannot claim that the result is robust over a meaningful finite-radius range. 

The correct interpretation is:

- small-chord IRS can reproduce a stage-level pattern;
- its finite-radius stability is unresolved;
- larger chord fractions confound local-response comparison with support departure.

This is not a reason to add more \(\eta\) values post hoc. It is a reason to define radius relative to layer scale and test a preregistered bias–support tradeoff.

## 3. The pABC structural-separation result is ill-conditioned

The raw corruption-shift JSON is [outputs/exp_p0_irs_corruption_shift_gpt2_seed20260712.json](https://github.com/ScottBlizzard/idle_1/blob/main/outputs/exp_p0_irs_corruption_shift_gpt2_seed20260712.json).

For duplicate-name IOI, the clean–corrupt NMH gap used as the normalization denominator is approximately

\[
0.421703.
\]

For pABC, it is approximately

\[
0.008002.
\]

Thus the pABC denominator is **52.7× smaller**. The reported normalized NMH “recoveries” of roughly \(-1.40\) to \(-2.31\) at layers 4–8 correspond to absolute patched-minus-corrupt shifts of only approximately

\[
-0.0112,\,-0.0149,\,-0.0172,\,-0.0185,\,-0.0142.
\]

The dramatic negative normalized values are therefore mostly a ratio instability, not a large independent structural failure. 

The same rows report behavioral restoration around \(0.920\)–\(0.939\) and IRS around \(0.149\)–\(0.176\). Those observations may still be descriptively interesting, but the claimed third axis—strongly failed structural recovery—has not been established.

**Required conclusion:** pABC does **not** currently demonstrate that strong behavioral restoration plus low IRS can coexist with strong structural non-restoration.

## 4. Name Mover Head recovery is not an independent structural ground truth

The implementation in [src/exp_p0_within_site_mechanism.py](https://github.com/ScottBlizzard/idle_1/blob/main/src/exp_p0_within_site_mechanism.py) averages selected attention-pattern probabilities from heads \(9.9\) and \(10.0\), measured from the final token to the indirect-object position. 

This is a separate readout from the final logit and is temporally eligible for interventions at layers 0–8. It is still not a structural ground truth:

- attention probability is not the head’s value-vector write;
- it is not the head’s direct logit contribution;
- it is not a path-specific causal effect through the published IOI circuit;
- the heads were selected from prior knowledge of the IOI circuit rather than discovered independently;
- averaging two attention coordinates can conceal compensating changes;
- the normalization is corruption-dependent and becomes unusable when the clean–corrupt gap is nearly zero.

The original IOI work identifies a multi-head circuit and evaluates faithfulness, completeness, and minimality using causal interventions, not attention probability alone. [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593) is therefore a standard the present witness does not meet. 

**Per-corruption verdict:**

- **Duplicate-name IOI:** the denominator is sufficiently large for a descriptive normalized coordinate, but the metric remains a circuit-correlate rather than structural recovery.
- **pABC:** the denominator is too small and the corruption changes the functional setting enough that the normalized score cannot be interpreted as recovery of the same structural target.

A valid replacement must use an **absolute path-specific causal effect**, with a preregistered minimum clean–corrupt effect size.

## 5. Statistical units are invalid for confirmatory inference

The aggregate report contains three sampling runs and 2,160 prompt–layer rows, with layers 4–7 selected as a “stable” region. The analysis code then computes correlations or significance using repeated layers and repeated prompts. 

These are not independent units:

- neighboring layers are ordered components of the same fixed model;
- the same prompts recur across layers;
- prompts share templates, names, donor pools, and model parameters;
- the three “seeds” resample data or probes; they are not three independently trained GPT-2 models;
- layers 4–7 were selected because they looked jointly stable on the same measurements later summarized.

Consequences:

1. A \(p\)-value based on 27 layer–seed rows treats correlated measurements as replication.
2. A \(p\)-value based on prompt–layer rows treats every layer copy of a prompt as a new observation.
3. The reported prompt-level significance can become extremely small through pseudo-replication even when the prompt-aggregated effect is weak.
4. Post-selecting the “stable” layer range invalidates confirmatory interpretations within that range.

The reported layer-stage pattern can be retained as a descriptive visualization. Its current significance calculations cannot support a general claim.

The leave-one-context-out analysis is more credible as a leakage check because it withholds larger semantic groups. It remains an evaluation on one task, one pretrained model, and a small number of contexts; it does not establish population-level generality.

## 6. Probe and context construction

The reference partition in [src/exp_p0_reference_crossfit.py](https://github.com/ScottBlizzard/idle_1/blob/main/src/exp_p0_reference_crossfit.py) randomizes IO-name items and partitions fit, calibration, and evaluation examples within context. That is preferable to fitting and evaluating on identical prompt instances. 

However, context-matched donor construction has two consequences:

- each evaluation item has an item-conditioned probe law rather than a common \(Q\);
- nearest or eligible donors can make probes easier by aligning them with the fitted clean geometry.

This does not automatically invalidate clean held-out comparisons. It does weaken any claim that the estimator measures a context-independent property of the mechanism. The estimand is:

> Agreement under this particular context- and donor-conditioned intervention-generation process.

That restriction must appear in every claim.

## 7. Tests validate algebra, not scientific validity

All eleven tests in [src/test_interventional_response.py](https://github.com/ScottBlizzard/idle_1/blob/main/src/test_interventional_response.py) and [src/test_validity_crossfit.py](https://github.com/ScottBlizzard/idle_1/blob/main/src/test_validity_crossfit.py) pass. They check synthetic response behavior, basic normalization, and generic split-conformal false-alarm behavior. 

They do not test:

- vector-output agreement between theory and implementation;
- endpoint-selection coverage;
- item-conditioned or without-replacement probe concentration;
- denominator instability;
- cluster-level statistical inference;
- post-selection;
- path-specific structural recovery;
- superiority to a single direction.

Passing them establishes implementation consistency for basic cases. It does not resolve the paper’s scientific validity.

---

# Main-Line Decision

## IRS disposition

**IRS should not remain the headline named method.**

The correct disposition is:

> Retain IRS, if at all, as one operational witness inside a broader paper about the identification boundaries between behavioral restoration, probe-indexed local interaction agreement, and structural mechanism recovery.

In the main text, use a descriptive name such as **probe-indexed local response discrepancy**. Keep “IRS” in code, tables, or an appendix only. Do not put IRS in the title.

If the experiment in the next section fails to show incremental information beyond both a fair single direction and a first-order multi-probe baseline, IRS should be removed from the main contribution entirely.

## Viable and non-viable main lines

### Potentially viable oral line

A paper becomes an oral candidate only if it contains both:

1. a probe-complete first-/second-order interaction-identification theorem for a non-vacuous structural class, including impossibility converses; and
2. a locked pretrained-model experiment showing that the mixed-order estimand predicts an absolute path-specific causal quantity beyond fair alternatives.

### Poster-level fallback

A defensible poster-level paper could instead be:

- a rigorous audit of why behavioral restoration, local response diagnostics, and selected circuit coordinates disagree;
- explicit about probe-law dependence;
- explicit that IRS did not beat a single direction;
- explicit that the strict robustness gate failed;
- stripped of pABC structural-separation and endpoint-coverage overclaims.

### Non-viable line

The following main line is dead:

> IRS is a robust, conformal, multi-directional certificate that distinguishes causally valid activation patching from invalid restoration.

The repository’s own results contradict every load-bearing adjective in that sentence.

## Three reviewer attacks most likely to kill an oral bid

### Attack 1: “This is Taylor expansion, random projections, and conformal ranking under a new acronym.”

A reviewer will observe that:

- zero-order non-identification is standard;
- the local theorem is Taylor transport;
- the trace identity is a standard quadratic-form expectation;
- split conformal is standard conditional on exchangeability;
- recent primary sources already treat intervention distributions, local response fields, multi-mediator interactions, and identification assumptions;
- IRS does not outperform the fair single direction.

Unless the paper proves the structural inverse theorem described above, this attack is correct.

### Attack 2: “The claimed structural counterexample is a denominator artifact, and the conformal endpoints are self-constructed.”

The pABC conclusion depends on dividing by a clean–corrupt NMH difference of only \(0.0080\). The absolute shifts are small, and the readout is attention probability rather than a path-specific causal effect. Separately, endpoint examples are adaptively moved toward the fitted clean reference set and then scored against that geometry. Calling their ranks clean-law conformal coverage is invalid.

This attack kills both the hierarchy’s strongest empirical example and the claimed finite-sample validity layer.

### Attack 3: “The apparent evidence comes from pseudo-replicated layers and prompts, while the actual independent comparison is negative.”

Layers are not independent models. Prompt copies across layers are not independent observations. Sampling seeds are not trained-model replications. The stable layer range is post-selected. Once attention is restricted to held-out contexts and the fair baseline, IRS provides no clear gain and fails the strict robustness gate.

This attack leaves the paper with a descriptive IOI plot rather than a general method result.

---

# Single Highest-Value Next Experiment

## Experiment

**Factorial local-interaction identification on the GPT-2-small Greater-Than circuit, with an absolute path-specific structural readout.**

This is one experiment, not a collection of loosely related tasks. It is the highest-information choice because it simultaneously tests:

- transfer beyond IOI;
- whether mixed mediator interactions matter;
- whether the proposed theorem has empirical content;
- whether multi-probe IRS adds information beyond a single interaction direction;
- whether local response agreement predicts a genuine path-causal quantity.

The published Greater-Than analysis identifies a circuit involving late MLPs and specific attention heads, making it possible to define a structural readout that is stronger than attention probability. [How Does GPT-2 Compute Greater-Than? Interpreting Mathematical Abilities in a Pre-Trained Language Model](https://arxiv.org/abs/2305.00586) reports important roles for MLPs 8–11 and heads including \(5.1\), \(5.5\), \(6.9\), \(7.10\), \(8.8\), \(8.11\), and \(9.1\). 

## Task and model

- **Model:** GPT-2 small only.
- **Task:** the published year-span Greater-Than task.
- **Evaluation cells:** combinations of held-out lexical template family, century prefix, and preregistered distance-to-threshold bin.
- **No model scaling and no second task.** This experiment is intended to resolve a mechanism question, not produce a benchmark average.

## Intervention units

Declare two mediator blocks before examining the confirmatory cells:

\[
M_1 =
\{
\text{outputs of heads }5.1,5.5,6.9,7.10,8.8,8.11,9.1
\}
\]

at the answer-relevant token position, and

\[
M_2 =
\{
\text{outputs of MLPs }8,9,10,11
\}
\]

at the final or answer position specified by the published circuit.

The experiment must estimate:

- first-order responses to \(M_1\) interventions;
- first-order responses to \(M_2\) interventions;
- mixed \(M_1\times M_2\) responses using the four-point factorial contrast.

Do not search over heads, layers, or token positions on the confirmatory split.

## Corruption

Use one corruption family:

> Replace the clean two-digit start-year suffix \(YY\) with a tokenization-matched alternative \(YY'\) within the same century, chosen from a preregistered threshold-distance stratum.

The corruption must preserve:

- tokenizer length;
- century prefix;
- lexical template;
- answer vocabulary;
- all non-year content.

The threshold-distance bins must be frozen before the development run. Do not introduce a second corruption if the first does not work.

## Target probe law

Define a product donor law from actual held-out clean mediator states:

\[
u_1\sim Q_{M_1}(\text{template},\text{century},\text{distance bin}),
\qquad
u_2\sim Q_{M_2}(\text{template},\text{century},\text{distance bin}).
\]

Requirements:

1. Donor-construction set \(D\) is disjoint from geometric fitting, normalization, calibration, development evaluation, and confirmation.
2. Donors are sampled using a fixed rule, not nearest-neighbor selection optimized separately for each candidate method.
3. The same donor pairs are used by every baseline.
4. Probe radii or swap magnitudes are logged.
5. A half-radius stability check is preregistered.
6. Probe-covariance spectra are reported for the declared mediator subspaces.

The target estimand is explicitly the response discrepancy under this conditional product law. No claim should extend beyond it.

## Structural ground truth/readout

Use an **absolute path-specific causal effect** through

\[
M_1\rightarrow M_2\rightarrow
\text{Greater-Than logit margin}.
\]

For each held-out cell \(c\), estimate a path/edge-patching quantity such as

\[
\operatorname{PSE}_{s,c}
=
\text{causal contribution of the declared }
M_1\!\rightarrow M_2
\text{ path to the correct-vs-incorrect logit margin}
\]

for \(s\in\{\text{patched},\text{target}\}\), and define

\[
T_c
=
\left|
\operatorname{PSE}_{\mathrm{patched},c}
-
\operatorname{PSE}_{\mathrm{target},c}
\right|.
\]

This is the structural target to be predicted.

Do **not** use:

- attention probability;
- activation cosine similarity;
- a ratio divided by a near-zero clean–corrupt gap;
- the final behavioral restoration score itself.

The path-specific intervention should follow an explicit causal-abstraction or path-patching definition rather than an informal “circuit activity” proxy. 

## Fair baselines

All methods must receive identical cells, donor pairs, radius budgets, and downstream model evaluations.

1. **Single clean–corrupt interaction direction.**
2. **Multi-probe first-order-only response discrepancy:** same probe law and number of directions, but no mixed \(M_1\times M_2\) term.
3. **PIE/INT-style factorial scalar decomposition:** an implementation aligned with the multi-mediator decomposition rather than a weak ablation baseline. 
4. **Behavioral restoration alone.**

The key comparison is the incremental value of the mixed interaction term over baseline 2, not merely over behavior.

## Statistical unit

Use the **held-out semantic cell**, not the prompt, probe, layer, head, or donor draw.

Predeclare **48 cells**:

- 16 development cells;
- 32 untouched confirmatory cells.

A cell is a unique combination of:

- template family;
- century prefix;
- threshold-distance bin.

Prompts and donor draws are nested repeated measurements within cells. All intervals, permutations, and bootstraps must resample at the cell level.

Stop the experiment if fewer than **40 total cells** or fewer than **28 confirmatory cells** survive tokenization and structural-readout admissibility checks. Do not silently replace failed cells after seeing outcomes.

## Decisive success threshold

The mixed first-/second-order method succeeds only if, on the untouched 32-cell confirmatory set:

1. It reduces cell-level RMSE for \(T_c\) by at least **20%** relative to the best fair baseline.
2. The paired cell-bootstrap 95% lower confidence bound on the relative RMSE reduction is at least **10%**.
3. In a preregistered cancellation subset—cells where first-order block effects oppose one another—the mixed method achieves AUROC at least **0.80**, with 95% lower bound at least **0.70**, for detecting high structural mismatch.
4. The result holds in both threshold-distance bins without selecting sites or bins after evaluation.
5. At least 90% of confirmatory cells have a well-conditioned structural target, defined before evaluation as either:
   \[
   |\operatorname{PSE}_{\mathrm{clean}}-
   \operatorname{PSE}_{\mathrm{corrupt}}|
   \ge 0.10
   \]
   logit-margin units, or at least \(0.25\) development-cell standard deviations.
6. The half-radius version preserves cell ranking with Spearman correlation at least **0.90** and changes the median score by no more than **20%**.

Meeting only the correlation threshold or only the point-estimate RMSE threshold is not success.

## Decisive failure threshold

The central result fails if any of the following occurs:

- confirmatory RMSE improvement is below **5%**;
- the confidence interval includes zero improvement;
- the best single-direction or first-order-only baseline matches the mixed method within the above uncertainty;
- no well-conditioned structural-mismatch cells are found;
- the claimed gain depends on post hoc mediator, layer, head, corruption, radius, or cell selection;
- mixed-response signal disappears at half radius;
- path-specific effects cannot be estimated without using the same behavioral outcome as both predictor and target.

On failure, IRS must be removed from the main contribution. The project may continue only as a negative-results or identification-audit paper.

---

# Seven-to-Ten GPU-Day Decision Tree

The budget below totals at most **10 GPU-days**. Early failure means stopping rather than reallocating the remaining budget to exploratory variants.

| Stage | Budget | Named ambiguity resolved | Pass branch | Stop branch |
|---|---:|---|---|---|
| 1. Structural theorem gate | 0 GPU-days | Is the proposed hierarchy backed by a non-tautological identification result? | Produce the first-/mixed-second-order theorem, converse, and a derived injectivity result for a substantive mediator graph class. Continue. | If \(\kappa>0\) is only assumed, or the result reduces to Taylor plus random projection, stop the oral project. At most retain a poster-level audit. |
| 2. Estimator and conformal repair | 0.5 GPU-days | Do theory, code, probe law, and coverage target describe the same estimand? | Fix vector-output averaging, derivative constants, radius logging, independent \(D/F/N/C\) splits, induced-law calibration, and cell-level analysis. Continue. | If an endpoint-law calibration cannot be specified without changing the scientific claim, delete endpoint coverage. If core scores cannot be reproduced after the fixes, stop. |
| 3. Structural-readout pilot | 1 GPU-day | Does Greater-Than provide a well-conditioned independent structural target? | On development cells, at least 90% satisfy the frozen absolute PSE condition and path estimates are stable across donor resampling. Continue. | If the PSE gap is near zero or path estimates are dominated by noise, stop. Do not substitute attention probability or a new task. |
| 4. Interaction-signal pilot | 1 GPU-day | Are mixed \(M_1\times M_2\) effects detectable at genuinely local radii? | Mixed contrast has signal-to-noise ratio at least 3 in 60% of development cells; half-radius rank correlation is at least 0.90. Continue. | If mixed effects vanish, are swamped by estimator variance, or exist only at off-support radii, stop the IRS line. |
| 5. Frozen development run | 2.5 GPU-days | Does the complete pipeline show enough prospective effect to justify confirmation? | Freeze all mediators, radii, normalizers, model-selection rules, and thresholds. Continue if mixed-order RMSE improvement over the best baseline is at least 10% on the 16 development cells. | If improvement is below 5%, stop. If it lies between 5% and 10%, classify the project as poster-level and do not use confirmation as a lottery ticket for an oral claim. |
| 6. Untouched confirmatory run | 4 GPU-days | Does the central result survive without post-selection? | Continue only if all decisive success conditions in the preceding section are met on the 32 untouched cells. | Any decisive failure condition terminates the oral line. No new corruption, site, or radius may be introduced. |
| 7. Prespecified finite-radius check | 1 GPU-day | Is the confirmed result a local interaction result rather than a radius artifact? | Half-radius stability passes in both threshold-distance bins. The project may proceed as an oral candidate. | If the main result changes materially with radius or loses one bin, downgrade to poster-level. |

## Branch outcomes

### Green branch: oral candidate

All of the following are true:

- theorem has a derived structural inverse and converse;
- structural readout is absolute and well-conditioned;
- mixed-order method wins against every fair baseline;
- result passes untouched cell-level confirmation;
- half-radius stability passes;
- no endpoint or conformal overclaim remains.

This branch supports an oral submission but does not guarantee one.

### Amber branch: poster only

Any of the following occurs:

- theorem is an impossibility result without a positive transformer-relevant class;
- mixed interactions are real but provide less than the decisive predictive gain;
- the result is informative but radius-sensitive;
- the experiment validates a distinction without establishing identification.

Frame the paper as an audit or boundary result.

### Red branch: stop

Stop the main project if:

- the theorem is tautological;
- the structural target is ill-conditioned;
- the mixed term has no stable signal;
- the single direction or first-order baseline matches the proposed method;
- the confirmatory interval includes no improvement;
- the result requires post hoc changes.

Do not respond to a red branch by adding models, seeds, tasks, or acronyms.

---

# Allowed, Conditional, and Prohibited Claims

## Currently supported claims

| Claim | Permissible scope |
|---|---|
| Pointwise behavioral restoration does not identify local response behavior without derivative assumptions. | A mathematical non-identification statement, not a claim that restoration is always misleading. |
| Under \(C^2\) regularity, local response disagreement is controlled by Jacobian disagreement plus a finite-radius remainder. | Only on the declared coordinates and local neighborhood. |
| For scalar outputs, the current IRS implementation estimates a normalized, probe-law-indexed finite-response discrepancy. | Conditional on the implemented chord law and normalization. |
| The trace identity makes IRS a probe-covariance-weighted Jacobian-discrepancy seminorm in the infinitesimal limit. | Not a basis-free or structural metric. |
| Generic composite split conformal ranking is valid for a fresh query exchangeable with the final calibration examples. | Conditional on the fitted score and preceding data splits. |
| Multi-directional IRS and a single clean–corrupt direction both recover the coarse IOI layer-stage pattern. | Descriptive result for GPT-2-small IOI. |
| IRS has not shown a clear advantage over the fair single-direction baseline. | Directly supported by the reported comparison. |
| The preregistered strict robustness gate fails. | Must be disclosed wherever robustness is discussed. |
| The corruption-shift result supports non-inferiority at best, not superiority. | Limited to the tested corruption and metrics. |
| Activation overlap is at most a compatibility diagnostic relative to a declared target distribution. | It is not a causal certificate. |

The theoretical and empirical bases for these statements are in the theory note, aggregate report, fair-baseline report, and stress summary. 

## Claims requiring one new central result

| Conditional claim | Required evidence |
|---|---|
| The three-level hierarchy is more than a synthesis. | The probe-complete interaction-identification theorem with a derived structural inverse and converse. |
| Mixed local response information detects structural disagreement missed by a fair single direction. | Locked Greater-Than result meeting the RMSE and cancellation thresholds. |
| The local-functional/structural separation occurs in a pretrained transformer. | A well-conditioned absolute path-specific causal readout; pABC NMH normalization is insufficient. |
| Higher-order mediator responses are necessary rather than decorative. | Incremental gain over the same multi-probe first-order-only baseline. |
| Probe completeness can be certified or diagnosed empirically. | Covariance-spectrum condition tied to the theorem and reported on the actual probe law. |
| Endpoint conformal ranks have formal validity. | Calibration examples generated exchangeably under the same induced endpoint law. |
| The method is finite-radius stable. | Prespecified half-radius confirmation under a common normalized-radius convention. |
| The contribution is oral-level. | All theory, structural-readout, baseline, statistical-unit, and confirmation gates must pass jointly. |

## Claims that must be prohibited

1. **“Activation overlap or IVS certifies causal validity.”**
2. **“Low IRS implies restoration of the same circuit, path, or mechanism.”**
3. **“The pABC experiment proves strong structural non-restoration.”**
4. **“Endpoint conformal acceptance above 0.99 shows that an endpoint is natural or on-manifold.”**
5. **“IRS outperforms a fair single-direction diagnostic.”**
6. **“IRS is robust across probe radii.”**
7. **“The current layer- or prompt-level \(p\)-values provide valid confirmatory evidence.”**
8. **“Three random seeds constitute independent model replication.”**
9. **“Name Mover Head attention probability is structural ground truth.”**
10. **“The result generalizes across transformers, tasks, circuits, or corruption families.”**
11. **“Composite conformal makes an adaptively selected endpoint valid under the clean reference law.”**
12. **“The current method is a causal certificate.”**
13. **“Multi-directionality itself guarantees coverage of relevant mechanism directions.”**
14. **“Behavioral, local-functional, and structural equivalence form a strict universal hierarchy with automatic implications.”**

---

# Recommended Framing

## Best final framing

The only plausible oral framing is:

> **Identification boundaries for neural interventions, with a positive probe-completeness theorem for a restricted multi-mediator class and a pretrained-model demonstration that mixed local interactions predict path-specific structural mismatch.**

The paper should not be framed as a new score. The score is an estimator inside the identification argument.

The scientific sequence should be:

1. pointwise restoration is non-identifying;
2. first-order local response agreement identifies only a probe-weighted operator restriction;
3. multiple mediators require mixed-order probes;
4. under explicit basis, graph, faithfulness, and probe-completeness assumptions, first-/mixed-second-order responses identify a defined local structural parameter;
5. rank-deficient probes and non-injective structural classes admit counterexamples;
6. the Greater-Than experiment tests the theorem against single-direction, first-order-only, and PIE/INT alternatives.

Do not describe the hierarchy as a ladder on which evidence automatically moves upward. Describe it as a set of **distinct equivalence relations**, with theorems specifying when one refines another.

## Title pattern

Use this only if every green-branch requirement is met:

> **When Does Behavioral Restoration Identify a Mechanism? Probe-Complete Local Interaction Tests for Neural Interventions**

A slightly more conservative pattern is:

> **Identification Boundaries for Neural Interventions: From Behavioral Restoration to Probe-Complete Local Interaction Equivalence**

Do not put “IRS,” “certificate,” “conformal validity,” or “causal restoration” in the title.

If the theorem or confirmatory experiment fails, use a poster-level title such as:

> **Restoration Is Not Mechanism Recovery: An Audit of Local Response Diagnostics in GPT-2 IOI**

## Conditional oral-level contribution statement

> Behavioral restoration after a neural intervention is a zero-order observation and generally does not identify the restored computation. We formalize behavioral, probe-indexed local interaction, and fixed-basis structural equivalence as distinct relations. For a declared multi-mediator causal class, we derive probe-completeness conditions under which central first-order and mixed-second-order interventional responses identify local path parameters up to finite-radius and sampling error, and prove converse non-identifiability under rank-deficient probes or non-injective structural maps. In GPT-2 small’s Greater-Than circuit, a locked product-probe experiment shows that the mixed-order estimator predicts independently measured path-specific causal mismatch beyond behavioral restoration, a fair clean–corrupt direction, an equally budgeted first-order multi-probe estimator, and existing factorial mediator-effect diagnostics.

That paragraph is **not currently supported**. It is the target contribution statement whose every clause must be earned by the decision tree.

---

# Final Go/No-Go Checklist

## Theory

- [ ] The hierarchy is defined as distinct equivalence relations, not rhetorical evidence levels.
- [ ] The zero-order theorem remains narrowly scoped.
- [ ] The local transport theorem states the exact intervention coordinates and probe law.
- [ ] The vector-output \(\sqrt{k}\) issue is fixed.
- [ ] Theory and code use the same output-axis aggregation.
- [ ] Actual chord radii are logged and appear in all finite-radius claims.
- [ ] Normalized-score bounds assume and verify a non-negligible target-response denominator.
- [ ] The probe-law theorem handles item-conditioned and without-replacement probes.
- [ ] Probe covariance spectra and covered subspaces are reported.
- [ ] A first-/mixed-second-order identification theorem is proved.
- [ ] Structural injectivity is derived for a substantive model class rather than assumed.
- [ ] Rank-deficiency and structural-noninjectivity converses are proved.

## Conformal and support logic

- [ ] Donor construction, score fitting, normalization, and final calibration use separated data roles.
- [ ] The target distribution is named precisely.
- [ ] Endpoints are calibrated under their own induced generation law, or all coverage language is removed.
- [ ] No endpoint rank is interpreted as naturalness or manifold membership.
- [ ] The effect of the coarse \(1/17\) \(p\)-value grid is disclosed.
- [ ] No marginal conformal statement is presented as a joint guarantee.

## Empirics

- [ ] The pABC normalized NMH result is removed as structural-separation evidence.
- [ ] Structural recovery is measured with an absolute path-specific causal effect.
- [ ] A minimum structural-effect condition is preregistered.
- [ ] The experiment is conducted on Greater-Than without post hoc task substitution.
- [ ] Mediator blocks, positions, corruption, probe law, and radii are frozen.
- [ ] The fair single-direction baseline is included.
- [ ] The equally budgeted first-order multi-probe baseline is included.
- [ ] A PIE/INT-style multi-mediator baseline is included.
- [ ] Statistical inference uses semantic cells rather than prompts, probes, layers, or sampling seeds.
- [ ] Confirmatory cells remain untouched until the pipeline is frozen.
- [ ] Mixed-order RMSE improvement is at least 20%, with a 95% lower bound of at least 10%.
- [ ] The cancellation-subset threshold is met.
- [ ] Half-radius stability passes in both preregistered bins.

## Claims

- [ ] No activation-overlap certificate claim remains.
- [ ] No “low IRS implies same mechanism” claim remains.
- [ ] No unsupported superiority or robustness claim remains.
- [ ] No layer/prompt pseudo-replication is used.
- [ ] No sampling seed is described as an independent model replication.
- [ ] No attention probability is called structural ground truth.
- [ ] No generalization beyond the tested model, task, mediator class, and probe law is implied.
- [ ] IRS is not the title-level contribution.

## Final decision

**As of 5 August 2026:**

- **ICLR oral:** **NO-GO.**
- **Headline IRS-method paper:** **NO-GO.**
- **Current pABC local-functional/structural separation claim:** **INVALID AS EVIDENCE.**
- **Current endpoint clean-law conformal interpretation:** **INVALID.**
- **Poster-level intervention-diagnostics audit after removing invalid claims:** **POSSIBLE.**
- **Oral-level rebuild:** proceed only through the gated theorem-plus-Greater-Than experiment above.

The project should continue toward an oral bid only if it obtains a new central identification result and the one locked experiment demonstrates information unavailable to the fair baselines. Otherwise, the scientifically correct outcome is to stop the method claim and publish, if worthwhile, a narrower negative or boundary result.