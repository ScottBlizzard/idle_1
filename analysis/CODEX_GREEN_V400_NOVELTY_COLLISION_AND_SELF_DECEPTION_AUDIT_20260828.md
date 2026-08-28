# GREEN v4 Novelty-Collision and Self-Deception Audit

**Audit date:** 2026-08-28  
**Scope:** public primary literature through 2026-08-28, local GREEN theory/protocol lineage, and the current v4 formal-prepare implementation  
**Evidence boundary:** no v4 real-row scientific outcome, development result, or confirmation result was opened  
**GitHub:** no push performed

## 1. Executive decision

GREEN v4 has **not** been completely pre-empted by one public paper. However, the earlier optimistic statement that there is no strong close work is no longer defensible. Most individual primitives have already appeared, and several 2026 papers are direct conceptual neighbours.

The following broad novelty claims must be abandoned:

- first demonstration that successful activation patching or restoration can misidentify a mechanism;
- first formal or certified mechanistic-interpretability method;
- first four-outcome mediator/bypass interaction functional;
- first Hessian or second-order diagnosis of patching error;
- first anytime-valid certificate for causal claims in mechanistic interpretability.

The remaining potentially novel claim is narrower:

> For a preregistered matched-bypass functional in an unmodified smooth Transformer tail, GREEN constructs a bit-exact relational interval-jet certificate for the function, derivative, full-interval curvature, and signed finite-radius remainder; it remains valid at every budget checkpoint and is independently replayed at higher precision. The certificate is scientifically useful only if it prospectively predicts an independent held-out mechanistic consequence that simpler patching, HVP, SNR, and generic verification baselines do not predict.

This is a real and potentially valuable contribution. It is currently a **conditional combination novelty**, not a demonstrated Oral-level scientific result.

## 2. Independent audit consensus

Three independent red-team tracks reached the same conclusion:

1. **Novelty/priority audit:** no full-system duplicate was found, but the four-branch interaction, continuous certified patching, second-order error analysis, and anytime causal-certification language all have strong prior art.
2. **Self-deception audit:** the numerical proof machinery is substantive, but it certifies a frozen intervention functional, not the semantic truth or uniqueness of a mechanism. The current Boundary Transition has two P0 design defects.
3. **ICLR reviewer/AC audit:** the current project is not submission-ready because the only complete manuscript still presents the superseded IVS story and v4 has no real scientific outcomes. The present recommendation is Reject/Weak Reject; Oral potential depends on a decisive independent experiment.

## 3. Closest prior art and collision level

| Work | What it already establishes | Collision with GREEN | What may remain distinct |
|---|---|---|---|
| [Vaidyanathan et al., *The Curse of Multiple Mediators*](https://arxiv.org/abs/2606.27510) | Transformer activation patching contains mediator-bypass interaction; gives the four-outcome contrast and a mixed second-directional-derivative interpretation; tests GPT-2 IOI and Greater-Than | **Very high** for the four-branch and second-order interpretation | Exact matched gate intervention semantics, rank-one gate operator, relational full-interval deterministic certificate, finite-radius signed remainder, and independent prediction |
| [Hadad et al., *Formal Mechanistic Interpretability*](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html) | ICLR 2026; provable circuit discovery and robust patching over continuous domains | **Very high** for “first formal/certified MI” | Smooth unmodified language Transformer, four-branch functional, first/second jets, signed remainder rather than circuit robustness/minimality |
| [Zhang and Wang, *When Attribution Patching Lies*](https://arxiv.org/abs/2606.09899) | Downstream nonlinearity causes attribution-patching error; provides reliability scores, error bounds, HVP correction, and 124M–9B experiments | **Very high** for “patching lies because of curvature” | Deterministic full-interval rather than approximate HVP bounds; joint matched-bypass target |
| [Grant et al., *Addressing Divergent Representations from Causal Interventions*](https://proceedings.iclr.cc/paper_files/paper/2026/hash/133e588e1429f9f1e25b215da145580e-Abstract-Conference.html) | ICLR 2026 Oral; divergent interventions can activate hidden pathways and yield misleadingly confirmatory behavior | **Fatal to the old broad Restoration Lies headline** | v4 must be framed around certifiability and independent consequence, not rediscovery of intervention divergence |
| [Makelov et al., *Interpretability Illusion for Subspace Activation Patching*](https://arxiv.org/abs/2311.17030) | ICLR 2024; a successful patch can activate a causally disconnected dormant parallel pathway | **Fatal to the old dormant-path novelty claim** | Same as above |
| [Asiaee, *Certified Interventional Fidelity*](https://arxiv.org/abs/2607.08349) | UAI 2026; anytime-valid statistical certification for intervention-based MI claims on GPT-2 IOI | **High terminology collision** | CIF certifies population estimands statistically; GREEN certifies one deterministic computation graph numerically |
| [Khemais, Part I](https://arxiv.org/abs/2608.03620) and [Part II](https://arxiv.org/abs/2608.03629) | Patching/ablation dissociation, exact interaction identities, second-order remainder, closed-form attention Jacobian bound, and real-model IOI test | **High** for interaction/remainder theory | Full multilayer smooth causal cone, exact LayerNorm/softmax/GELU interval propagation, four correlated branches |
| [Somani, *Towards Verifiable Transformers*](https://arxiv.org/abs/2605.24033) | Solver-checkable circuit explanations and continuous residual perturbation robustness at GPT-2 scale | **High** for “verifiable Transformer mechanism” | The verified GPT-2-scale object is modified/calibrated; GREEN targets a frozen original-model path functional |
| [Anani et al., *Certified Circuits*](https://arxiv.org/abs/2602.22968) | ICML 2026; provably stable circuit inclusion under concept-dataset edits, including GPT-2 Greater-Than | **Moderate/high naming and benchmark collision** | Different guarantee: dataset stability rather than exact nonlinear functional evaluation |
| [Olivieri and Pérez Rodríguez, *Transformer Field Theory*](https://arxiv.org/abs/2605.25225) | Response operators, local linear regimes, propagation, Green-operator slices, and transfer | **Moderate/high** for response-operator framing | Deterministic nonlinear certificate and matched-bypass identification |
| [Palumbo et al., *Validating Mechanistic Interpretations: An Axiomatic Approach*](https://proceedings.mlr.press/v267/palumbo25a.html) | ICML 2025; compositional axioms for validating mechanistic interpretations | **Moderate** for broad validation framing | Local quantitative proof object rather than a full high-level interpretation axiom system |
| [Méloux et al., *Is Mechanistic Interpretability Identifiable?*](https://arxiv.org/abs/2502.20914) | Multiple circuits/interpretations may explain the same behavior | **High semantic warning** | GREEN can certify a declared local functional but must not claim uniqueness of the true mechanism |

## 4. Formal mapping that must be written before submission

The paper must explicitly map

\[
\Psi(t)=PAT_J(t)-PAT_B(t)-TAR_J(t)+TAR_B(t)
\]

to the mediated interaction used by Vaidyanathan et al.:

\[
Y(a)-Y(a',M(a))-Y(a,M(a'))+Y(a').
\]

The mapping must answer, algebraically rather than rhetorically:

1. Are the four quantities identical under a renaming of intervention states?
2. If not, which consistency, path, anchoring, or affine-control assumptions differ?
3. Which GREEN gate operator or theorem is not available from the existing causal-mediation interaction result?
4. Does GREEN certify the existing interaction estimand, or define a genuinely different path functional?

If the functionals are equivalent, the estimand must be credited to the mediation/interaction literature. Novelty then resides in deterministic certification and predictive use.

## 5. What is still plausibly new

No located primary source combines all of the following:

1. a preregistered PAT/TAR matched-bypass functional on a frozen, unmodified language Transformer;
2. exact IEEE-bit import into MPFR;
3. a shared relational causal-cone DAG that preserves four-branch cancellation;
4. outward interval jets for values, first derivatives, and full-interval second derivatives through LayerNorm, softmax, attention, and GELU;
5. a signed endpoint-curvature identity for finite-radius error rather than a sampled or local Hessian approximation;
6. an anytime, budget-monotone deterministic enclosure;
7. non-constructive independent higher-precision full-history replay;
8. an outcome-blind prediction of an independent held-out mechanistic consequence.

Items 1–7 form a defensible formal-methods contribution. Item 8 is necessary to convert it into a strong mechanistic-science contribution.

## 6. P0 self-deception defects

### 6.1 Low-amplitude confidence bound is directionally wrong

The binding v4 corrigendum currently requires the low-amplitude bin to satisfy:

> simultaneous 95% **lower** confidence bound \(\le 0.20\).

This does not establish a low success probability. A wide interval around a 50% or even higher observed success rate can still have a lower bound below 0.20.

The necessary condition is:

\[
\boxed{\text{low-bin simultaneous 95% UCB}\le 0.20.}
\]

The high-amplitude condition \(\mathrm{LCB}\ge0.80\) is directionally correct.

**Required action:** issue an outcome-blind protocol corrigendum before any real Boundary outcome is opened, and add a negative unit test in which a 50% low-bin success rate cannot pass.

### 6.2 The current Boundary Transition is close to definitional

The current predictor is amplitude \(j/2\); the outcome is whether the same certificate becomes resolved and passes P13. Amplitude directly changes the endpoint signal, finite-difference error, curvature remainder, interval width, and set-SNR. A monotone transition can therefore arise in a one-dimensional affine sensor, random network, or semantically irrelevant path.

As written, the result may establish only:

> an algorithmic resolution or power curve for the certificate.

It does not by itself establish a Transformer mechanism transition.

**Required action:** make `certificate_resolved` an intermediate/secondary variable. The primary endpoint must be an independent held-out transport, behavior, or circuit consequence that was not used to construct the certificate.

### 6.3 Numerical truth is not semantic mechanism truth

The interval theorem can prove that a declared exact-real intervention graph encloses \(\Psi,\Psi',\Psi''\). It cannot prove that:

- the intervention direction lies on the natural activation support or tangent space;
- the declared branch decomposition is the unique correct high-level mechanism;
- the gate is necessary or sufficient in natural computation;
- the complete reasoning chain has been restored;
- an equivalent parameterization gives the same semantic story.

A minimal counterexample is \(y=a+b\) with natural data restricted to \(a=b=x\). Exact certificates of \(\partial y/\partial a=1\) and \(\partial y/\partial b=1\) are both numerically correct, but neither proves that the unique true mechanism is branch \(a\) or branch \(b\).

**Required language:** “certified local path functional,” not “certified true mechanism” or “complete mechanism recovery.”

### 6.4 Cross-version garden of forking paths

The project moved from IVS to IRS, then matched-bypass operator recovery, v2 `STOP_ORAL`, v3 `POSTER_ONLY`, and now v4 Joint Witness/Boundary Transition. Each move has a scientific rationale, and negative results were preserved, which is good. Nevertheless, protocol-level outcome blindness does not remove project-level hypothesis search.

**Required action:** freeze a terminal thesis matrix now. Specify which v4 outcomes end the Oral line rather than triggering an unlimited v5 rescue, and disclose the negative lineage in the final paper.

## 7. Other high-priority scientific risks

1. **The Joint Witness is numerically independent, not scientifically independent.** It shares the row, model, hooks, directions, contrast, and estimand with the component estimator. A separate held-out consequence is still needed.
2. **Access-regime ambiguity.** The central estimator is response-only, but the complete method requires weights, graph extraction, primitive semantics, MPFR, and AD adjudication. The paper must not market the whole system as black-box or response-only.
3. **Strong baselines are not yet binding.** Exact activation/path patching, AtP*, Integrated Gradients, HVP/MS-HVP, raw SNR, and at least one generic verifier must be frozen before real outcomes.
4. **Single model/task.** GPT-2 Small Greater-Than alone cannot support broad claims about Transformer mechanisms. One different mechanism/task and one different model or architecture should be mandatory.
5. **Exact-real versus runtime semantics.** The certificate covers the extracted exact-real graph. It is not automatically a bit-level proof of the PyTorch/TransformerLens GPU program. Independent graph extraction or stronger trace equivalence is needed.
6. **Resource calibration is not scientific evidence.** B4/B8/B16/B32 can establish runtime feasibility and audit integrity only. It says nothing about real certificate width, P13, or mechanism truth.

## 8. Binding experiments needed for an Oral case

Before real outcomes, freeze the following:

1. **Independent primary endpoint:** held-out path transport or behavioral consequence, never consumed by certificate construction.
2. **Prospective prediction:** GREEN labels rows/conditions as reliable, unreliable, or unresolved before the independent endpoint is measured.
3. **Strong baseline gate:** exact patching, AtP*, IG, HVP/MS-HVP, raw SNR/analytic power, and a generic verifier.
4. **Difficulty-matched nulls:** random gate/direction, branch permutation, bypass-only, random-weight model, affine scalar sensor, and a known dormant/decoy path; match signal amplitude, curvature, cone depth, and compute difficulty where possible.
5. **Off-manifold check:** natural-support or tangent-admissibility diagnostic for intervention directions.
6. **External replication:** a different task/mechanism and a different model/architecture, with no threshold tuning.
7. **Project-level kill rule:** no further main-claim replacement after v4 outcomes. A failed result remains a publishable negative result or ends the line.

The decisive novelty test is:

> On the same preregistered instances, ordinary patching/AtP*/IG/HVP/raw SNR or a generic verifier makes a confident mechanistic prediction; GREEN prospectively warns that the prediction is invalid or unresolved; an independent held-out causal consequence then shows that GREEN was right and the simpler method was wrong.

If this pattern replicates across a second task/model, the project has an Oral-level scientific story. If the only result is that more amplitude or more budget makes the interval exclude zero, it is a rigorous calibration/verification paper, not an Oral-level mechanism discovery.

## 9. Reviewer-style level assessment

### Current state

| Dimension | Assessment |
|---|---:|
| Engineering and audit discipline | 8/10 |
| Theorem implementation | 7/10 |
| Real v4 scientific evidence | 2/10 |
| Manuscript alignment with v4 | 2/10 |
| Current submission competitiveness | 3–4/10 |

Current AC-style recommendation: **Reject / Weak Reject**. This reflects absent v4 evidence and a superseded manuscript, not a conclusion that the remaining idea is worthless.

### Conditional ceiling

| Final evidence | Likely level |
|---|---|
| Formal theorem, tests, and a small number of valid certificates | Borderline/Poster |
| Many nontrivial certificates, but Boundary is only self-resolution | Poster |
| Prospective failure detection validated by an independent endpoint and strong controls | Strong Accept territory |
| Same result replicated without retuning across model/task, with a surprising mechanism conclusion | Oral discussion territory |

## 10. Immediate execution decision

1. **Continue** the already-running outcome-blind synthetic resource calibration. It is not contaminated by this scientific audit.
2. **Do not open** real v4 Boundary outcomes under the current §13.5 protocol.
3. **Do not push** this audit or other changes to GitHub unless explicitly requested.
4. **Next local task:** write an outcome-blind v4 scientific corrigendum that:
   - fixes low-bin UCB direction;
   - makes the independent held-out consequence primary;
   - freezes strong baselines and matched falsifiers;
   - maps GREEN algebraically to the 2026 mediator-interaction literature;
   - freezes project-level termination criteria.
5. Have the corrigendum independently red-teamed before any scientific opening. GPTPro is not yet required to discover the issue; it may be used later for final theory/protocol adjudication if the corrected estimand or causal mapping remains unresolved.

## 11. Final verdict

- **Was the entire v4 idea already done?** No.
- **Were major primitives and the old headline already done?** Yes.
- **Is the project merely self-deception?** No; the mathematics, negative-result preservation, and numerical engineering are substantive.
- **Is the current Oral thesis established?** No.
- **Can Oral ambition remain?** Yes, but only through a narrow and difficult route: deterministic finite-radius relational certification must prospectively change and correctly predict an independent mechanistic conclusion that simpler methods miss, and that result must replicate.

