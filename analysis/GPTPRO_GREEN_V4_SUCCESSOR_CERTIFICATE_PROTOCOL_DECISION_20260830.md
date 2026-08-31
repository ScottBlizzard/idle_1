# GPTPRO GREEN v4 Successor Certificate Protocol Decision

**Proposed repository path:** `analysis/GPTPRO_GREEN_V4_SUCCESSOR_CERTIFICATE_PROTOCOL_DECISION_20260830.md`  
**Decision date:** 2026-08-30  
**Repository:** `ScottBlizzard/idle_1`  
**Audited working commit:** `91741e32bfaad05667fb2d4c8d1de26373e90a15`  
**Outcome-blind reconstruction cutoff:** `6aab5837bf3950b86b166ef938097e9739810d9b` and earlier  
**Parent protocols:** `GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1` and `GREEN_V400_SILENT_FAILURE_GT_REPLICATION_PREPARE_V1`  
**Successor protocol:** `GREEN-V410-SFC-JWTEC-20260830`  
**Decision-rule ID:** `GREEN_V410_SFC_JWTEC_DECISION_V1`  
**Attempt:** `A1`, one scientific attempt, no outcome-dependent retry

---

## 1. Executive verdict

### Binding verdict: `CONDITIONAL_GO`

The already-opened GREEN v4.0.0 development execution cannot be completed retroactively as a valid primary GREEN analysis. Its prediction and endpoint data are protocol-diagnostic evidence only. The frozen v4.0.0 contract lacks a mathematically defined mapping from the Joint Witness construction to one status for each present IOI or Greater-Than SFC site. That omission changes coverage, denominators, acceptance, and success, and is therefore scientific rather than merely clerical.

A new successor route is nevertheless scientifically defensible. This decision freezes the missing estimand, direction aggregation, radius schedule, P13 role, row classifier, causal graph, identities, resource policy, queue order, and endpoint firewall before any successor certificate row, reserve qualification result, or confirmation outcome is opened.

The successor may proceed only if all of the following pre-execution gates pass:

1. the confirmation seal audit proves that no confirmation prediction, certificate, replay, clean-validity, endpoint, or analysis outcome has been produced or accessed;
2. every schema, source file, graph constructor, queue, decision rule, and resource manifest in this document is frozen by canonical SHA-256;
3. all theorem, graph, identity, classifier, no-clobber, firewall, and fault-injection tests pass;
4. a deterministic synthetic-only resource calibration selects a feasible fixed leaf budget;
5. the untouched reserve P13 qualification passes separately for IOI and Greater-Than;
6. no forbidden development-derived tuning has occurred.

Failure of any gate is terminal for `GREEN-V410-SFC-JWTEC-20260830/A1`. Codex must emit the specified STOP receipt and must not search for a friendlier rule, threshold, aggregation, radius, subset, or compute budget.

### Explicit classification

`SCIENTIFIC_AMBIGUITY` **and** `PURE_ENGINEERING`.

- `SCIENTIFIC_AMBIGUITY`: v4.0.0 did not define the site estimand, eight-direction aggregation, final status semantics, P13-to-site mapping, radius binding, or SFC graph identity.
- `PURE_ENGINEERING`: after this document is frozen, implementation of the specified graph, queues, serializers, workers, receipts, and tests contains no remaining scientific choice.

### Terminal ruling on v4.0.0

`STOP_PRIMARY_ANALYSIS_MISSING_PRESPECIFIED_GREEN_INPUTS` remains binding for v4.0.0 development. No file in the v4.0.0 execution lineage may be edited to make that run appear complete.

---

## 2. Audit boundary and provenance

### 2.1 Phase I — outcome-blind scientific reconstruction

The Phase I rule below was reconstructed from commit `6aab5837...` and earlier materials, including the v1.36, v2.0.0, v3.0.0, and v4.0.0 decisions and corrigenda; the shared SFC decision specification; the IOI and Greater-Than task specifications; the current SFC response adapters; the matched-bypass branch implementation; the direction binding code; and the pre-outcome universe and queue manifests.

No later development endpoint value, success rate, AUROC, failure fraction, certificate width, or prospective confirmation content was used to choose:

- the site quantity;
- the sign orientation;
- the eight-direction aggregation;
- the radius panel;
- the threshold;
- the status mapping;
- P13's role;
- the primary denominator;
- confirmation eligibility.

The current outcome-blind audit was used only to enumerate missing definitions.

### 2.2 Phase II — eligibility and implementation audit

Only after the Phase I rule was fixed was the post-cutoff state inspected. That state establishes that:

- development prediction and endpoint fleets ran;
- the shared analyzer did not run because GREEN statuses and Greater-Than clean-validity records were absent;
- no certificate queue was in the activated plans;
- the older certificate implementation was prepare/synthetic-only and used incompatible row and graph identities;
- confirmation remained locked according to the repository's execution records.

These facts determine versioning and artifact eligibility. They do not alter the Phase I estimand.

### 2.3 Decision-provenance labels

Every scientific rule in this document is one of:

- `INHERITED`: uniquely fixed by pre-outcome materials;
- `RECONSTRUCTED`: uniquely implied by several pre-outcome components that were not previously assembled in one executable rule;
- `SUCCESSOR_DECISION`: newly fixed here because pre-outcome materials admitted multiple scientifically consequential choices.

No `SUCCESSOR_DECISION` may be changed after this document's canonical hash is committed.

### 2.4 Evidence ledger

The binding evidence set includes, at minimum, the following repository objects at the stated cutoffs. Later files are used only in Phase II.

| Role | Repository object | Phase |
|---|---|---|
| Current ambiguity audit | `analysis/CODEX_GREEN_V400_OUTCOME_BLIND_CERTIFICATE_PROTOCOL_AUDIT_20260830.md` | locator only in Phase I; eligibility in Phase II |
| v3 terminal theorem/protocol | `analysis/GPTPRO_GREEN_V300_TERMINAL_SCIENTIFIC_DECISION_20260826.md` | Phase I |
| v4 theorem and implementation corrigendum | `analysis/GPTPRO_GREEN_V400_BINDING_CORRIGENDUM_20260826.md` | Phase I |
| Execution consistency lock | `analysis/CODEX_GREEN_V400_EXECUTION_CONSISTENCY_LOCK_20260826.md` | Phase I |
| Anytime resource theory | `analysis/GREEN_V400_ANYTIME_CERTIFICATE_RESOURCE_POLICY_V1_20260827.md` | Phase I |
| Mission and novelty audits | `analysis/CODEX_GREEN_V400_ORIGINAL_MISSION_ALIGNMENT_CORRIGENDUM_20260828.md`; `analysis/CODEX_GREEN_V400_NOVELTY_COLLISION_AND_SELF_DECEPTION_AUDIT_20260828.md` | Phase I |
| Shared population and analyzer rule | `configs/green_v400_shared_decision_spec.json`; `analysis/green_v400_shared_decision_analyzer.py` | Phase I |
| IOI/GT task protocols | `configs/green_v400_silent_failure_challenge_prepare.json`; `configs/green_v400_greater_than_silent_failure_prepare.json` | Phase I |
| Untouched universes | `configs/green_v400_ioi_untouched_universe.json`; `configs/green_v400_greater_than_untouched_universe.json` | Phase I |
| Task response maps | `src/green_v400_ioi_response_adapter.py`; `src/green_v400_greater_than_response_adapter.py` | Phase I |
| Matched-bypass semantics | `src/green_v400_matched_bypass_adapter.py`; `src/green_v400_four_branch_baseline.py`; `src/green_bridge_v400_branch_semantics.py` | Phase I |
| Gate and model constants | `src/green_bridge_spec.py` and frozen model/tokenizer manifests | Phase I |
| Direction binding | `src/green_v400_direction_binding.py` and frozen direction registries | Phase I |
| Formal graph/certificate implementation | `src/green_bridge_v400_relational_graph.py`; `src/green_bridge_v400_certificate.py`; `src/green_bridge_v400_gpt2_program.py`; schema and tests | Phase I for available primitives; Phase II for reuse ruling |
| Activated development state | `analysis/GREEN_V400_DEVELOPMENT_EXECUTION_STATUS_20260829.md`; `analysis/CODEX_GREEN_V400_DEVELOPMENT_COMPLETENESS_AND_CONFIRMATION_AUTHORIZATION_20260829.md` | Phase II only |
| Development authorization | `configs/green_v400_development_authorization_20260829.json` and activated plans/receipts | Phase II only |

The history is interpreted as evidence, not as executable instructions that supersede this decision.

---

## 3. Phase I outcome-blind reconstruction

### 3.1 Inherited objects

The following are inherited without alteration:

1. **Model.** GPT-2 Small, frozen checkpoint revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, with the existing model-manifest tensor hashes and TransformerLens source binding.
2. **Candidate sites.** `blocks.{layer}.hook_resid_post`, layers `0,1,...,8`, at the task-specific earlier token position stored in the immutable site manifest.
3. **Direction payload law.** Eight precommitted float32 GREEN directions per site, width 768, nominal norm `0.001`, generated and verified under the existing rowwise PCG64DXSM law. The exact little-endian float32 bytes and tensor SHA-256, not an idealized real vector, define each direction.
4. **Structural readout.** Selected layer-10 MLP post-activation coordinates
   
   ```text
   (2326, 1138, 2287, 606, 2848, 2305, 46, 2659, 946, 1616)
   ```
   
   at `blocks.10.mlp.hook_post`, final sequence position.
5. **Matched bypass.** The bypass freezes only those selected gate-post coordinates to their branch-specific `t=0` values. Nonselected MLP coordinates, attention, residual bypasses, and all other downstream computation remain live.
6. **Task orientation.** A larger scalar response means more clean-correct behavior.
7. **Radius family.** The inherited local witness radii are `1`, `1/2`, and `1/4` in the dimensionless amplitude multiplying the frozen direction vector.
8. **Official precision.** 384-bit outward MPFR is official; 512-bit execution is an audit on the exact frozen official partition and must nest inside the 384-bit enclosure.
9. **P13 numerical thresholds.** Median contraction ratio at most `0.20` and 90th percentile at most `0.50`.
10. **Shared analysis.** High restoration means ordinary restoration at least `0.80`; only `CERTIFIED_POSITIVE` accepts; minimum GREEN coverage is `0.50`; method-invalid rate must not exceed `0.05`; warning and resource rows remain in the intent-to-analyze denominator.

### 3.2 Scientific ambiguity that cannot be inherited

The pre-outcome record does not fix any of the following:

- whether raw Joint Witness sign, absolute magnitude, equivalence, or a vote is the site claim;
- how eight directions form one site result;
- how the old donor-level P13 contraction gate becomes a site status;
- whether the old 17-radius boundary-transition plan applies to the SFC norm-`0.001` directions;
- how the old fixed block-8/final-token TensorProgram maps to layers 0–8 and earlier token positions;
- the SFC certificate-row identity and graph crosswalk;
- the handling of malformed, mixed, missing, duplicated, or resource-limited direction records.

Those choices can change acceptance count, coverage, denominator composition, selective risk, and the paper verdict. They are scientific ambiguity.

### 3.3 Binding successor choices

This document makes the following result-blind successor decisions:

- `S1`: certify a **normalized matched-bypass local transport-equivalence margin**, not the raw sign of one derivative;
- `S2`: use all eight directions jointly through an RMS panel statistic; no voting, union, intersection of signs, or best-direction selection;
- `S3`: use exactly the three inherited radii `{1,1/2,1/4}`; the old 17-radius plan is inapplicable;
- `S4`: retain P13 only as a separate cohort-level proof-contraction qualification gate;
- `S5`: use a new SFC causal-cone graph and identity system;
- `S6`: make sound finite-budget intervals reportable without a separate width-tolerance pass; interval width affects `UNRESOLVED`, not validity;
- `S7`: use an untouched reserve qualification cohort before opening confirmation;
- `S8`: version the route as v4.1.0 rather than editing v4.0.0 in place.

---

## 4. Scientific estimand

### 4.1 Site, center, and direction

For one immutable site row `s`, let:

- `task(s) ∈ {ioi, greater_than}`;
- `l_s ∈ {0,...,8}` be the layer;
- `p_s` be the immutable token position;
- `x_s ∈ R^768` be the clean center activation captured at `blocks.{l_s}.hook_resid_post[p_s]` under the frozen float64 response-evaluation model;
- `d_{s,k} ∈ R^768`, `k=0,...,7`, be the exact real interpretation of the frozen float32 direction payload;
- `t` be a dimensionless scalar.

The intervention value is

\[
    z_{s,k}(t)=x_s+t d_{s,k}.
\]

The nominal displacement norms at the three witness radii are `0.001`, `0.0005`, and `0.00025`; the exact claim is defined by the hashed payload bytes.

### 4.2 Task response maps

#### IOI

Let `i_s` be the single-token indirect-object name ID and `r_s` the single-token subject name ID. At the final prompt position,

\[
 C_{\mathrm{IOI}}(L)=L[-1,i_s]-L[-1,r_s].
\]

The token IDs must be distinct, in vocabulary, and each name must tokenize to exactly one token under the frozen tokenizer. A violation is `INVALID`.

#### Greater-Than

Let `y_s ∈ {0,...,98}` be the clean start-year suffix and let `u_0,...,u_{99}` be the frozen suffix token IDs. Define

\[
 a_j(y_s)=
 \begin{cases}
   -1/(y_s+1), & j\le y_s,\\
   +1/(99-y_s), & j>y_s.
 \end{cases}
\]

At the final prompt position,

\[
 C_{\mathrm{GT}}(L)=\sum_{j=0}^{99} a_j(y_s)L[-1,u_j].
\]

This is the exact frozen balanced mean-logit contrast. The 100 IDs must be unique and valid.

### 4.3 Pattern, target, joint, control, baseline, and contrast branches

For each site and direction, define four scalar response functions.

- `PAT_J`: corrupt-token context; overwrite the site with `z_{s,k}(t)`; selected layer-10 gates are live.
- `PAT_B`: same corrupt-token context and site overwrite; selected layer-10 gate posts at the final position are frozen to the values captured in that context at `t=0`.
- `TAR_J`: clean-token context; overwrite the same site with `z_{s,k}(t)`; selected gates are live.
- `TAR_B`: same clean-token context and site overwrite; selected gate posts are frozen to their clean-context `t=0` values.

`PAT` is the behaviorally restored pattern/recipient context. `TAR` is the natural clean target context. `J` is the live joint branch. `B` is the matched-bypass control branch. The **baseline** of every branch is its value at `t=0`.

Let the task-oriented scalar outputs be

\[
 C^{PJ}_{s,k}(t),\quad C^{PB}_{s,k}(t),\quad
 C^{TJ}_{s,k}(t),\quad C^{TB}_{s,k}(t).
\]

The branch-specific gate-mediated local effects are

\[
 G^P_{s,k}(t)=C^{PJ}_{s,k}(t)-C^{PB}_{s,k}(t),
\]

\[
 G^T_{s,k}(t)=C^{TJ}_{s,k}(t)-C^{TB}_{s,k}(t).
\]

The Joint Witness relational contrast is

\[
 \Psi_{s,k}(t)=G^P_{s,k}(t)-G^T_{s,k}(t)
 =C^{PJ}_{s,k}(t)-C^{PB}_{s,k}(t)-C^{TJ}_{s,k}(t)+C^{TB}_{s,k}(t).
\]

The branch weights and order are therefore exactly

```text
PAT_J:+1, PAT_B:-1, TAR_J:-1, TAR_B:+1
```

The matched anchors imply, in exact graph semantics,

\[
 C^{PJ}_{s,k}(0)=C^{PB}_{s,k}(0),\qquad
 C^{TJ}_{s,k}(0)=C^{TB}_{s,k}(0),\qquad
 \Psi_{s,k}(0)=0.
\]

Failure of either anchor identity is an implementation invalidity, never a negative scientific result.

### 4.4 Direction-level local quantities

Define

\[
 j_{s,k}=\Psi'_{s,k}(0),\qquad
 p_{s,k}=\frac{d}{dt}C^{PJ}_{s,k}(0),\qquad
 q_{s,k}=\frac{d}{dt}C^{TJ}_{s,k}(0).
\]

Raw sign is oriented as follows:

- `j_{s,k}>0`: selected-gate mediation is locally stronger in PAT than TAR in that oriented direction;
- `j_{s,k}<0`: it is locally weaker;
- `j_{s,k}=0`: exact local equality for that scalar direction.

Neither raw sign is intrinsically good or bad, because replacing `d_{s,k}` by `-d_{s,k}` reverses every derivative. Raw sign therefore cannot determine site acceptance.

### 4.5 Eight-direction site estimand

With `K=8`, define

\[
 J_s=\sqrt{\frac1K\sum_{k=0}^{K-1}j_{s,k}^2},
\]

\[
 P_s=\sqrt{\frac1K\sum_{k=0}^{K-1}p_{s,k}^2},\qquad
 Q_s=\sqrt{\frac1K\sum_{k=0}^{K-1}q_{s,k}^2}.
\]

Let

\[
 \varepsilon=10^{-12},\qquad
 D_s=\max(P_s,Q_s,\varepsilon),
\]

and define the normalized matched-bypass transport residual

\[
 R_s=\frac{J_s}{D_s}.
\]

Equivalently, define the signed transport-equivalence margin

\[
 M_s=\frac15D_s-J_s.
\]

The scientific assertion is:

- **positive certificate:** `M_s >= 0`, equivalently `R_s <= 0.20`;
- **negative certificate:** `M_s < 0`, equivalently `R_s > 0.20`.

This is invariant to independently reversing any of the eight direction signs. Mixed direction signs are retained through the squared RMS and are never voted away.

### 4.6 What a positive certificate asserts

For one site, `CERTIFIED_POSITIVE` asserts that, on the exact eight-direction GREEN panel and at the derivative point `t=0`, the RMS discrepancy between PAT and TAR selected-gate-mediated sensitivities is no more than 20% of the larger RMS full local response sensitivity in PAT or TAR.

It does **not** assert:

- exact equality in all directions;
- equivalence for arbitrary directions in the 768-dimensional space;
- equality of internal representations;
- global behavior under large perturbations;
- that the selected gates are the only causal route;
- guaranteed success on the sealed endpoint panel.

### 4.7 What a negative certificate asserts

`CERTIFIED_NEGATIVE` asserts that the exact normalized panel residual is strictly greater than 20%. It is a rigorous transport warning. It is not the assertion that every raw `j_{s,k}` is negative.

### 4.8 Relationship to restoration and the endpoint

Ordinary restoration is inherited from the parent prediction packet. It tests whether inserting the clean center into the corrupt context restores the clean-correct task score. It is not used to construct `R_s`; it only defines the prespecified high-restoration analysis stratum.

For a sealed endpoint direction `e_{s,m}`, let

\[
 T_{s,m}=C^{TJ}_{s,m}(1)-C^{TJ}_{s,m}(0),
\]

\[
 P^{\mathrm{end}}_{s,m}=C^{PJ}_{s,m}(1)-C^{PJ}_{s,m}(0).
\]

The primary endpoint is the inherited symmetric normalized full-response mismatch

\[
 E_s=
 \frac{\operatorname{RMS}_m(P^{\mathrm{end}}_{s,m}-T_{s,m})}
 {\max(\operatorname{RMS}_m(P^{\mathrm{end}}_{s,m}),
       \operatorname{RMS}_m(T_{s,m}),10^{-12})}.
\]

The certificate and endpoint are deliberately different:

- the certificate uses public GREEN directions, a local derivative, and the matched-bypass relational mechanism;
- the endpoint uses hidden directions, finite displacement at `t=1`, and the full response field.

The empirical claim under test is that rigorous local matched-bypass transport equivalence predicts lower held-out full transport error at matched coverage. No theorem in this protocol identifies `R_s` with `E_s`.

### 4.9 Required validity assumptions

A site certificate is scientifically valid only if all of the following hold:

1. model, checkpoint, tokenizer, task metric, token IDs, and selected-gate hashes match the frozen manifests;
2. clean and corrupt token tensors are immutable and equal in shape;
3. the clean center, full controlled-hook tensors, and directions match their exact payload hashes;
4. the site layer is below gate layer 10;
5. all stochastic behavior is disabled;
6. the four branches differ only by the declared context and gate-freeze intervention;
7. the downstream graph is complete and algebraically equivalent to the frozen tail computation;
8. interval primitives outwardly enclose real values and first two derivatives;
9. every dyadic partition is complete and split exactly at zero;
10. 512-bit audit intervals nest inside 384-bit official intervals;
11. no endpoint payload, endpoint direction, or endpoint outcome enters graph construction, certification, classification, or P13 qualification.

---

## 5. Direction and radius semantics

### 5.1 Direction treatment

Certification is direction-wise at the one-dimensional graph level and jointly aggregated at the site level.

- Each direction receives its own exact scalar control `t` and its own certificate record.
- Exactly eight ordinals `0,...,7` are required.
- No direction may be dropped because it is difficult, sign-inconsistent, low-signal, wide, or resource-intensive.
- The site statistic is the deterministic RMS formula in Section 4.5.
- There is no majority vote, all-positive intersection, any-positive union, median sign, maximum-risk substitution, or direction counting.
- The result is not a multivariate interval over the span or convex hull of the directions.

### 5.2 Radius panel

The only official radius panel is

\[
 H=\left\{1,\frac12,\frac14\right\}.
\]

Its ID is

```text
GREEN_V410_RADIUS_PANEL_T1_T1_2_T1_4_V1
```

The older 17-radius `CertificatePlan` is not mapped to SFC and is forbidden for this successor.

### 5.3 Signed-secant witness interval

For each scalar function

```text
f ∈ {Psi, PAT_J, TAR_J, PAT_B, TAR_B}
```

and each `h ∈ H`, let `E_f(-h)`, `E_f(0)`, and `E_f(+h)` be certified value intervals. Define the outward central secant interval

\[
 W_f(h)=\frac{E_f(+h)-E_f(-h)}{2h}.
\]

Let `K_{f,+}(h)` enclose

\[
 f(h)-f(0)-h f'(0),
\]

and let `K_{f,-}(h)` enclose

\[
 f(-h)-f(0)+h f'(0).
\]

Then

\[
 K_{f,\mathrm{sec}}(h)=K_{f,+}(h)-K_{f,-}(h)
\]

and the official radius-specific derivative interval is

\[
 I_f(h)=W_f(h)-\frac{K_{f,\mathrm{sec}}(h)}{2h}.
\]

The remainders use the budget-monotone signed-curvature recurrence. Every refinement intersects the new sound enclosure with the preceding one. Empty intersection is `INVALID_EMPTY_SOUND_INTERSECTION`.

The official cross-radius interval is

\[
 I_f=I_f(1)\cap I_f(1/2)\cap I_f(1/4).
\]

All three radii are mandatory. No radius is selected after observing width or sign.

### 5.4 Precision policy

- 384-bit intervals are the only official intervals used by the classifier.
- The exact final 384-bit partitions are replayed at 512 bits.
- Every 512-bit node, endpoint, remainder, derivative, and final interval must be a subset of its 384-bit counterpart, allowing only the declared outward-rounding comparison rule.
- 512-bit results may not replace or tighten the official 384-bit result.
- Independent float64 AD/JVP and finite-difference routes are validation routes only. They must overlap the official interval where specified; they are never intersected into the official certificate.

### 5.5 Interval aggregation for the site ratio

For any closed interval `I=[a,b]`, define

\[
 \operatorname{sq}(I)=
 \begin{cases}
 [0,\max(a^2,b^2)],&a\le0\le b,\\
 [\min(a^2,b^2),\max(a^2,b^2)],&\text{otherwise}.
 \end{cases}
\]

For direction intervals `I^J_{s,k}`, `I^P_{s,k}`, `I^Q_{s,k}`, compute with outward arithmetic

\[
 \mathcal J_s=\sqrt{\frac18\sum_k\operatorname{sq}(I^J_{s,k})}=[J_s^-,J_s^+],
\]

and analogously `mathcal P_s=[P_s^-,P_s^+]` and `mathcal Q_s=[Q_s^-,Q_s^+]`.

Then

\[
 \mathcal D_s=
 [\max(P_s^-,Q_s^-,10^{-12}),
  \max(P_s^+,Q_s^+,10^{-12})]
 =[D_s^-,D_s^+].
\]

Because `D_s^- > 0`,

\[
 \mathcal R_s=
 \left[\frac{J_s^-}{D_s^+},\frac{J_s^+}{D_s^-}\right]
 =[R_s^-,R_s^+].
\]

The official status boundaries are:

- `R_s^+ <= 0.20` → `CERTIFIED_POSITIVE`;
- `R_s^- > 0.20` → `CERTIFIED_NEGATIVE`;
- otherwise → `UNRESOLVED`.

An exact interval `[0.20,0.20]` is positive. An interval with lower endpoint `0.20` and upper endpoint greater than `0.20` is unresolved. No midpoint rule is permitted.

---

## 6. Executable P13 specification

### 6.1 Binding role

P13 remains a **cohort-level proof-contraction qualification gate only**. It is not a per-site transport rule and may not be broadcast into `green_status`.

The site transport decision is Section 5.5. P13 answers a different question: whether relational interval evaluation yields the preregistered contraction over an independent qualification cohort.

### 6.2 Qualification cohort

Use only the parent `unused_reserve` universes. No development or confirmation row is eligible.

#### IOI

From the 64 prompt rows under template `reserve_0`, compute

```text
SHA256("GREEN-V410-P13-QUALIFICATION\0ioi\0" || prompt_row_id)
```

and select the 12 smallest `(hash, prompt_row_id)` pairs. Include all nine layers and all eight GREEN direction ordinals. Expected size: 12 prompt clusters, 108 sites, 864 direction records.

#### Greater-Than

Create the 12 strata

```text
century ∈ {10,18,20}
× distance_bin ∈ {near,far}
× orientation ∈ {up,down}
```

Within each stratum, among all prompt rows from reserve nouns `ceremony`, `committee`, `federation`, and `workshop`, select the smallest

```text
SHA256("GREEN-V410-P13-QUALIFICATION\0greater_than\0" || prompt_row_id)
```

with `prompt_row_id` as tie break. Include all nine layers and all eight directions. Expected size: 12 prompt clusters, 108 sites, 864 direction records.

The qualification direction seed domains are newly frozen as

```text
GREEN_V410_P13_IOI_GREEN_PANEL
GREEN_V410_P13_GT_GREEN_PANEL
```

with the inherited float32 rowwise PCG64DXSM generator, width 768, count 8, and nominal norm `0.001`. Their exact payloads and bindings must be committed before any qualification model execution.

### 6.3 Direction-level contraction quantity

For each qualification site-direction, build:

1. one shared relational graph for `Psi`;
2. four branch-isolated graphs with no cross-branch common-subexpression sharing.

For each branch `b`, compute an independent float64 response-only center using the fixed central secant at `h_c=1/4`:

\[
 \widehat c_b=\frac{C_b(1/4)-C_b(-1/4)}{1/2}.
\]

Let `I_b=[l_b,u_b]` be the official branch-isolated derivative interval. Define

\[
 b_b=\max(|\widehat c_b-l_b|,|\widehat c_b-u_b|).
\]

With signs `(+1,-1,-1,+1)`, define

\[
 \widehat\theta=\widehat c_{PJ}-\widehat c_{PB}-\widehat c_{TJ}+\widehat c_{TB},
\]

\[
 B_{\mathrm{box}}=b_{PJ}+b_{PB}+b_{TJ}+b_{TB}.
\]

Let `I_W=[l_W,u_W]` be the shared relational interval for `Psi'(0)`. Define

\[
 B_W=\max(|\widehat\theta-l_W|,|\widehat\theta-u_W|),
\]

\[
 B_{\mathrm{JW}}=\min(B_{\mathrm{box}},B_W),
\]

\[
 \rho_{s,k}=B_{\mathrm{JW}}/B_{\mathrm{box}}.
\]

Boundary rules:

- if `B_box > 0`, evaluate the ratio normally;
- if `B_box = 0` and `B_W = 0`, set `rho=0`;
- if `B_box = 0` and `B_W > 0`, emit `INVALID_P13_ZERO_BOX_INCONSISTENT` and fail the task qualification.

The response-only centers must be finite. Each independent AD derivative must overlap its corresponding official interval. Failure is invalidity, not a poor contraction result.

### 6.4 Direction-to-site and site-to-task aggregation

For each qualification site,

\[
 \rho_s=\max_{k=0,...,7}\rho_{s,k}.
\]

This worst-direction aggregation is fixed to prevent eight correlated directions from being treated as independent donor rows.

For each task, sort the 108 site values

\[
 \rho_{(1)}\le\cdots\le\rho_{(108)}.
\]

Define the exact sample median

\[
 \operatorname{median}=\frac{\rho_{(54)}+\rho_{(55)}}2
\]

and the nearest-rank 90th percentile

\[
 p90=\rho_{(\lceil0.90\cdot108\rceil)}=\rho_{(98)}.
\]

The task passes P13 iff

\[
 \operatorname{median}\le0.20
 \quad\text{and}\quad
 p90\le0.50.
\]

Equality passes. IOI and Greater-Than must pass separately. A pooled value is diagnostic only.

### 6.5 P13 prerequisites and failure handling

P13 is `PASS` only if all 108 sites and all 864 direction records for the task are present, unique, valid, fully computed at all radii and precisions, and graph/hash checks pass. Any `RESOURCE_INCONCLUSIVE`, `INVALID`, missing record, duplicate, conflict, or malformed interval makes that task `P13_FAIL`.

A failed P13 task receipt locks all confirmation queues for both tasks. No site classifier may be invoked without a valid two-task P13 PASS receipt.

P13 values and qualification site statuses are never input to threshold selection, direction selection, graph selection, confirmation sampling, or resource-budget changes.

---

## 7. Total status classifier

### 7.1 Status domain

The only final site statuses are:

```text
CERTIFIED_POSITIVE
CERTIFIED_NEGATIVE
UNRESOLVED
INVALID
RESOURCE_INCONCLUSIVE
```

`CERTIFIED_NULL` does not exist and is forbidden in schemas and analyzers.

Every status packet must also contain a `reason_code` and `invalid_scope`:

```text
invalid_scope ∈ {NONE, TASK_PRECONDITION, METHOD, RUN_CONTRACT}
```

### 7.2 Precedence

Precedence is strict:

```text
RUN_CONTRACT INVALID
> TASK_PRECONDITION INVALID
> METHOD INVALID
> RESOURCE_INCONCLUSIVE
> mathematical positive/negative/unresolved classification
```

An invalid condition can never be downgraded to resource or unresolved.

### 7.3 Deterministic pseudocode

```python
def classify_site(
    site_identity,
    direction_certificate_records,
    p13_specification,
    validity_flags,
    clean_task_validity,
    resource_flags,
):
    # 1. Global/run-contract checks.
    if not schema_and_protocol_ids_exact(site_identity):
        return INVALID("RUN_IDENTITY_MISMATCH", scope="RUN_CONTRACT")
    if not p13_specification.is_exact_two_task_pass_receipt:
        return INVALID("P13_GLOBAL_NOT_PASSED", scope="RUN_CONTRACT")
    if not validity_flags.frozen_source_schema_graph_hashes_match:
        return INVALID("FROZEN_HASH_MISMATCH", scope="RUN_CONTRACT")
    if validity_flags.phase_manifest_conflict:
        return INVALID("PHASE_MANIFEST_CONFLICT", scope="RUN_CONTRACT")

    # 2. Task precondition.
    if site_identity.task == "greater_than":
        if clean_task_validity is MISSING_OR_MALFORMED:
            return INVALID("GT_CLEAN_VALIDITY_MISSING", scope="RUN_CONTRACT")
        if clean_task_validity is False:
            return INVALID("GT_CLEAN_TASK_INVALID", scope="TASK_PRECONDITION")
        if clean_task_validity is not True:
            return INVALID("GT_CLEAN_VALIDITY_BAD_DOMAIN", scope="RUN_CONTRACT")
    elif site_identity.task == "ioi":
        if clean_task_validity not in (None, "NOT_APPLICABLE"):
            return INVALID("IOI_CLEAN_VALIDITY_MUST_BE_NA", scope="RUN_CONTRACT")
    else:
        return INVALID("UNKNOWN_TASK", scope="RUN_CONTRACT")

    # 3. Expected record set and hard validity.
    if not isinstance(direction_certificate_records, list):
        return INVALID("DIRECTION_RECORD_CONTAINER_MALFORMED", scope="METHOD")
    if records_have_duplicate_ids_or_conflicting_payloads(direction_certificate_records):
        return INVALID("DUPLICATE_OR_CONFLICTING_DIRECTION_RECORD", scope="METHOD")
    if set(record.direction_ordinal for record in direction_certificate_records) != set(range(8)):
        return INVALID("DIRECTION_PANEL_INCOMPLETE", scope="METHOD")
    if any(not record.identity_matches(site_identity) for record in direction_certificate_records):
        return INVALID("DIRECTION_IDENTITY_MISMATCH", scope="METHOD")
    if any(record.status == "INVALID" for record in direction_certificate_records):
        return INVALID("DIRECTION_CERTIFICATE_INVALID", scope="METHOD")
    if validity_flags.any_hard_site_graph_numeric_invalidity:
        return INVALID(validity_flags.first_canonical_reason, scope="METHOD")

    # 4. Resource state. Missing records were handled above and are not resource.
    if resource_flags.any_required_child_resource_inconclusive:
        return RESOURCE_INCONCLUSIVE(resource_flags.first_canonical_reason)
    if any(record.status == "RESOURCE_INCONCLUSIVE" for record in direction_certificate_records):
        return RESOURCE_INCONCLUSIVE("DIRECTION_RESOURCE_INCONCLUSIVE")

    # 5. Every required record must be a complete interval record.
    if any(record.status != "INTERVAL_COMPUTED" for record in direction_certificate_records):
        return INVALID("UNKNOWN_DIRECTION_TERMINAL_STATE", scope="METHOD")
    if any(not record.has_exact_radii({1, 1/2, 1/4}) for record in direction_certificate_records):
        return INVALID("RADIUS_PANEL_INCOMPLETE", scope="METHOD")
    if any(not record.has_valid_intervals_for({"psi", "pat_joint", "tar_joint"})
           for record in direction_certificate_records):
        return INVALID("SCALAR_INTERVAL_MISSING_OR_MALFORMED", scope="METHOD")

    # 6. Fixed interval arithmetic from Section 5.5.
    ratio_interval = outward_panel_ratio(direction_certificate_records)
    if ratio_interval.is_malformed_or_nonfinite:
        return INVALID("SITE_RATIO_INTERVAL_INVALID", scope="METHOD")

    if ratio_interval.upper <= 0.20:
        return CERTIFIED_POSITIVE("JWTEC_MARGIN_NONNEGATIVE")
    if ratio_interval.lower > 0.20:
        return CERTIFIED_NEGATIVE("JWTEC_MARGIN_STRICTLY_NEGATIVE")
    return UNRESOLVED("JWTEC_INTERVAL_STRADDLES_BOUNDARY")
```

### 7.4 Greater-Than clean-task validity

For one clean prompt, let `pi_j` be the frozen float64 softmax probability assigned to suffix token `u_j` at the final position. Define

\[
 V_{\mathrm{clean}}=
 \sum_{j>y_s}\pi_j-\sum_{j\le y_s}\pi_j.
\]

`clean_task_valid=True` iff `V_clean > 0`. Equality is false. This is computed once per prompt row and referenced by all nine layer sites. It uses no corruption, GREEN direction, certificate, restoration, or endpoint.

### 7.5 Missing, malformed, duplicated, and conflicting records

- Missing expected direction or radius: `INVALID`, never resource.
- Explicit worker cap exhaustion with a valid receipt: `RESOURCE_INCONCLUSIVE`.
- Nonfinite endpoint, interval, center, direction, score, or hash field: `INVALID`.
- Duplicate byte-identical record at an already committed final path: validate and treat as the same immutable record.
- Duplicate identity with nonidentical bytes: `INVALID_CONFLICTING_ARTIFACT`, phase hard stop.
- Unknown enum value: `INVALID`.
- Empty sound-interval intersection: `INVALID`, not negative or unresolved.

---

## 8. Denominator, coverage, and acceptance policy

### 8.1 Primary intent-to-analyze population

After the endpoint phase is valid, the task-specific primary denominator is

```text
all immutable confirmation site rows
with endpoint_status == VALID
and ordinary_restoration >= 0.80
and, for Greater-Than, clean_task_valid == True.
```

The endpoint population is shared by all methods. Any endpoint-invalid row invalidates the affected phase for every method; it is not removed only for GREEN.

Within the primary population, every one of these statuses remains in the denominator:

```text
CERTIFIED_POSITIVE
CERTIFIED_NEGATIVE
UNRESOLVED
RESOURCE_INCONCLUSIVE
INVALID with invalid_scope == METHOD
```

`INVALID/TASK_PRECONDITION` Greater-Than rows are excluded by the precommitted clean-task gate and do not count as method invalid. Any `RUN_CONTRACT` invalidity stops the phase before analysis.

### 8.2 Acceptance and coverage

Only `CERTIFIED_POSITIVE` is accepted.

\[
 \operatorname{coverage}=\frac{N_{\mathrm{CERTIFIED\_POSITIVE}}}{N_{\mathrm{primary}}}.
\]

Minimum coverage remains `0.50` for each task.

The GREEN method-invalid rate is

\[
 \frac{N_{\mathrm{INVALID, METHOD}}}{N_{\mathrm{primary}}}
\]

and must not exceed `0.05`.

`UNRESOLVED` and `RESOURCE_INCONCLUSIVE` are warnings, not abstentions removed from the denominator.

### 8.3 Shared selective-risk analyzer

The inherited analyzer logic remains:

- GREEN accepts its positive rows;
- every baseline accepts exactly the same number `K` of rows with lowest frozen baseline risk, ties by `row_id`;
- endpoint values are not used to select rows;
- prompt rows are bootstrap clusters;
- IOI and Greater-Than use the same GREEN status rule and thresholds.

The successor analyzer may be a hash-pinned wrapper around the existing analyzer, but it must additionally validate the successor schema, invalid scope, P13 receipt, certificate manifest, and endpoint-transition receipt before delegating to the inherited statistics.

---

## 9. Causal graph and identity contract

### 9.1 Executable downstream causal cone

For a site after `blocks.l.hook_resid_post`, the formal graph begins at the controlled-hook cut and contains the entire downstream cone:

1. the full sequence residual tensor at the cut for the PAT or TAR context;
2. the affine replacement `x_s+t d_{s,k}` at the exact site position;
3. every subsequent transformer block `l+1,...,11`;
4. all token positions, attention scores, masks, softmaxes, value mixing, residual additions, LayerNorms, MLPs, and nonlinearities that can influence the final position;
5. the layer-10 selected-gate live/frozen intervention;
6. final LayerNorm, unembedding, and the exact task scalar output.

The graph must not inject only at the final token. It must not use the old fixed block-8 patch graph for layers 0–8. It must not drop earlier token positions after attention can transmit their effect.

The t-independent controlled-hook tensors are exact serialized float64 payloads produced by the frozen response-evaluation checkpoint. Every float64 value is imported into MPFR as its exact dyadic value. The formal claim is about the explicitly implemented hook intervention using these payloads. Full-model hook execution and tail replay must satisfy the parity tests in Section 15.

### 9.2 Graph outputs

Each relational graph exports exactly these named scalars and jets:

```text
PAT_J
PAT_B
TAR_J
TAR_B
PSI = PAT_J - PAT_B - TAR_J + TAR_B
```

The graph compiler must retain a signed dependency map showing that the `PSI` output is algebraically identical to the branch formula. Common-subexpression elimination is allowed only when the compiler emits a reversible provenance map and equivalence tests pass.

### 9.3 Graph validity checks

A graph is valid only if:

- hook, layer, position, shape, model, tokenizer, task metric, selected gates, and payload hashes match the site identity;
- the site is strictly upstream of gate layer 10;
- every output has a complete topological dependency closure;
- `PAT_J(0)=PAT_B(0)` and `TAR_J(0)=TAR_B(0)` are enclosed around zero at both precisions;
- native full-model, native tail, float64 AD, and interval routes agree under the frozen parity contract at `t∈{0,±1,±1/2,±1/4}`;
- the 512-bit graph and partition identities equal the 384-bit identities except precision;
- no endpoint direction, endpoint path, endpoint key, or endpoint outcome is in the graph manifest or process environment.

### 9.4 Canonical serialization

Canonical JSON is UTF-8 with:

```python
json.dumps(value, sort_keys=True, separators=(",", ":"),
           ensure_ascii=False, allow_nan=False)
```

SHA-256 is lowercase hexadecimal over those bytes. Tensor hashing uses an explicit dtype/shape header, one NUL byte, and canonical contiguous little-endian bytes.

### 9.5 Site identity

Every successor site identity contains at least:

```json
{
  "schema_version": "green-v410-sfc-jwtec-site-identity-v1",
  "protocol_id": "GREEN-V410-SFC-JWTEC-20260830",
  "parent_protocol_id": "<exact-v400-task-protocol-id>",
  "phase": "qualification|confirmation",
  "task": "ioi|greater_than",
  "parent_prompt_row_id": "<immutable>",
  "parent_site_row_id": "<immutable>",
  "layer": 0,
  "hook_family": "resid_post",
  "hook_name": "blocks.0.hook_resid_post",
  "token_position": 0,
  "clean_tokens_sha256": "<sha256>",
  "corrupt_tokens_sha256": "<sha256>",
  "clean_center_tensor_sha256": "<sha256>",
  "pat_controlled_hook_tensor_sha256": "<sha256>",
  "tar_controlled_hook_tensor_sha256": "<sha256>",
  "green_direction_binding_sha256": "<sha256>",
  "selected_gate_spec_sha256": "<sha256>",
  "task_metric_spec_sha256": "<sha256>",
  "model_manifest_sha256": "<sha256>",
  "tokenizer_manifest_sha256": "<sha256>",
  "graph_semantics_id": "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1"
}
```

The `site_identity_sha256` is the SHA-256 of this canonical object.

### 9.6 Direction certificate identity

For each ordinal, define

```text
certificate_row_id = SHA256(canonical_json({
  schema_version,
  protocol_id,
  task,
  phase,
  parent_site_row_id,
  site_identity_sha256,
  direction_ordinal,
  direction_payload_sha256,
  radius_panel_id,
  branch_semantics_id,
  graph_manifest_sha256,
  resource_manifest_sha256,
  decision_rule_sha256
}))
```

### 9.7 Site certificate identity

```text
site_certificate_id = SHA256(canonical_json({
  schema_version,
  protocol_id,
  task,
  phase,
  parent_site_row_id,
  site_identity_sha256,
  sorted_direction_certificate_row_ids,
  p13_two_task_pass_receipt_sha256,
  decision_rule_sha256,
  classifier_source_sha256
}))
```

Exactly eight sorted direction IDs are required.

### 9.8 Reuse of older graph materials

Older v4 certificate rows, row specs, TensorPrograms, real-row serializers, and graph identities are scientifically ineligible and may not be crosswalked by layer or prompt heuristics.

Reusable only after new hash binding:

- interval-number and interval-jet primitives;
- exact LayerNorm, GELU, softmax, affine, and reduction kernels;
- MPFR context code;
- budget-monotone partition machinery;
- synthetic theorem fixtures;
- canonical JSON/tensor hashing utilities.

---

## 10. Protocol versioning and eligibility ruling

### 10.1 Version

This is not an in-place v4.0.0 repair.

```text
Protocol name: GREEN v4.1.0 — SFC Joint-Witness Transport Equivalence Certificate
Protocol ID: GREEN-V410-SFC-JWTEC-20260830
Branch: codex/green-v410-sfc-jwtec-successor
Decision rule: GREEN_V410_SFC_JWTEC_DECISION_V1
Attempt: A1
```

All new artifacts use v4.1.0 schema names. Existing v4.0.0 files remain immutable historical evidence.

### 10.2 Development status

All opened v4.0.0 development predictions, replay records, endpoints, aggregate diagnostics, and later summaries are `DIAGNOSTIC_ONLY_FOR_V410`.

They may be discussed as historical evidence that motivated a successor protocol, but they may not be used to:

- choose or validate the v4.1 site threshold;
- choose the RMS aggregation;
- choose radii or directions;
- choose P13 sample or thresholds;
- choose resource budget;
- choose graph simplifications;
- estimate v4.1 coverage;
- tune confirmation baselines or thresholds;
- decide whether to continue after P13.

No v4.1 primary claim may be made on development.

### 10.3 Confirmation eligibility

The existing confirmation split remains eligible for one v4.1 confirmation attempt only if a machine-verifiable seal audit passes after this decision is frozen.

The seal audit must establish:

1. no confirmation prediction, Grant, clean-validity, certificate, replay, endpoint, analyzer, or model-session output exists;
2. no execution receipt authorizes confirmation;
3. no confirmation job ID appears in worker logs, caches, temporary paths, supervisor state, or phase ledgers except immutable prepare/queue manifests;
4. no endpoint direction payload was materialized in a prediction or certificate process;
5. the immutable confirmation site and direction hashes still match the parent manifests;
6. two independent audit implementations produce byte-identical canonical summaries.

Reading already-public prepare manifests and prompt text does not contaminate confirmation. Producing or reading model-derived confirmation values does.

If the seal fails, `GREEN-V410-SFC-JWTEC-20260830/A1` stops. The existing confirmation and the reserve used for P13 are not eligible as a replacement. A later attempt would require an entirely new, independently frozen prompt universe, split, GREEN direction panel, endpoint panel, and protocol document. This decision does not authorize an automatic fallback universe.

---

## 11. Artifact reuse and regeneration

### 11.1 Mechanically reusable

Subject to exact hash validation and a passing seal audit:

- immutable IOI and Greater-Than prompt and site manifests;
- parent confirmation GREEN direction tensors and bindings;
- parent hidden endpoint direction commitments and sealed payloads;
- model, checkpoint, tokenizer, task metric, and selected-gate manifests;
- unopened parent confirmation prediction, Grant, replay, and endpoint job payloads;
- existing response adapters and empirical baseline source only where their source hashes are explicitly adopted;
- existing shared analyzer statistics through a successor validation wrapper.

Adoption requires a `green-v410-adoption-receipt-v1` recording parent path, parent hash, semantic role, successor role, and a proof that no prohibited field changed.

### 11.2 Diagnostic-only

- all v4.0.0 development outputs;
- current development completeness and status reports;
- development endpoint values and diagnostic AUROCs;
- older donor, boundary-transition, and formal-prepare certificate rows.

### 11.3 Must be generated anew

- v4.1 scientific config and canonical decision hash;
- reserve qualification selection manifest and direction registry;
- P13 qualification queues, graph manifests, records, and task receipts;
- confirmation seal receipts;
- site-to-certificate identity crosswalk;
- full downstream-cone graph materializer;
- real-row interval serializer;
- certificate worker and coordinator;
- resource calibration and frozen resource manifest;
- confirmation certificate queues;
- Greater-Than clean-validity queue and receipts;
- v4.1 site status packets and status manifest;
- endpoint-transition authorization binding the frozen status manifest;
- successor analyzer wrapper and final decision receipt.

---

## 12. Production implementation contract

### 12.1 Schema and queue names

Required schemas:

```text
green-v410-sfc-jwtec-protocol-spec-v1
green-v410-sfc-jwtec-adoption-receipt-v1
green-v410-sfc-jwtec-confirmation-seal-v1
green-v410-sfc-jwtec-resource-calibration-v1
green-v410-sfc-jwtec-resource-manifest-v1
green-v410-sfc-jwtec-site-identity-v1
green-v410-sfc-jwtec-graph-manifest-v1
green-v410-sfc-jwtec-direction-certificate-v1
green-v410-sfc-jwtec-site-certificate-v1
green-v410-sfc-jwtec-p13-task-receipt-v1
green-v410-sfc-jwtec-p13-two-task-receipt-v1
green-v410-sfc-jwtec-clean-validity-v1
green-v410-sfc-jwtec-status-manifest-v1
green-v410-sfc-jwtec-endpoint-transition-v1
green-v410-sfc-jwtec-final-analysis-receipt-v1
```

Required queue IDs:

```text
GREEN_V410_P13_IOI_CERTIFICATE_QUEUE_A1
GREEN_V410_P13_GT_CERTIFICATE_QUEUE_A1
GREEN_V410_CONFIRMATION_SEAL_AUDIT_QUEUE_A1
GREEN_V410_CONFIRM_IOI_PREDICTION_ADOPTION_QUEUE_A1
GREEN_V410_CONFIRM_GT_PREDICTION_ADOPTION_QUEUE_A1
GREEN_V410_CONFIRM_GT_CLEAN_VALIDITY_QUEUE_A1
GREEN_V410_CONFIRM_IOI_GRANT_ADOPTION_QUEUE_A1
GREEN_V410_CONFIRM_GT_GRANT_ADOPTION_QUEUE_A1
GREEN_V410_CONFIRM_IOI_REPLAY_ADOPTION_QUEUE_A1
GREEN_V410_CONFIRM_GT_REPLAY_ADOPTION_QUEUE_A1
GREEN_V410_CONFIRM_IOI_CERTIFICATE_QUEUE_A1
GREEN_V410_CONFIRM_GT_CERTIFICATE_QUEUE_A1
GREEN_V410_CONFIRM_NONENDPOINT_FREEZE_QUEUE_A1
GREEN_V410_CONFIRM_IOI_ENDPOINT_ADOPTION_QUEUE_A1
GREEN_V410_CONFIRM_GT_ENDPOINT_ADOPTION_QUEUE_A1
GREEN_V410_CONFIRM_MERGE_ANALYZE_QUEUE_A1
```

Expected fixed confirmation certificate queue sizes are:

- IOI: 1,152 site jobs and 9,216 direction records;
- Greater-Than: 3,456 site jobs and at most 27,648 direction records. The queue contains all 3,456 sites; clean-invalid prompts emit task-precondition site records without formal direction work.

Greater-Than clean validity has 384 prompt jobs, each referenced by nine site rows.

### 12.2 Queue construction

Queue builders are pure functions of frozen manifests and must:

1. sort by `(task, phase, prompt_row_id, layer, site_row_id)`;
2. emit one site job containing the exact eight direction identities;
3. include all source, schema, graph, model, tokenizer, gate, task metric, resource, and decision hashes;
4. contain no endpoint payload or endpoint path for prediction/certificate queues;
5. assert exact expected counts and no duplicate identities;
6. write a canonical queue manifest and SHA-256 before execution;
7. forbid command-line overrides of any scientific or resource field.

### 12.3 Real-row serialization

The real-row serializer must be a new v4.1 module. It must:

- accept only `qualification` or `confirmation` roles under v4.1;
- reject `development` real rows;
- import exact model-manifest constants and float64 controlled-hook payloads;
- import exact float32 directions as dyadic rationals;
- emit the five-output graph and reversible provenance map;
- serialize no endpoint key, direction, value, or path;
- validate graph node count before any MPFR evaluation;
- emit a graph manifest even when a later worker becomes resource-inconclusive.

### 12.4 Certificate worker

A certificate site worker must:

1. validate the site job and all hashes;
2. for Greater-Than, read only the clean-validity receipt; if false, emit `INVALID/TASK_PRECONDITION` without materializing directions or formal graph work;
3. for each ordinal `0,...,7`, construct one graph and execute all three radii in fixed order `1,1/2,1/4`;
4. complete the entire 384-bit official phase before starting 512-bit audit for that direction;
5. use the budget-monotone exact priority and tie breaks;
6. emit one immutable direction record;
7. after all eight records, invoke the frozen classifier and emit one site packet;
8. never read endpoint material;
9. never use a scientific midpoint, approximate sign, or partial direction panel.

### 12.5 Deterministic resource calibration and ceilings

The historical `max_cells=262144` and 17-radius plan is not executable and is retired.

Before any qualification real row, run a closed synthetic actual-shape calibration at candidate final leaf budgets

```text
{4, 8, 16, 32}
```

using six fixed graph profiles:

```text
IOI layer 0, IOI layer 4, IOI layer 8,
GT layer 0, GT layer 4, GT layer 8
```

The calibration fixture generator is fixed as `numpy-PCG64DXSM` with unsigned 64-bit seed `41020260830` and domain separator `GREEN_V410_RESOURCE_CALIBRATION_V1`. For each profile, run exactly the five domains below at each precision and candidate budget, giving 30 cold processes per precision and candidate:

```text
[-1, 0]
[0, 1]
[-1, -1/2]
[1/2, 1]
[511/1024, 513/1024]
```

The domain-to-fixture mapping is exact:

```text
[-1, 0]                 -> affine
[0, 1]                  -> positive_curvature
[-1, -1/2]              -> negative_curvature
[1/2, 1]                -> signed_cancellation
[511/1024, 513/1024]    -> deep_dyadic
```

For each `(profile, fixture_kind)`, derive a child seed as the first eight bytes, interpreted little-endian unsigned, of `SHA256("GREEN_V410_RESOURCE_CALIBRATION_V1\0" || uint64_le(41020260830) || "\0" || profile || "\0" || fixture_kind)`. Generate independent standard-normal float64 PAT and TAR controlled-hook tensors, then overwrite the site vector with the generated clean center. Generate the float32 direction by standard-normal draw followed by float64 normalization to `0.001` and cast to float32; verify it under the frozen direction-binding tolerance. The synthetic branch nonlinearity modifier used to force the four fixture kinds is specified by exact coefficients in the calibration config as `affine:(0,0)`, `positive_curvature:(+1/16,0)`, `negative_curvature:(-1/16,0)`, `signed_cancellation:(+1/16,-1/16)`, and `deep_dyadic:(+1/32,+1/64)`, applied only in the closed test wrapper and never in production graphs. No real prompt, center, direction, response, or endpoint artifact may be read.

The selected uniform leaf budget `L*` is the largest candidate satisfying all of:

```text
all theorem and nesting checks pass;
max_depth <= 24;
graph_nodes <= 2,000,000;
1.25 * maximum observed process-tree RSS <= 64 GiB;
1.25 * projected complete direction wall time <= 85,800 seconds;
no cgroup OOM, timeout, malformed checkpoint, or nondeterministic replay;
byte-identical replay under the frozen scheduler.
```

For `R=3` radii and uniform `L*`, the maximum formal passes per completed direction are recorded as

\[
 N_{384}=3(2L^*+1),\qquad
 N_{512}=3(L^*+3),\qquad
 N_{\mathrm{total}}=9L^*+12.
\]

Hard production limits:

```text
precision_official_bits: 384
precision_audit_bits: 512
max_depth: 24
max_graph_nodes: 2,000,000
memory.max per direction child: 68,719,476,736 bytes
cumulative wall limit per direction child: 85,800 seconds
safety guardband: 1.25
leaf budget: mechanically selected L*
radius count: 3
attempt_index: 1
```

If no candidate passes, emit `STOP_RESOURCE_LOCK_INFEASIBLE`. After `L*` is frozen, it may not be increased or decreased using qualification, confirmation, sign, interval width, P13, or coverage information.

No separate absolute or relative width threshold is required for `INTERVAL_COMPUTED`. A complete sound interval at `L*` is valid; if it cannot decide the site margin, the site is `UNRESOLVED`.

### 12.6 Retry and resume

There is one scientific attempt. Resource-inconclusive and invalid rows are never retried with changed budgets.

One process relaunch is allowed only when all conditions hold:

- failure code is exactly one of `HOST_REBOOT`, `WORKER_PROCESS_LOST`, `TRANSIENT_FILESYSTEM_EIO`, or `TRANSIENT_FILESYSTEM_ESTALE`;
- no final direction record was committed;
- a byte-validated checkpoint exists;
- the same job ID, process image, source hashes, partition, priority queue, counters, limits, and output paths are reused;
- completed/charged passes and cumulative wall time are not refunded.

Cgroup OOM, wall limit, leaf limit, depth limit, interval invalidity, nesting failure, or unknown process death is terminal for the direction. `max_process_launches=2` including the initial launch.

### 12.7 Atomic no-clobber outputs

Every output uses:

1. a same-filesystem temporary path `.<name>.partial.<job_id>.<pid>`;
2. complete schema and hash validation;
3. file `fsync`;
4. atomic no-replace publication;
5. directory `fsync`;
6. immutable final permissions.

If a final path already exists:

- byte-identical canonical content is accepted as the same artifact;
- any difference is `INVALID_CONFLICTING_ARTIFACT` and phase hard stop.

No `os.replace` may overwrite an existing final production artifact.

### 12.8 Validation receipts

Every queue and phase emits a canonical receipt containing:

- queue/phase ID;
- protocol and attempt;
- expected and observed counts;
- sorted artifact IDs and hashes;
- source and schema hashes;
- model/tokenizer/gate/task/direction/resource/decision hashes;
- start and completion timestamps as metadata only;
- terminal state;
- first canonical failure code;
- proof that endpoint access was disabled where required.

A downstream phase accepts only the exact upstream receipt hash listed in its plan.

### 12.9 Endpoint firewall

The endpoint route is physically and logically separated.

Before the endpoint key or endpoint payload can be materialized, the following must exist and be frozen:

- confirmation seal PASS;
- two-task P13 PASS;
- all confirmation prediction and Grant packets;
- Greater-Than clean-validity manifest;
- all numerical replay receipts;
- all v4.1 site status packets;
- complete non-endpoint status manifest;
- source/schema/queue/resource/decision hashes;
- endpoint-transition receipt.

Certificate processes must run under an environment and schema that do not contain endpoint paths or keys. Endpoint processes receive no GREEN directions, certificate widths, P13 values, or status values. The transition receipt binds the status-manifest hash without exposing its contents to endpoint selection logic.

### 12.10 Analyzer ingestion

The merge step must create exactly one row per immutable parent site with:

```text
row_id = parent_site_row_id
prompt_row_id
task
ordinary_restoration
clean_task_valid or NOT_APPLICABLE
green_status
green_reason_code
green_invalid_scope
site_certificate_id
baseline_risk_scores
endpoint_status
heldout_transport_absolute_rmse
heldout_transport_symmetric_normalized_error
```

The wrapper rejects:

- missing or duplicate parent site IDs;
- certificate IDs not in the frozen status manifest;
- invalid endpoint populations;
- nonfinite restoration/baseline/endpoint values;
- any status outside the five-state domain;
- missing P13, seal, transition, source, or schema receipts.

The analyzer never recomputes a status.

---

## 13. Canonical machine-readable protocol specification

The following object is normative. Hash fields in angle brackets are mechanically computed from frozen files; they are not discretionary values.

```json
{
  "schema_version": "green-v410-sfc-jwtec-protocol-spec-v1",
  "protocol_id": "GREEN-V410-SFC-JWTEC-20260830",
  "parent_commit": "91741e32bfaad05667fb2d4c8d1de26373e90a15",
  "outcome_blind_cutoff_commit": "6aab5837bf3950b86b166ef938097e9739810d9b",
  "attempt_index": 1,
  "scientific_retry_allowed": false,
  "tasks": ["ioi", "greater_than"],
  "site_layers": [0,1,2,3,4,5,6,7,8],
  "site_hook_family": "resid_post",
  "gate": {
    "layer": 10,
    "hook": "blocks.10.mlp.hook_post",
    "position": -1,
    "selected_indices": [2326,1138,2287,606,2848,2305,46,2659,946,1616]
  },
  "directions": {
    "count": 8,
    "width": 768,
    "payload_dtype": "float32-little-endian",
    "nominal_norm": 0.001,
    "generator": "numpy-PCG64DXSM-rowwise-normal-v1"
  },
  "radii": ["1", "1/2", "1/4"],
  "precision_bits": {"official": 384, "audit": 512},
  "branch_order": ["PAT_J", "PAT_B", "TAR_J", "TAR_B"],
  "branch_weights": [1,-1,-1,1],
  "site_estimand": {
    "j": "d/dt Psi(0)",
    "p": "d/dt PAT_J(0)",
    "q": "d/dt TAR_J(0)",
    "aggregation": "RMS across exactly eight directions",
    "denominator": "max(RMS(p), RMS(q), 1e-12)",
    "ratio_threshold": "1/5"
  },
  "status_rule": {
    "CERTIFIED_POSITIVE": "ratio_interval.upper <= 1/5",
    "CERTIFIED_NEGATIVE": "ratio_interval.lower > 1/5",
    "UNRESOLVED": "otherwise after valid complete intervals",
    "INVALID": "hard contract, task, graph, identity, or numeric invalidity",
    "RESOURCE_INCONCLUSIVE": "declared resource ceiling before complete interval",
    "CERTIFIED_NULL": "FORBIDDEN"
  },
  "p13": {
    "role": "two-task qualification cohort gate only",
    "site_aggregation": "max over eight direction contraction ratios",
    "task_median_max": "1/5",
    "task_p90_max": "1/2",
    "sample_sites_per_task": 108,
    "sample_prompt_clusters_per_task": 12,
    "all_records_required": true
  },
  "primary_analysis": {
    "restoration_min": "4/5",
    "greater_than_clean_valid_required": true,
    "accept_status": "CERTIFIED_POSITIVE",
    "minimum_coverage": "1/2",
    "maximum_method_invalid_rate": "1/20",
    "warning_rows_remain_in_denominator": true
  },
  "resource_selector": {
    "candidate_leaf_budgets": [4,8,16,32],
    "selection": "largest passing candidate",
    "max_depth": 24,
    "max_graph_nodes": 2000000,
    "memory_max_bytes": 68719476736,
    "direction_wall_max_seconds": 85800,
    "guardband": "5/4",
    "max_process_launches": 2
  },
  "development_use": "DIAGNOSTIC_ONLY",
  "confirmation": {
    "eligible_only_if_seal_passes": true,
    "one_shot": true,
    "endpoint_after_status_manifest": true
  },
  "required_hashes": {
    "decision_document_sha256": "<computed>",
    "decision_rule_sha256": "<computed>",
    "protocol_config_sha256": "<computed>",
    "schema_bundle_sha256": "<computed>",
    "source_bundle_sha256": "<computed>",
    "resource_manifest_sha256": "<computed>",
    "p13_two_task_receipt_sha256": "<computed-before-confirmation>",
    "confirmation_seal_sha256": "<computed-before-confirmation>"
  }
}
```

---

## 14. Required tests and acceptance criteria

### 14.1 Mathematical and interval tests

1. affine functions: exact derivative singleton at every radius;
2. quadratic and cubic functions: signed-secant remainder encloses exact derivative;
3. positive and negative curvature partitions;
4. cross-radius intersection nonempty for valid functions;
5. deliberately inconsistent enclosures produce `INVALID`, never unresolved;
6. refinement can only tighten reportable intervals;
7. 512-bit audit nests inside 384-bit official intervals;
8. interval square/RMS/max/division includes high-precision ground truth;
9. exact boundary `[0.20,0.20]` is positive;
10. `[0.20,0.20+δ]` is unresolved;
11. lower bound strictly above `0.20` is negative;
12. independently reversing any subset of directions leaves the site status unchanged;
13. mixed raw signs are aggregated by RMS, not vote;
14. no `CERTIFIED_NULL` can be serialized.

### 14.2 P13 tests

1. golden four-branch center and bound calculation;
2. `B_JW=min(B_box,B_W)` exactness;
3. zero-box boundary cases;
4. `rho_s=max` over eight directions;
5. exact 108-value median and nearest-rank p90;
6. equality at `0.20` and `0.50` passes;
7. any missing/resource/invalid qualification record fails the task;
8. P13 value cannot enter the site classifier except through the global PASS receipt;
9. reserve selection golden hashes and exact counts.

### 14.3 Graph tests

For IOI and Greater-Than at layers 0, 4, and 8:

1. full-model hook and tail replay parity at `t=0,±1,±1/2,±1/4`;
2. exact task metric parity;
3. `PAT_J(0)=PAT_B(0)` and `TAR_J(0)=TAR_B(0)`;
4. selected gates frozen only in B branches;
5. nonselected gates and residual bypass remain live;
6. dependency closure reaches all relevant token positions;
7. branch permutation changes the signed contrast as expected;
8. omitted downstream node or earlier token position is detected;
9. old fixed block-8 graph is rejected for a current SFC identity;
10. endpoint identifiers and paths are absent.

### 14.4 Identity and schema tests

1. canonical JSON golden vectors;
2. little-endian tensor hash golden vectors;
3. site, direction, graph, and site-certificate ID golden vectors;
4. ordering invariance where allowed and ordering sensitivity where required;
5. duplicate identical artifact idempotence;
6. duplicate conflicting artifact phase stop;
7. malformed enum, NaN, infinity, negative widths, or reversed bounds rejected;
8. exact expected queue counts;
9. parent-to-successor adoption receipt verifies no semantic drift.

### 14.5 Classifier-totality tests

Exhaustively enumerate combinations of:

- task domain;
- clean validity states;
- zero through nine direction records;
- duplicate/conflict flags;
- interval/resource/invalid child statuses;
- well-formed and malformed ratio intervals;
- global P13 pass/fail/missing.

Every case must return exactly one of the five statuses and one canonical reason code. Invalid precedence must dominate resource precedence.

### 14.6 Resource, retry, and no-clobber tests

1. candidate budget selector chooses the largest passing candidate deterministically;
2. no candidate produces the STOP receipt;
3. cgroup OOM → resource-inconclusive, no 512 start;
4. wall limit → resource-inconclusive;
5. process relaunch whitelist and cumulative counters;
6. no refund of charged passes;
7. byte-identical crash resume;
8. no-replace atomic publication under concurrent writers;
9. conflicting final bytes stop the phase;
10. larger posthoc budget cannot upgrade an official result.

### 14.7 Firewall and transition tests

1. certificate worker cannot parse an endpoint payload;
2. endpoint environment is unavailable before transition receipt;
3. endpoint transition fails if any site status is missing;
4. endpoint transition fails if P13 or seal receipt mismatches;
5. endpoint worker receives no GREEN status values;
6. analyzer refuses data without the endpoint-transition hash;
7. development outputs cannot enter the v4.1 analysis path;
8. confirmation seal detects seeded forbidden artifacts in every output category.

### 14.8 Analyzer tests

1. primary population exactly implements restoration and clean-validity gates;
2. warning/resource/method-invalid statuses remain in denominator;
3. task-precondition invalid rows are excluded but reported;
4. only positive rows accepted;
5. matched baseline count equals GREEN accepted count;
6. ties break by `row_id` without endpoint values;
7. prompt-cluster bootstrap treats layers as dependent;
8. minimum coverage and invalid-rate gates are exact;
9. invalid endpoint invalidates all methods.

### 14.9 Acceptance standard

All binding tests must pass with:

```text
0 failed
0 error
0 xfailed
0 xpassed
0 skipped binding tests
0 unreviewed warnings
```

Platform-specific nonbinding tests may be skipped only if listed in a frozen waiver manifest before real execution. No waiver may cover mathematical, graph, identity, classifier, firewall, resource, or confirmation-seal tests.

---

## 15. Transition gates and immediate stop conditions

### 15.1 Gate order

```text
T0 decision/config/schema/source freeze
T1 theorem and synthetic tests
T2 deterministic resource calibration and resource-manifest freeze
T3 confirmation seal audit
T4 reserve direction freeze
T5 IOI and GT P13 qualification
T6 confirmation non-endpoint authorization
T7 prediction, Grant, clean-validity, replay, graph, and certificate completion
T8 site-status manifest freeze
T9 endpoint transition authorization
T10 sealed endpoint execution
T11 merge and one-shot analyzer
T12 final scientific verdict
```

No gate may be reordered to expose endpoint information earlier.

### 15.2 Immediate STOP conditions

Any of the following stops A1:

- confirmation seal failure;
- no feasible resource candidate;
- any binding test failure;
- either task fails P13;
- source/schema/hash drift after freeze;
- a confirmation artifact predating authorization;
- endpoint material in prediction/certificate process;
- queue count or identity mismatch;
- conflicting no-clobber artifact;
- graph parity or causal-cone failure;
- 512/384 nesting failure;
- attempt to tune threshold, radius, aggregation, budget, subset, or status mapping using real results;
- attempt to rerun qualification or confirmation under the same protocol after a scientific or resource failure.

---

## 16. Forbidden actions

Codex and operators are forbidden to:

1. edit v4.0.0 artifacts or describe v4.1 as retroactive completion;
2. infer site status from restoration, endpoint error, a baseline score, Grant divergence, or empirical four-branch samples;
3. use raw derivative-sign voting;
4. use fewer than eight directions or fewer than three radii;
5. select a best direction, radius, precision, partition, or interval after seeing values;
6. use the old 17-radius plan;
7. reuse old certificate row identities or the fixed block-8/final-token TensorProgram;
8. introduce `CERTIFIED_NULL`;
9. remove unresolved, resource, or method-invalid rows from the primary denominator;
10. relabel malformed/missing/conflicting artifacts as resource-inconclusive;
11. increase compute for promising rows or after P13/coverage inspection;
12. materialize endpoint directions before the frozen status manifest;
13. expose endpoint values to the classifier or certificate worker;
14. use development outcomes in successor resource or scientific decisions;
15. overwrite a final artifact;
16. start confirmation if either task P13 fails;
17. create an automatic fallback confirmation universe if the seal fails;
18. weaken a gate to obtain an ICLR-positive result.

---

## 17. Reviewer-facing justification

### 17.1 Prospectivity

The successor rule is prospective with respect to every v4.1 certificate result, reserve P13 result, and confirmation outcome. It is not the original v4.0.0 preregistration. The paper must state that the original v4.0.0 development route revealed a protocol incompleteness because no site-level certificate mapping had been frozen. Development was retired as diagnostic, and a complete successor rule was frozen before an untouched one-shot confirmation.

### 17.2 Avoiding a favorable post-outcome rule

The rule is anchored in pre-outcome objects:

- the four-branch matched-bypass functional;
- the three inherited local radii;
- the eight frozen GREEN directions;
- the task response maps;
- the endpoint's symmetric normalized scale and 0.20 effect-size boundary;
- the original P13 contraction thresholds;
- the shared coverage and analysis rule.

The site rule is sign-reversal invariant, uses all directions, admits no direction/radius selection, and is frozen before any certificate outcome exists. Reserve P13 uses only proof widths and is separate from confirmation endpoints.

### 17.3 Confirmation validity

Confirmation can constitute a valid untouched test if and only if the seal audit passes. The correct description is **prospective successor confirmation**, not untouched continuation of the original v4.0.0 analysis.

### 17.4 Defensible claims

If both tasks satisfy the frozen coverage, invalidity, and selective-risk gates, the defensible central claim is:

> A rigorous relational certificate of local matched-bypass transport equivalence can prospectively select behaviorally restored interventions with lower held-out full-response transport error than strong empirical predictors at matched coverage, across IOI and Greater-Than.

Additional defensible claims may include budget-monotone soundness and the distinction between certified mismatch, unresolved proof width, and resource inconclusiveness.

The paper may not claim that GREEN proves global mechanism identity, certifies all directions, or rescues the original v4.0.0 development analysis.

### 17.5 ICLR ambition

The central GREEN/Joint Witness contribution remains scientifically strong enough for an ambitious ICLR submission only if:

- the full downstream-cone certificate is genuinely operational;
- coverage is at least 0.50 on both tasks;
- invalidity is at most 0.05;
- confirmation selective risk beats the strong frozen comparators under the existing simultaneous criteria;
- the protocol history is disclosed without euphemism.

A failure of those gates is a valid negative or engineering-limit result, not permission for another same-data main-claim rewrite.

---

## INSTRUCTIONS TO CODEX

Execute these instructions in order. Do not make any scientific choice not written below.

1. Create branch `codex/green-v410-sfc-jwtec-successor` from commit `91741e32bfaad05667fb2d4c8d1de26373e90a15`. Do not modify any existing v4.0.0 artifact.
2. Add this decision document at `analysis/GPTPRO_GREEN_V4_SUCCESSOR_CERTIFICATE_PROTOCOL_DECISION_20260830.md` and compute its canonical file SHA-256.
3. Create `configs/green_v410_sfc_jwtec_protocol.json` containing the normative object in Section 13. Replace only mechanically computed hash placeholders. Do not alter any scientific value.
4. Implement the listed v4.1 schemas and canonical validators. Reject NaN, infinity, unknown fields where the schema is closed, unknown enum values, malformed bounds, and noncanonical hashes.
5. Implement the confirmation seal with two independent scanners. Scan repository outputs, `/mnt/sdb/ccj/iclr_1_runs`, temporary files, caches, logs, supervisor state, phase ledgers, and receipts. Commit both canonical reports and a comparison receipt. If either finds a forbidden confirmation artifact or the reports differ, emit `STOP_CONFIRMATION_SEAL_FAILED` and stop.
6. Implement the full downstream-cone graph materializer for `resid_post` layers 0–8. Use the exact selected gates and task response maps in this document. Reject the old fixed block-8 graph for v4.1 identities.
7. Implement five graph outputs: `PAT_J`, `PAT_B`, `TAR_J`, `TAR_B`, and `PSI`. Enforce branch weights `[1,-1,-1,1]` and branch-specific `t=0` gate anchors.
8. Implement direction certificates at radii `1`, `1/2`, and `1/4`, 384-bit official and 512-bit audit, with the signed-secant and budget-monotone intersection equations in Section 5. Do not add or remove radii.
9. Implement the deterministic synthetic resource calibration over leaf budgets `{4,8,16,32}`. Use the exact guardband and hard ceilings in Section 12.5. Select the largest passing candidate. If none passes, emit `STOP_RESOURCE_LOCK_INFEASIBLE` and stop. Freeze `L*` by hash before any reserve model row.
10. Implement all tests in Section 14. Run the complete binding suite. If any binding test fails, stop. Do not waive a binding test.
11. Build the P13 IOI and Greater-Than qualification cohorts exactly as Section 6.2. Freeze their prompt, site, direction, graph, queue, and resource hashes before execution. Assert 12 prompt clusters, 108 sites, and 864 directions per task.
12. Execute each P13 queue once. Compute response-only centers at `h=1/4`, independent branch boxes, relational witness radii, worst-direction site ratios, exact task medians, and nearest-rank p90 values. Any missing, invalid, or resource record fails the task. If either task fails, emit `STOP_P13_QUALIFICATION_FAILED` and stop without opening confirmation.
13. Issue the canonical two-task P13 PASS receipt. Bind its hash into every confirmation site job and classifier invocation.
14. Build adoption receipts for immutable parent confirmation site manifests, GREEN directions, prediction jobs, Grant jobs, replay jobs, and sealed endpoint jobs. Reuse only byte-identical semantics. Do not reuse development output values.
15. Build the fixed confirmation queues with the IDs and expected counts in Section 12.1. The Greater-Than site queue must contain all 3,456 sites even though clean-invalid sites later short-circuit.
16. Run confirmation prediction, Grant, and Greater-Than clean-validity routes under non-endpoint authorization. Clean validity is strict `correct_suffix_probability_mass > incorrect_suffix_probability_mass`; equality is false. Freeze the prompt-level clean-validity manifest.
17. Run the inherited numerical replay gates. Any replay-invalid phase stops before endpoint and before scientific analysis.
18. Materialize confirmation site graph manifests and run the IOI and Greater-Than certificate queues. For clean-invalid Greater-Than prompts, emit `INVALID/TASK_PRECONDITION` site packets without formal direction work. For every task-valid site, require eight complete direction records.
19. Invoke only the classifier in Section 7. Do not derive status in a worker, analyzer, notebook, or ad hoc script. Emit immutable site packets and then a complete status manifest.
20. Validate that the status manifest contains exactly 1,152 IOI and 3,456 Greater-Than site rows, with unique parent site IDs and one of the five statuses. Validate method-invalid and task-precondition scopes separately. Do not inspect or summarize status counts before the endpoint-transition receipt is committed.
21. Freeze all non-endpoint artifacts and issue `GREEN_V410_CONFIRM_NONENDPOINT_FREEZE_A1`. The receipt must bind the complete status-manifest hash without exposing endpoint payloads.
22. Issue the endpoint-transition authorization only after every prerequisite hash matches. Materialize endpoint keys and payloads in separate endpoint processes. Endpoint workers must receive no certificate widths, P13 values, GREEN directions, or status values.
23. Execute each sealed endpoint job once under the inherited replay and no-clobber rules. Any invalid endpoint invalidates the affected phase for every method.
24. Merge one row per parent site using Section 12.10. Validate all identities, receipts, finite values, and shared endpoint population.
25. Run the successor analyzer wrapper exactly once. It may delegate statistical calculations to the frozen shared analyzer only after validating v4.1 contracts. Preserve coverage `>=0.50`, method-invalid rate `<=0.05`, matched-coverage selection, prompt-cluster bootstrap, and the existing task-specific Oral gates.
26. Emit a final canonical decision receipt containing the task results, gate results, all input hashes, source/schema hashes, and the first failed gate if any.
27. Report the protocol history exactly: v4.0.0 development primary analysis was blocked; its outputs are diagnostic; v4.1 is the first complete formal site-certificate protocol; confirmation was one-shot and eligible only under the seal receipt.
28. After any STOP condition, do not modify thresholds, aggregation, radii, directions, P13 rules, graph semantics, resource budget, status mapping, denominator, or confirmation population under `GREEN-V410-SFC-JWTEC-20260830/A1`.
