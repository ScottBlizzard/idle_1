<!-- filename: analysis/GPTPRO_GREEN_V136_TERMINAL_DECISION_20260825.md -->

# GPTPro GREEN v1.3.6 Terminal Scientific and Protocol Decision — 2026-08-25

## Document status

| Field | Binding value |
|---|---|
| Repository | `ScottBlizzard/idle_1` |
| Exact reviewed commit | `3bdeac04a16724461f266705ef250a6357ced1cf` |
| Reviewed branch | `main` |
| Predecessor scientific identity | `green-bridge-v1.3.6` |
| Predecessor outcome | Immutable development `STOP_ORAL` |
| First terminal gate | `12_DEVELOPMENT_SURVIVAL` |
| Confirmation exposure | None |
| Classification | **A + B + C + D, with the scopes defined below** |
| Newly authorized scientific identity | `green-bridge-v2.0.0` |
| New protocol | `structural-envelope-matched-bypass-setid-v2.0.0` |
| New attempt identity | Attempt 1; retry forbidden |
| New GPU execution | **Authorized exactly once, conditional on every prerequisite in this document** |
| Confirmation | Closed until a newly frozen v2.0.0 development result explicitly returns `OPEN_CONFIRMATION` |
| Central theoretical claim | Preserved without weakening |
| Fixed-rank donor PCA | Permanently terminated |
| PIE | Baseline and post hoc diagnostic only; never the central estimator |

---

# 1. Binding executive decision

The v1.3.6 development result is a **valid and terminal scientific failure of the frozen v1.3.6 protocol**. The run completed its authorized computation, produced all expected tensor and energy records, and stopped because no development cell survived the preregistered tensor-admissibility conjunction. The old output root, ledgers, Parquet files, audit files, and `STOP_ORAL` verdict are immutable. They may not be rerun, overwritten, deleted, relabelled, or treated as a technical dry run.

That terminal result is **not a clean falsification of the matched-bypass theorem**. The exact theorem concerns infinitesimal, per-gate derivatives. The executable v1.3.6 certification instead applied finite-radius Richardson estimates and rejected gates using fixed relative tolerances of `0.15` for factorization and `0.05` for white-box agreement. Those two tolerances are not derived from the numerical uncertainty quantities that the same implementation computes and serializes. They are preregistered heuristic tolerances: valid as v1.3.6 rules, but not proof-level consequences of its numerical model.

The correct scientific interpretation is therefore:

1. **A — yes:** v1.3.6 validly failed its frozen development protocol.
2. **B — yes:** the result is evidence of a remaining mismatch between the exact infinitesimal estimand and the finite-radius certification procedure, or of a genuine local structural contradiction that the present two-scale procedure cannot distinguish from finite-radius bias.
3. **C — yes, with a crucial qualification:** requiring every selected gate to be point-identified or certified null is stronger than the per-gate theorem requires. It is sufficient for a point estimate of the complete ten-gate sum, but not necessary for a rigorous set-valued estimate. **All ten gates must still be accounted for.**
4. **D — yes:** a new, outcome-blind, proof-derived partial-identification and robust-aggregation protocol is scientifically warranted.

Exactly one fresh scientific identity, `green-bridge-v2.0.0`, is authorized. It shall:

- preserve the basis-free ambient rank-one matched-bypass operator;
- preserve the exact LayerNorm structural envelope;
- preserve the ten gates, intervention sites, target construction, matched control, contrast, base radii, statistical performance thresholds, and independent energy target;
- replace the unjustified factorization and white-box cutoffs with uncertainty-derived compatibility inequalities;
- add a fixed third dyadic stencil scale;
- distinguish structural contradiction from lack of point identifiability;
- account for unresolved gates through rigorous contribution intervals rather than silently dropping them;
- evaluate the mixed predictor through worst-case interval RMSE and robust interval AUROC;
- use only previously unopened v1.3.6 confirmation cells, deterministically divided into a fresh v2.0.0 development subset and a still-locked v2.0.0 confirmation subset.

The strong raw PIE correlation remains scientifically inadmissible. It may appear only as a clearly labelled exploratory postmortem. It must not determine thresholds, radii, gate classes, aggregation rules, split selection, or confirmation access.

---

# 2. Evidence reviewed

The audit began with:

- `analysis/GPTPRO_GREEN_V136_TERMINAL_HANDOFF_20260825.md`

and independently traced its claims through:

1. `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/TERMINAL_AUDIT.md`
2. `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/terminal_admissibility_audit.json`
3. `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/analyze_terminal.py`
4. `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/dev_tensor_scores.parquet`
5. `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/dev_energy_targets.parquet`
6. `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/dev_cells.json`
7. `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/dev_result.json`
8. `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/development_multigpu_merge.json`
9. `src/exp_green_bridge_gpt2.py`
10. `src/green_bridge_spec.py`
11. `src/green_bridge_numerics.py`
12. `src/matched_bypass_gate.py`
13. every `CODEX_GREEN_V132` through `CODEX_GREEN_V136` decision
14. `analysis/GPTPRO_GREEN_V13_MANUAL_TAIL_DECISION_20260825.md`
15. the binding v1.2 structural-envelope decision
16. the active launcher and cell-level analysis implementation.

The Parquet blobs are authenticated by their committed SHA-256 values, the exact multigpu merge manifest, and the deterministic `analyze_terminal.py` audit whose derived counts, distributions, and correlations agree with `dev_result.json`, `dev_cells.json`, and the terminal handoff. The repository browser exposes the Parquet inputs as binary blobs rather than a native tabular view, so independent row-level verification here rests on the immutable input hashes plus a source-level audit of the committed deterministic analyzer—not on accepting the prose handoff as authority.

The v1.3.2–v1.3.6 decisions were also checked individually. They corrected batch-shape equivalence, anchor recentering, exact batch-one multigpu execution, response-field pairing, and direct-bypass orientation. Their explicit contract was to preserve the scientific payload; none authorized a change to the theorem, selected gates, target, estimator definition, radii, or scientific cutoffs.

---

# 3. Independently verified v1.3.6 terminal facts

## 3.1 Execution and provenance

The reviewed evidence establishes:

- all 168 frozen contract tests passed;
- prepare passed;
- development used eight physical GPUs;
- exact endpoint batch size was one;
- all eight workers completed;
- each worker produced 32 records;
- the merge produced 128 tensor rows and 128 energy rows;
- confirmation was never opened;
- the scientific payload hash remained
  `60ca5e9e221064f288a1993ee3cbf42e99330bbf6f9008946a25556438cbc3d3`;
- the v1.3.6 spec hash was
  `cb771c59e91b4fc553ef73a1c7a116ec0ee55f499ce46a2f91e4c600cd8bd41d`.

The frozen source artifacts are:

| Artifact | SHA-256 |
|---|---|
| `dev_tensor_scores.parquet` | `660788dde8bc5df1d057db31b4dc1065b222ac7777efc0e4c6220e09f1ed81ff` |
| `dev_energy_targets.parquet` | `23a99b6998ec2c51184ae26b8f86a7656247ff2091e251752c1fccd06295e593` |
| `dev_cells.json` | `1294a76d6d79c81f240c20c4257aa6b0fe76457d46b30cfc5d5699e27759ae1f` |
| `dev_result.json` | `2e15531d62bd5cc1162980fdaa2643a7300b362eb6b11ff5b94bb3d623c37277` |
| `development_multigpu_merge.json` | `31dbc71fbeaa40f313be6078a627082050aa5a338e132d3a6ed7343869eaad7a` |

These values agree across the terminal audit and merge evidence.

## 3.2 Exact terminal localization

The failure was not caused by missing records:

| Record class | Produced | Admissible |
|---|---:|---:|
| Energy | 128/128 | 128/128 |
| Tensor | 128/128 | 0/128 |

Every one of the 16 development cells contained eight admissible energy rows and zero admissible tensor rows. Consequently:

```text
outer stop gate   = 12_DEVELOPMENT_SURVIVAL
development verdict = STOP_ORAL
surviving cells   = 0 / 16
conditioned cells = 0
SNR cells         = 0
confirmation      = never opened
```

Because no cell survived, mixed RMSE, best-baseline RMSE, and relative gain were correctly left undefined rather than calculated on inadmissible rows.

## 3.3 Gate-level failure pattern

Across 1,280 gate audits in each system:

| System | Active identified | Certified null | Invalid | All-valid items | Admissible systems |
|---|---:|---:|---:|---:|---:|
| `tar` | 708 | 6 | 566 | 0/128 | 0/128 |
| `pat` | 702 | 7 | 571 | 2/128 | 2/128 |

The code declared a system complete only when every one of the ten selected gates was non-invalid. It then additionally required at least three active gates and common-frame bypass disagreement no greater than `0.15`. Therefore a single invalid gate eliminated the system, and a single inadmissible system eliminated the tensor row.

Among the 1,137 invalid gate audits, the non-exclusive active-criterion failure counts were:

| Criterion | Failure count |
|---|---:|
| White-box agreement | 851 |
| Factorization residual | 823 |
| Curvature SNR | 229 |
| Inferred tensor SNR | 74 |
| Tensor symmetric change | 62 |
| Gate-response SNR | 30 |
| Richardson change | 11 |
| Tensor cosine | 4 |

Factorization and white-box agreement failed together in 635 invalid gates. Their invalid-gate medians were approximately:

| Metric | `pat` median | `tar` median | Frozen cutoff |
|---|---:|---:|---:|
| Factorization residual | `0.18493` | `0.18981` | `0.15` |
| Relative white-box error | `0.07157` | `0.06522` | `0.05` |

Meanwhile, full/half tensor cosine was usually near one and Richardson change was usually small. This rules out a simple “random numerical chaos” explanation. It is consistent with a persistent finite-radius bias, an underestimated numerical enclosure, or an actual estimand/implementation contradiction that requires a sharper local test.

## 3.4 Inadmissible structured signal

The terminal audit reports:

| Diagnostic | Spearman correlation with behavioral target |
|---|---:|
| Raw PIE, item level | `0.612954` |
| Raw PIE, 16-cell means | approximately `0.961765` |
| Raw first-order, 16-cell means | approximately `0.876471` |

The audit itself explicitly warns that all tensor rows are protocol-inadmissible and that these summaries cannot support a scientific claim.

This signal is useful only for one conclusion:

> The experiment did not reveal an absence of structure.

It cannot support any of the following:

- that PIE is the correct estimator;
- that PIE should replace the matched-bypass estimator;
- that the v1.3.6 thresholds should be relaxed;
- that confirmation should open;
- that the raw correlation estimates generalization;
- that any v2.0.0 rule should be selected because it retrospectively admits these rows.

---

# 4. Exact theory and estimand audit

## 4.1 The preserved theorem

For a gate \(j\), system \(s\), output response \(Y\), residual perturbation \(x\), and gate perturbation \(z\), define at the center:

\[
G_{sj}=\partial_z Y^P_{sj}(0,0),
\]

\[
C_{sj}=\partial_z^2Y^P_{sj}(0,0),
\]

\[
H^P_{sj}(u)
=
D_x\partial_zY^P_{sj}(0,0)[u],
\]

\[
H^C_{sj}(u)
=
D_x\partial_zY^C_{sj}(0,0)[u].
\]

The matched-bypass difference is

\[
\Delta H_{sj}(u)
=
H^P_{sj}(u)-H^C_{sj}(u).
\]

The structural theorem gives

\[
\boxed{
\Delta H_{sj}(u)
=
C_{sj}\langle g_{sj},u\rangle
}
\]

where \(g_{sj}\) is the ambient residual-space gradient of the selected gate preactivation.

For a structural frame

\[
Q_{sj}=[q_1,\ldots,q_5],
\]

define

\[
A_{sj,i}
=
\langle g_{sj},q_i\rangle.
\]

Whenever \(C_{sj}\neq0\),

\[
\boxed{
A_{sj,i}
=
\frac{
\langle C_{sj},\Delta H_{sj}(q_i)\rangle
}{
\|C_{sj}\|_2^2
}.
}
\]

The frame representation of the path operator is

\[
P^{Q}_{sj}
=
A_{sj}G_{sj}^{\top},
\]

and the basis-free ambient operator is

\[
\boxed{
\mathcal P_{sj}
=
G_{sj}g_{sj}^{\top}.
}
\]

The exact LayerNorm structural envelope establishes that \(g_{sj}\) lies in the architecture-derived five-vector frame, up to the frozen numerical structural residual. Thus the frame probes are complete without donor PCA.

## 4.2 What v1.3.6 actually estimated

For each gate, v1.3.6 evaluated central finite differences at a base radius and at one half of that radius, then formed

\[
\widehat T_R
=
\frac{4\widehat T_{h/2}-\widehat T_h}{3}
\]

for each derivative object

\[
T\in\{G,C,J,H^P,H^C\}.
\]

It then computed

\[
\widehat{\Delta H}
=
\widehat H^P-\widehat H^C,
\]

\[
\widehat A_i
=
\frac{
\langle\widehat C,\widehat{\Delta H}_i\rangle
}{
\|\widehat C\|_2^2
},
\]

and

\[
\widehat P
=
\widehat A\widehat G^\top.
\]

This point estimator is the correct finite-difference implementation of the theorem-derived estimator. `identify_gate` itself is not the defect. It performs the specified rank-one identification and computes the factorization residual of the estimated mixed-response tensor.

The scientific problem lies in how the finite estimate was certified.

---

# 5. Why the `0.15` and `0.05` rules are not mathematically justified

## 5.1 What the numerical model actually provides

The active numerical module computes absolute uncertainty quantities:

\[
\epsilon_G,\qquad
\epsilon_C,\qquad
\epsilon_J,\qquad
\epsilon_{\Delta H,i},
\]

and then derives:

\[
A_{\max,i},
\qquad
\epsilon_{A,i},
\qquad
\epsilon_{P,i},
\qquad
\epsilon_{P,F}.
\]

These quantities depend on:

- the duplicated-logit error floor \(\epsilon_y\);
- the residual and gate stencil radii;
- rich-versus-half discrepancies;
- the dimensions of the output and structural frame;
- exact error propagation through the inverse based on \(\|\widehat C\|-\epsilon_C\).

In particular, the inverse is deemed admissible only if

\[
\|\widehat C\|_2>\epsilon_C,
\]

and the code derives an upper bound on the true coordinate magnitude and an absolute error bound for every identified coordinate.

## 5.2 What the classifier does instead

The classifier separately requires

\[
R_{\mathrm{fac}}
\le 0.15
\]

and

\[
\|\widehat A-A^{WB}\|_2
\le
0.05\max(\|A^{WB}\|_2,10^{-6}),
\]

with a separate absolute rule when the white-box norm is very small. It does not derive either cutoff from \(\epsilon_C\), \(\epsilon_{\Delta H}\), or \(\epsilon_A\).

That disconnect is mathematically decisive.

A fixed relative factorization tolerance can be:

- too strict when \(\|\Delta H\|\) is small but the absolute residual is fully explained by numerical uncertainty;
- too loose when \(\|\Delta H\|\) is large and the residual substantially exceeds the propagated numerical error.

Likewise, a fixed five-percent white-box tolerance can be:

- arbitrarily strict near a small white-box coordinate;
- arbitrarily permissive for a large coordinate;
- unrelated to the response-derived uncertainty \(\epsilon_A\).

There is no theorem or numerical derivation in the reviewed code that turns the existing error quantities into the constants `0.15` and `0.05`.

## 5.3 Binding threshold verdict

The frozen values:

```text
factorization_residual_max = 0.15
whitebox_a_relative_max = 0.05
whitebox_a_small_absolute_max = 1e-4
```

were valid preregistered v1.3.6 rules. They remain part of the historical protocol and explain its valid STOP.

They are **not** proof-level tolerances and may not be reused as active v2.0.0 identity tests.

They shall be replaced by compatibility inequalities whose right-hand sides are derived from the numerical error envelope. The new normalized threshold is exactly:

\[
\boxed{1.0}
\]

because the test asks whether an observed discrepancy is inside or outside a derived uncertainty bound—not whether it is within a retrospectively convenient percentage.

---

# 6. Binding classification of A–D

| Question | Binding answer | Scope |
|---|---|---|
| **A. Valid scientific failure?** | **Yes** | v1.3.6 failed its own frozen development protocol and remains terminal. |
| **B. Mathematical or estimand mismatch?** | **Yes, evidence of one** | The exact local theorem was certified through finite-radius estimates plus heuristic tolerances. The evidence does not distinguish finite-radius bias from a genuine local contradiction. |
| **C. All-ten completeness unnecessarily strong?** | **Yes for point completeness; no for accounting** | The theorem is per-gate. Every gate need not be point-identified, but every gate’s contribution must be represented by a point estimate, a certified null, or a rigorous interval. |
| **D. New proof-valid protocol needed?** | **Yes** | A bound-certified three-scale response estimator, explicit contradiction class, partial identification, and robust interval aggregation are warranted. |

The answer is therefore not “A or B or C or D.” It is:

\[
\boxed{
\text{A is the terminal protocol result; B, C, and D are the scientific diagnosis.}
}
\]

---

# 7. Why all-ten point completeness is stronger than the theorem

The theorem identifies one gate at a time. It does not state that all ten selected gates must have nonzero curvature or sufficiently high numerical SNR on every item.

For the full selected-gate predictor,

\[
\Theta_s
=
\sum_{j=1}^{10}\theta_{sj},
\]

a point estimate of \(\Theta_s\) requires every term to be either:

- point-identified; or
- proven negligible.

The v1.3.6 rule implemented precisely that sufficient condition: every gate had to be active-identified or certified target-null. Any other state was called `invalid`, and one invalid gate eliminated the whole system.

This is stronger than necessary. Suppose one gate has curvature too small to invert reliably. The theorem does not imply that the gate’s contribution is zero. It implies only that the response-based coordinate is not point-identifiable from that mixed derivative.

A rigorous alternative is:

\[
\theta_j\in[-U_j,U_j]
\]

for an outcome-independent upper bound \(U_j\).

Then the complete sum is set-identified:

\[
\Theta_s
\in
\sum_{j=1}^{10}I_{sj},
\]

where \(I_{sj}\) is:

- a centered uncertainty interval for an active identified gate;
- a narrow null interval for a certified null gate;
- a wider symmetric interval for an unresolved but bounded gate.

This preserves all-ten accounting. It does not omit any gate, replace it by an unjustified zero, or use white-box information as a signed point estimate.

---

# 8. Authorization of a new scientific protocol

## 8.1 New identity

The new protocol shall use:

```text
SCHEMA_VERSION
    green-bridge-v2.0.0

PROTOCOL_ID
    structural-envelope-matched-bypass-setid-v2.0.0

PARENT_PROTOCOL_ID
    structural-envelope-matched-bypass-v1.3.6

DECISION_ID
    GPTPRO-GREEN-V136-TERMINAL-SETID-v1-20260825

PROTOCOL_RUN_ID
    green-bridge-v2.0.0-one-shot

OUTPUT_ROOT
    outputs/green_bridge_v200

ATTEMPT_INDEX
    1

RETRY_ALLOWED
    false

PHASE_ALL_ALLOWED
    false
```

Associated schemas shall be:

```text
green-bridge-manifest-v2.0.0
green-bridge-terminal-v2.0.0
green-bridge-prepare-v2.0.0
green-bridge-development-v2.0.0
green-bridge-confirmation-v2.0.0
green-bridge-frozen-analysis-v2.0.0
green-bridge-cell-setid-v2.0.0
```

This is not:

- v1.3.6 attempt 2;
- v1.3.7;
- a retry of the old root;
- a threshold amendment to the old data;
- a retrospective reanalysis that can open the old confirmation split.

The major version is intentional because the scientific admissibility and aggregation rules change.

## 8.2 Scientific invariants

The following remain unchanged:

- model and model revision;
- tokenizer;
- frozen TransformerLens source;
- float32 scientific forward execution;
- exact endpoint batch size one;
- ten selected MLP gates;
- block-8 patch site;
- block-10 residual intervention site;
- block-10 gate preactivation intervention site;
- path, control, and joint semantics;
- final raw-logit endpoint;
- exact LayerNorm structural-frame construction;
- common frame dimension \(4\);
- per-gate frame dimension \(5\);
- all-gate frame dimension \(14\);
- base residual radius;
- base gate radius `0.20`;
- physical target vectors;
- contrast vectors;
- residual-bypass subtraction in the independent target;
- first-order and PIE baseline definitions;
- energy-target definition and thresholds;
- materiality floors;
- SNR threshold `20`;
- active-gate minimum `3`;
- null contribution ceiling `0.005`;
- common-bypass disagreement ceiling `0.15`;
- nonnegative affine baseline calibration;
- development performance thresholds;
- confirmation performance thresholds;
- 100,000-replicate stratified bootstrap;
- bootstrap seed `20260805`;
- no pseudoinverse;
- no ridge;
- no gate replacement;
- no donor PCA.

## 8.3 Authorized scientific changes

Only these scientific changes are authorized:

1. add a fixed quarter-radius GateJet evaluation;
2. make the fine Richardson estimator the immutable point estimator;
3. derive factorization and white-box admissibility from propagated numerical bounds;
4. replace heuristic gate-level scale metrics with uncertainty-ball overlap;
5. add `unresolved-bounded` and `structural-contradiction` gate classes;
6. replace all-ten point completeness with all-ten set accounting;
7. aggregate gate intervals into item and cell intervals;
8. use worst-case interval RMSE and interval AUROC lower bounds;
9. use a newly frozen development/confirmation split drawn only from the previously unopened v1.3.6 confirmation population;
10. scale cell-count thresholds by their original frozen proportions, rounded upward.

No other scientific delta is authorized.

---

# 9. Predecessor immutability and contamination firewall

## 9.1 Immutable predecessor

The following must remain byte-for-byte unchanged:

```text
outputs/green_bridge_v136/
analysis/GREEN_V136_TERMINAL_AUDIT_20260825/
analysis/GPTPRO_GREEN_V136_TERMINAL_HANDOFF_20260825.md
```

The v2.0.0 runner must verify the five SHA-256 values in Section 3.1 before creating the v2.0.0 root.

It must also verify:

```text
v1.3.6 verdict == STOP_ORAL
v1.3.6 first terminal gate == 12_DEVELOPMENT_SURVIVAL
v1.3.6 tensor count == 128
v1.3.6 energy count == 128
v1.3.6 surviving cells == 0
v1.3.6 confirmation opened == false
v1.3.6 scientific payload hash ==
    60ca5e9e221064f288a1993ee3cbf42e99330bbf6f9008946a25556438cbc3d3
v1.3.6 spec hash ==
    cb771c59e91b4fc553ef73a1c7a116ec0ee55f499ce46a2f91e4c600cd8bd41d
```

Any mismatch is terminal:

```text
STOP 00_PREDECESSOR_IMMUTABILITY
```

## 9.2 Forbidden predecessor inputs

The v2.0.0 scientific runner, classifier, aggregator, and analysis code must not load:

```text
outputs/green_bridge_v136/dev_tensor_scores.parquet
outputs/green_bridge_v136/dev_energy_targets.parquet
outputs/green_bridge_v136/dev_cells.json
analysis/GREEN_V136_TERMINAL_AUDIT_20260825/terminal_admissibility_audit.json
```

They may be read only by the predecessor hash verifier.

The v1.3.6 raw behavior, PIE values, factorization values, and white-box values may not be passed into any v2.0.0 threshold, radius, split, calibration, or development decision.

Synthetic unit tests must be used for the new rules. A retrospective “would v1.3.6 pass v2.0.0?” analysis is forbidden before the v2.0.0 protocol terminates.

---

# 10. Fresh v2.0.0 development and confirmation split

The original v1.3.6 development cells are outcome-exposed and excluded from all v2.0.0 inferential phases.

The 32 previously unopened v1.3.6 confirmation cells comprise 16 noun-century groups, each with both `near` and `far` cells. They shall be divided at the noun-century group level so that the two distance bins never cross phase boundaries.

## 10.1 Frozen ranking rule

For every original confirmation noun-century group, compute:

```python
sha256(
    f"green-v200-resplit-20260825|{noun}|{century:02d}".encode("utf-8")
).hexdigest()
```

Sort lexicographically by this digest.

The first four groups are v2.0.0 development. The remaining twelve groups are v2.0.0 confirmation.

## 10.2 Exact development groups

| Rank | Noun | Century | SHA-256 rank key |
|---:|---|---:|---|
| 1 | `dynasty` | 16 | `066e4d0fbd2636a5de7c5587fea60ba6d83c2173fd8a1a3b9598806973ed2596` |
| 2 | `dynasty` | 12 | `0fe1884c7d56deb8cdcb34d7b4eea65b398a9fdf03fc638f8bbc4a422c6ff6b6` |
| 3 | `reign` | 14 | `169c8a45b7aea24c90ce94ecefc84aa0588e34831b5074b9a57a4c5380373b51` |
| 4 | `warfare` | 14 | `36d63a7d439059d0877995705989132fddb35d1cfc9381be110925cacc8776c4` |

These four groups produce:

```text
8 development cells
64 development tensor records
64 development energy records
```

## 10.3 Exact confirmation groups

| Noun | Century | SHA-256 rank key |
|---|---:|---|
| `treaty` | 12 | `5419d9cb8844c61db83ae2eae7243dbd16a9c2bf5ee7967401eafd7f70f2475a` |
| `warfare` | 12 | `57822f1c018d9552848007996257f81da49ebef54f6e4559dc84fe13312ed2b4` |
| `expedition` | 14 | `5f5f6555263c3ee9052d9f1240096f6004091201fdd805b4ede1769481fcc321` |
| `kingdom` | 12 | `6c27075f448a87bd7bdb373924e72caba1816a5033625dbe92d1c59d1977dae8` |
| `treaty` | 16 | `6c8d9da9bd864657ac675f5b68f65e22f44ac97b0ccec1a4acfa8987a513fb77` |
| `kingdom` | 16 | `8571c8283f76806da63c769868b6a34448f6f02ae86d57f8a13db6597cecde00` |
| `campaign` | 14 | `9942a20d23a6fb97e7f33390172c7049ebe78341bde1b047a28ae64e997d431b` |
| `siege` | 16 | `a19e2bc49bf4b522ae28f500cd6596f5c492e8f817008f0c5985341e55c45741` |
| `reign` | 12 | `aa4cd1c743ab745f1278738367fcdd5a3937d36082f92b10aa3144c990001af4` |
| `siege` | 14 | `c39c88f5f37a424b7196cf99a4d34062f6150740ede6ab7c97abdd36d2d76d01` |
| `campaign` | 16 | `e1d35b6e9b3ec70687d8ed270afec74fc565c633b0d0a43011d507323df4f939` |
| `expedition` | 16 | `f7fcfdc5e4306cca1d4b0309c086dc0d6e033b72ce1236d8bb6d1986c362351f` |

These groups produce:

```text
24 confirmation cells
192 confirmation tensor records
192 confirmation energy records
```

## 10.4 Canonical split hash

Canonical JSON must be encoded with:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
```

with schema:

```text
green-bridge-v2.0.0-resplit-v1
```

and the exact fields:

```text
schema
salt
source_split
development_groups
confirmation_groups
distance_bins
roles
records_per_role_per_cell
```

The required SHA-256 is:

```text
f012a286801bc3e3e937b390f0a62d7e92f8d5a21ba59d7e53478ae911e72cfc
```

Any mismatch is terminal:

```text
STOP 07_SPLIT_FIREWALL
```

---

# 11. New fixed three-scale response estimator

## 11.1 Frozen scales

For every gate and item, retain the existing base radii:

\[
h_x,\qquad h_z=0.20.
\]

Evaluate GateJets at exactly:

\[
\rho_0=1,\qquad
\rho_1=\frac12,\qquad
\rho_2=\frac14.
\]

Thus:

\[
(h_x,h_z),
\quad
\left(\frac{h_x}{2},\frac{h_z}{2}\right),
\quad
\left(\frac{h_x}{4},\frac{h_z}{4}\right).
\]

No alternative radius, radius sweep, per-gate search, or retry radius is allowed.

## 11.2 Coarse and fine Richardson estimates

For every derivative object

\[
T\in\{G,C,J,H^P,H^C\},
\]

define:

\[
R^{(c)}_T
=
\frac{4T_{\rho_1}-T_{\rho_0}}{3},
\]

\[
R^{(f)}_T
=
\frac{4T_{\rho_2}-T_{\rho_1}}{3}.
\]

The scientific point estimator is always:

\[
\boxed{
\widehat T=R^{(f)}_T.
}
\]

The executor may not select the coarse or fine estimator based on:

- factorization outcome;
- white-box outcome;
- behavior;
- PIE;
- baseline performance;
- survival rate;
- which candidate makes a gate pass.

The coarse estimate exists solely for numerical enclosure and radius-stability auditing.

## 11.3 Coarse and fine numerical balls

Generalize the existing frozen Richardson error propagation so that it can be called separately on:

```text
coarse pair = base, half
fine pair   = half, quarter
```

For the fine pair, call the same endpoint-noise formulas using:

\[
h_x^{(f)}=\frac{h_x}{2},
\qquad
h_z^{(f)}=\frac{h_z}{2}.
\]

The current endpoint-noise coefficients remain unchanged:

\[
\eta_G=\frac{3\epsilon_y}{h_z},
\]

\[
\eta_C=\frac{64\epsilon_y}{3h_z^2},
\]

\[
\eta_J=\frac{3\epsilon_y}{h_x},
\]

\[
\eta_H=\frac{17\epsilon_y}{3h_xh_z}.
\]

The duplicate-output floor remains:

\[
\epsilon_y\ge10^{-7}.
\]

For each coarse and fine pair, compute the existing rich-versus-small-scale discrepancies plus the corresponding endpoint-noise terms.

Extend the numerical structure to include:

\[
\epsilon_J
\]

in addition to the existing:

\[
\epsilon_G,\epsilon_C,\epsilon_{\Delta H},\epsilon_A,
\epsilon_P,\epsilon_{P,F}.
\]

## 11.4 Dyadic overlap gate

For every derivative object, the coarse and fine uncertainty balls must overlap.

For vectors:

\[
\|R^{(f)}_T-R^{(c)}_T\|_2
\le
\epsilon_T^{(f)}+\epsilon_T^{(c)}.
\]

For \(J\):

\[
\|R^{(f)}_J-R^{(c)}_J\|_F
\le
\epsilon_J^{(f)}+\epsilon_J^{(c)}.
\]

For every row \(i\) of \(\Delta H\):

\[
\left\|
R^{(f)}_{\Delta H,i}
-
R^{(c)}_{\Delta H,i}
\right\|_2
\le
\epsilon_{\Delta H,i}^{(f)}
+
\epsilon_{\Delta H,i}^{(c)}.
\]

No cosine or symmetric-percent threshold substitutes for this condition.

Failure means:

```text
gate label = numerical-invalid
```

and cannot be converted to `unresolved-bounded`.

## 11.5 Final downstream uncertainty

For each object centered on the fine estimate, use the tighter of two valid bounds:

\[
\boxed{
\bar\epsilon_T
=
\min\left(
\epsilon_T^{(f)},
\;
\|R^{(f)}_T-R^{(c)}_T\|+\epsilon_T^{(c)}
\right).
}
\]

Use the corresponding norm for vectors, matrices, and rowwise mixed derivatives.

This follows directly from:

\[
\|T-R^{(f)}_T\|
\le\epsilon_T^{(f)}
\]

and

\[
\|T-R^{(f)}_T\|
\le
\|T-R^{(c)}_T\|
+
\|R^{(c)}_T-R^{(f)}_T\|.
\]

No behavior enters this choice.

---

# 12. Prepare-only automatic-differentiation enclosure audit

The existing Richardson propagation is an explicit numerical model, not a universal theorem about every smooth function. v2.0.0 therefore adds an independent prepare-only falsification audit.

## 12.1 Audit population

Construct 40 strata:

```text
2 systems × 10 gate slots × 2 distance bins = 40
```

Use only metadata from the outcome-exposed v1.3.6 development population.

For each stratum, select the record with minimum:

```python
sha256(
    f"green-v200-ad-audit|{pair_digest}|{system}|{gate_slot}|{distance_bin}"
    .encode("utf-8")
).hexdigest()
```

The selector may read:

- pair digests;
- prompts;
- system labels;
- gate slots;
- distance-bin labels.

It may not read:

- behavioral values;
- PIE;
- first-order values;
- tensor admissibility;
- gate labels;
- factorization residuals;
- white-box residuals;
- development cell outcomes.

## 12.2 Audit computation

Add a differentiable, prepare-only tail implementation that:

- executes the same path and control response maps;
- uses the same stored model weights and anchors;
- uses the exact full unembedding endpoint;
- casts the local tail and exact float32-stored parameters to float64 solely for this numerical audit;
- computes \(G,C,J,H^P,H^C\) at the center;
- uses two independent derivative routes:
  - forward-over-forward;
  - reverse-over-forward.

The float64 audit is not the scientific point estimator and cannot be serialized into development tensor scores.

For every audited derivative object, require:

1. the two AD routes are mutually compatible under their outward-rounded discrepancy bound;
2. the AD value lies inside the coarse Richardson ball;
3. the AD value lies inside the fine Richardson ball;
4. where curvature is invertible, the AD mixed derivative satisfies the matched-bypass factorization and white-box identity under the same bound-derived compatibility rules used below.

Any miss is:

```text
STOP 08_AD_ENCLOSURE
```

There is no inflation factor, percentile, pass-rate threshold, retraining, radius change, or second audit panel.

All 40 strata must pass or be mathematically non-invertible without contradiction.

---

# 13. New factorization compatibility rule

Let the fine point estimates be:

\[
\widehat C,
\qquad
\widehat{\Delta H}_i,
\qquad
\widehat A_i.
\]

Let the final numerical bounds satisfy:

\[
\|C-\widehat C\|_2\le\bar\epsilon_C,
\]

\[
\|\Delta H_i-\widehat{\Delta H}_i\|_2
\le
\bar\epsilon_{\Delta H,i},
\]

\[
|A_i-\widehat A_i|
\le
\bar\epsilon_{A,i},
\]

\[
|A_i|\le A_{\max,i}.
\]

Under the theorem,

\[
\Delta H_i=A_iC.
\]

Therefore:

\[
\widehat{\Delta H}_i-\widehat A_i\widehat C
=
(\widehat{\Delta H}_i-\Delta H_i)
+
(A_i-\widehat A_i)\widehat C
+
A_i(C-\widehat C).
\]

By the triangle inequality:

\[
\boxed{
\left\|
\widehat{\Delta H}_i-\widehat A_i\widehat C
\right\|_2
\le
B^{\mathrm{fac}}_i
}
\]

where

\[
\boxed{
B^{\mathrm{fac}}_i
=
\bar\epsilon_{\Delta H,i}
+
\bar\epsilon_{A,i}\|\widehat C\|_2
+
A_{\max,i}\bar\epsilon_C.
}
\]

Define:

\[
R^{\mathrm{fac}}_i
=
\begin{cases}
0,&q_i=0,\ B_i=0,\\
+\infty,&q_i>0,\ B_i=0,\\
q_i/B_i,&B_i>0,
\end{cases}
\]

where

\[
q_i=
\left\|
\widehat{\Delta H}_i-\widehat A_i\widehat C
\right\|_2.
\]

The gate passes factorization compatibility iff:

\[
\boxed{
\max_iR^{\mathrm{fac}}_i\le1.
}
\]

All scalar comparisons use:

```python
residual <= math.nextafter(bound, math.inf)
```

There is no `0.15` factorization threshold in active v2.0.0 code.

---

# 14. New white-box compatibility rules

## 14.1 White-box role

The white-box structural gradient remains:

- an audit of response identification;
- a source of absolute upper bounds for unresolved gates.

It remains forbidden as:

- the signed point estimator;
- a replacement for response-derived \(\widehat A\);
- a replacement for the matched-bypass mixed response;
- a way to fill an unidentified point coordinate.

## 14.2 Coordinate error envelope

For every item, gate, and system, compute both:

- analytic LayerNorm gate-gradient coordinates;
- float64 autograd coordinates.

Require:

```text
max coordinate absolute difference <= 1e-10
```

and the existing structural-envelope and shift-null preflight bounds.

Set:

\[
\epsilon_{WB,i}=10^{-10}
\]

for every structural-frame coordinate.

If the formula/autograd difference exceeds `1e-10`, the item is structurally invalid. The threshold may not be inflated from the observation.

## 14.3 Response-versus-white-box compatibility

The theorem gives:

\[
A_i=A^{WB}_i.
\]

Therefore:

\[
\boxed{
|\widehat A_i-A^{WB}_i|
\le
\bar\epsilon_{A,i}+\epsilon_{WB,i}.
}
\]

Define the coordinatewise ratio:

\[
R^{WB}_i
=
\frac{
|\widehat A_i-A^{WB}_i|
}{
\bar\epsilon_{A,i}+\epsilon_{WB,i}
}
\]

with the same exact zero-denominator branch used for factorization.

The gate passes iff:

\[
\boxed{
\max_iR^{WB}_i\le1.
}
\]

There is no five-percent relative rule and no separate small-norm `1e-4` exception.

## 14.4 Direct white-box factorization identity

Also require:

\[
\boxed{
\left\|
\widehat{\Delta H}_i
-
A^{WB}_i\widehat C
\right\|_2
\le
B^{WB\text{-}\mathrm{fac}}_i
}
\]

where

\[
\boxed{
B^{WB\text{-}\mathrm{fac}}_i
=
\bar\epsilon_{\Delta H,i}
+
|A^{WB}_i|\bar\epsilon_C
+
\epsilon_{WB,i}
\left(
\|\widehat C\|_2+\bar\epsilon_C
\right).
}
\]

This is an independent theorem-level check. It does not use the response-derived \(\widehat A\) on the right-hand side.

## 14.5 Shift-null coordinate

Replace the old rule

```text
max(1e-4, 5 epsilon_A)
```

with:

\[
\boxed{
|\widehat A_{\mathrm{shift}}|
\le
\bar\epsilon_{A,\mathrm{shift}}
+
\epsilon_{WB,\mathrm{shift}}.
}
\]

Failure under an admissible inverse is a structural contradiction.

---

# 15. v2.0.0 gate classes

Every selected gate receives exactly one of four labels.

## 15.1 `active-identified`

A gate is `active-identified` iff all of the following hold:

1. center no-op audit passes;
2. all coarse/fine derivative balls overlap;
3. the fine curvature inverse is admissible;
4. curvature RMS is at least `5e-4`;
5. curvature norm is at least `20 × epsilon_C`;
6. gate-response RMS is at least `5e-4`;
7. gate-response norm is at least `20 × epsilon_G`;
8. factorization compatibility ratio is at most `1`;
9. response-versus-white-box ratio is at most `1`;
10. direct white-box factorization ratio is at most `1`;
11. shift-null compatibility passes;
12. operator norm is at least `20 × epsilon_P_F`;
13. every value is finite.

The point operator remains:

\[
\widehat{\mathcal P}_j
=
\widehat G_j
\widehat g_j^\top,
\qquad
\widehat g_j=Q_j\widehat A_j.
\]

## 15.2 `certified-target-null`

A gate is `certified-target-null` iff:

1. all numerical jet overlap audits pass;
2. no structural contradiction has been observed;
3. the response upper envelope satisfies the existing low-response rule;
4. its full target-contribution upper bound is at most:

\[
0.005.
\]

Its point contribution is zero and its uncertainty interval is its certified null bound.

## 15.3 `unresolved-bounded`

A gate is `unresolved-bounded` iff:

1. all numerical jet overlap audits pass;
2. no factorization or white-box contradiction is observed in a regime where the inverse is admissible;
3. the gate is not active-identified;
4. the gate is not certified null;
5. a finite structural upper bound on its target contribution exists.

Typical reasons include:

- curvature inverse not admissible;
- curvature materiality below the floor;
- response materiality below the floor;
- operator SNR below `20`;
- response coordinates not point-identifiable.

An unresolved gate is not called null.

## 15.4 `structural-contradiction` or `numerical-invalid`

A gate is invalid if any of these occurs:

- coarse/fine numerical balls do not overlap;
- an AD prepare enclosure fails;
- the inverse is admissible but factorization exceeds its derived bound;
- the inverse is admissible but white-box agreement exceeds its derived bound;
- direct white-box factorization exceeds its derived bound;
- shift-null identity exceeds its derived bound;
- center, frame, hook, or endpoint audits fail;
- any required quantity is nonfinite.

A contradictory gate may not be converted to `unresolved-bounded`.

---

# 16. Rigorous unresolved-gate contribution bound

Let:

- \(\ell\) be the frozen output contrast;
- \(v\) be the frozen physical residual target;
- \(\widehat G_j\) be the fine response estimate;
- \(\bar\epsilon_{G,j}\) be its uncertainty;
- \(g^{WB}_j\) be the analytic ambient gate gradient.

The true contribution is:

\[
\theta_j
=
(\ell^\top G_j)(g_j^\top v).
\]

The white-box gradient is used only to bound the second factor.

Set:

\[
\epsilon_{g,WB}
=
\sqrt{768}\times10^{-10}.
\]

Then:

\[
|\ell^\top G_j|
\le
|\ell^\top\widehat G_j|
+
\|\ell\|_2\bar\epsilon_{G,j},
\]

and

\[
|g_j^\top v|
\le
|(g_j^{WB})^\top v|
+
\epsilon_{g,WB}\|v\|_2.
\]

Therefore:

\[
\boxed{
U_j
=
\left(
|\ell^\top\widehat G_j|
+
\|\ell\|_2\bar\epsilon_{G,j}
\right)
\left(
|(g_j^{WB})^\top v|
+
\epsilon_{g,WB}\|v\|_2
\right).
}
\]

An unresolved gate contributes the interval:

\[
\boxed{
I_j=[-U_j,U_j].
}
\]

The signed center is exactly zero. The white-box sign is not used.

---

# 17. Active-gate interval

For an active gate, use the existing structural-envelope contraction-error derivation with the v2.0.0 fine uncertainty:

\[
\widehat\theta_j
=
\ell^\top
\widehat{\mathcal P}_jv.
\]

Let \(B_j\) be the existing active contraction bound computed from:

- \(\|\ell\|_2\);
- \(\|Q_j^\top v\|_2\);
- \(\|v\|_2\);
- \(\bar\epsilon_{P,F,j}\);
- \(\|\widehat G_j\|_2\);
- \(\bar\epsilon_{G,j}\);
- the exact structural-envelope residual.

Then:

\[
\boxed{
I_j
=
[\widehat\theta_j-B_j,\widehat\theta_j+B_j].
}
\]

No point estimate is retained unless its complete error interval is serialized.

---

# 18. All-ten system accounting

For each system:

\[
\widehat\Theta_s
=
\sum_{j:\mathrm{active}}
\widehat\theta_{sj}.
\]

Define:

\[
B_s
=
\sum_{j:\mathrm{active}}B_{sj}
+
\sum_{j:\mathrm{null}}U_{sj}
+
\sum_{j:\mathrm{unresolved}}U_{sj}.
\]

Then:

\[
\boxed{
I_s
=
[\widehat\Theta_s-B_s,\widehat\Theta_s+B_s].
}
\]

A system is `set-admissible` iff:

1. all ten gates are one of:
   - `active-identified`;
   - `certified-target-null`;
   - `unresolved-bounded`;
2. no gate is contradictory or numerically invalid;
3. at least three gates are active-identified;
4. the existing common-frame bypass-disagreement rule remains at most `0.15`;
5. the system interval is finite.

A system is separately marked `point-complete` iff every gate is active or certified null.

The primary v2.0.0 analysis uses `set-admissible`, not `point-complete`.

---

# 19. Item and cell interval construction

## 19.1 Item interval

For one tensor item:

\[
I_{\mathrm{tar}}
=
[L_{\mathrm{tar}},U_{\mathrm{tar}}],
\]

\[
I_{\mathrm{pat}}
=
[L_{\mathrm{pat}},U_{\mathrm{pat}}].
\]

The signed system-difference interval is:

\[
I_\Delta
=
[
L_{\mathrm{pat}}-U_{\mathrm{tar}},
\;
U_{\mathrm{pat}}-L_{\mathrm{tar}}
].
\]

The mixed magnitude interval is:

\[
I_M=
\begin{cases}
[0,\max(|L_\Delta|,|U_\Delta|)],&
0\in I_\Delta,\\[4pt]
[\min(|L_\Delta|,|U_\Delta|),\max(|L_\Delta|,|U_\Delta|)],&
0\notin I_\Delta.
\end{cases}
\]

## 19.2 Cell aggregation order

Preserve the current scientific aggregation order:

1. average signed `pat` system contributions over tensor rows;
2. average signed `tar` system contributions over tensor rows;
3. subtract those means;
4. take the absolute value.

For intervals, use the corresponding Minkowski operations:

\[
\bar I_{s,c}
=
\left[
\frac1n\sum_iL_{s,i},
\;
\frac1n\sum_iU_{s,i}
\right],
\]

then form the difference and magnitude interval.

This preserves the current `abs(mean(theta_pat)-mean(theta_tar))` estimand rather than changing it to a mean of per-item absolute effects. The existing implementation uses exactly that order.

A cell survives iff:

```text
set-admissible tensor rows >= 6
admissible energy rows       >= 6
```

## 19.3 Cell set-identification SNR

For a cell interval

\[
I_{M,c}=[L_c,U_c],
\]

define:

\[
m_c=\frac{L_c+U_c}{2},
\qquad
r_c=\frac{U_c-L_c}{2},
\]

\[
\boxed{
\mathrm{SNR}^{set}_c
=
\frac{m_c}{\max(r_c,10^{-8})}.
}
\]

The development SNR threshold remains `3`.

---

# 20. Robust performance analysis

## 20.1 Worst-case interval RMSE

For target \(y_c\) and predictor interval \([L_c,U_c]\), the maximum possible squared error is:

\[
e_{c,\max}^2
=
\max\left(
(y_c-L_c)^2,
(y_c-U_c)^2
\right).
\]

Define:

\[
\boxed{
\mathrm{RMSE}^{worst}_{mixed}
=
\sqrt{
\frac1N
\sum_ce_{c,\max}^2
}.
}
\]

This is the active mixed-predictor error in development and confirmation.

The interval midpoint RMSE may be serialized only as a diagnostic.

## 20.2 Baselines

The baseline set remains:

```text
behavioral
single
first_order
pie
```

The current nonnegative affine calibration and development leave-one-cell-out fitting remain unchanged.

PIE remains one baseline among four. It receives no privileged calibration, threshold, split, or selection rule.

## 20.3 Robust relative gain

Define:

\[
\boxed{
\mathrm{gain}
=
1-
\frac{
\mathrm{RMSE}^{worst}_{mixed}
}{
\mathrm{RMSE}_{best\ baseline}
}.
}
\]

The best baseline is selected by the same frozen development LOOCV rule.

## 20.4 Robust cancellation AUROC lower bound

For a positive cell \(p\) with interval \([L_p,U_p]\) and a negative cell \(n\) with interval \([L_n,U_n]\), the pair is certainly correctly ranked only if:

\[
L_p>U_n.
\]

Define:

\[
\boxed{
\mathrm{AUC}_{LB}
=
\frac1{N_+N_-}
\sum_{p,n}
\left[
\mathbf1(L_p>U_n)
+
\frac12\mathbf1(L_p=U_n)
\right].
}
\]

Use this lower-bound AUC, not the interval midpoint AUC, for the active confirmation criterion and its stratified bootstrap.

---

# 21. Frozen v2.0.0 thresholds

## 21.1 Retired active thresholds

These v1.3.6 thresholds remain historical metadata but are inactive in v2.0.0:

```text
factorization_residual_max       = 0.15
whitebox_a_relative_max          = 0.05
whitebox_a_small_absolute_max    = 1e-4
tensor_cosine_min                = 0.95
tensor_symmetric_change_max      = 0.25
richardson_change_max            = 0.25
```

The last three may be serialized as diagnostics, but gate certification uses uncertainty-ball overlap.

## 21.2 New identity thresholds

```text
factorization_compatibility_ratio_max       = 1.0
whitebox_compatibility_ratio_max             = 1.0
whitebox_factorization_ratio_max             = 1.0
dyadic_ball_overlap_ratio_max                = 1.0
whitebox_coordinate_abs_error_max            = 1e-10
quarter_radius_multiplier                    = 0.25
AD audit required strata                     = 40
AD audit permitted misses                    = 0
```

These values are not tuned to v1.3.6 outcomes. The value `1.0` means “inside the derived bound.”

## 21.3 Unchanged scientific thresholds

```text
curvature_rms_min                     = 5e-4
curvature_snr_min                     = 20
gate_response_rms_min                 = 5e-4
gate_response_snr_min                 = 20
tensor_snr_min                        = 20
active_gates_min                      = 3
certified_null_contribution_max       = 0.005
bypass_disagreement_max               = 0.15
cell_set_snr_min                      = 3
conditioning_absolute                 = unchanged
conditioning_dev_sd                   = unchanged
development_stop_below                = 0.05
confirmation_open_gain_min            = 0.10
confirmation_relative_gain_min        = 0.20
confirmation_relative_lcb_min         = 0.10
confirmation_absolute_gain_min        = 0.01
per_bin_relative_gain_min             = 0.10
per_bin_absolute_gain_min             = 0.005
cancellation_auroc_min                = 0.80
cancellation_auroc_lcb_min            = 0.70
half_radius_spearman_min              = 0.90
half_radius_change_max                = 0.20
bootstrap_replicates                  = 100000
bootstrap_seed                        = 20260805
```

The confirmation radius checks shall compare cell predictors derived from the coarse and fine Richardson operators.

## 21.4 Count thresholds after fresh resplit

These are derived by preserving the old frozen proportions and rounding upward.

### Development

\[
\left\lceil
\frac{15}{16}\times8
\right\rceil
=8.
\]

```text
development cells total               = 8
development surviving cells min       = 8
development conditioned cells min     = 8
development SNR cells min             = ceil((10/16) × 8) = 5
```

### Confirmation

\[
\left\lceil
\frac{28}{32}\times24
\right\rceil
=21,
\]

\[
\left\lceil
\frac{29}{32}\times24
\right\rceil
=22,
\]

\[
\left\lceil
\frac{14}{16}\times12
\right\rceil
=11.
\]

```text
confirmation cells total              = 24
confirmation technical cells min      = 21
confirmation oral cells min           = 22
confirmation cells per distance min   = 11
combined dev+confirm technical min     = ceil((40/48) × 32) = 27
```

The cancellation count requirements remain unchanged rather than being scaled down:

```text
cancellation total min                 = 8
cancellation per class min             = 3
cancellation per distance bin min      = 3
```

---

# 22. Exact source-code changes

## 22.1 `src/green_bridge_spec.py`

Add or replace active constants with:

```python
SCHEMA_VERSION = "green-bridge-v2.0.0"
PROTOCOL_ID = "structural-envelope-matched-bypass-setid-v2.0.0"
PARENT_PROTOCOL_ID = "structural-envelope-matched-bypass-v1.3.6"
DECISION_ID = "GPTPRO-GREEN-V136-TERMINAL-SETID-v1-20260825"
PROTOCOL_RUN_ID = "green-bridge-v2.0.0-one-shot"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "green_bridge_v200"

ATTEMPT_INDEX = 1
RETRY_ALLOWED = False
QUARTER_RADIUS_MULTIPLIER = 0.25

FACTOR_COMPATIBILITY_RATIO_MAX = 1.0
WHITEBOX_COMPATIBILITY_RATIO_MAX = 1.0
WHITEBOX_FACTORIZATION_RATIO_MAX = 1.0
DYADIC_BALL_OVERLAP_RATIO_MAX = 1.0
WHITEBOX_COORDINATE_ABS_ERROR_MAX = 1.0e-10
AD_AUDIT_STRATA = 40
AD_AUDIT_PERMITTED_MISSES = 0
```

Add the exact split tables and expected split hash from Section 10.

Move the old `0.15`, `0.05`, `1e-4`, `0.95`, and `0.25` gate-level certification constants into:

```python
HISTORICAL_V136_THRESHOLDS
```

They may not be referenced by active v2.0.0 classification code.

Update count thresholds exactly as in Section 21.4.

Retain all model, gate, site, structural-frame, target, base-radius, baseline, and statistical constants.

## 22.2 `src/green_bridge_numerics.py`

Retain existing historical functions.

Add:

```python
@dataclass(frozen=True)
class ScaleNumericalBoundsV200:
    epsilon_G: float
    epsilon_C: float
    epsilon_J: float
    epsilon_delta_H: np.ndarray
    A_max: np.ndarray
    epsilon_A: np.ndarray
    epsilon_P: np.ndarray
    epsilon_P_F: float
    inverse_admissible: bool
```

Add:

```python
@dataclass(frozen=True)
class DyadicEnclosureV200:
    coarse: ScaleNumericalBoundsV200
    fine: ScaleNumericalBoundsV200
    final_epsilon_G: float
    final_epsilon_C: float
    final_epsilon_J: float
    final_epsilon_delta_H: np.ndarray
    overlap_G: bool
    overlap_C: bool
    overlap_J: bool
    overlap_delta_H: np.ndarray
```

Add exactly these functions:

```python
richardson_pair_bounds_v200(...)
dyadic_enclosure_v200(...)
factorization_compatibility_v200(...)
whitebox_compatibility_v200(...)
whitebox_factorization_compatibility_v200(...)
shift_null_compatibility_v200(...)
unresolved_gate_contraction_bound_v200(...)
minkowski_sum_interval(...)
subtract_intervals(...)
absolute_value_interval(...)
worst_case_interval_rmse(...)
robust_interval_auc_lower_bound(...)
```

`factorization_compatibility_v200` must implement Section 13 literally.

`whitebox_compatibility_v200` and `whitebox_factorization_compatibility_v200` must implement Section 14 literally.

No denominator may be protected by a hidden empirical floor. Use explicit zero branches and outward-rounded comparison.

## 22.3 `src/matched_bypass_gate.py`

Do not change the algebra in:

```python
identify_gate
```

Do not add:

- a pseudoinverse;
- ridge regularization;
- a learned inverse;
- donor PCA;
- rank truncation.

Add versioned dataclasses:

```python
GateScaleIdentificationV200
GateCertificationV200
GateContributionIntervalV200
```

The point estimate remains the result of `identify_gate(fine_richardson_jet)`.

## 22.4 `src/green_bridge_response_ad.py`

Create a new prepare-only module containing:

```python
response_gate_jet_forward_ad64(...)
response_gate_jet_reverse_ad64(...)
select_ad_audit_panel_v200(...)
audit_richardson_enclosure_v200(...)
```

Requirements:

- no import from `analyze_terminal.py`;
- no Parquet input;
- no behavioral or baseline field;
- exact full unembedding;
- path/control code isolation retained;
- float64 used only here;
- outputs written only to prepare audit artifacts;
- no AD tensor enters development prediction.

## 22.5 `src/exp_green_bridge_gpt2.py`

Do not modify the historical v1.3.6 functions in a way that changes predecessor reproducibility.

Add:

```python
_gate_jet_triplet_v200(...)
_classify_gate_v200(...)
_mixed_system_v200(...)
_tensor_item_v200(...)
_energy_item_v200(...)
_run_split_v200(...)
_aggregate_cells_v200(...)
```

### `_gate_jet_triplet_v200`

Call `_jet_at_radius_physical` at:

```text
1.0
0.5
0.25
```

and return:

```text
base
half
quarter
coarse_richardson
fine_richardson
coarse_bounds
fine_bounds
dyadic_enclosure
```

### `_classify_gate_v200`

Implement the four classes from Section 15.

It must not reference:

```python
THRESHOLDS.factorization_residual_max
THRESHOLDS.whitebox_a_relative_max
THRESHOLDS.whitebox_a_small_absolute_max
THRESHOLDS.tensor_cosine_min
THRESHOLDS.tensor_symmetric_change_max
THRESHOLDS.richardson_change_max
```

### `_mixed_system_v200`

Return at least:

```text
theta_center
theta_lower
theta_upper
theta_coarse
theta_fine
active_gates
null_gates
unresolved_gates
contradictory_gates
numerical_invalid_gates
set_complete
point_complete
set_admissible
bypass_disagreement
gates
```

All ten gates must appear in `gates`.

### `_tensor_item_v200`

Return at least:

```text
set_admissible
point_complete
theta_tar_center
theta_tar_lower
theta_tar_upper
theta_pat_center
theta_pat_lower
theta_pat_upper
behavioral
single
first_order
pie
cancellation_dx
cancellation_dz
mixed_audit
```

PIE remains calculated exactly as before and stored only under the baseline field.

### `_aggregate_cells_v200`

Filter tensor rows by `set_admissible`.

Preserve the current aggregation order and write:

```text
mixed_lower
mixed_upper
mixed_center
mixed_coarse
mixed_fine
error_bound
snr
```

where:

```python
error_bound = 0.5 * (mixed_upper - mixed_lower)
snr = 0.5 * (mixed_upper + mixed_lower) / max(error_bound, 1e-8)
```

## 22.6 `src/green_bridge_dataset.py`

Add:

```python
build_green_bridge_v200_splits(...)
```

It must:

- reconstruct the original v1.3.6 confirmation groups from frozen source definitions;
- apply the exact group-level split in Section 10;
- verify the exact hash;
- produce only 8 development and 24 confirmation cells;
- keep `near` and `far` cells from one noun-century group in the same phase;
- reject any v1.3.6 development group;
- write no model response during split generation.

## 22.7 `src/analyze_green_bridge.py`

Retain historical v1 analysis functions.

Add:

```python
development_decision_v200(...)
freeze_confirmation_v200(...)
confirmation_decision_v200(...)
worst_case_interval_rmse(...)
robust_interval_auroc(...)
```

Development must compare the worst-case mixed RMSE with the same frozen baseline LOOCV RMSE.

Confirmation must:

- use the frozen development baseline calibration;
- use worst-case mixed RMSE;
- bootstrap robust relative gain;
- use interval-AUC lower bounds;
- compare coarse and fine cell predictors for radius stability;
- enforce the scaled count thresholds.

## 22.8 Multigpu worker and launcher

Update `src/green_bridge_multigpu_worker.py` with a versioned v2.0.0 worker mode.

Create:

```text
src/launch_green_bridge_v200.sh
```

by copying the v1.3.6 launcher.

Required changes:

```text
runtime root:
    /mnt/sdb/ccj/iclr_1_runs/green_bridge_v200_runtime

test log:
    /tmp/green_bridge_v200_contract_${PHASE}.log

run log:
    /tmp/green_bridge_v200_${PHASE}.log

output root:
    outputs/green_bridge_v200
```

Retain:

- coordinator physical GPU `4`;
- worker physical GPUs `0` through `7`;
- environment `green_bridge_20260805`;
- PyTorch `2.7.1`;
- the frozen requirements lock;
- TransformerLens source hashes;
- deterministic environment variables;
- exact endpoint batch size one.

Do not modify `src/launch_green_bridge_v136.sh`.

---

# 23. Required unit and regression tests

All 168 existing tests must remain semantically enforced. Tests containing historical v1.3.6 threshold literals shall be revised to assert that those thresholds remain archived but inactive; they may not simply be deleted or skipped.

Add exactly these 32 tests:

1. `V200FactorizationBoundsTests.test_exact_rank_one_inside_derived_bound`
2. `V200FactorizationBoundsTests.test_residual_one_ulp_above_bound_fails`
3. `V200FactorizationBoundsTests.test_active_classifier_has_no_factorization_point_one_five`
4. `V200WhiteboxBoundsTests.test_componentwise_whitebox_envelope_passes`
5. `V200WhiteboxBoundsTests.test_componentwise_whitebox_excess_fails`
6. `V200WhiteboxBoundsTests.test_active_classifier_has_no_whitebox_point_zero_five`
7. `V200WhiteboxBoundsTests.test_direct_whitebox_factorization_triangle_bound`
8. `V200WhiteboxBoundsTests.test_shift_null_uses_epsilon_a_plus_epsilon_wb`
9. `V200StencilTests.test_radii_are_base_half_quarter`
10. `V200StencilTests.test_fine_richardson_is_always_primary`
11. `V200StencilTests.test_dyadic_overlap_uses_uncertainty_balls`
12. `V200StencilTests.test_no_estimator_selection_uses_behavior_or_baseline`
13. `V200ADAuditTests.test_audit_reader_has_no_behavioral_fields`
14. `V200ADAuditTests.test_panel_has_exactly_forty_strata`
15. `V200ADAuditTests.test_ad_value_outside_enclosure_stops_prepare`
16. `V200GateClassTests.test_noninvertible_gate_becomes_unresolved_bounded`
17. `V200GateClassTests.test_bound_exceedance_becomes_structural_contradiction`
18. `V200GateClassTests.test_unresolved_gate_point_center_is_zero`
19. `V200GateClassTests.test_unresolved_whitebox_use_is_absolute_bound_only`
20. `V200SystemTests.test_all_ten_gates_are_accounted`
21. `V200SystemTests.test_invalid_gate_cannot_enter_interval_sum`
22. `V200SystemTests.test_active_gate_minimum_remains_three`
23. `V200IntervalTests.test_system_interval_is_minkowski_sum`
24. `V200IntervalTests.test_cell_interval_preserves_abs_of_mean_difference`
25. `V200IntervalTests.test_worst_case_rmse_uses_farthest_endpoint`
26. `V200IntervalTests.test_robust_auc_is_pairwise_lower_bound`
27. `V200BaselineTests.test_pie_remains_baseline_only`
28. `V200FirewallTests.test_v136_development_rows_are_forbidden_inputs`
29. `V200FirewallTests.test_v200_split_groups_and_sha256_are_exact`
30. `V200FirewallTests.test_confirmation_artifacts_forbidden_before_open`
31. `V200PredecessorTests.test_v136_terminal_hashes_and_stop_are_immutable`
32. `V200TheoryTests.test_fixed_rank_donor_pca_remains_terminated`

Required result:

```text
Ran 200 tests
OK
```

No skip, expected failure, monkeypatched threshold, or exclusion is authorized.

---

# 24. Required prepare artifacts

A successful v2.0.0 prepare must durably write:

```text
outputs/green_bridge_v200/run_ledger.json
outputs/green_bridge_v200/predecessor_v136_manifest.json
outputs/green_bridge_v200/model_fingerprint.json
outputs/green_bridge_v200/v200_split.json
outputs/green_bridge_v200/scientific_delta_v200.json
outputs/green_bridge_v200/gate04_legacy_panel.json
outputs/green_bridge_v200/hook_audit.json
outputs/green_bridge_v200/manual_tail_equivalence.json
outputs/green_bridge_v200/structural_frame_preflight.json
outputs/green_bridge_v200/response_ad_enclosure_audit_v200.json
outputs/green_bridge_v200/three_scale_numerical_preflight_v200.json
outputs/green_bridge_v200/hardware_plan.json
outputs/green_bridge_v200/throughput_preflight.json
outputs/green_bridge_v200/prepare_result.json
outputs/green_bridge_v200/manifest.json
outputs/green_bridge_v200/sha256sums.txt
```

`scientific_delta_v200.json` must enumerate every authorized scientific change in Section 8.3 and prove equality of all frozen invariants in Section 8.2.

A successful `prepare_result.json` must contain:

```json
{
  "schema_version": "green-bridge-prepare-v2.0.0",
  "verdict": "PREPARE_PASS",
  "attempt_index": 1,
  "retry_allowed": false,
  "development_started": false,
  "confirmation_started": false,
  "first_failed_gate": null
}
```

After prepare, the following must not exist:

```text
dev_tensor_scores.parquet
dev_energy_targets.parquet
dev_cells.json
dev_result.json
frozen_analysis.json
confirm_tensor_scores.parquet
confirm_energy_targets.parquet
confirm_cells.json
confirm_result.json
```

---

# 25. Prepare gates

Prepare passes only if all of the following hold:

1. reviewed commit is an ancestor of the execution commit;
2. branch is `main`;
3. worktree is clean;
4. v2.0.0 root was absent before launch;
5. attempt index is one;
6. retry is false;
7. every predecessor hash passes;
8. v1.3.6 confirmation is proven unopened;
9. all 200 CPU tests pass;
10. frozen Python, PyTorch, CUDA, TransformerLens, and model hashes pass;
11. exact endpoint batch size one is enforced;
12. Gate-04 passes;
13. manual-tail raw-logit equivalence passes unchanged;
14. structural-frame dimensions and residuals pass unchanged;
15. split hash equals
    `f012a286801bc3e3e937b390f0a62d7e92f8d5a21ba59d7e53478ae911e72cfc`;
16. all 40 AD audit strata pass;
17. all three-scale preflight jets are finite;
18. coarse/fine balls overlap on the preflight panel;
19. no behavioral or baseline field is read by the AD or numerical preflight;
20. peak allocated memory is at most 20 GB per RTX 4090;
21. projected development time is at most 24 GPU hours;
22. no selected-column, lower-precision, larger-batch, or reduced-gate fallback is used;
23. all prepare artifacts are present and hashed;
24. no development or confirmation artifact exists.

---

# 26. Development gates

Development shall produce exactly:

```text
64 tensor records
64 energy records
8 cells
```

The eight workers must each receive exactly:

```text
8 tensor records
8 energy records
```

using deterministic role-stratified round-robin assignment after sorting by:

```python
sha256(
    f"green-v200-worker|{role}|{pair_digest}".encode("utf-8")
).hexdigest()
```

Development cell rules:

```text
set-admissible tensor rows per surviving cell >= 6
admissible energy rows per surviving cell     >= 6
surviving cells                               >= 8
conditioned cells                             >= 8
set-SNR >= 3 cells                            >= 5
```

Performance rule:

```text
robust relative gain < 0.05
    STOP_ORAL

0.05 <= robust relative gain < 0.10
    POSTER_ONLY
    confirmation remains closed

robust relative gain >= 0.10
and every technical gate passes
    OPEN_CONFIRMATION
```

No midpoint-only result may open confirmation.

Development must write:

```text
dev_tensor_scores.parquet
dev_energy_targets.parquet
development_multigpu_merge.json
dev_cells.json
dev_result.json
```

Only an `OPEN_CONFIRMATION` result may create:

```text
frozen_analysis.json
```

The frozen file must contain:

- exact source and protocol hashes;
- exact v2 split hash;
- baseline calibration;
- robust analysis definitions;
- bootstrap seed and count;
- confirmation cell identities;
- confirmation retry count zero.

---

# 27. Confirmation gates

Confirmation remains forbidden unless:

```text
dev_result.verdict == OPEN_CONFIRMATION
```

and every frozen-analysis hash passes.

Confirmation shall produce exactly:

```text
192 tensor records
192 energy records
24 cells
```

Each of eight workers receives:

```text
24 tensor records
24 energy records
```

Technical requirements:

```text
surviving cells              >= 21
conditioned cells            >= 21
cells per distance bin       >= 11
combined dev+confirm cells   >= 27
```

Oral-result requirements:

```text
surviving cells                     >= 22
conditioned cells                   >= 22
robust relative gain                >= 0.20
bootstrap 2.5% relative-gain LCB    >= 0.10
robust absolute gain                >= 0.01
per-bin robust relative gain        >= 0.10
per-bin relative-gain LCB           > 0
per-bin robust absolute gain        >= 0.005
robust cancellation AUROC lower     >= 0.80
robust cancellation AUROC LCB       >= 0.70
coarse/fine cell Spearman            >= 0.90
coarse/fine median symmetric change <= 0.20
same radius conditions per bin
```

The confirmation result is terminal whether it passes or fails.

---

# 28. Required provenance and artifact rules

Every phase must atomically serialize:

```text
execution_commit
review_commit
source_sha256
protocol_sha256
requirements_sha256
model_sha256
transformer_lens_source_sha256
predecessor_sha256
split_sha256
scientific_delta_sha256
artifact_sha256
```

`sha256sums.txt` must cover every regular file in the output root except itself.

Before development and confirmation, rerun:

```bash
(
  cd outputs/green_bridge_v200
  sha256sum -c sha256sums.txt
)
```

and reverify all v1.3.6 predecessor hashes.

No prior manifest entry may be mutated between phases. Continuation phases append new phase artifacts and write a newly versioned manifest generation whose predecessor-manifest hash is recorded.

Worker endpoint batches remain transactional and one-shot:

```text
one item
one role
one immutable batch ID
one committed endpoint artifact
no resubmission
```

A crash after a phase claim is terminal.

---

# 29. Exact implementation commands

Run from the repository root.

## 29.1 Establish the reviewed base

```bash
set -euo pipefail

git switch main

test "$(git rev-parse HEAD)" = \
"3bdeac04a16724461f266705ef250a6357ced1cf"

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"
```

## 29.2 Verify predecessor artifacts

```bash
sha256sum -c <<'EOF'
660788dde8bc5df1d057db31b4dc1065b222ac7777efc0e4c6220e09f1ed81ff  outputs/green_bridge_v136/dev_tensor_scores.parquet
23a99b6998ec2c51184ae26b8f86a7656247ff2091e251752c1fccd06295e593  outputs/green_bridge_v136/dev_energy_targets.parquet
1294a76d6d79c81f240c20c4257aa6b0fe76457d46b30cfc5d5699e27759ae1f  outputs/green_bridge_v136/dev_cells.json
2e15531d62bd5cc1162980fdaa2643a7300b362eb6b11ff5b94bb3d623c37277  outputs/green_bridge_v136/dev_result.json
31dbc71fbeaa40f313be6078a627082050aa5a338e132d3a6ed7343869eaad7a  outputs/green_bridge_v136/development_multigpu_merge.json
EOF
```

Verify terminal state:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/green_bridge_v136")

dev = json.loads(
    (root / "dev_result.json").read_text(encoding="utf-8")
)
merge = json.loads(
    (root / "development_multigpu_merge.json").read_text(
        encoding="utf-8"
    )
)

assert dev["verdict"] == "STOP_ORAL"
assert dev["n_surviving_cells"] == 0
assert dev["spec_sha256"] == (
    "cb771c59e91b4fc553ef73a1c7a116ec0ee55f499ce46a2f91e4c600cd8bd41d"
)
assert merge["tensor_count"] == 128
assert merge["energy_count"] == 128
assert merge["tensor_parquet_sha256"] == (
    "660788dde8bc5df1d057db31b4dc1065b222ac7777efc0e4c6220e09f1ed81ff"
)
assert merge["energy_parquet_sha256"] == (
    "23a99b6998ec2c51184ae26b8f86a7656247ff2091e251752c1fccd06295e593"
)

for forbidden in (
    "confirm_tensor_scores.parquet",
    "confirm_energy_targets.parquet",
    "confirm_cells.json",
    "confirm_result.json",
    "frozen_analysis.json",
):
    assert not (root / forbidden).exists(), forbidden
PY
```

## 29.3 Implement the exact v2.0.0 changes

Modify only the files authorized in Section 22 and add this decision document.

Run:

```bash
python src/test_green_bridge_contract.py \
  2>&1 |
  tee /tmp/green_bridge_v200_contract_precommit.log

grep -F "Ran 200 tests" \
  /tmp/green_bridge_v200_contract_precommit.log

grep -F "OK" \
  /tmp/green_bridge_v200_contract_precommit.log

git diff --check
```

Verify the old launcher is unchanged:

```bash
git diff --exit-code \
  3bdeac04a16724461f266705ef250a6357ced1cf \
  -- src/launch_green_bridge_v136.sh
```

Review the complete scientific patch:

```bash
git diff -- \
  analysis/GPTPRO_GREEN_V136_TERMINAL_DECISION_20260825.md \
  src/green_bridge_spec.py \
  src/green_bridge_numerics.py \
  src/matched_bypass_gate.py \
  src/green_bridge_response_ad.py \
  src/green_bridge_dataset.py \
  src/exp_green_bridge_gpt2.py \
  src/analyze_green_bridge.py \
  src/green_bridge_multigpu_worker.py \
  src/test_green_bridge_contract.py \
  src/launch_green_bridge_v200.sh
```

## 29.4 Commit the new identity

```bash
git add \
  analysis/GPTPRO_GREEN_V136_TERMINAL_DECISION_20260825.md \
  src/green_bridge_spec.py \
  src/green_bridge_numerics.py \
  src/matched_bypass_gate.py \
  src/green_bridge_response_ad.py \
  src/green_bridge_dataset.py \
  src/exp_green_bridge_gpt2.py \
  src/analyze_green_bridge.py \
  src/green_bridge_multigpu_worker.py \
  src/test_green_bridge_contract.py \
  src/launch_green_bridge_v200.sh

git diff --cached --check

git commit -m \
  "Add GREEN v2.0 bound-certified set-identification protocol"

EXECUTION_COMMIT="$(git rev-parse HEAD)"
printf '%s\n' "$EXECUTION_COMMIT"

git merge-base --is-ancestor \
  3bdeac04a16724461f266705ef250a6357ced1cf \
  "$EXECUTION_COMMIT"

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

test ! -e outputs/green_bridge_v200
```

The resulting `EXECUTION_COMMIT` is the only authorized v2.0.0 execution commit.

---

# 30. Exact prepare command

```bash
set -euo pipefail

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

test ! -e outputs/green_bridge_v200

bash src/launch_green_bridge_v200.sh 4 prepare
```

This command may be issued once.

A nonzero exit, interruption after ledger creation, partial root, or terminal `STOP` consumes the attempt.

---

# 31. Mechanical prepare-pass verification

```bash
set -euo pipefail

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("outputs/green_bridge_v200")

prepare = json.loads(
    (root / "prepare_result.json").read_text(encoding="utf-8")
)
manifest = json.loads(
    (root / "manifest.json").read_text(encoding="utf-8")
)
ledger = json.loads(
    (root / "run_ledger.json").read_text(encoding="utf-8")
)
split = json.loads(
    (root / "v200_split.json").read_text(encoding="utf-8")
)
ad = json.loads(
    (root / "response_ad_enclosure_audit_v200.json").read_text(
        encoding="utf-8"
    )
)

assert prepare["verdict"] == "PREPARE_PASS"
assert prepare["first_failed_gate"] is None

assert ledger["protocol_run_id"] == \
    "green-bridge-v2.0.0-one-shot"
assert ledger["attempt_index"] == 1
assert ledger["retry_allowed"] is False
assert ledger["development_started"] is False
assert ledger["confirmation_started"] is False

assert manifest["schema_version"] == \
    "green-bridge-manifest-v2.0.0"
assert manifest["protocol_id"] == \
    "structural-envelope-matched-bypass-setid-v2.0.0"
assert manifest["parent_protocol_id"] == \
    "structural-envelope-matched-bypass-v1.3.6"

assert split["sha256"] == \
    "f012a286801bc3e3e937b390f0a62d7e92f8d5a21ba59d7e53478ae911e72cfc"
assert len(split["development_groups"]) == 4
assert len(split["confirmation_groups"]) == 12

assert ad["required_strata"] == 40
assert ad["completed_strata"] == 40
assert ad["misses"] == 0
assert ad["behavioral_fields_read"] is False

for name, expected in manifest["artifact_sha256"].items():
    path = root / name
    assert path.is_file(), name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (name, actual, expected)

for forbidden in (
    "dev_tensor_scores.parquet",
    "dev_energy_targets.parquet",
    "dev_cells.json",
    "dev_result.json",
    "frozen_analysis.json",
    "confirm_tensor_scores.parquet",
    "confirm_energy_targets.parquet",
    "confirm_cells.json",
    "confirm_result.json",
):
    assert not (root / forbidden).exists(), forbidden
PY

(
  cd outputs/green_bridge_v200
  sha256sum -c sha256sums.txt
)
```

Re-run the predecessor hash checks before development.

---

# 32. Exact development command

Development is authorized only after Section 31 passes.

```bash
set -euo pipefail

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

bash src/launch_green_bridge_v200.sh 4 development
```

This command may be issued once.

---

# 33. Mechanical development verification

```bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/green_bridge_v200")

merge = json.loads(
    (root / "development_multigpu_merge.json").read_text(
        encoding="utf-8"
    )
)
result = json.loads(
    (root / "dev_result.json").read_text(encoding="utf-8")
)
ledger = json.loads(
    (root / "run_ledger.json").read_text(encoding="utf-8")
)

assert merge["tensor_count"] == 64
assert merge["energy_count"] == 64
assert merge["exact_batch_size"] == 1

assert ledger["development_started"] is True
assert ledger["confirmation_started"] is False

assert result["verdict"] in {
    "STOP_ORAL",
    "POSTER_ONLY",
    "OPEN_CONFIRMATION",
}

if result["verdict"] == "OPEN_CONFIRMATION":
    assert result["n_surviving_cells"] >= 8
    assert result["n_conditioned_cells"] >= 8
    assert result["n_snr_cells"] >= 5
    assert result["robust_relative_gain"] >= 0.10
    assert (root / "frozen_analysis.json").is_file()
else:
    assert not (root / "frozen_analysis.json").exists()
PY

(
  cd outputs/green_bridge_v200
  sha256sum -c sha256sums.txt
)
```

A `STOP_ORAL` or `POSTER_ONLY` result is terminal and leaves confirmation closed.

---

# 34. Exact confirmation command

Confirmation is authorized only when:

```text
dev_result.verdict == OPEN_CONFIRMATION
```

and all frozen hashes pass.

```bash
set -euo pipefail

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

bash src/launch_green_bridge_v200.sh 4 confirmation
```

This command may be issued once.

The result is terminal.

---

# 35. Explicit STOP conditions

The v2.0.0 runner must write a durable terminal result and stop on any of the following:

1. reviewed commit is not an ancestor;
2. branch is not `main`;
3. worktree is dirty;
4. v2.0.0 root existed before launch;
5. attempt index is not one;
6. retry is enabled;
7. a predecessor hash differs;
8. v1.3.6 STOP state differs;
9. evidence exists that v1.3.6 confirmation was accessed;
10. environment or package hash differs;
11. TransformerLens source differs;
12. model or tokenizer differs;
13. any CPU test fails;
14. test count is not 200;
15. Gate-04 fails;
16. manual-tail endpoint equivalence fails;
17. structural-frame preflight fails;
18. split hash differs;
19. an old development group enters v2.0.0;
20. a confirmation group enters development beyond the four frozen groups;
21. AD audit reads a behavioral or baseline field;
22. fewer than 40 AD strata complete;
23. any AD enclosure miss occurs;
24. any preflight derivative is nonfinite;
25. any preflight coarse/fine ball fails to overlap;
26. throughput exceeds 24 GPU hours;
27. memory exceeds 20 GB;
28. exact endpoint batch size differs from one;
29. any fallback precision, radius, gate set, or batch size is attempted;
30. any worker is missing;
31. endpoint coverage is not exact;
32. an endpoint batch is repeated;
33. a worker artifact hash differs;
34. development record counts are not 64/64;
35. confirmation record counts are not 192/192;
36. a contradictory gate is aggregated;
37. any selected gate is omitted from a system interval;
38. PIE enters the matched-bypass point or interval estimator;
39. v1.3.6 rows are scored under v2.0.0 before v2.0.0 terminates;
40. development survival is below eight cells;
41. development conditioning is below eight cells;
42. development SNR count is below five;
43. development robust gain is below `0.10`;
44. confirmation is attempted after `POSTER_ONLY` or `STOP_ORAL`;
45. frozen-analysis hashes differ;
46. confirmation technical survival is below 21;
47. any confirmation oral criterion fails;
48. any phase crashes after its ledger claim;
49. any source or protocol file changes between phases;
50. any proposal is made to retry the failed phase under the same identity.

No STOP in this list authorizes v2.0.1.

---

# 36. Authorized actions

The executor is authorized to:

- implement only the changes in this decision;
- add the quarter-radius GateJet evaluation;
- add the prepare-only AD enclosure audit;
- replace the three unjustified white-box/factorization thresholds with derived bounds;
- add set-valued gate and cell aggregation;
- create the exact fresh split;
- run 200 CPU tests;
- create one clean execution commit;
- run v2.0.0 prepare once;
- run development once after prepare passes;
- run confirmation once only after `OPEN_CONFIRMATION`;
- report a terminal negative result honestly.

---

# 37. Forbidden actions

The executor is forbidden to:

- rerun v1.3.6;
- modify or delete v1.3.6 artifacts;
- reinterpret v1.3.6 as an engineering dry run;
- apply v2.0.0 retrospectively to v1.3.6 rows before termination;
- inspect old confirmation responses;
- use raw PIE correlation to set any rule;
- replace the matched-bypass estimator with PIE;
- add PIE to the operator center;
- use white-box \(A\) as the signed point estimate;
- silently omit unresolved gates;
- relabel contradiction as unresolved;
- restore donor PCA;
- change selected gates;
- change sites;
- change the target;
- change the contrast;
- change the base radius;
- add a radius search;
- choose coarse versus fine per outcome;
- use a pseudoinverse;
- add ridge;
- lower the active-gate minimum;
- relax materiality or SNR thresholds;
- relax development or confirmation performance thresholds;
- reduce the bootstrap;
- change the bootstrap seed;
- change precision in the scientific forward;
- change endpoint batch size;
- reuse a partial phase;
- retry any v2.0.0 phase;
- open confirmation without the exact frozen development verdict.

---

# 38. Paper-level interpretation

## 38.1 What v1.3.6 supports

The paper may state:

- the complete engineering pipeline executed;
- the exact batch-one/manual-tail equivalence contract passed;
- the exact LayerNorm structural envelope passed;
- all energy-target locality checks passed;
- the frozen tensor certification failed at development;
- the dominant failures were factorization and white-box agreement under heuristic relative cutoffs;
- no confirmation result exists.

## 38.2 What v1.3.6 does not support

The paper may not state:

- that the mixed estimator predicts behavior;
- that PIE validates the main claim;
- that the 0.962 diagnostic is confirmatory;
- that the theorem was empirically established;
- that no mechanistic signal exists;
- that the exact theorem was falsified.

## 38.3 Handling PIE

PIE may appear in an exploratory appendix or failed-pilot postmortem with all of the following labels:

```text
post hoc
inadmissible rows
development only
not used for design
not used for confirmation
not evidence for the registered claim
```

Its strong diagnostic correlation can motivate separate future work, but not a v2.0.0 design choice.

## 38.4 If v2.0.0 also fails

If v2.0.0 fails prepare, development, or confirmation:

- no further run is authorized by this document;
- the paper retains the exact matched-bypass identification theorem;
- the basis-free ambient operator and exact LayerNorm envelope remain the theoretical contribution;
- the empirical section must report that the frozen experiments did not establish a robust predictive result;
- any paper-level claim must distinguish theorem, structural validation, failed development evidence, and absent confirmation;
- fixed-rank donor PCA remains terminated.

The theoretical height can be preserved through a rigorous theorem, exact architectural probe completeness, and an honest negative empirical certification result. It cannot be preserved by overstating post hoc behavior correlations.

---

# 39. Mechanical executor checklist

## 39.1 Predecessor

- [ ] Repository begins at exact commit `3bdeac04a16724461f266705ef250a6357ced1cf`.
- [ ] Branch is `main`.
- [ ] Worktree is clean.
- [ ] All five v1.3.6 artifact hashes pass.
- [ ] v1.3.6 verdict is `STOP_ORAL`.
- [ ] v1.3.6 surviving-cell count is zero.
- [ ] v1.3.6 scientific payload hash is exact.
- [ ] v1.3.6 spec hash is exact.
- [ ] No v1.3.6 confirmation artifact exists.
- [ ] Old output and audit directories remain untouched.

## 39.2 Implementation

- [ ] New identity is v2.0.0, not v1.3.7.
- [ ] Quarter-radius multiplier is exactly `0.25`.
- [ ] Fine Richardson is always the point estimator.
- [ ] No behavior-dependent estimator selection exists.
- [ ] Factorization uses the derived rowwise bound.
- [ ] White-box comparison uses the derived coordinatewise bound.
- [ ] Direct white-box factorization is implemented.
- [ ] Shift-null uses the derived bound.
- [ ] Coarse/fine derivative balls must overlap.
- [ ] `unresolved-bounded` is distinct from null.
- [ ] Structural contradiction remains hard invalid.
- [ ] All ten gates are represented.
- [ ] White-box values enter unresolved gates only through absolute bounds.
- [ ] PIE remains baseline-only.
- [ ] Donor PCA remains absent.
- [ ] Historical v1.3.6 code and launcher remain reproducible.

## 39.3 Split

- [ ] Exact four development groups are present.
- [ ] Exact twelve confirmation groups are present.
- [ ] Both distance bins remain together per noun-century group.
- [ ] Split hash is `f012a286801bc3e3e937b390f0a62d7e92f8d5a21ba59d7e53478ae911e72cfc`.
- [ ] No old development cell enters v2.0.0.
- [ ] Confirmation responses remain unread.

## 39.4 Tests

- [ ] All 168 previous assertions remain enforced.
- [ ] All 32 new tests exist.
- [ ] Test result is exactly `Ran 200 tests`.
- [ ] Test result is `OK`.
- [ ] No test is skipped.
- [ ] No threshold is monkeypatched.

## 39.5 Prepare

- [ ] New root is absent before launch.
- [ ] Prepare command is issued once.
- [ ] All environment hashes pass.
- [ ] Gate-04 passes.
- [ ] Manual-tail equivalence passes.
- [ ] Structural-frame preflight passes.
- [ ] All 40 AD strata complete.
- [ ] AD audit reads no behavior or baseline.
- [ ] AD misses equal zero.
- [ ] Three-scale preflight passes.
- [ ] Memory is at most 20 GB.
- [ ] Forecast is at most 24 GPU hours.
- [ ] All prepare artifacts are present and hashed.
- [ ] No development or confirmation artifact exists.

## 39.6 Development

- [ ] Predecessor hashes are rechecked.
- [ ] Source and manifest hashes are frozen.
- [ ] Development command is issued once.
- [ ] Exactly 64 tensor rows are produced.
- [ ] Exactly 64 energy rows are produced.
- [ ] No endpoint batch repeats.
- [ ] Every system accounts for all ten gates.
- [ ] Contradictory gates are never aggregated.
- [ ] Worst-case interval RMSE is used.
- [ ] PIE remains a baseline.
- [ ] Confirmation opens only for `OPEN_CONFIRMATION`.

## 39.7 Confirmation

- [ ] Frozen analysis exists and hashes pass.
- [ ] Development verdict is exactly `OPEN_CONFIRMATION`.
- [ ] Confirmation command is issued once.
- [ ] Exactly 192 tensor rows are produced.
- [ ] Exactly 192 energy rows are produced.
- [ ] Robust interval criteria are used.
- [ ] All survival, gain, per-bin, AUROC, bootstrap, and radius gates are enforced.
- [ ] Final result is treated as terminal.
- [ ] No retry follows.

**BINDING VERDICT: GREEN v1.3.6 remains an immutable valid development STOP; it does not by itself falsify the matched-bypass theorem, and exactly one fresh GREEN v2.0.0 three-scale, uncertainty-bound, all-ten set-identification run is authorized under this document, with confirmation otherwise closed.**
