<!-- filename: analysis/GPTPRO_GREEN_V200_CORRIGENDUM_DECISION_20260825.md -->

# GPTPro GREEN v2.0.0 Binding Corrigendum and Pre-Launch Protocol Decision — 2026-08-25

## Document status

| Field | Binding value |
|---|---|
| Repository | `ScottBlizzard/idle_1` |
| Reviewed branch | `codex/green-v200` |
| Exact reviewed implementation commit | `159520a24b1a7903110f3c457234d1ebf254710b` |
| Binding predecessor decision | `analysis/GPTPRO_GREEN_V136_TERMINAL_DECISION_20260825.md` |
| Implementation/blocker report | `analysis/GREEN_V200_IMPLEMENTATION_BLOCKERS_20260825.md` |
| Existing protocol identity | `green-bridge-v2.0.0` |
| Existing run identity | `green-bridge-v2.0.0-one-shot` |
| Official v2.0.0 output root | Absent |
| Attempt consumed | No |
| Fresh v2.0.0 development response observed | No |
| Fresh v2.0.0 confirmation response observed | No |
| Review classification | **Binding pre-launch corrigendum** |
| Reviewed commit launch status | **Not authorized** |
| Corrected descendant launch status | **Conditionally authorized exactly once** |
| Issue 1 decision | **Option B: replace the incorrect split digest** |
| Issue 2 decision | **Option B: replace the mathematically incomplete enclosure before launch** |
| Issue 3 decision | **Float64 `cfg.dtype` correction approved only in an isolated AD model, not by mutating the active scientific model** |
| Central theoretical claim | Preserved without weakening |
| Fixed-rank donor PCA | Permanently terminated |
| PIE | Baseline and post hoc diagnostic only |
| Confirmation | Closed unless corrected v2.0.0 development returns `OPEN_CONFIRMATION` |

---

# 1. Binding executive verdict

The GREEN v2.0.0 one-shot **must not be launched from commit**:

```text
159520a24b1a7903110f3c457234d1ebf254710b
```

The implementation has two pre-launch defects that affect binding protocol identity:

1. the frozen split digest does not match the split payload specified by the predecessor decision;
2. the prescribed Richardson “balls” are not valid enclosures of the fine-Richardson derivative estimate around the independent float64 derivative reference.

The split defect is clerical but terminal if left unfixed. The numerical defect is scientific: the current implementation treats a rich-versus-small-scale discrepancy plus endpoint-repeatability terms as a proof-valid truncation-error enclosure, but neither the code nor the predecessor derivation supplies the smoothness or remainder bound needed for that conclusion. The non-inferential smoke case is a valid falsification of the claimed enclosure construction because the independent AD derivative lies outside both the coarse and fine balls while all coarse/fine overlap ratios also exceed one. The smoke record was outside the new inferential population and outside the frozen AD panel, so using it to detect a defective universal numerical claim creates no behavioral-outcome contamination.

The existing one-shot remains unconsumed because:

- `outputs/green_bridge_v200` has not been created;
- no v2.0.0 development or confirmation response has been evaluated;
- the only numerical diagnostic used a legacy donor record outside the v2.0.0 inferential population;
- 199 of 200 tests pass, with the sole current failure being the authoritative split-hash inconsistency.

Therefore this document does **not** create attempt 2, does **not** authorize a retry, and does **not** create a new inferential population. It corrects the still-unlaunched v2.0.0 protocol and requires a new clean implementation commit.

The binding scientific choices are:

1. Replace the incorrect split digest

   ```text
   f012a286801bc3e3e937b390f0a62d7e92f8d5a21ba59d7e53478ae911e72cfc
   ```

   with

   ```text
   0873915c966bef8f54b83d4151a9d7c75b577da5dfc17ee093b9f5c58a9590f7
   ```

   while preserving every group, row, record count, role, distance bin, and contamination boundary.

2. Retain the immutable fine-Richardson estimator as the scientific point estimator.

3. Retire coarse/fine “ball overlap” as a hard numerical-admissibility claim. Coarse/fine differences remain mandatory diagnostics and retain their confirmatory radius-stability role.

4. Replace the defective numerical enclosure with a **dual-route float64-AD-certified fine-Richardson enclosure** evaluated for every tensor item, system, and gate.

5. Preserve the four gate classes, but classify a gate as:
   - `numerical-invalid` when the dual-route numerical reference is not trustworthy;
   - `structural-contradiction` when the exact matched-bypass identity fails beyond the corrected numerical bounds;
   - `unresolved-bounded` when the identity is compatible but the response estimator is not point-identifiable;
   - `active-identified` or `certified-target-null` only under the existing proof-derived response and contribution rules.

6. Approve float64 `model.cfg.dtype` inside an isolated local-tail AD clone. Mutating the live float32 scientific model through `model.double()` and then attempting to restore it is forbidden.

7. Correct the additional protocol defects found in:
   - predecessor identity metadata;
   - confirmation baseline selection;
   - confirmation technical cell-count enforcement;
   - v2 analysis CLI dispatch;
   - center-failure gate accounting;
   - v2 throughput estimation;
   - protocol-file hashing;
   - launcher environment mutation.

The central contribution remains:

> **Matched-bypass derivatives identify a basis-invariant ambient rank-one path operator, made probe-complete by the exact LayerNorm structural envelope.**

The predecessor decision explicitly preserves this theorem, the ten-gate accounting rule, the fixed third scale, robust interval aggregation, and the prohibition on replacing the matched-bypass estimator with PIE or donor PCA.

---

# 2. Evidence and contamination boundary

## 2.1 Evidence reviewed

The binding review covered:

1. `analysis/GPTPRO_GREEN_V136_TERMINAL_DECISION_20260825.md`
2. `analysis/GREEN_V200_IMPLEMENTATION_BLOCKERS_20260825.md`
3. `src/green_bridge_spec.py`
4. `src/green_bridge_numerics.py`
5. `src/matched_bypass_gate.py`
6. `src/green_bridge_response_ad.py`
7. `src/green_bridge_dataset.py`
8. `src/exp_green_bridge_gpt2.py`
9. `src/analyze_green_bridge.py`
10. `src/green_bridge_multigpu_worker.py`
11. `src/test_green_bridge_contract.py`
12. `src/launch_green_bridge_v200.sh`

The implementation does contain the intended v2.0.0 identity, three fixed scales, fine-Richardson point estimate, four gate classes, all-ten interval accounting, robust interval analysis, deterministic resplitting, eight-GPU worker assignment, predecessor checks, and a float64 AD module. The blocker report records exactly 200 tests, with 199 passing after strengthening the split test to hash the actual payload.

## 2.2 Binding contamination statement

The following information may be used in this corrigendum:

- source code;
- protocol documents;
- split metadata;
- predecessor hashes and terminal state;
- legacy donor numerical smoke diagnostics;
- GPU memory and AD-route engineering diagnostics;
- synthetic unit-test outputs.

The following remain forbidden inputs to protocol design or implementation choices:

- v1.3.6 behavioral outcomes;
- v1.3.6 PIE correlations;
- v1.3.6 tensor survival outcomes as a threshold-calibration set;
- any unopened v1.3.6 confirmation response;
- any fresh v2.0.0 development response;
- any fresh v2.0.0 confirmation response.

The v1.3.6 PIE signal remains inadmissible for setting radii, bounds, classes, aggregation rules, or confirmation access. The predecessor decision already freezes PIE as a baseline and post hoc diagnostic only.

> **No fresh development or confirmation response may be inspected, computed, materialized, cached, or summarized until this corrigendum has been implemented, committed, hashed, tested, and frozen through a successful corrected prepare phase.**

---

# 3. Issue 1 — binding split-hash corrigendum

## 3.1 Decision

Issue 1 is resolved by **Option B**.

The digest in the predecessor decision is incorrect for the specified payload. It is replaced by:

```text
V200_SPLIT_SHA256 =
0873915c966bef8f54b83d4151a9d7c75b577da5dfc17ee093b9f5c58a9590f7
```

The inactive and forbidden digest is:

```text
f012a286801bc3e3e937b390f0a62d7e92f8d5a21ba59d7e53478ae911e72cfc
```

The implementation currently freezes the incorrect digest even though it contains the intended four development and twelve confirmation group identities.

## 3.2 Exact JSON types

The canonical payload has these exact types:

| Field | Exact JSON type |
|---|---|
| `schema` | string |
| `salt` | string |
| `source_split` | string |
| `development_groups` | array of four objects |
| `confirmation_groups` | array of twelve objects |
| each `noun` | string |
| each `century` | integer |
| each `rank_key` | string |
| `distance_bins` | array of two strings |
| `roles` | array of two strings |
| `records_per_role_per_cell` | integer |

The serializer is exactly:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
```

The UTF-8 bytes of that returned string are hashed directly with SHA-256. There is:

- no trailing newline;
- no indentation;
- no byte-order mark;
- no tuple representation;
- no conversion of centuries to strings;
- no wrapper object;
- no omitted `rank_key`;
- no sorted reordering of the group arrays beyond the explicitly frozen row order.

## 3.3 Exact canonical JSON string

The following is the complete literal canonical string:

```json
{"confirmation_groups":[{"century":12,"noun":"treaty","rank_key":"5419d9cb8844c61db83ae2eae7243dbd16a9c2bf5ee7967401eafd7f70f2475a"},{"century":12,"noun":"warfare","rank_key":"57822f1c018d9552848007996257f81da49ebef54f6e4559dc84fe13312ed2b4"},{"century":14,"noun":"expedition","rank_key":"5f5f6555263c3ee9052d9f1240096f6004091201fdd805b4ede1769481fcc321"},{"century":12,"noun":"kingdom","rank_key":"6c27075f448a87bd7bdb373924e72caba1816a5033625dbe92d1c59d1977dae8"},{"century":16,"noun":"treaty","rank_key":"6c8d9da9bd864657ac675f5b68f65e22f44ac97b0ccec1a4acfa8987a513fb77"},{"century":16,"noun":"kingdom","rank_key":"8571c8283f76806da63c769868b6a34448f6f02ae86d57f8a13db6597cecde00"},{"century":14,"noun":"campaign","rank_key":"9942a20d23a6fb97e7f33390172c7049ebe78341bde1b047a28ae64e997d431b"},{"century":16,"noun":"siege","rank_key":"a19e2bc49bf4b522ae28f500cd6596f5c492e8f817008f0c5985341e55c45741"},{"century":12,"noun":"reign","rank_key":"aa4cd1c743ab745f1278738367fcdd5a3937d36082f92b10aa3144c990001af4"},{"century":14,"noun":"siege","rank_key":"c39c88f5f37a424b7196cf99a4d34062f6150740ede6ab7c97abdd36d2d76d01"},{"century":16,"noun":"campaign","rank_key":"e1d35b6e9b3ec70687d8ed270afec74fc565c633b0d0a43011d507323df4f939"},{"century":16,"noun":"expedition","rank_key":"f7fcfdc5e4306cca1d4b0309c086dc0d6e033b72ce1236d8bb6d1986c362351f"}],"development_groups":[{"century":16,"noun":"dynasty","rank_key":"066e4d0fbd2636a5de7c5587fea60ba6d83c2173fd8a1a3b9598806973ed2596"},{"century":12,"noun":"dynasty","rank_key":"0fe1884c7d56deb8cdcb34d7b4eea65b398a9fdf03fc638f8bbc4a422c6ff6b6"},{"century":14,"noun":"reign","rank_key":"169c8a45b7aea24c90ce94ecefc84aa0588e34831b5074b9a57a4c5380373b51"},{"century":14,"noun":"warfare","rank_key":"36d63a7d439059d0877995705989132fddb35d1cfc9381be110925cacc8776c4"}],"distance_bins":["near","far"],"records_per_role_per_cell":8,"roles":["tensor","energy"],"salt":"green-v200-resplit-20260825","schema":"green-bridge-v2.0.0-resplit-v1","source_split":"green-bridge-v1.3.6-confirmation"}
```

Its SHA-256 is:

```text
0873915c966bef8f54b83d4151a9d7c75b577da5dfc17ee093b9f5c58a9590f7
```

The blocker report independently records the same payload, serializer, corrected digest, 8 development cells, 24 confirmation cells, 64/64 development records, 192/192 confirmation records, and zero overlap with the old development groups.

## 3.4 Split invariants that do not change

The correction changes no group allocation.

### Development groups

```text
dynasty / 16
dynasty / 12
reign   / 14
warfare / 14
```

### Confirmation groups

```text
treaty     / 12
warfare    / 12
expedition / 14
kingdom    / 12
treaty     / 16
kingdom    / 16
campaign   / 14
siege      / 16
reign      / 12
siege      / 14
campaign   / 16
expedition / 16
```

The following remain exact:

```text
development cells                  8
confirmation cells                24
development tensor records        64
development energy records        64
confirmation tensor records      192
confirmation energy records      192
records per role per cell          8
distance bins             near, far
roles                  tensor, energy
```

Both distance bins for one noun-century group must remain in the same phase.

---

# 4. Issue 2 — Richardson enclosure ruling

## 4.1 Current implementation

For a derivative object \(T\), the implementation forms:

\[
R_T^{(c)}
=
\frac{4T_{1/2}-T_1}{3},
\]

\[
R_T^{(f)}
=
\frac{4T_{1/4}-T_{1/2}}{3}.
\]

This fine-Richardson primacy is correct and must remain unchanged. The source explicitly makes the fine estimate from the half and quarter stencils and passes it to `identify_gate`.

The current numerical implementation then constructs, for example,

\[
\epsilon_G^{(f)}
=
\|R_G^{(f)}-G_{1/4}\|_2
+
10\eta_G,
\]

with analogous expressions for \(C\), \(J\), and \(\Delta H\), and requires the coarse and fine balls to overlap. It subsequently takes the minimum of the fine radius and the distance-to-coarse plus the coarse radius.

The current classifier makes non-overlap a hard `numerical-invalid` result, and the prepare preflight stops before AD whenever any of the prescribed coarse/fine balls do not overlap.

## 4.2 Why the current construction is not a valid enclosure

The quantity

\[
\|R_T^{(f)}-T_{1/4}\|
\]

is an observed scale difference. Without an independently established remainder model, it is not an upper bound on

\[
\|R_T^{(f)}-T^\star\|,
\]

where \(T^\star\) is the infinitesimal derivative.

A Richardson error bound requires assumptions such as:

- a valid asymptotic expansion over the used radii;
- bounded higher derivatives over the full stencil region;
- a proven sign or contraction property for successive remainder terms;
- an interval-arithmetic evaluation of the required derivative remainder.

None of these is established in the current numerical module. Its own docstring describes the construction as converting endpoint noise and scale differences into uncertainty bounds, but the code simply adds those quantities rather than proving that they dominate the derivative remainder.

The legacy-donor smoke case produced:

| Object | Coarse/fine center-distance ratio |
|---|---:|
| \(G\) | `2.3718` |
| \(C\) | `3.0029` |
| \(J\) | `2.6480` |
| \(\Delta H\) rows | `2.7523`–`3.1550` |

The independent float64 AD derivative was also outside both prescribed balls; for example, the fine \(C\) discrepancy was `0.26483988` against a prescribed bound of `0.06437722`.

This does not establish that the fine estimator is scientifically useless. It establishes that the claimed ball is too small and is not a justified enclosure.

## 4.3 Binding choice

Issue 2 is resolved by **Option B**:

> The numerical enclosure is mathematically incomplete and must be corrected before launch.

Launching with the known defect and calling the resulting prepare STOP “scientifically intended” is forbidden. That would consume the one-shot on an audit already known not to implement its stated mathematical purpose.

## 4.4 Rejected corrections

The following proposed repairs are explicitly rejected.

### Scalar float32 ULP multiplier

Adding

\[
K\cdot\operatorname{ulp}(Y)
\]

to every endpoint error is not sufficient. The derivative errors involve:

- cancellation between signed endpoints;
- multiple matrix multiplications;
- LayerNorm;
- activation nonlinearities;
- residual additions;
- second derivatives;
- mixed derivatives;
- Richardson subtraction and extrapolation.

A scalar raw-logit ULP allowance does not formally propagate through that graph.

### Empirical inflation factor

Multiplying the old balls by `3.2`, `4`, a smoke-derived maximum, a percentile, or any observed pass-rate factor is forbidden. That would be calibration to a diagnostic outcome rather than a mathematical correction.

### Radius search

Trying additional radii until coarse/fine overlap or white-box compatibility passes is forbidden.

### Choosing coarse instead of fine

The smoke case indicates that the coarse estimator can be closer to AD than the fine estimator. That does not authorize selecting the better-looking estimator. Fine-Richardson remains the immutable point estimator.

### Treating coarse/fine distance as a theorem-backed remainder

A three-scale difference alone cannot provide a universal remainder bound without an added regularity theorem. No such theorem is currently available in the protocol.

---

# 5. Corrected numerical contract

## 5.1 Scientific point estimator remains unchanged

For every tensor record, system, gate, and derivative object:

\[
\boxed{
\widehat T
=
R_T^{(f)}
=
\frac{4T_{1/4}-T_{1/2}}{3}
}
\]

for:

\[
T\in
\left\{
G,\ C,\ J,\ H^P,\ H^C
\right\}.
\]

The matched-bypass response is:

\[
\widehat{\Delta H}
=
\widehat H^P-\widehat H^C.
\]

No AD tensor replaces \(\widehat T\) as the scientific point estimate.

## 5.2 Why an item-level AD certificate is required

A prepare-only panel can detect a defective numerical formula, but a finite panel cannot furnish a universal error bound for unseen items without an empirical calibration step. Such calibration is forbidden.

Therefore the corrected protocol must evaluate an independent numerical derivative reference for every:

```text
tensor item × system × selected gate
```

The AD reference is used only to:

- certify numerical fidelity;
- compute an outcome-blind uncertainty radius;
- test the exact matched-bypass identity;
- distinguish structural contradiction from finite-difference non-identifiability.

It must not:

- replace the fine-Richardson point estimator;
- use behavioral outcomes;
- use PIE;
- use baseline scores;
- select a radius;
- select an estimator;
- change a gate;
- enter the center of any serialized contribution interval.

This converts the AD module from “prepare-only” to **certification-only and phase-local**. It remains isolated from behavioral analysis.

## 5.3 Dual-route AD reference

For each derivative object define:

\[
T_F
=
\text{forward-over-forward float64 AD result},
\]

\[
T_R
=
\text{reverse-over-forward float64 AD result}.
\]

Define the midpoint reference:

\[
\boxed{
T_A
=
\frac{T_F+T_R}{2}.
}
\]

Use these norms:

| Object | Norm |
|---|---|
| \(G\) | vector \(L_2\) |
| \(C\) | vector \(L_2\) |
| \(J\) | Frobenius |
| each row \(\Delta H_i\) | rowwise vector \(L_2\) |

For \(\Delta H\), all route quantities are computed row by row.

## 5.4 Frozen float64 route-consistency guard

Set:

```python
FLOAT64_UNIT_ROUNDOFF = 2.0 ** -53
AD_ROUTE_OPERATION_BUDGET = 65_536
AD_ROUTE_GAMMA = (
    AD_ROUTE_OPERATION_BUDGET * FLOAT64_UNIT_ROUNDOFF
    / (
        1.0
        - AD_ROUTE_OPERATION_BUDGET
        * FLOAT64_UNIT_ROUNDOFF
    )
)
```

Thus:

\[
u_{64}=2^{-53},
\]

\[
N_{\mathrm{AD}}=65536,
\]

\[
\boxed{
\gamma_{\mathrm{AD}}
=
\frac{N_{\mathrm{AD}}u_{64}}
{1-N_{\mathrm{AD}}u_{64}}
=
7.275957614236365\times10^{-12}.
}
\]

For each object define:

\[
d_T
=
\|T_F-T_R\|,
\]

\[
s_T
=
\max
\left(
1,
\|T_F\|,
\|T_R\|
\right).
\]

Require:

\[
\boxed{
d_T
\le
\operatorname{up}
\left(
2\gamma_{\mathrm{AD}}s_T
\right).
}
\]

For every \(\Delta H_i\), apply this independently.

This is a fixed implementation-consistency guard, not a fitted scientific tolerance. The operation budget is frozen before all inferential outcomes and is more than twenty times the largest single GPT-2 reduction dimension used by this local tail.

If the route guard fails:

```text
gate label = numerical-invalid
reason = ad-route-disagreement
```

During prepare, any route failure is:

```text
STOP 08_AD_ROUTE_CONSISTENCY
```

## 5.5 AD reference radius

Define:

\[
\boxed{
r_{A,T}
=
\operatorname{up}
\left(
\frac{d_T}{2}
+
\gamma_{\mathrm{AD}}s_T
\right).
}
\]

This covers both:

- the observed route disagreement around the midpoint;
- the frozen float64 route-consistency allowance.

For \(\Delta H\), compute \(r_{A,\Delta H,i}\) rowwise.

## 5.6 Fine-pair endpoint-repeatability terms

Retain the duplicate-endpoint quantity:

\[
\epsilon_y
=
\max
\left(
10^{-7},
\text{observed duplicate raw-logit maximum error}
\right).
\]

It is now interpreted only as a deterministic endpoint-repeatability contribution. It is not treated as a full float32 truncation-error model.

For the fine Richardson pair define:

\[
h_x^{(f)}
=
\frac{h_x}{2},
\]

\[
h_z^{(f)}
=
\frac{h_z}{2}.
\]

Then retain the existing propagation coefficients:

\[
\eta_G^{(f)}
=
\frac{3\epsilon_y}{h_z^{(f)}},
\]

\[
\eta_C^{(f)}
=
\frac{64\epsilon_y}
{3(h_z^{(f)})^2},
\]

\[
\eta_J^{(f)}
=
\frac{3\epsilon_y}{h_x^{(f)}},
\]

\[
\eta_H^{(f)}
=
\frac{17\epsilon_y}
{3h_x^{(f)}h_z^{(f)}}.
\]

The output-space contributions are:

\[
\nu_G
=
10\eta_G^{(f)},
\]

\[
\nu_C
=
10\eta_C^{(f)},
\]

\[
\nu_J
=
\sqrt{500}\eta_J^{(f)},
\]

\[
\nu_{\Delta H}
=
20\eta_H^{(f)}.
\]

These dimension factors preserve the existing 100-dimensional output and five-row structural-frame calculations. The current source already uses the same coefficients and passes the half-scale radii for the fine pair.

## 5.7 Corrected fine-Richardson uncertainty

The active uncertainty around the fine estimator is:

\[
\boxed{
\bar\epsilon_G
=
\operatorname{up}
\left(
\|\widehat G-G_A\|_2
+
r_{A,G}
+
\nu_G
\right)
}
\]

\[
\boxed{
\bar\epsilon_C
=
\operatorname{up}
\left(
\|\widehat C-C_A\|_2
+
r_{A,C}
+
\nu_C
\right)
}
\]

\[
\boxed{
\bar\epsilon_J
=
\operatorname{up}
\left(
\|\widehat J-J_A\|_F
+
r_{A,J}
+
\nu_J
\right)
}
\]

and, for every structural-frame row \(i\),

\[
\boxed{
\bar\epsilon_{\Delta H,i}
=
\operatorname{up}
\left(
\|
\widehat{\Delta H}_i
-
\Delta H_{A,i}
\|_2
+
r_{A,\Delta H,i}
+
\nu_{\Delta H}
\right).
}
\]

These equations contain no empirical multiplier. They follow from the triangle inequality around a frozen independent computational derivative reference.

The uncertainty may be large. A large uncertainty is not repaired or tuned away. It causes:

- non-invertibility;
- insufficient operator SNR;
- an `unresolved-bounded` classification;
- wider system and cell intervals;
- possible development failure.

That is the scientifically correct consequence.

## 5.8 Status of the coarse estimator

The coarse estimate remains mandatory:

\[
R_T^{(c)}
=
\frac{4T_{1/2}-T_1}{3}.
\]

For every gate, serialize:

\[
d_T^{cf}
=
\|R_T^{(f)}-R_T^{(c)}\|.
\]

Also serialize the historical coarse/fine overlap ratio as:

```text
diagnostic_only = true
active_admissibility_gate = false
```

A ratio above one is no longer a prepare STOP or a gate-level `numerical-invalid` result.

Coarse and fine cell predictors remain subject to the frozen confirmatory radius-stability requirements:

```text
Spearman >= 0.90
median symmetric change <= 0.20
```

This preserves the third-scale purpose without pretending that scale disagreement is itself a proof-valid error enclosure.

---

# 6. Exact matched-bypass theorem audit under the corrected reference

## 6.1 AD-level structural identity

For structural coordinate \(i\), the exact theorem gives:

\[
\Delta H_i=A_iC.
\]

Let:

- \(A_i^{WB}\) be the analytic LayerNorm white-box coordinate;
- \(\epsilon_{WB}=10^{-10}\);
- \(C_A\) be the dual-route AD midpoint;
- \(\Delta H_{A,i}\) be the dual-route AD midpoint;
- \(r_{A,C}\) and \(r_{A,\Delta H,i}\) be their route radii.

Define:

\[
q_i^{AD}
=
\left\|
\Delta H_{A,i}
-
A_i^{WB}C_A
\right\|_2.
\]

The proof-derived bound is:

\[
\boxed{
B_i^{AD}
=
r_{A,\Delta H,i}
+
|A_i^{WB}|r_{A,C}
+
\epsilon_{WB}
\left(
\|C_A\|_2+r_{A,C}
\right).
}
\]

Require:

\[
\boxed{
q_i^{AD}
\le
\operatorname{up}
\left(
B_i^{AD}
\right)
\quad
\text{for all }i.
}
\]

This test does not require curvature inversion.

Failure means:

```text
gate label = structural-contradiction
reason = ad-matched-bypass-factorization
```

This is the decisive theorem-level audit. It distinguishes:

- a finite-difference estimator that is too noisy or biased;
- an actual failure of the matched-bypass identity in the executable map.

## 6.2 Fine-estimator inverse admissibility

Define:

\[
c_{\mathrm{low}}
=
\operatorname{down}
\left(
\|\widehat C\|_2-\bar\epsilon_C
\right).
\]

The response inverse is admissible iff:

\[
\boxed{
c_{\mathrm{low}}>0.
}
\]

When \(c_{\mathrm{low}}\le0\):

- do not call `identify_gate`;
- do not use a pseudoinverse;
- do not add ridge regularization;
- do not declare a structural contradiction;
- classify the gate as either `certified-target-null` or `unresolved-bounded`, provided all non-inversion audits pass.

## 6.3 Identified coordinate bounds

When \(c_{\mathrm{low}}>0\), define:

\[
\boxed{
A_{\max,i}
=
\operatorname{up}
\left(
\frac{
\|\widehat{\Delta H}_i\|_2
+
\bar\epsilon_{\Delta H,i}
}{
c_{\mathrm{low}}
}
\right).
}
\]

For the response estimate:

\[
\widehat A_i
=
\frac{
\langle\widehat C,\widehat{\Delta H}_i\rangle
}{
\|\widehat C\|_2^2
},
\]

define:

\[
\boxed{
\bar\epsilon_{A,i}
=
\operatorname{up}
\left(
\frac{
\bar\epsilon_{\Delta H,i}
+
A_{\max,i}\bar\epsilon_C
}{
\|\widehat C\|_2
}
\right).
}
\]

For the rank-one operator coordinate:

\[
\widehat P_i
=
\widehat A_i\widehat G^\top,
\]

define:

\[
\boxed{
\bar\epsilon_{P,i}
=
\operatorname{up}
\left(
\bar\epsilon_GA_{\max,i}
+
\|\widehat G\|_2\bar\epsilon_{A,i}
\right).
}
\]

Finally:

\[
\boxed{
\bar\epsilon_{P,F}
=
\operatorname{up}
\left(
\|
\bar\epsilon_P
\|_2
\right).
}
\]

The algebra in `identify_gate` remains unchanged. The current implementation correctly uses the rank-one response estimator without a pseudoinverse or learned replacement. The correction is to its uncertainty certificate, not to the central operator.

---

# 7. Corrected compatibility tests

## 7.1 Fine-estimator factorization

Define:

\[
q_i^{fac}
=
\left\|
\widehat{\Delta H}_i
-
\widehat A_i\widehat C
\right\|_2.
\]

The corrected bound is:

\[
\boxed{
B_i^{fac}
=
\bar\epsilon_{\Delta H,i}
+
\bar\epsilon_{A,i}
\|\widehat C\|_2
+
A_{\max,i}\bar\epsilon_C.
}
\]

Require:

\[
q_i^{fac}
\le
\operatorname{up}
\left(
B_i^{fac}
\right).
\]

This retains the proof-derived factorization rule from the predecessor decision but supplies valid uncertainty inputs.

## 7.2 Response-versus-white-box coordinate agreement

Require:

\[
\boxed{
|
\widehat A_i-A_i^{WB}
|
\le
\operatorname{up}
\left(
\bar\epsilon_{A,i}+\epsilon_{WB}
\right).
}
\]

The inactive v1.3.6 relative threshold `0.05` remains historical only.

## 7.3 Direct white-box factorization

Define:

\[
q_i^{WB\text{-}fac}
=
\left\|
\widehat{\Delta H}_i
-
A_i^{WB}\widehat C
\right\|_2.
\]

Require:

\[
\boxed{
q_i^{WB\text{-}fac}
\le
\operatorname{up}
\left[
\bar\epsilon_{\Delta H,i}
+
|A_i^{WB}|\bar\epsilon_C
+
\epsilon_{WB}
\left(
\|\widehat C\|_2+\bar\epsilon_C
\right)
\right].
}
\]

## 7.4 Shift-null identity

For the exact LayerNorm shift-null coordinate:

\[
\boxed{
|\widehat A_{\mathrm{shift}}|
\le
\operatorname{up}
\left(
\bar\epsilon_{A,\mathrm{shift}}
+
\epsilon_{WB}
\right).
}
\]

No empirical absolute floor is authorized.

---

# 8. Zero-denominator and outward-rounding rules

## 8.1 Required primitives

Add exactly these helpers:

```python
def round_up(value: float) -> float:
    value64 = np.float64(value)
    if np.isnan(value64):
        raise ValueError("cannot outward-round NaN")
    return float(np.nextafter(value64, np.inf))


def round_down(value: float) -> float:
    value64 = np.float64(value)
    if np.isnan(value64):
        raise ValueError("cannot outward-round NaN")
    return float(np.nextafter(value64, -np.inf))


def add_up(*values: float) -> float:
    total = np.float64(0.0)
    for value in values:
        if not np.isfinite(value) or value < 0:
            raise ValueError("add_up expects finite nonnegative values")
        total = np.nextafter(total + np.float64(value), np.inf)
    return float(total)


def multiply_up(left: float, right: float) -> float:
    if (
        not np.isfinite(left)
        or not np.isfinite(right)
        or left < 0
        or right < 0
    ):
        raise ValueError(
            "multiply_up expects finite nonnegative values"
        )
    return round_up(np.float64(left) * np.float64(right))


def norm_up(value) -> float:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("norm input must be finite")
    return round_up(np.linalg.norm(array))


def subtract_down(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("subtract_down expects finite values")
    return round_down(np.float64(left) - np.float64(right))
```

## 8.2 Compatibility ratio

Use exactly:

```python
def compatibility_ratio(
    residual: float,
    bound: float,
) -> float:
    if (
        np.isnan(residual)
        or np.isnan(bound)
        or residual < 0.0
        or bound < 0.0
    ):
        raise ValueError(
            "residual and bound must be nonnegative"
        )

    if bound == 0.0:
        return 0.0 if residual == 0.0 else math.inf

    return float(residual / bound)
```

Pass iff:

```python
residual <= round_up(bound)
```

## 8.3 Required branches

| Condition | Binding result |
|---|---|
| residual \(=0\), bound \(=0\) | pass, ratio `0.0` |
| residual \(>0\), bound \(=0\) | fail, ratio `inf` |
| negative bound | implementation error |
| NaN | implementation error |
| positive infinity in an active point certificate | not active-identifiable |
| \(c_{\mathrm{low}}\le0\) | non-invertible; null/unresolved candidate |
| AD route guard failure | numerical-invalid |
| AD theorem identity failure | structural-contradiction |

## 8.4 Interval arithmetic

For every contribution or prediction interval:

- sum lower endpoints with downward rounding;
- sum upper endpoints with upward rounding;
- divide lower means downward;
- divide upper means upward;
- subtract intervals as:

  \[
  [L_1,U_1]-[L_2,U_2]
  =
  [
  \operatorname{down}(L_1-U_2),
  \operatorname{up}(U_1-L_2)
  ];
  \]

- take absolute-value intervals as:

  \[
  |[L,U]|
  =
  \begin{cases}
  [0,\operatorname{up}(\max(-L,U))],&L\le0\le U,\\
  [\operatorname{down}(\min(|L|,|U|)),
  \operatorname{up}(\max(|L|,|U|))],&\text{otherwise}.
  \end{cases}
  \]

The current interval architecture—Minkowski all-ten aggregation, absolute value after the `pat`/`tar` mean difference, worst-case RMSE, and robust interval AUROC—is scientifically correct in structure and must be retained. The current development code already uses worst-case interval RMSE.

---

# 9. Issue 3 — float64 AD dtype ruling

## 9.1 Decision

The finding that TransformerLens consults `model.cfg.dtype` is accepted.

Setting the AD model’s configuration dtype to float64 is **approved and required**. The reduction of forward/reverse route differences from approximately \(10^{-7}\) to \(10^{-14}\)–\(10^{-16}\) is strong evidence that the original route disagreement was an internal dtype inconsistency rather than a mathematical derivative mismatch.

The current engineering implementation is nevertheless not approved because it executes:

```python
model.double()
model.cfg.dtype = torch.float64
...
model.float()
model.cfg.dtype = scientific_cfg_dtype
```

on the active scientific model.

## 9.2 Required correction

The active float32 scientific model must never be cast in place.

Create an isolated AD-local model containing deep copies of exactly the required tail components:

```text
block 10
block 11
ln_final
unembed
output-softcap configuration
required model configuration fields
```

The clone must:

```python
ad_model = copy.deepcopy(local_tail)
ad_model.to(device=device, dtype=torch.float64)
ad_model.cfg = copy.deepcopy(local_tail.cfg)
ad_model.cfg.dtype = torch.float64
```

Every floating-point anchor tensor passed to the AD model must be a newly allocated float64 tensor.

The active model must remain:

```text
parameter dtype = float32
buffer dtype = frozen original
model.cfg.dtype = float32
```

before, during, and after AD evaluation.

## 9.3 Integrity assertions

Before constructing the AD clone, hash:

```text
all active model parameter bytes
all active model buffer bytes
serialized active model.cfg fields
```

After destroying the clone, recompute the hashes and require exact equality.

Prepare artifact fields:

```json
{
  "active_model_parameter_hash_before": "...",
  "active_model_parameter_hash_after": "...",
  "active_model_buffer_hash_before": "...",
  "active_model_buffer_hash_after": "...",
  "active_model_config_hash_before": "...",
  "active_model_config_hash_after": "...",
  "active_model_unchanged": true,
  "ad_local_parameter_dtype": "float64",
  "ad_local_cfg_dtype": "float64",
  "scientific_model_dtype": "float32"
}
```

Any mismatch is:

```text
STOP 08B_AD_MODEL_ISOLATION
```

## 9.4 Exception safety

The clone must be created and destroyed within an exception-safe context manager:

```python
with isolated_ad_tail_v200(
    scientific_model,
    anchor,
) as ad_tail:
    ...
```

The context manager must not depend on “restoring” modified scientific parameters. It must mutate only the isolated object.

---

# 10. Four-class gate classification after the corrigendum

## 10.1 `numerical-invalid`

A gate is `numerical-invalid` if:

- any base, half, quarter, coarse, fine, forward-AD, or reverse-AD tensor is nonfinite;
- the forward/reverse AD route guard fails;
- the active float32 model changes during AD;
- endpoint, hook, frame, or raw-logit equivalence fails.

Coarse/fine non-overlap alone is no longer sufficient.

## 10.2 `structural-contradiction`

A gate is `structural-contradiction` if:

- the AD-level matched-bypass factorization exceeds \(B_i^{AD}\);
- the fine response inverse is admissible and any corrected:
  - factorization;
  - response/white-box;
  - direct white-box factorization;
  - shift-null

  inequality fails;
- the analytic/autograd LayerNorm coordinate error exceeds \(10^{-10}\).

A structural contradiction may not become unresolved.

## 10.3 `unresolved-bounded`

A gate is `unresolved-bounded` if:

- all numerical and structural-identity checks pass;
- it is not certified null;
- it cannot satisfy active identification because of:
  - non-invertible curvature;
  - insufficient materiality;
  - insufficient response SNR;
  - insufficient operator SNR;
  - a wide but finite corrected error interval;
- its full target-contribution upper bound is finite.

Its signed point center remains exactly zero.

## 10.4 `active-identified`

The gate uses:

\[
\widehat{\mathcal P}
=
\widehat G\widehat g^\top,
\]

where \(\widehat g=Q\widehat A\), and all active materiality, SNR, compatibility, and contribution conditions remain unchanged.

## 10.5 `certified-target-null`

A gate is null only if its complete outward-rounded target-contribution upper bound is at most:

\[
0.005.
\]

Non-invertibility alone is not nullity.

---

# 11. Contribution bounds and all-ten accounting

## 11.1 Unresolved contribution

Let:

- \(\ell\) be the frozen output contrast;
- \(v\) be the frozen physical target vector;
- \(g^{WB}\) be the analytic ambient gradient;
- \(\epsilon_{g,WB}=\sqrt{768}\times10^{-10}\).

Define:

\[
\boxed{
U_j
=
\operatorname{up}
\left[
\left(
|\ell^\top\widehat G_j|
+
\|\ell\|_2\bar\epsilon_{G,j}
\right)
\left(
|(g_j^{WB})^\top v|
+
\epsilon_{g,WB}\|v\|_2
\right)
\right].
}
\]

The unresolved interval is:

\[
\boxed{
I_j=[-U_j,U_j].
}
\]

White-box sign is not used as the center.

## 11.2 Active contribution

For an active gate:

\[
\widehat\theta_j
=
\ell^\top\widehat{\mathcal P}_jv.
\]

Retain the structural-envelope contraction error, using corrected \(\bar\epsilon_{P,F}\) and \(\bar\epsilon_G\), and outward-round its result to \(B_j\).

Then:

\[
I_j
=
[
\operatorname{down}(\widehat\theta_j-B_j),
\operatorname{up}(\widehat\theta_j+B_j)
].
\]

## 11.3 Null contribution

A null gate receives:

\[
I_j=[-U_j,U_j]
\]

with:

\[
U_j\le0.005.
\]

Its point center is zero.

## 11.4 System interval

All ten selected gates must occur exactly once in:

```text
gates
```

For one system:

\[
I_s
=
\sum_{j=1}^{10}I_j.
\]

A system is set-admissible only if:

- no gate is numerical-invalid;
- no gate is structurally contradictory;
- every gate is active, null, or unresolved;
- at least three gates are active;
- the common-frame bypass disagreement remains at most `0.15`;
- every interval endpoint is finite.

The implementation’s existing all-ten set-accounting architecture is retained. No gate may be silently dropped.

---

# 12. Additional implementation defects and binding repairs

## 12.1 Stale predecessor identity

`green_bridge_spec.py` currently declares `PREDECESSOR_RUN` as v1.3.5 with first failure `11_MULTIGPU_WORKER`, even though the parent protocol is v1.3.6.

Replace it with:

```python
PREDECESSOR_RUN = {
    "schema_version":
        "green-bridge-v1.3.6",
    "protocol_id":
        "structural-envelope-matched-bypass-v1.3.6",
    "protocol_run_id":
        "green-bridge-v1.3.6-one-shot",
    "attempt_index": 1,
    "retry_allowed": False,
    "verdict": "STOP_ORAL",
    "first_failed_gate":
        "12_DEVELOPMENT_SURVIVAL",
    "scientific_spec_sha256":
        "60ca5e9e221064f288a1993ee3cbf42e99330bbf6f9008946a25556438cbc3d3",
    "frozen_spec_sha256":
        "cb771c59e91b4fc553ef73a1c7a116ec0ee55f499ce46a2f91e4c600cd8bd41d",
    "confirmation_started": False,
    "artifact_sha256": {
        "dev_tensor_scores.parquet":
            "660788dde8bc5df1d057db31b4dc1065b222ac7777efc0e4c6220e09f1ed81ff",
        "dev_energy_targets.parquet":
            "23a99b6998ec2c51184ae26b8f86a7656247ff2091e251752c1fccd06295e593",
        "dev_cells.json":
            "1294a76d6d79c81f240c20c4257aa6b0fe76457d46b30cfc5d5699e27759ae1f",
        "dev_result.json":
            "2e15531d62bd5cc1162980fdaa2643a7300b362eb6b11ff5b94bb3d623c37277",
        "development_multigpu_merge.json":
            "31dbc71fbeaa40f313be6078a627082050aa5a338e132d3a6ed7343869eaad7a",
    },
}
```

## 12.2 Protocol-file omissions

Add both:

```text
analysis/GREEN_V200_IMPLEMENTATION_BLOCKERS_20260825.md
analysis/GPTPRO_GREEN_V200_CORRIGENDUM_DECISION_20260825.md
```

to `PROTOCOL_FILES`.

The current protocol list ends at the predecessor decision and therefore does not hash either the blocker evidence or this binding correction.

## 12.3 Center-failure accounting bug

When the center audit fails, the current return reports:

```text
contradictory_gates = 10
numerical_invalid_gates = 0
```

even though the individual gate rows are numerical-invalid.

Replace it with:

```text
contradictory_gates = 0
numerical_invalid_gates = 10
```

and:

```text
reason = center-noop-failure
```

A center executable-equivalence failure is numerical invalidity, not evidence against the theorem.

## 12.4 Confirmation baseline reselection

Development correctly selects the baseline with minimum development LOOCV RMSE.

The current confirmation code recomputes the best baseline on confirmation, and repeats that selection inside every bootstrap replicate and distance bin.

This is forbidden confirmation-set model selection.

`freeze_confirmation_v200` must serialize:

```json
{
  "frozen_best_baseline": "<development-selected name>",
  "frozen_best_baseline_loocv_rmse": 0.0,
  "baseline_calibration": {
    "...": {
      "alpha": 0.0,
      "beta": 0.0
    }
  }
}
```

Confirmation must use only:

```python
best_name = frozen["frozen_best_baseline"]
best_prediction = predictions[best_name]
```

for:

- overall RMSE;
- relative gain;
- absolute gain;
- all 100,000 bootstrap replicates;
- both distance-bin calculations;
- all per-bin bootstrap replicates.

Other baselines may be reported as diagnostic columns but may not become the comparator after development.

## 12.5 Confirmation technical counts

Before computing confirmation performance, require all three:

```text
surviving cells       >= 21
conditioned cells     >= 21
cells in near bin     >= 11
cells in far bin      >= 11
```

The current v2 function checks only total surviving cells at its early technical gate.

After combining development and confirmation, require:

```text
combined surviving cells >= 27
```

The oral-level requirements remain:

```text
surviving cells       >= 22
conditioned cells     >= 22
```

plus every existing gain, bootstrap, bin, AUROC, and radius-stability criterion.

## 12.6 Analysis CLI dispatch

The current CLI dispatches the historical `development_decision` and `confirmation_decision`, not the v2.0.0 functions.

Add:

```python
parser.add_argument(
    "--protocol-version",
    choices=("v136", "v200"),
    required=True,
)
```

Dispatch exactly:

```python
if args.protocol_version == "v200":
    if args.phase == "development":
        result = development_decision_v200(payload)
    else:
        result = confirmation_decision_v200(
            payload,
            frozen,
        )
else:
    ...
```

The launcher and all v2.0.0 subprocess calls must pass:

```text
--protocol-version v200
```

## 12.7 Throughput preflight

The current prepare path reads the historical v1.3.6 throughput artifact, changes its schema label, and writes it as v2.0.0.

That is invalid because v2.0.0 adds:

- a third finite-difference scale;
- a different tensor-item call count;
- dual-route AD certification.

Replace it with an actual v2.0.0 benchmark.

### Benchmark population

Use eight legacy donor records outside every v2.0.0 inferential cell:

- four `near`;
- four `far`;
- selected by minimum hash:

  ```python
  sha256(
      (
          "green-v200-corrigendum-throughput|"
          + pair_digest
      ).encode("utf-8")
  ).hexdigest()
  ```

No behavioral or baseline field may be loaded.

### Benchmark workload

For every selected record execute:

- the complete `_tensor_item_v200` finite-difference workload;
- all twenty system/gate AD certificates;
- interval construction;
- one complete energy-item workload.

### Hard limits

Require:

```text
peak allocated memory per RTX 4090 <= 20 GiB
projected total eight-GPU scientific wall time <= 24 hours
selected-projection fallback = false
lower-precision fallback = false
reduced-gate fallback = false
radius fallback = false
```

The forecast must separately report:

```text
prepare seconds
development seconds
confirmation seconds
total seconds
finite-difference seconds
AD certification seconds
analysis seconds
```

A failed forecast is terminal. It does not authorize removing AD certification.

## 12.8 Launcher environment mutation

The launcher currently executes three `pip install` commands every phase.

Remove them.

The launcher must only validate the already-frozen environment through:

```bash
python -m pip check
```

and an exact version/hash audit.

No package installation, upgrade, dependency resolution, or network package access may occur during prepare, development, or confirmation.

---

# 13. Protocol identity after this corrigendum

Because no official root exists and no one-shot phase has been claimed, the protocol identity remains:

```text
SCHEMA_VERSION
    green-bridge-v2.0.0

PROTOCOL_ID
    structural-envelope-matched-bypass-setid-v2.0.0

PARENT_PROTOCOL_ID
    structural-envelope-matched-bypass-v1.3.6

PROTOCOL_RUN_ID
    green-bridge-v2.0.0-one-shot

ATTEMPT_INDEX
    1

RETRY_ALLOWED
    false

OUTPUT_ROOT
    outputs/green_bridge_v200
```

Add:

```python
DECISION_ID = (
    "GPTPRO-GREEN-V136-TERMINAL-SETID-v1-20260825"
)

CORRIGENDUM_ID = (
    "GPTPRO-GREEN-V200-CORRIGENDUM-v1-20260825"
)

AMENDMENT_ID = CORRIGENDUM_ID

NUMERICAL_ERROR_CONTRACT = (
    "dual-route-ad-certified-fine-richardson-v1"
)
```

The manifest must contain both the original decision and this corrigendum.

The reviewed commit is not an execution commit. The corrected clean descendant commit becomes the only authorized execution commit.

---

# 14. Updated operation counts

The current finite-difference counts are internally consistent with:

```text
tensor items total                   256
energy items total                   256
finite calls per tensor item       5,220
tensor finite-tail calls       1,336,320
energy finite-tail calls           3,840
total finite-tail calls         1,340,160
JVP invocations                      768
full-model evaluations             1,664
raw invocations                1,342,592
effective units                1,343,360
development effective units      335,840
confirmation effective units   1,007,520
```

These values are already present in the implementation.

Add the following separately; AD calls must not be disguised as finite-tail units:

```text
prepare AD gate-system certificates         40
development AD gate-system certificates  1,280
confirmation AD gate-system certificates 3,840
total AD gate-system certificates        5,160

AD routes per gate-system                     2
prepare AD GateJet routes                    80
development AD GateJet routes             2,560
confirmation AD GateJet routes            7,680
total AD GateJet routes                   10,320

derivative objects per route                   5
prepare top-level AD derivative calls         400
development top-level AD derivative calls  12,800
confirmation top-level AD derivative calls 38,400
total top-level AD derivative calls        51,600
```

The five derivative objects are:

```text
G
C
J
H_path
H_control
```

Any operation-count artifact omitting these AD calls is invalid.

---

# 15. File-by-file binding implementation instructions

## 15.1 `analysis/GPTPRO_GREEN_V200_CORRIGENDUM_DECISION_20260825.md`

Add this document verbatim.

## 15.2 `src/green_bridge_spec.py`

Make all of these changes:

1. replace `V200_SPLIT_SHA256` with:

   ```python
   V200_SPLIT_SHA256 = (
       "0873915c966bef8f54b83d4151a9d7c75"
       "b577da5dfc17ee093b9f5c58a9590f7"
   )
   ```

2. replace the stale predecessor object with the v1.3.6 object in Section 12.1;

3. add:

   ```python
   CORRIGENDUM_ID = (
       "GPTPRO-GREEN-V200-CORRIGENDUM-v1-20260825"
   )

   AMENDMENT_ID = CORRIGENDUM_ID

   NUMERICAL_ERROR_CONTRACT = (
       "dual-route-ad-certified-fine-richardson-v1"
   )

   FLOAT64_UNIT_ROUNDOFF = 2.0 ** -53
   AD_ROUTE_OPERATION_BUDGET = 65_536
   AD_ROUTE_GAMMA = (
       AD_ROUTE_OPERATION_BUDGET
       * FLOAT64_UNIT_ROUNDOFF
       / (
           1.0
           - AD_ROUTE_OPERATION_BUDGET
           * FLOAT64_UNIT_ROUNDOFF
       )
   )
   ```

4. move:

   ```text
   DYADIC_BALL_OVERLAP_RATIO_MAX
   ```

   into a diagnostic/historical namespace;

5. retain:

   ```text
   QUARTER_RADIUS_MULTIPLIER = 0.25
   FACTORIZATION_COMPATIBILITY_RATIO_MAX = 1.0
   WHITEBOX_COMPATIBILITY_RATIO_MAX = 1.0
   WHITEBOX_FACTORIZATION_RATIO_MAX = 1.0
   ```

6. add all AD operation counts from Section 14 to the frozen spec;

7. include the corrected split hash, corrigendum ID, and numerical contract in `FROZEN_SPEC`.

## 15.3 `src/green_bridge_numerics.py`

Retain historical v1 functions for predecessor reproducibility.

Add:

```python
@dataclass(frozen=True)
class ADRouteCertificateV200:
    forward: GateJet
    reverse: GateJet
    reference: GateJet
    route_difference_G: float
    route_difference_C: float
    route_difference_J: float
    route_difference_delta_H: np.ndarray
    route_radius_G: float
    route_radius_C: float
    route_radius_J: float
    route_radius_delta_H: np.ndarray
    route_pass_G: bool
    route_pass_C: bool
    route_pass_J: bool
    route_pass_delta_H: np.ndarray


@dataclass(frozen=True)
class ADCertifiedEnclosureV200:
    fine_jet: GateJet
    ad_reference: GateJet
    epsilon_G: float
    epsilon_C: float
    epsilon_J: float
    epsilon_delta_H: np.ndarray
    inverse_lower_bound: float
    inverse_admissible: bool
    A_max: np.ndarray
    epsilon_A: np.ndarray
    epsilon_P: np.ndarray
    epsilon_P_F: float
```

Add:

```text
round_up
round_down
add_up
multiply_up
norm_up
subtract_down
ad_route_certificate_v200
ad_certified_enclosure_v200
ad_matched_bypass_compatibility_v200
```

`ad_certified_enclosure_v200` must implement Section 5.7 literally.

`dyadic_enclosure_v200` becomes diagnostic-only. No active classifier may consume its `overlap_*` fields.

Update all:

```text
active_envelope_contraction_bound
certified_null_bound
unresolved_gate_contraction_bound_v200
minkowski_sum_interval
subtract_intervals
absolute_value_interval
worst_case_interval_rmse
```

to use outward rounding.

## 15.4 `src/matched_bypass_gate.py`

Do not alter:

```python
identify_gate
extrapolate_gate_jet
operator_action
reconstruct_cotangent
```

Do not add:

- pseudoinverse;
- ridge;
- rank truncation;
- donor PCA;
- learned alignment;
- estimator switching.

Extend v2 dataclasses only as needed to serialize:

```text
fine point estimate
AD certificate
corrected uncertainty
gate class
contribution interval
```

## 15.5 `src/green_bridge_response_ad.py`

Change the module description from:

```text
prepare-only
```

to:

```text
outcome-blind numerical-certification module
```

It remains forbidden from importing:

```text
analyze_green_bridge
pandas
pyarrow
behavioral fields
PIE
baseline fields
```

Add:

```text
isolated_ad_tail_v200
active_model_integrity_hash_v200
ad_route_certificate_v200
```

`build_ad_response_functions_v200` must receive only the isolated float64 tail.

Retain both independent routes.

Remove the old requirement that AD must lie inside the coarse and fine historical balls.

The 40-stratum prepare audit must instead require:

- route consistency;
- finite tensors;
- exact local-tail endpoint;
- AD matched-bypass compatibility;
- white-box coordinate compatibility;
- unchanged active scientific model.

During development and confirmation, the same AD certificate must be computed for every tensor gate/system before behavioral fields are attached to the result row.

## 15.6 `src/green_bridge_dataset.py`

Preserve the group allocation and record construction.

Change only the active split digest and strengthen validation to compare:

```python
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
actual = hashlib.sha256(
    canonical.encode("utf-8")
).hexdigest()
```

Require:

```text
actual == 087391...
```

Also require exact equality between the generated payload and the literal typed schema in Section 3.

## 15.7 `src/exp_green_bridge_gpt2.py`

Make all of these changes:

1. add the blocker report and corrigendum to `PROTOCOL_FILES`;

2. import the corrected AD certificate and enclosure functions;

3. retain `_gate_jet_triplet_v200` fine primacy;

4. replace active `dyadic_enclosure` use with:

   ```text
   coarse_fine_diagnostic
   ad_certified_enclosure
   ```

5. compute the AD certificate for each gate before `_classify_gate_v200`;

6. prohibit `_classify_gate_v200` from using behavioral, PIE, first-order, or baseline values;

7. apply the four-class rules in Section 10;

8. fix center-failure counts;

9. store only numerical summaries needed for certification—no full AD prediction tensor is used as the point estimator;

10. update prepare AD artifacts;

11. replace the copied v1.3.6 throughput report with an actual v2.0.0 benchmark;

12. serialize all finite and AD operation counts;

13. reverify the active model hash after every worker’s AD clone is destroyed;

14. ensure a structural contradiction is never converted to unresolved;

15. preserve exact batch-one finite execution and deterministic eight-GPU assignment.

The existing assignment algorithm—sorting by the role-specific hash and distributing round-robin across physical GPUs 0–7—is correct and must remain unchanged.

## 15.8 `src/analyze_green_bridge.py`

Make all of these changes:

1. preserve development worst-case interval RMSE;

2. freeze the development-selected best baseline name;

3. use only that baseline in confirmation;

4. enforce all technical count gates before performance analysis;

5. retain robust interval AUROC;

6. retain 100,000 bootstrap replicates and seed `20260805`;

7. fix CLI version dispatch;

8. reject a frozen-analysis payload lacking:

   ```text
   frozen_best_baseline
   source_sha256
   protocol_sha256
   split_sha256
   execution_commit
   development_result_sha256
   confirmation_retries = 0
   ```

## 15.9 `src/green_bridge_multigpu_worker.py`

For every worker:

1. load one immutable float32 scientific model;

2. capture and hash its state;

3. create one isolated float64 AD-tail template;

4. create record-specific float64 anchors without changing the scientific model;

5. execute finite fine-Richardson first;

6. execute AD certification before attaching behavioral data;

7. destroy AD intermediates after each gate or bounded group of gates;

8. recheck scientific-model hashes before writing the worker artifact;

9. preserve exact role-stratified assignments and batch-one endpoint ledgers.

## 15.10 `src/test_green_bridge_contract.py`

Implement the test changes in Section 16.

## 15.11 `src/launch_green_bridge_v200.sh`

Remove:

```bash
python -m pip install --upgrade "pip==25.1.1"
python -m pip install "torch==2.7.1" ...
python -m pip install -r ...
```

The current launcher performs these mutations before checking source hashes.

Replace with:

```bash
python -m pip check
```

and exact package-version validation.

Retain:

```text
coordinator GPU 4
worker GPUs 0–7
offline Hugging Face mode
frozen conda environment
deterministic CUDA environment
phase choices prepare/development/confirmation
```

---

# 16. Required tests

## 16.1 Existing tests

All existing 200 tests remain, subject to these revisions:

- split test expects the corrected actual payload digest;
- dyadic-overlap test verifies diagnostic serialization, not a hard gate;
- AD enclosure test verifies the corrected fine-to-AD uncertainty;
- predecessor test uses v1.3.6;
- protocol-file test includes the blocker report and this corrigendum;
- center-failure test expects ten numerical-invalid gates;
- confirmation test freezes the development-selected baseline;
- launcher test forbids package installation.

## 16.2 Twenty additional tests

Add exactly these tests:

1. `V200SplitCorrigendumTests.test_literal_canonical_payload_hash_is_087391`
2. `V200SplitCorrigendumTests.test_f012_digest_is_not_active`
3. `V200ADCertificateTests.test_local_tail_cfg_dtype_is_float64`
4. `V200ADCertificateTests.test_active_scientific_model_state_is_bitwise_unchanged`
5. `V200ADCertificateTests.test_route_guard_uses_gamma_65536`
6. `V200ADCertificateTests.test_route_excess_is_numerical_invalid`
7. `V200ADCertificateTests.test_fine_richardson_remains_point_center`
8. `V200ADCertificateTests.test_fine_error_is_ad_distance_plus_route_and_endpoint_terms`
9. `V200ADCertificateTests.test_coarse_fine_nonoverlap_is_diagnostic_only`
10. `V200ADCertificateTests.test_ad_whitebox_factorization_excess_is_structural_contradiction`
11. `V200OutwardRoundingTests.test_zero_bound_zero_residual_passes`
12. `V200OutwardRoundingTests.test_zero_bound_positive_residual_fails`
13. `V200OutwardRoundingTests.test_nonpositive_inverse_lower_bound_is_unresolved_not_contradiction`
14. `V200ConfirmationFreezeTests.test_best_baseline_is_frozen_from_development`
15. `V200ConfirmationFreezeTests.test_confirmation_cannot_reselect_best_baseline`
16. `V200ConfirmationFreezeTests.test_bootstrap_and_per_bin_use_frozen_baseline`
17. `V200ConfirmationFreezeTests.test_technical_counts_include_conditioned_bins_and_combined`
18. `V200ThroughputContractTests.test_v200_benchmark_executes_fine_tensor_and_dual_ad`
19. `V200OperationCountTests.test_dual_ad_route_counts_are_exact`
20. `V200AnalysisCLITests.test_cli_requires_v200_dispatch`

Required result:

```text
Ran 220 tests
OK
```

No test may be:

- skipped;
- marked expected failure;
- removed;
- weakened to source-string presence only when executable behavior can be tested;
- passed by monkeypatching an active threshold.

---

# 17. Corrected prepare gates

Prepare may begin only after:

```text
corrected implementation committed
branch main
clean worktree
official root absent
all 220 tests pass
predecessor hashes pass
```

Prepare must pass all of the following.

## 17.1 Identity and provenance

```text
schema_version == green-bridge-v2.0.0
protocol_id ==
    structural-envelope-matched-bypass-setid-v2.0.0
protocol_run_id ==
    green-bridge-v2.0.0-one-shot
attempt_index == 1
retry_allowed == false
corrigendum_id exact
numerical_error_contract exact
execution commit is corrected descendant
```

## 17.2 Split firewall

```text
canonical digest == 087391...
development groups exact
confirmation groups exact
old v1.3.6 development overlap == 0
development records == 128 total
confirmation records == 384 total
```

No confirmation prompt may be run.

## 17.3 Existing equivalence gates

Retain:

- Gate-04 HF/TransformerLens audit;
- same-TransformerLens no-op;
- manual-tail raw-logit equivalence;
- exact batch-one equivalence;
- structural-frame dimensions;
- structural-envelope residuals;
- analytic/autograd LayerNorm agreement;
- repeated-frame determinism.

## 17.4 Forty-stratum AD prepare panel

For every one of the 40 frozen strata:

- both AD routes complete;
- local AD config dtype is float64;
- active scientific model remains byte-identical;
- all derivatives are finite;
- route consistency passes;
- AD matched-bypass factorization passes;
- white-box coordinates pass;
- fine-Richardson uncertainty is computed;
- coarse/fine distances are serialized;
- no behavior or baseline field is read.

Required misses:

```text
0
```

Coarse/fine historical ball non-overlap is recorded but does not itself fail prepare.

## 17.5 Throughput and memory

The corrected actual benchmark must pass:

```text
peak allocated <= 20 GiB
projected total eight-GPU wall time <= 24 hours
```

No fallback is allowed.

## 17.6 Prepare artifacts

Required artifacts include:

```text
run_ledger.json
predecessor_v136_manifest.json
model_fingerprint.json
v200_split.json
scientific_delta_v200.json
numerical_contract_v200.json
gate04_legacy_panel.json
hook_audit.json
manual_tail_equivalence.json
structural_frame_preflight.json
response_ad_route_audit_v200.json
response_ad_theorem_audit_v200.json
three_scale_numerical_preflight_v200.json
active_model_integrity_audit_v200.json
hardware_plan.json
throughput_preflight.json
operation_counts_v200.json
prepare_result.json
manifest.json
sha256sums.txt
```

After prepare, no development or confirmation result artifact may exist.

---

# 18. Development consequences

Development remains:

```text
8 cells
64 tensor records
64 energy records
```

Every worker receives:

```text
8 tensor records
8 energy records
```

Development performs:

```text
1,280 gate-system AD certificates
2,560 AD GateJet routes
12,800 top-level AD derivative calls
```

For every tensor record:

1. finite base/half/quarter jets are computed;
2. fine Richardson is fixed;
3. dual-route AD is computed;
4. corrected uncertainty is built;
5. theorem identity is tested;
6. all ten gates are classified and accounted for;
7. only then may the row’s behavioral and baseline fields be attached.

The development gates remain:

```text
surviving cells >= 8
conditioned cells >= 8
set-SNR cells >= 5
```

Performance remains:

```text
robust relative gain < 0.05
    STOP_ORAL

0.05 <= robust relative gain < 0.10
    POSTER_ONLY

robust relative gain >= 0.10
and every technical gate passes
    OPEN_CONFIRMATION
```

Only `OPEN_CONFIRMATION` creates `frozen_analysis.json`.

The frozen analysis must contain the exact development-selected baseline name.

---

# 19. Confirmation consequences

Confirmation remains completely closed unless corrected development returns:

```text
OPEN_CONFIRMATION
```

Confirmation contains:

```text
24 cells
192 tensor records
192 energy records
```

It performs:

```text
3,840 gate-system AD certificates
7,680 AD GateJet routes
38,400 top-level AD derivative calls
```

Before performance analysis, require:

```text
surviving cells >= 21
conditioned cells >= 21
near cells >= 11
far cells >= 11
combined development+confirmation surviving cells >= 27
```

The active baseline comparator is the development-frozen baseline only.

The oral-level requirements remain unchanged:

```text
surviving cells >= 22
conditioned cells >= 22
robust relative gain >= 0.20
relative-gain bootstrap LCB >= 0.10
robust absolute gain >= 0.01
per-bin robust relative gain >= 0.10
per-bin relative-gain LCB > 0
per-bin robust absolute gain >= 0.005
robust cancellation AUROC lower bound >= 0.80
robust cancellation AUROC bootstrap LCB >= 0.70
coarse/fine cell Spearman >= 0.90
coarse/fine median symmetric change <= 0.20
```

The confirmation result is terminal.

---

# 20. Exact implementation and execution commands

## 20.1 Establish the reviewed implementation

```bash
set -euo pipefail

cd /home/ccj/workspace_1/idle_1_green_bridge

git switch codex/green-v200

test "$(git rev-parse HEAD)" = \
"159520a24b1a7903110f3c457234d1ebf254710b"

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

test ! -e outputs/green_bridge_v200
```

## 20.2 Implement only this corrigendum

Modify only the authorized files:

```text
analysis/GPTPRO_GREEN_V200_CORRIGENDUM_DECISION_20260825.md
src/green_bridge_spec.py
src/green_bridge_numerics.py
src/matched_bypass_gate.py
src/green_bridge_response_ad.py
src/green_bridge_dataset.py
src/exp_green_bridge_gpt2.py
src/analyze_green_bridge.py
src/green_bridge_multigpu_worker.py
src/test_green_bridge_contract.py
src/launch_green_bridge_v200.sh
```

## 20.3 Run the contract

```bash
python src/test_green_bridge_contract.py \
  2>&1 |
  tee /tmp/green_bridge_v200_corrigendum_contract.log

grep -F "Ran 220 tests" \
  /tmp/green_bridge_v200_corrigendum_contract.log

grep -F "OK" \
  /tmp/green_bridge_v200_corrigendum_contract.log

git diff --check
```

## 20.4 Verify forbidden historical changes

```bash
git diff --exit-code \
  159520a24b1a7903110f3c457234d1ebf254710b \
  -- \
  src/launch_green_bridge_v136.sh \
  analysis/GPTPRO_GREEN_V136_TERMINAL_DECISION_20260825.md \
  analysis/GREEN_V136_TERMINAL_AUDIT_20260825
```

## 20.5 Commit the corrected protocol

```bash
git add \
  analysis/GPTPRO_GREEN_V200_CORRIGENDUM_DECISION_20260825.md \
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
  "Apply binding GREEN v2.0.0 numerical corrigendum"

CORRECTED_COMMIT="$(git rev-parse HEAD)"
printf '%s\n' "$CORRECTED_COMMIT"

git merge-base --is-ancestor \
  159520a24b1a7903110f3c457234d1ebf254710b \
  "$CORRECTED_COMMIT"

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"
```

The printed `CORRECTED_COMMIT` is the sole authorized execution commit.

## 20.6 Fast-forward main

```bash
git switch main

git merge --ff-only codex/green-v200

test "$(git rev-parse HEAD)" = "$CORRECTED_COMMIT"

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"

test ! -e outputs/green_bridge_v200
```

## 20.7 Validate the environment without mutation

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate green_bridge_20260805

python -m pip check

python - <<'PY'
import importlib.metadata

expected = {
    "torch": "2.7.1",
    "transformer-lens": "3.6.0",
    "transformers": "5.13.0",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "pandas": "2.2.3",
    "pyarrow": "19.0.1",
    "threadpoolctl": "3.6.0",
}

actual = {
    name: importlib.metadata.version(name)
    for name in expected
}

assert actual == expected, (actual, expected)
PY
```

## 20.8 Launch corrected prepare exactly once

```bash
bash src/launch_green_bridge_v200.sh 4 prepare
```

This is the first and only authorized claim of the v2.0.0 one-shot.

## 20.9 Development

Only after every corrected prepare gate passes:

```bash
bash src/launch_green_bridge_v200.sh 4 development
```

## 20.10 Confirmation

Only after:

```text
dev_result.verdict == OPEN_CONFIRMATION
```

and all frozen hashes pass:

```bash
bash src/launch_green_bridge_v200.sh 4 confirmation
```

---

# 21. Explicit STOP conditions

The corrected protocol must STOP on any of the following:

1. execution begins from commit `159520...`;
2. corrected commit is not a descendant of `159520...`;
3. execution branch is not `main`;
4. worktree is dirty;
5. official root existed before prepare;
6. attempt identity differs;
7. retry is enabled;
8. predecessor hash differs;
9. predecessor terminal state differs;
10. evidence of v1.3.6 confirmation access exists;
11. active split hash is `f012...`;
12. corrected split hash is not `087391...`;
13. any frozen group changes;
14. an old development group enters v2.0.0;
15. a confirmation group enters development beyond the four frozen groups;
16. any of the 220 tests fails;
17. test count is not 220;
18. launcher mutates the Python environment;
19. package or TransformerLens hashes differ;
20. active scientific model is mutated during AD;
21. AD local config dtype is not float64;
22. scientific model dtype is not float32;
23. either AD route is nonfinite;
24. an AD route-consistency guard fails;
25. any of the 40 prepare AD theorem checks fails;
26. any LayerNorm structural-envelope audit fails;
27. manual-tail equivalence fails;
28. exact batch-one execution fails;
29. memory exceeds 20 GiB;
30. projected total wall time exceeds 24 hours;
31. a lower-precision fallback is attempted;
32. a radius search is attempted;
33. a coarse-versus-fine estimator choice is attempted;
34. an empirical enclosure inflation factor is introduced;
35. any structural contradiction is relabelled unresolved;
36. any selected gate is omitted;
37. PIE enters the matched-bypass point estimate or interval center;
38. white-box \(A\) becomes the signed response point estimate;
39. a worker assignment changes;
40. an endpoint batch is repeated;
41. development record counts differ from 64/64;
42. confirmation record counts differ from 192/192;
43. behavior is read before numerical certification;
44. development does not satisfy survival/conditioning/SNR gates;
45. confirmation is attempted after `STOP_ORAL`;
46. confirmation is attempted after `POSTER_ONLY`;
47. confirmation reselects a baseline;
48. confirmation technical count gates fail;
49. any confirmation oral criterion fails;
50. a phase crashes after its ledger claim;
51. any phase is retried under the same identity.

No STOP authorizes v2.0.1 or another run.

---

# 22. Authorized actions

The executor is authorized to:

- correct the split digest;
- implement the dual-route AD-certified fine-Richardson uncertainty;
- use isolated float64 AD tails;
- preserve fine-Richardson as the point estimator;
- retain coarse/fine diagnostics;
- correct phase-lock and confirmation-analysis defects;
- update the tests to exactly 220;
- create one corrected clean commit;
- fast-forward that commit to `main`;
- run corrected prepare exactly once;
- run development once after prepare passes;
- run confirmation once only after `OPEN_CONFIRMATION`.

---

# 23. Forbidden actions

The executor is forbidden to:

- launch commit `159520...`;
- create the output root before the corrected commit is frozen;
- consume the one-shot to test the known split failure;
- inspect fresh development responses before prepare passes;
- inspect fresh confirmation responses before confirmation opens;
- keep `f012...` active;
- alter the group allocation;
- tune the enclosure from the legacy smoke ratios;
- multiply bounds by an empirical constant;
- use an empirical pass-rate target;
- search radii;
- switch to the coarse estimator;
- switch the point estimate to AD;
- switch the point estimate to white-box coordinates;
- switch the estimator to PIE;
- restore donor PCA;
- add pseudoinverse or ridge;
- mutate the live scientific model to float64;
- change selected gates;
- change sites;
- change target vectors;
- change contrasts;
- change base radii;
- change performance thresholds;
- change bootstrap settings;
- reduce all-ten accounting;
- retry a consumed phase.

---

# 24. Final executor checklist

## 24.1 Corrigendum identity

- [ ] This decision document is added verbatim.
- [ ] `CORRIGENDUM_ID` is exact.
- [ ] `NUMERICAL_ERROR_CONTRACT` is exact.
- [ ] v2.0.0 remains attempt 1.
- [ ] retry remains false.
- [ ] official output root remains absent.

## 24.2 Split

- [ ] Active split hash is `0873915c966bef8f54b83d4151a9d7c75b577da5dfc17ee093b9f5c58a9590f7`.
- [ ] `f012...` is inactive.
- [ ] The canonical payload equals Section 3.3.
- [ ] Four development groups are exact.
- [ ] Twelve confirmation groups are exact.
- [ ] Record counts are exact.
- [ ] Old development overlap is zero.
- [ ] Both distance bins stay together.

## 24.3 Numerical correction

- [ ] Fine Richardson remains the point estimator.
- [ ] Coarse estimate is diagnostic only.
- [ ] No old dyadic-overlap hard gate remains.
- [ ] Both float64 AD routes are evaluated.
- [ ] Route guard uses \(\gamma_{65536}\).
- [ ] AD midpoint and route radius are computed.
- [ ] Endpoint-repeatability terms are added.
- [ ] Corrected fine uncertainty uses Section 5.7.
- [ ] AD matched-bypass identity uses Section 6.1.
- [ ] No empirical inflation factor exists.
- [ ] No radius search exists.

## 24.4 AD isolation

- [ ] AD local-tail parameters are float64.
- [ ] AD local-tail `cfg.dtype` is float64.
- [ ] Scientific model parameters remain float32.
- [ ] Scientific `cfg.dtype` remains float32.
- [ ] No in-place `model.double()` is called on the scientific model.
- [ ] Before/after model hashes are exact.
- [ ] AD clone is destroyed exception-safely.

## 24.5 Gate logic

- [ ] Numerical invalidity and structural contradiction are distinct.
- [ ] Coarse/fine non-overlap alone is not invalid.
- [ ] Structural contradiction cannot become unresolved.
- [ ] Non-invertibility does not imply nullity.
- [ ] All ten gates are represented.
- [ ] Active intervals use corrected bounds.
- [ ] Unresolved centers are zero.
- [ ] Null contribution is at most `0.005`.
- [ ] PIE remains baseline-only.
- [ ] Donor PCA remains terminated.

## 24.6 Analysis and phase lock

- [ ] Development uses worst-case interval RMSE.
- [ ] Development freezes the best baseline name.
- [ ] Confirmation uses only that baseline.
- [ ] Bootstrap cannot reselect a baseline.
- [ ] Per-bin analysis cannot reselect a baseline.
- [ ] Confirmation technical counts are complete.
- [ ] CLI dispatches v2.0.0 functions.
- [ ] Confirmation remains inaccessible before `OPEN_CONFIRMATION`.

## 24.7 Throughput and operation counts

- [ ] Finite operation counts match Section 14.
- [ ] AD operation counts match Section 14.
- [ ] Actual v2.0.0 workload is benchmarked.
- [ ] Historical v1.3.6 throughput is not relabelled.
- [ ] Memory is at most 20 GiB.
- [ ] Total eight-GPU forecast is at most 24 hours.
- [ ] No fallback is used.

## 24.8 Tests and commit

- [ ] Exactly 220 tests run.
- [ ] All 220 pass.
- [ ] No tests are skipped.
- [ ] Worktree is clean.
- [ ] Corrected commit descends from `159520...`.
- [ ] Corrected commit is fast-forwarded to `main`.
- [ ] Corrected commit is recorded in the manifest.
- [ ] Old v1.3.6 artifacts remain immutable.

## 24.9 Execution

- [ ] No fresh development response has been inspected.
- [ ] No fresh confirmation response has been inspected.
- [ ] Prepare is launched once.
- [ ] Development is launched only after prepare passes.
- [ ] Confirmation is launched only after `OPEN_CONFIRMATION`.
- [ ] No phase is retried.

**BINDING VERDICT: commit `159520a24b1a7903110f3c457234d1ebf254710b` is not launch-authorized; the split digest is corrected to `0873915c966bef8f54b83d4151a9d7c75b577da5dfc17ee093b9f5c58a9590f7`, the defective Richardson balls are replaced by the exact dual-route AD-certified fine-Richardson contract above, isolated float64 `cfg.dtype` is approved, and the still-unconsumed GREEN v2.0.0 one-shot may be launched exactly once only from a clean corrected descendant commit after all 220 tests and every pre-launch condition in this document pass.**