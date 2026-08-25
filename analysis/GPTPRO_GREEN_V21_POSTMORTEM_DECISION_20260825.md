<!-- filename: analysis/GPTPRO_GREEN_V21_POSTMORTEM_DECISION_20260825.md -->

# GPTPro GREEN v2.0.0 Postmortem and GREEN v3.0.0 Prepare-Only Scientific Decision — 2026-08-25

## Document status

| Field | Binding value |
|---|---|
| Repository | `ScottBlizzard/idle_1` |
| Reviewed branch | `codex/green-v200` |
| Exact postmortem evidence commit | `ef09fce529553d5a3d236852a288cde02b88418a` |
| Immutable official execution commit | `e52e082296c33a10557636706e572147136fce34` |
| Immutable predecessor protocol | `green-bridge-v2.0.0-one-shot` |
| Immutable predecessor verdict | `STOP_ORAL` |
| First failed gate | `12_DEVELOPMENT_SURVIVAL` |
| Confirmation state | Closed; never started |
| v2 retry | Forbidden |
| Retrospective threshold change | Forbidden |
| Primary diagnosis | **Finite-radius response identifiability and target-functional mismatch, not a falsification of the exact matched-bypass identity** |
| Ranked explanation | **D > B > E > C > A** |
| Successor identity | `green-bridge-v3.0.0-one-shot` |
| Successor scientific focus | Curvature-controlled identifiability and held-out causal transport |
| Action authorized by this document | Read-only postmortem, clean v3.0.0 implementation, tests, and **prepare only** |
| Development authorization | Not granted |
| Confirmation authorization | Not granted |
| Fixed-rank donor PCA | Permanently terminated |
| PIE | Baseline or explicitly post hoc diagnostic only |
| Server storage rule | All data, cache, runtime, logs, temporary files, and outputs under `/mnt/sdb` |

---

# 1. Binding verdict

The official GREEN v2.0.0 result is valid, terminal, and immutable:

```text
verdict = STOP_ORAL
phase = development
first_failed_gate = 12_DEVELOPMENT_SURVIVAL
n_surviving_cells = 0
n_conditioned_cells = 0
n_snr_cells = 0
confirmation_open = false
confirmation_started = false
```

The completed run produced exactly 64 tensor records and 64 energy records. All eight workers completed, all active-model integrity checks passed, every energy record was admissible, and no formal development cell survived because no tensor record met the frozen system-level set-admissibility rule. The archive records 7 `active-identified`, 1,262 `certified-target-null`, 11 `unresolved-bounded`, zero `numerical-invalid`, zero `structural-contradiction`, zero AD-route failures, zero AD-theorem failures, and zero white-box-coordinate failures across 1,280 gate-system classifications. Confirmation never opened.

The v2.0.0 result is **not** invalidated by an implementation defect. The source implements the frozen rule exactly: a system is set-admissible only when all ten gates are accounted for, at least three gates are `active-identified`, common-frame bypass disagreement passes, and the system interval is finite. The observed data contained 121 systems with zero active gates, seven systems with one active gate, and none with at least three.

The v2.0.0 result also does **not** falsify the exact local matched-bypass identity. Every AD theorem certificate passed, with no route failures, no structural contradictions, and residual-to-bound ratios orders of magnitude below one on the frozen prepare panel. The detailed prepare audit reports representative theorem residuals around \(10^{-17}\)–\(10^{-16}\) against bounds around \(10^{-10}\), while forward/reverse AD discrepancies were approximately \(10^{-15}\)–\(10^{-14}\).

The immediate v2 failure is instead the conjunction of three facts:

1. The response-only finite estimator was almost never sufficiently conditioned to produce a narrow point-identified operator. Curvature SNR was the dominant active-condition failure, occurring 1,163 times among inverse-capable nonactive classifications.
2. The v2 target and cell aggregation use a signed-mean functional,
   \[
   \left|\frac1n\sum_i d_i\right|,
   \]
   while the earlier strong PIE and behavioral diagnostics average itemwise magnitudes,
   \[
   \frac1n\sum_i |d_i|.
   \]
   Those are different estimands and can diverge sharply under sign cancellation. The current source constructs row-level `behavioral` and `pie` as absolute item contrasts, but constructs the formal cell mixed predictor and independent energy target by averaging signed `pat` and `tar` quantities separately, subtracting, and only then taking the absolute value.
3. Even after a deliberately permissive read-only counterfactual admitted point-complete rows, every cell interval crossed zero, every interval SNR was exactly \(1\), midpoint RMSE was approximately \(0.0047467\), worst-case RMSE was approximately \(0.0106438\), the best baseline LOOCV RMSE was approximately \(0.0008111\), and robust relative gain was approximately \(-12.12\). Thus merely lowering the three-active-gate requirement would not rescue the frozen behavioral bridge.

A new scientific version is justified, but it must not be a softer rerun of v2.0.0. The successor shall be a major-version change:

```text
green-bridge-v3.0.0
```

Its primary experiment shall directly test whether a response-only matched-bypass operator predicts **held-out gate-mediated causal transport**, while separately testing the theorem’s curvature-controlled detectability boundary. Behavioral prediction becomes a secondary, explicitly separated endpoint rather than the primary certification target.

This document authorizes:

1. exact read-only postmortem analyses on already exposed v1.3.6 and v2.0.0 development artifacts;
2. implementation of the immutable v3.0.0 transport protocol;
3. all required tests;
4. one clean v3.0.0 prepare execution.

This document does not authorize v3.0.0 development or confirmation.

---

# 2. Evidence boundary and audit scope

## 2.1 Evidence reviewed

The decision is based on the exact postmortem commit and the following repository evidence:

```text
analysis/GPTPRO_GREEN_V200_CORRIGENDUM_DECISION_20260825.md
analysis/GREEN_V200_DEVELOPMENT_TERMINAL_STOP_20260825.md
analysis/GREEN_V200_DEVELOPMENT_TERMINAL_DIAGNOSTIC_20260825.json
analysis/archive/green_v200_stop_20260825/
src/green_bridge_spec.py
src/green_bridge_numerics.py
src/matched_bypass_gate.py
src/green_bridge_response_ad.py
src/green_bridge_dataset.py
src/green_bridge_structural_frame.py
src/green_bridge_path_target.py
src/green_bridge_tail.py
src/exp_green_bridge_gpt2.py
src/analyze_green_bridge.py
src/green_bridge_multigpu_worker.py
src/test_green_bridge_contract.py
src/launch_green_bridge_v200.sh
analysis/GREEN_V136_TERMINAL_AUDIT_20260825/
```

The terminal report states that 28 selected server artifacts were copied into the committed archive and verified against the formal root’s `sha256sums.txt`. The official Parquet hashes match the multigpu merge manifest.

## 2.2 Binary-artifact limitation and mandatory remedy

The repository browser exposes the committed Parquet files as binary artifacts rather than decoded rows. This review independently verified:

- their committed hashes and row counts through the terminal reports and manifests;
- every arithmetic conclusion available from the machine-readable terminal JSON;
- the source code that serializes, parses, classifies, and aggregates those rows;
- the exact mathematical consequence of the interval and aggregation implementation.

It does not claim to have independently decoded every binary Parquet row inside the browser interface.

For that reason, the exact row-level decompositions in Section 7 are mandatory read-only Codex analyses before any v3.0.0 prepare. Their outputs become part of the frozen postmortem evidence bundle. They may diagnose v2.0.0, but may not alter its official verdict or tune v3.0.0 success thresholds.

---

# 3. Immutable v2.0.0 facts and implementation audit

## 3.1 Prepare was successful

The official prepare passed:

```text
220 / 220 tests
40 / 40 frozen AD strata
0 AD route misses
0 theorem misses
0 white-box misses
exact manual-tail equivalence
exact active-model integrity
three-scale diagnostics
actual throughput preflight
peak allocation = 7.4576 GiB
projected total runtime = 1,367.95 seconds
```

The scientific float32 model remained byte-identical while the isolated local AD tail used float64. No precision, radius, projection, gate-count, or batching fallback was used.

## 3.2 Fine Richardson was the scientific point estimator

For every derivative object,

\[
T\in\{G,C,J,H^P,H^C\},
\]

the code evaluated full, half, and quarter stencils and formed:

\[
R_T^{(c)}
=
\frac{4T_{1/2}-T_1}{3},
\]

\[
\boxed{
R_T^{(f)}
=
\frac{4T_{1/4}-T_{1/2}}{3}
}
\]

with \(R_T^{(f)}\) passed to `identify_gate`. Coarse/fine overlap was serialized as diagnostic-only and did not determine admissibility.

This part of the protocol was implemented correctly.

## 3.3 The numerical uncertainty was dominated by finite-radius discrepancy

The v2 AD-certified enclosure is centered on the fine Richardson estimate and includes:

\[
\bar\epsilon_G
=
\|\widehat G-G_A\|_2+r_{A,G}+\nu_G,
\]

\[
\bar\epsilon_C
=
\|\widehat C-C_A\|_2+r_{A,C}+\nu_C,
\]

\[
\bar\epsilon_J
=
\|\widehat J-J_A\|_F+r_{A,J}+\nu_J,
\]

\[
\bar\epsilon_{\Delta H,i}
=
\|\widehat{\Delta H}_i-\Delta H_{A,i}\|_2
+r_{A,\Delta H,i}
+\nu_{\Delta H}.
\]

The duplicate endpoint audit observed zero discrepancy in all 96 repetitions and therefore used the frozen floor \(\epsilon_y=10^{-7}\).

At the fine gate radius \(h_z=0.1\), the curvature endpoint contribution is:

\[
10\eta_C
=
10\cdot
\frac{64\epsilon_y}{3h_z^2}
=
10\cdot
\frac{64\cdot10^{-7}}{3\cdot0.1^2}
\approx
0.0021333.
\]

The detailed AD audit reports typical \(\bar\epsilon_C\) values around \(0.26\)–\(0.36\), while route discrepancies are approximately \(10^{-15}\). Therefore more than 99% of typical curvature uncertainty is attributable to the finite-Richardson-versus-AD discrepancy, not endpoint repeatability or float64 route disagreement.

All 40 frozen preflight strata also had `coarse_fine_overlap = false` at multipliers \(1, 1/2, 1/4\), even though the protocol correctly treated this as diagnostic-only.

This is evidence of finite-radius nonlocality or poor response-estimator conditioning at the chosen radii. It is not evidence of an AD arithmetic defect.

## 3.4 Gate classification was internally consistent

The v2 classifier follows this order:

1. AD route certification;
2. exact AD matched-bypass identity;
3. response inverse, when admissible;
4. factorization, white-box, direct-white-box-factorization, and shift-null compatibility;
5. active materiality and SNR;
6. complete contribution upper bound;
7. `certified-target-null` if that upper bound is at most `0.005`;
8. otherwise `unresolved-bounded` if the upper bound is finite.

A failed AD route becomes `numerical-invalid`. A failed AD theorem or bound-certified identity becomes `structural-contradiction`. Neither occurred in the official development run.

The active rule requires:

\[
\frac{\|\widehat C\|_2}{10}
\ge 5\times10^{-4},
\qquad
\|\widehat C\|_2
\ge 20\bar\epsilon_C,
\]

\[
\frac{\|\widehat G\|_2}{10}
\ge 5\times10^{-4},
\qquad
\|\widehat G\|_2
\ge 20\bar\epsilon_G,
\]

\[
\|\widehat P\|_F
\ge 20\bar\epsilon_{P,F}.
\]

Curvature SNR, rather than materiality, was overwhelmingly the failing condition.

## 3.5 “Certified target null” did not mean exact zero

The complete unresolved/null contribution bound is derived from:

\[
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
\]

A gate was labelled `certified-target-null` when:

\[
U_j\le0.005.
\]

The official null-bound distribution was:

| Quantile | Bound |
|---|---:|
| Minimum | \(1.48\times10^{-8}\) |
| 25th percentile | \(5.50\times10^{-5}\) |
| Median | \(1.82\times10^{-4}\) |
| 75th percentile | \(4.96\times10^{-4}\) |
| 90th percentile | \(1.35\times10^{-3}\) |
| 95th percentile | \(2.45\times10^{-3}\) |
| 99th percentile | \(4.71\times10^{-3}\) |
| Maximum | \(6.83\times10^{-3}\) |

The `0.005` null ceiling is therefore much larger than the best-baseline cell RMSE scale of approximately \(8.11\times10^{-4}\). Summing ten symmetric gate bounds for each of two systems can produce an interval far wider than the target being predicted.

The label was valid under the frozen protocol, but the phrase “null” is operationally misleading at the behavioral target scale. A more precise interpretation is:

> Not separately point-identifiable and certified to have scalar target contribution below the frozen absolute ceiling.

The successor shall not reuse this absolute ceiling.

## 3.6 Why every permissive cell had SNR exactly one

For a signed interval:

\[
I_\Delta=[L,U]
\]

that crosses zero, the v2 absolute-value transform returns:

\[
|I_\Delta|
=
[0,\max(|L|,|U|)].
\]

Let the transformed interval be \([0,M]\). Its midpoint and half-width are:

\[
m=\frac M2,
\qquad
r=\frac M2.
\]

Therefore:

\[
\boxed{
\mathrm{SNR}_{set}
=
\frac m r
=
1
}
\]

for every nondegenerate zero-crossing interval.

The diagnostic observation that all eight permissive cells had interval SNR exactly one is therefore not a coding accident. It proves that every signed cell interval crossed zero. The implementation’s `absolute_value_interval`, interval midpoint, half-width, and SNR formula produce this result mechanically.

## 3.7 No execution-invalidating implementation defect was found

The audit found no defect that legally invalidates the official v2.0.0 STOP.

The following are scientific operationalization issues, not execution defects:

- the active-gate minimum;
- the absolute null ceiling;
- symmetric zero-centered unresolved intervals;
- signed-mean cell aggregation;
- use of disjoint tensor and energy role samples;
- the choice of behavior as the terminal target.

One minor reporting ambiguity should be corrected in future reports: the prepare report mentions both a 40-stratum AD panel and 160 gate-system certificates. The latter belongs to the actual eight-record throughput workload, not the frozen 40-stratum theorem panel. Both counts can be correct, but future artifacts must label them separately. This does not alter the v2 result.

---

# 4. Ranked scientific diagnosis

## 4.1 Ranking

| Rank | Explanation | Assessment |
|---:|---|---|
| 1 | **D. Target scale, normalization, or aggregation mismatch** | Strongly supported |
| 2 | **B. Theory correct but contribution target or active-set rule incompatible with actual signal scale** | Strongly supported |
| 3 | **E. Earlier and v2 evidence concern distinct regimes; a scoped theorem and new experiment are required** | Strongly supported as the strategic synthesis |
| 4 | **C. Proof-valid intervals are excessively conservative** | Partly supported, but secondary |
| 5 | **A. Matched-bypass theory is falsified in the tested regime** | Rejected by current evidence |

These explanations are not mutually exclusive. D and B jointly explain the terminal result; E is the correct successor strategy.

---

## 4.2 Explanation A — the matched-bypass theory is genuinely falsified

### Existing evidence

The current evidence rejects A for the exact local theorem.

The AD audit tests:

\[
\Delta H_i=A_i^{WB}C
\]

for every structural coordinate \(i\). The official run recorded:

```text
AD route failures = 0
AD theorem failures = 0
white-box coordinate failures = 0
structural contradictions = 0
```

across the full 1,280 development gate-system classifications. The prepare audit also completed 40 outcome-blind strata with zero misses and very small theorem residual ratios.

### What v2 did falsify

The following empirical proposition failed:

> At the frozen radii, with the frozen response-only inverse, frozen active-set rule, frozen all-ten interval construction, and frozen signed cell target, the matched-bypass predictor produces admissible cells and outperforms the baseline family.

That proposition is not the same as the exact local derivative theorem.

### Decisive next analysis

Run both:

\[
T_j^{AD}(u)
=
\left(
D_xY_j^P(0,0)-D_xY_j^C(0,0)
\right)[u]
\]

and:

\[
\mathcal P_j^{AD}u
=
G_j^{AD}
(g_j^{WB})^\top u.
\]

If:

\[
\left\|
T_j^{AD}(u)-\mathcal P_j^{AD}u
\right\|
\]

exceeds the outward-rounded dual-route and white-box bound on any valid held-out direction, the executable transport theorem is contradicted.

Also test the all-ten composition:

\[
\ell^\top
\sum_{j=1}^{10}
\mathcal P_j^{AD}v
\]

against the independently implemented joint, direct-bypass-subtracted target JVP.

### What would support A

A is supported only if one of the following occurs:

- exact AD path-minus-control transport violates \(Gg^\top\);
- exact all-ten first-order composition fails beyond numerical bounds;
- the structural-frame envelope fails on held-out complement directions;
- a failure survives independent route, hook, endpoint, and white-box audits.

### Classification

At present:

```text
A = rejected
```

---

## 4.3 Explanation B — the theorem is correct, but the active-set or contribution target is incompatible with the signal scale

### Existing evidence

B is strongly supported.

Only 7 of 1,280 gate-system classifications were active, while 1,163 nonactive inverse-capable cases failed curvature SNR. Materiality failures were zero. This means the observed response was not absent merely because every gate had negligible raw magnitude; rather, the finite response inverse was generally too uncertain relative to curvature.

The mathematical bottleneck is:

\[
\widehat A_i
=
\frac{
\langle\widehat C,\widehat{\Delta H}_i\rangle
}{
\|\widehat C\|_2^2
}.
\]

If the lower curvature bound

\[
c_-=\|\widehat C\|_2-\bar\epsilon_C
\]

is nonpositive or small, the coordinate error expands as:

\[
\bar\epsilon_{A,i}
\lesssim
\frac{
\bar\epsilon_{\Delta H,i}
+
A_{\max,i}\bar\epsilon_C
}{
\|\widehat C\|_2
}.
\]

The exact operator can be real and nonzero while response-only recovery remains unidentifiable.

### Why three active gates was not the whole problem

The three-active-gate rule caused the official survival failure, but relaxing it post hoc would not rescue the experiment. The permissive read-only diagnostic still produced:

```text
set-SNR cells = 0 / 8
midpoint RMSE ≈ 0.0047467
worst-case RMSE ≈ 0.0106438
best baseline RMSE ≈ 0.0008111
robust gain ≈ -12.12
```

Therefore B is not “the active threshold was slightly too high.” It is:

> The response-only inverse and scalar contribution operationalization were mismatched to the curvature and target scale.

### Decisive next analyses

- Decompose each gate’s uncertainty into finite-radius, AD-route, and endpoint components.
- Compare the response estimator with:
  - exact AD \(G\) and white-box \(g\);
  - exact direct path-minus-control transport;
  - the independent joint target.
- Measure how held-out transport error varies with:
  \[
  \kappa_C=\frac{\|C\|}{\bar\epsilon_C}.
  \]

### What would falsify B

B is weakened if:

- exact AD and white-box operators also have negligible direct transport;
- response-identifiable gates show the same large held-out error as unidentifiable gates;
- reducing finite-radius bias outcome-blindly does not improve response recovery at any admissible radius;
- the all-ten exact operator sum fails independently.

### Classification

```text
B = strongly supported
```

---

## 4.4 Explanation C — the interval construction is proof-valid but excessively conservative

### Existing evidence

C is supported only as a secondary explanation.

The interval construction is conservative because it:

- adds per-gate bounds by Minkowski sum;
- centers null and unresolved gates at zero;
- sums ten gates for each of two systems;
- subtracts system intervals;
- takes an absolute-value image;
- produces \([0,U]\) whenever the signed interval crosses zero.

This can magnify uncertainty substantially.

However, the primary uncertainty terms were not generated by endpoint noise or AD roundoff. They were dominated by the observed discrepancy between the fine finite estimator and the AD derivative. Furthermore, the counterfactual midpoint itself underperformed the baseline badly. Thus interval conservatism cannot by itself explain the failure.

### Decisive next analyses

- Quantify fractions:
  \[
  f_{\mathrm{FD}}
  =
  \frac{\|\widehat T-T_A\|}{\bar\epsilon_T},
  \quad
  f_{\mathrm{route}}
  =
  \frac{r_{A,T}}{\bar\epsilon_T},
  \quad
  f_{\mathrm{endpoint}}
  =
  \frac{\nu_T}{\bar\epsilon_T}.
  \]
- Compare:
  - exact AD point predictions;
  - fine finite point predictions;
  - active-only point predictions;
  - full interval predictions.
- Compute the added width from:
  - gatewise sum;
  - two-system subtraction;
  - absolute-value transformation.

### What would support C as the primary cause

C becomes primary only if:

- exact or finite point centers predict the target accurately;
- most error arises only after uncertainty aggregation;
- a tighter theorem-derived dependence-aware bound, without changing the point estimator, restores held-out predictive performance.

The existing midpoint counterfactual argues against this.

### Classification

```text
C = partially supported, not primary
```

---

## 4.5 Explanation D — inferential target, normalization, or aggregation mismatch

### Existing evidence

D is the best-supported diagnosis.

For a tensor record, the source serializes:

\[
\mathrm{behavioral}_i
=
|b_{pat,i}-b_{tar,i}|,
\]

\[
\mathrm{PIE}_i
=
|\mathrm{PIE}_{pat,i}-\mathrm{PIE}_{tar,i}|.
\]

Cell baselines are then:

\[
M_c
=
\frac1n\sum_i |\delta_i|.
\]

By contrast, the mixed system contributions remain signed until cell aggregation:

\[
A_c^{mixed}
=
\left|
\frac1n\sum_i\theta_{pat,i}
-
\frac1n\sum_i\theta_{tar,i}
\right|.
\]

The independent energy target is likewise:

\[
A_c^{target}
=
\left|
\frac1m\sum_i t_{pat,i}
-
\frac1m\sum_i t_{tar,i}
\right|.
\]

The tensor and energy roles use disjoint records within each cell.

In general:

\[
\left|\mathbb E[d]\right|
\le
\mathbb E[|d|],
\]

with equality only when \(d\) has a stable sign almost surely.

The earlier strong cell-mean PIE correlation therefore concerned a magnitude functional, not the signed-mean functional used by the v2 independent target. The earlier audit’s group means were direct means of row-level absolute fields.

### Decisive next analyses

For every exposed cell and every estimator, compute all three frozen functionals:

\[
A_c
=
\left|
\frac1n\sum_i d_i
\right|,
\]

\[
M_c
=
\frac1n\sum_i |d_i|,
\]

\[
R_c
=
\sqrt{
\frac1n\sum_i d_i^2
}.
\]

Also compute the cancellation ratio:

\[
\chi_c
=
\frac{A_c}
{\max(M_c,\tau_c)},
\]

where:

\[
\tau_c
=
10^{-12}
\]

for a pure diagnostic and is never used as a protocol threshold.

Compare:

- PIE \(A/M/R\);
- behavioral \(A/M/R\);
- matched-bypass \(A/M/R\);
- independent target \(A/M/R\);
- same-role and disjoint-role estimates.

### What would falsify D

D is weakened if:

- \(A_c\), \(M_c\), and \(R_c\) are nearly identical across all cells;
- item signs are stable;
- same-role and cross-role estimates agree tightly;
- the earlier PIE correlation remains equally strong for the signed-mean functional;
- direct transport and joint composition still fail after removing the functional mismatch.

### Classification

```text
D = most likely immediate scientific cause
```

---

## 4.6 Explanation E — earlier and v2 evidence concern distinct regimes

### Existing evidence

E is strongly supported as the strategic conclusion.

The earlier evidence used:

- heuristic gate admissibility;
- raw itemwise magnitude diagnostics;
- 16-cell mean correlations;
- no valid tensor rows under the registered protocol.

The v2 experiment used:

- dual-route AD-certified uncertainty;
- strict response point-identification;
- all-ten set accounting;
- a signed-mean independent target;
- disjoint tensor and energy role records;
- robust worst-case interval performance.

These are not merely two estimators applied to the same estimand. They are different inferential regimes.

### Scientific implication

The next paper-level experiment should not pretend that v2 was nearly successful. It should make the regime distinction explicit:

1. exact local structural identity;
2. response-only recoverability;
3. held-out causal transport;
4. all-ten joint composition;
5. behavioral or task-level aggregation.

The first four can be tested without conflating them with the fifth.

### Classification

```text
E = strongly supported and adopted
```

---

# 5. Primary scientific diagnosis

The most likely causal chain is:

\[
\boxed{
\text{large-radius finite-response bias}
\rightarrow
\text{curvature-controlled nonidentifiability}
\rightarrow
\text{almost no point operators}
\rightarrow
\text{wide zero-crossing all-ten intervals}
}
\]

combined with:

\[
\boxed{
\mathbb E|d|
\neq
|\mathbb E d|
}
\]

and disjoint tensor/energy item sampling at a target scale below many per-gate upper bounds.

The exact evidence anchors are:

- 1,163 curvature-SNR active-condition failures;
- only 7 active gate-system classifications;
- 1,262 null labels under an absolute `0.005` ceiling;
- median null bound \(1.82\times10^{-4}\), 95th percentile \(2.45\times10^{-3}\);
- zero AD theorem failures;
- zero endpoint-repeatability error;
- typical \(\epsilon_C\approx0.3\), dominated by fine-versus-AD discrepancy;
- all 40 original scale-overlap diagnostics false;
- every permissive absolute cell interval crossing zero;
- midpoint and worst-case predictors both substantially worse than the best baseline.

## 5.1 Falsifiers of this diagnosis

The preferred diagnosis is falsified if the mandatory read-only postmortem finds any of the following:

1. Exact AD direct transport violates:
   \[
   D_xY^P-D_xY^C=Gg^\top
   \]
   beyond certified numerical error.
2. The exact all-ten operator sum fails the independent joint target.
3. Fine-versus-AD discrepancy is not the dominant uncertainty source.
4. Signed-mean, mean-absolute, and RMS functionals are nearly equivalent.
5. The response estimator remains inaccurate on held-out transport even in a globally calibrated, numerically local radius regime with:
   \[
   \|C\|/\epsilon_C\gg1.
   \]
6. Matched control does not improve transport prediction relative to an unmatched mixed-derivative estimator.

Findings 1 or 2 halt GREEN before any v3 prepare. Findings 3–6 materially revise the successor design and require another GPT Pro decision.

---

# 6. Theory status and strengthened main line

## 6.1 The exact factorization theorem remains

Let:

- \(x\in\mathbb R^{768}\) be the residual perturbation;
- \(z\in\mathbb R\) be a selected-gate preactivation perturbation;
- \(Y_j^P(x,z)\in\mathbb R^{100}\) be the path response;
- \(Y_j^C(x,z)\in\mathbb R^{100}\) be the matched control;
- \(a_j(x)\) be the selected gate preactivation;
- \(g_j=\nabla_xa_j(0)\);
- \(G_j=\partial_zY_j^P(0,0)\);
- \(C_j=\partial_z^2Y_j^P(0,0)\).

Then:

\[
\boxed{
D_x\partial_zY_j^P(0,0)[u]
-
D_x\partial_zY_j^C(0,0)[u]
=
C_j\langle g_j,u\rangle.
}
\]

The v2 AD audit supports this identity in the executed map.

## 6.2 The ambient path operator remains

Define:

\[
\boxed{
\mathcal P_j
=
G_jg_j^\top.
}
\]

This is basis-invariant. A frame changes only the coordinates used to identify \(g_j\), not the ambient operator.

The response-derived frame coordinates are:

\[
A_{j,i}
=
\langle g_j,q_{j,i}\rangle
=
\frac{
\langle C_j,\Delta H_{j,i}\rangle
}{
\|C_j\|_2^2
},
\]

when \(C_j\ne0\).

The implementation’s `identify_gate` uses exactly:

\[
\widehat A
=
\widehat{\Delta H}\widehat C
/
\|\widehat C\|_2^2,
\]

\[
\widehat P
=
\widehat A\widehat G^\top,
\]

without donor PCA, pseudoinverse, ridge, or learned alignment.

## 6.3 The exact LayerNorm structural envelope remains

The selected gate gradient lies in the span generated by:

- the shift direction;
- the centered residual anchors;
- the gate-specific \(\gamma\odot W_{\mathrm{in},j}\) atom.

The common frame has dimension four, and the per-gate frame appends one gate atom to obtain dimension five. The frame is constructed endpoint-blindly and deterministically.

Fixed-rank donor PCA remains permanently terminated.

## 6.4 New transport theorem

Write the selected-gate path locally as:

\[
Y_j^P(x,z)
=
F_j
\left(
x,\,
\phi(a_j(x)+z)
\right),
\]

and the matched control as:

\[
Y_j^C(x,z)
=
F_j
\left(
x,\,
\phi(a_j(0)+z)
\right).
\]

The control retains the residual perturbation but removes the \(x\to a_j\to\phi(a_j)\) route. The source implements exactly this distinction: path mode recomputes the selected postactivation from the perturbed preactivation, whereas control mode uses the anchored preactivation plus \(z\), independent of \(x\).

By the chain rule:

\[
D_xY_j^P(0,0)
=
D_xF_j
+
D_pF_j\,\phi'(a_j(0))g_j^\top,
\]

\[
D_xY_j^C(0,0)
=
D_xF_j.
\]

Therefore:

\[
\boxed{
D_xY_j^P(0,0)
-
D_xY_j^C(0,0)
=
G_jg_j^\top
=
\mathcal P_j.
}
\]

This yields a direct, held-out falsifier of the response-identified operator.

## 6.5 New curvature-controlled detectability theorem

Suppose the response-only observations satisfy:

\[
\|\widehat C-C\|_2\le\epsilon_C,
\]

\[
\|
\widehat{\Delta H}_i-A_iC
\|_2
\le\epsilon_{H,i}.
\]

Define:

\[
c_-
=
\|\widehat C\|_2-\epsilon_C.
\]

### Point-identifiable regime

When:

\[
c_->0,
\]

a finite upper bound is:

\[
A_{\max,i}
=
\frac{
\|\widehat{\Delta H}_i\|_2+\epsilon_{H,i}
}{
c_-
},
\]

and the response coordinate error obeys:

\[
\boxed{
|\widehat A_i-A_i|
\le
\frac{
\epsilon_{H,i}
+
A_{\max,i}\epsilon_C
}{
\|\widehat C\|_2
}.
}
\]

### Non-point-identifiable regime

When:

\[
0\in
B(\widehat C,\epsilon_C),
\]

response-only point identification is impossible without an additional prior bound on \(A_i\).

To see this, let an observed mixed response \(h\) be compatible with the uncertainty set. For arbitrarily large \(M\), choose:

\[
A_M=M,
\qquad
C_M=\frac hM.
\]

Then:

\[
A_MC_M=h,
\]

while:

\[
\|C_M\|\to0
\]

as \(M\to\infty\). If zero curvature is inside the admissible curvature ball, sufficiently small \(C_M\) remains compatible. Thus \(A\) is unbounded even though the product \(AC\) is observed.

This distinction strengthens the theory:

> The operator is structurally defined independently of response recoverability. Matched-bypass response identification exhibits a curvature-controlled detectability boundary.

## 6.6 Revised central claim

The paper must not state without qualification that every path operator is practically identifiable from finite response probes.

The strongest defensible theory-first successor claim is:

> **Matched-bypass derivatives factor gate-mediated residual transport into a basis-invariant ambient rank-one operator. Exact LayerNorm geometry provides a complete five-vector probe frame, while response-only recovery undergoes a curvature-controlled detectability transition. When recoverable, the identified operator predicts held-out path-minus-control transport and composes across gates.**

This is at least as theoretically ambitious as the original claim. It adds:

- a direct transport theorem;
- a detectability theorem;
- a partial-identification boundary;
- a new falsifiable transport experiment.

---

# 7. Authorized read-only postmortem analyses

All outputs shall be created under:

```text
analysis/GREEN_V21_POSTMORTEM_20260825/
```

All scratch data, decoded Parquet copies, temporary arrays, logs, and caches shall be under:

```text
/mnt/sdb/ccj/iclr_1_postmortem/green_v21/
```

The official v2 root and committed archive are read-only.

Every output must contain:

```json
{
  "postmortem_schema": "...",
  "postmortem_commit": "ef09fce529553d5a3d236852a288cde02b88418a",
  "official_execution_commit": "e52e082296c33a10557636706e572147136fce34",
  "official_verdict": "STOP_ORAL",
  "official_verdict_unchanged": true,
  "confirmation_data_accessed": false,
  "usable_for_threshold_selection": false,
  "source_artifact_sha256": {},
  "analysis_script_sha256": "..."
}
```

No postmortem output may be copied into the v2 root.

---

## 7.1 Analysis 01 — integrity reconstruction

### Purpose

Reconstruct the official terminal counts directly from the archived files before any scientific postmortem.

### Inputs

```text
analysis/archive/green_v200_stop_20260825/sha256sums.txt
analysis/archive/green_v200_stop_20260825/result.json
analysis/archive/green_v200_stop_20260825/dev_result.json
analysis/archive/green_v200_stop_20260825/dev_cells.json
analysis/archive/green_v200_stop_20260825/manifest.json
analysis/archive/green_v200_stop_20260825/run_ledger.json
analysis/archive/green_v200_stop_20260825/development_multigpu_merge.json
analysis/archive/green_v200_stop_20260825/dev_tensor_scores.parquet
analysis/archive/green_v200_stop_20260825/dev_energy_targets.parquet
```

### Required checks

```text
all listed SHA-256 values pass
tensor rows = 64
energy rows = 64
cells = 8
gate-system rows = 1,280
class counts sum to 1,280
verdict = STOP_ORAL
first failed gate = 12_DEVELOPMENT_SURVIVAL
confirmation_open = false
confirmation_started = false
confirmation artifacts absent
```

### Aggregation unit

Run and artifact.

### Output

```text
analysis/GREEN_V21_POSTMORTEM_20260825/01_integrity_reconstruction.json
```

Schema:

```text
green-v21-postmortem-integrity-v1
```

### Interpretation

This is proof-checking, not exploratory analysis.

### Decision enabled

Any mismatch terminates all GREEN successor work and requires return to GPT Pro.

---

## 7.2 Analysis 02 — gate-certificate decomposition

### Purpose

Determine whether nonactivity is concentrated in particular gates, systems, distance bins, orientations, or cells, and separate curvature, response, and operator detectability.

### Inputs and exact fields

From:

```text
dev_tensor_scores.parquet
```

use:

```text
pair_digest
cell_id
distance_bin
orientation
mixed_audit
```

Parse:

```text
mixed_audit["tar"]["gates"]
mixed_audit["pat"]["gates"]
```

For every gate use:

```text
gate_slot
gate_index
label
reason
curvature_norm
gate_response_norm
epsilon_C
epsilon_G
epsilon_J
epsilon_delta_H
inverse_lower_bound
inverse_admissible
epsilon_A
epsilon_P_F
null_bound
unresolved_bound
dyadic_overlap
ad_route_passed
ad_matched_bypass
factorization
whitebox
whitebox_factorization
shift
whitebox_coordinate_error
contribution_center
contribution_error
contribution_lower
contribution_upper
```

Fields absent because a branch was not reached shall be recorded as `null` with an explicit `field_not_applicable_reason`.

### Formulas

\[
s_C
=
\begin{cases}
\infty,&\epsilon_C=0,\ \|C\|>0,\\
0,&\epsilon_C=0,\ \|C\|=0,\\
\|C\|/\epsilon_C,&\epsilon_C>0,
\end{cases}
\]

\[
s_G
=
\begin{cases}
\infty,&\epsilon_G=0,\ \|G\|>0,\\
0,&\epsilon_G=0,\ \|G\|=0,\\
\|G\|/\epsilon_G,&\epsilon_G>0,
\end{cases}
\]

and, when available:

\[
s_P
=
\frac{\|\widehat P\|_F}{\epsilon_{P,F}}.
\]

If \(\|\widehat P\|_F\) is not serialized, reconstruct it from:

\[
\|\widehat P\|_F
=
\|\widehat A\|_2\|\widehat G\|_2.
\]

### Aggregation unit

Gate-system-item, then summaries by:

```text
gate_slot
gate_index
system
distance_bin
orientation
cell_id
label
reason
```

### Outputs

```text
02_gate_certificate_rows.parquet
02_gate_certificate_summary.json
```

Schemas:

```text
green-v21-gate-certificate-row-v1
green-v21-gate-certificate-summary-v1
```

### Interpretation class

Protocol-design evidence.

### Decision enabled

Determines whether v2 failed globally or through a few pathological gates and whether curvature detectability is the correct successor axis.

---

## 7.3 Analysis 03 — uncertainty-source decomposition

### Purpose

Quantify how much of every uncertainty radius came from finite-radius discrepancy, AD route radius, and endpoint repeatability.

### Inputs

Archived development records, anchors, frozen model fingerprint, structural inputs, radii, `noise_audit_dev.json`, and v2 source code.

No behavioral, PIE, target-performance, or baseline field may be loaded.

### Formulas

For:

\[
T\in\{G,C,J,\Delta H_1,\ldots,\Delta H_5\},
\]

compute:

\[
d_{FD,T}
=
\|\widehat T-T_A\|,
\]

\[
b_{route,T}=r_{A,T},
\]

\[
b_{endpoint,T}=\nu_T,
\]

\[
\epsilon_T
=
d_{FD,T}+b_{route,T}+b_{endpoint,T}.
\]

Then:

\[
f_{FD,T}
=
\frac{d_{FD,T}}{\epsilon_T},
\quad
f_{route,T}
=
\frac{b_{route,T}}{\epsilon_T},
\quad
f_{endpoint,T}
=
\frac{b_{endpoint,T}}{\epsilon_T}.
\]

When \(\epsilon_T=0\), require all three numerators to be zero and set all fractions to zero.

### Aggregation unit

Gate-system-item-object.

### Outputs

```text
03_uncertainty_source_rows.parquet
03_uncertainty_source_summary.json
```

### Interpretation class

Proof-checking and protocol-design evidence.

### Decision enabled

If endpoint or route terms dominate contrary to the committed audits, stop and return. If finite-radius discrepancy dominates, retain the v3 numerical-locality design.

---

## 7.4 Analysis 04 — exact direct transport identity

### Purpose

Test the new transport theorem on exposed v2 development anchors without using behavior.

### Required implementation

Extend the read-only AD evaluator to compute:

\[
J_j^P=D_xY_j^P(0,0),
\]

\[
J_j^C=D_xY_j^C(0,0).
\]

For each frozen frame direction and each deterministic held-out direction \(u\), compute:

\[
T_j^{AD}(u)
=
(J_j^P-J_j^C)u,
\]

\[
T_j^{op}(u)
=
G_j^{AD}
(g_j^{WB})^\top u.
\]

### Residual

\[
r_{j,u}
=
\|T_j^{AD}(u)-T_j^{op}(u)\|_2.
\]

### Bound

Use outward-rounded propagation of:

- forward/reverse AD route radii for \(J^P,J^C,G\);
- the \(10^{-10}\) white-box-coordinate bound;
- structural-envelope residual.

No finite-Richardson term enters this exact AD theorem check.

### Inputs

```text
development anchor cache
development structural frames
model fingerprint
selected gates
isolated float64 AD tail
```

Forbidden inputs:

```text
behavioral
pie
single
first_order
dev_cells performance fields
```

### Aggregation unit

Gate-system-item-direction.

### Outputs

```text
04_exact_transport_identity.parquet
04_exact_transport_identity.json
```

Schema:

```text
green-v21-exact-transport-identity-v1
```

### Interpretation class

Decisive theorem proof-check.

### Decision enabled

Any bound-certified failure stops GREEN and requires return to GPT Pro before v3 implementation.

---

## 7.5 Analysis 05 — exact all-ten joint composition

### Purpose

Test whether the sum of exact per-gate path operators equals the independently implemented joint, direct-bypass-subtracted first-order target.

### Formula

For the frozen physical vector \(v\) and output contrast \(\ell\):

\[
\theta_{op}^{AD}
=
\ell^\top
\sum_{j=1}^{10}
G_j^{AD}
(g_j^{WB})^\top v.
\]

Compute the independent target:

\[
\theta_{joint}^{AD}
=
\ell^\top
D_xY_{\mathrm{joint,bypass-sub}}(0)[v].
\]

Residual:

\[
r_{joint}
=
|\theta_{op}^{AD}-\theta_{joint}^{AD}|.
\]

### Bound

Outward-rounded sum of:

- ten per-gate AD route bounds;
- white-box gradient error;
- target-JVP dual-route error;
- exact structural-envelope residual.

### Inputs

```text
development anchor cache
target vectors
contrast vectors
isolated AD tail
green_bridge_path_target.py
```

### Aggregation unit

Item-system and item-level `pat - tar` contrast.

### Outputs

```text
05_exact_joint_composition.parquet
05_exact_joint_composition.json
```

### Interpretation class

Decisive theorem and implementation proof-check.

### Decision enabled

Any valid residual above its bound stops GREEN before v3 prepare.

---

## 7.6 Analysis 06 — estimator ladder

### Purpose

Localize failure to finite-radius estimation, curvature inversion, gate zero-centering, or target aggregation.

### Estimators

Compute, without changing the official result:

1. `fine_response`:
   \[
   \widehat P^{fine}
   =
   \widehat G^{fine}
   (\widehat g^{fine})^\top.
   \]

2. `coarse_response`:
   \[
   \widehat P^{coarse}.
   \]

3. `ad_response_whitebox`:
   \[
   P^{oracle}
   =
   G^{AD}(g^{WB})^\top.
   \]

4. `fine_G_whitebox_g`:
   \[
   P^{G-only}
   =
   \widehat G^{fine}(g^{WB})^\top.
   \]

5. `active_only_v2`:
   official v2 point centers.

6. `all_gate_response_where_invertible`:
   every finite inverse, irrespective of the frozen SNR rule, clearly labelled post hoc.

7. `zero_centered_v2`:
   official active/null/unresolved point centers.

### Targets

Compare each to:

- exact direct transport;
- independent joint AD target;
- independent finite energy target;
- formal v2 cell target.

### Aggregation unit

Gate-system-item, item, and cell.

### Outputs

```text
06_estimator_ladder_rows.parquet
06_estimator_ladder_summary.json
```

### Interpretation class

Exploratory and protocol-design evidence.

### Decision enabled

- Oracle succeeds, response estimators fail: response-identifiability mismatch.
- Fine \(G\)+white-box \(g\) succeeds, response \(g\) fails: curvature-inverse failure.
- All estimators fail exact transport: theorem or implementation concern.
- Exact transport succeeds but behavioral target fails: target/regime mismatch.

No ladder member may be selected as the v3 point estimator based on these results.

---

## 7.7 Analysis 07 — null and unresolved mass

### Purpose

Measure how much uncertainty is introduced by gates called null or unresolved relative to exact signal and cell target scale.

### Formulas

For each system:

\[
U_{\mathrm{null}}
=
\sum_{j:\mathrm{null}}U_j,
\]

\[
U_{\mathrm{unresolved}}
=
\sum_{j:\mathrm{unresolved}}U_j,
\]

\[
A_{\mathrm{active}}
=
\left|
\sum_{j:\mathrm{active}}\widehat\theta_j
\right|.
\]

Ratios:

\[
R_{\mathrm{null}}
=
\frac{
U_{\mathrm{null}}
}{
\max(
|\theta_{\mathrm{joint}}^{AD}|,
B_{\mathrm{joint}}
)
},
\]

\[
R_{\mathrm{unresolved}}
=
\frac{
U_{\mathrm{unresolved}}
}{
\max(
|\theta_{\mathrm{joint}}^{AD}|,
B_{\mathrm{joint}}
)
}.
\]

If both denominator terms are zero, return zero only when the numerator is zero; otherwise return infinity.

### Inputs

`mixed_audit` gate contribution fields and Analysis 05 exact joint targets.

### Aggregation unit

System-item and cell.

### Outputs

```text
07_null_unresolved_mass.parquet
07_null_unresolved_mass.json
```

### Interpretation class

Protocol-design evidence.

### Decision enabled

Determines whether the absolute `0.005` null rule is grossly mismatched to the target scale.

---

## 7.8 Analysis 08 — aggregation-functional audit

### Purpose

Separate signed mean, mean absolute magnitude, and RMS magnitude.

### Item-level signed quantities

Recover or recompute signed quantities before `abs` for:

```text
behavioral contrast
PIE contrast
single contrast
matched-bypass contrast
independent joint target contrast
```

### Cell functionals

For each signed item value \(d_i\):

\[
A_c
=
\left|
\frac1n\sum_i d_i
\right|,
\]

\[
M_c
=
\frac1n\sum_i|d_i|,
\]

\[
R_c
=
\sqrt{
\frac1n\sum_i d_i^2
},
\]

\[
\chi_c
=
\frac{A_c}{\max(M_c,10^{-12})}.
\]

### Report

For every estimator/target pair:

- Spearman correlation;
- Pearson correlation;
- RMSE;
- sign-stability fraction;
- cancellation ratio;
- near/far summaries;
- orientation summaries.

No functional may be selected because it correlates best.

### Inputs

```text
dev_tensor_scores.parquet
dev_energy_targets.parquet
archived anchors, if signed pre-absolute values require recomputation
```

### Aggregation unit

Item and cell.

### Outputs

```text
08_aggregation_functionals.parquet
08_aggregation_functionals.json
```

### Interpretation class

Exploratory and protocol-design evidence.

### Decision enabled

Confirms or rejects explanation D and determines how behavioral endpoints must be separately reported in the paper.

---

## 7.9 Analysis 09 — role-sampling audit

### Purpose

Determine whether disjoint tensor and energy role sampling adds substantial variance at the observed target scale.

### Method

Using the exposed development pool only, perform a read-only bootstrap with:

```text
replicates = 100,000
seed = 20260825
cluster = noun-century group
```

Compare:

1. official disjoint-role signed-mean target;
2. same-role estimates where technically reconstructable;
3. pooled-role estimates;
4. orientation-balanced estimates;
5. distance-stratified estimates.

### Metrics

For each cell:

\[
\Delta_{\mathrm{role}}
=
\hat\theta_{\mathrm{same}}
-
\hat\theta_{\mathrm{disjoint}},
\]

and its cluster-bootstrap interval.

### Inputs

Tensor and energy Parquets plus role metadata.

### Aggregation unit

Cell and noun-century group.

### Output

```text
09_role_sampling_audit.json
```

### Interpretation class

Exploratory.

### Decision enabled

Large role effects support D/E. Small effects localize failure elsewhere.

---

## 7.10 Analysis 10 — set-SNR geometry audit

### Purpose

Prove mechanically that the diagnostic SNR of one came from zero-crossing signed intervals rather than a reporting defect.

### Inputs

For every counterfactual surviving cell:

```text
theta_tar_lower
theta_tar_upper
theta_pat_lower
theta_pat_upper
mixed_lower
mixed_upper
error_bound
snr
```

### Checks

Reconstruct:

\[
I_\Delta
=
I_{pat}-I_{tar},
\]

\[
I_M=|I_\Delta|.
\]

Record whether:

\[
0\in I_\Delta.
\]

Verify:

\[
I_M=[0,U]
\implies
\mathrm{SNR}=1
\]

within exact float serialization.

### Output

```text
10_set_snr_geometry.json
```

### Interpretation class

Proof-checking.

### Decision enabled

Distinguishes mathematical conservatism from a code bug.

---

## 7.11 Analysis 11 — v1.3.6/v2 regime bridge

### Purpose

Determine whether the earlier strong diagnostics and v2.0.0 address different regimes.

### Inputs

Only already exposed development artifacts from:

```text
analysis/GREEN_V136_TERMINAL_AUDIT_20260825/
analysis/archive/green_v200_stop_20260825/
```

No confirmation artifacts.

### Analyses

Compare by version:

- radii;
- gate labels;
- curvature and response scales;
- factorization metrics;
- \(A/M/R\) functionals;
- PIE;
- behavioral contrast;
- independent target;
- cell definitions;
- system and role sampling.

### Outputs

```text
11_regime_bridge_rows.parquet
11_regime_bridge.json
```

### Interpretation class

Exploratory historical analysis.

### Decision enabled

Supports or rejects E.

These results may appear in a postmortem appendix but may not choose v3 thresholds.

---

## 7.12 Analysis 12 — reporting consistency

### Purpose

Separate panel counts, throughput counts, and scientific operation counts.

### Required checks

```text
frozen prepare panel:
    40 gate-system certificates
    80 GateJet routes

throughput preflight:
    160 gate-system certificates
    320 GateJet routes

development:
    1,280 gate-system certificates
    2,560 GateJet routes
```

Verify the corresponding timing and operation-count artifacts.

### Output

```text
12_reporting_consistency.json
```

### Interpretation class

Reporting-only.

### Decision enabled

Corrects future prose and tables. It does not alter v2.0.0.

---

# 8. Successor protocol decision

## 8.1 Identity

The successor is a major version because its primary scientific endpoint changes.

```text
SCHEMA_VERSION
    green-bridge-v3.0.0

PROTOCOL_ID
    structural-envelope-matched-bypass-transport-v3.0.0

PARENT_PROTOCOL_ID
    structural-envelope-matched-bypass-setid-v2.0.0

DECISION_ID
    GPTPRO-GREEN-V21-POSTMORTEM-TRANSPORT-v1-20260825

PROTOCOL_RUN_ID
    green-bridge-v3.0.0-one-shot

ATTEMPT_INDEX
    1

RETRY_ALLOWED
    false

PHASE_ALL_ALLOWED
    false

RESUME_ALLOWED
    false

AUTHORIZED_PHASES_UNDER_THIS_DOCUMENT
    prepare only
```

The formal external root shall be:

```text
/mnt/sdb/ccj/iclr_1_runs/<execution-id>/outputs/green_bridge_v300
```

No formal v3 root may be created until all read-only postmortem gates and all tests pass.

## 8.2 What is inherited unchanged

The following remain frozen from v2.0.0:

```text
model ID and revision
tokenizer
TransformerLens source commit and source hashes
float32 scientific execution
isolated float64 AD audit
exact full unembedding
manual-tail equivalence
exact batch-one endpoint execution
selected ten gates
block-8 patch semantics
block-10 residual intervention site
block-10 gate-preactivation intervention site
path/control/joint semantics
direct residual-bypass subtraction in the joint target
output contrast construction
exact LayerNorm structural frame
common-frame dimension = 4
per-gate frame dimension = 5
all-gate frame dimension = 14
basis-free ambient rank-one operator
no pseudoinverse
no ridge
no donor PCA
no learned alignment
no outcome-adaptive estimator selection
development/confirmation phase firewall
one-shot semantics
```

## 8.3 Allowed scientific changes

Only the following scientific changes are authorized:

1. make held-out path-minus-control transport the primary target;
2. extend GateJet/AD evaluation with \(J^C\);
3. add the curvature-detectability theorem and recoverability analysis;
4. perform outcome-blind global radius calibration on legacy donor records;
5. use deterministic held-out frame mixtures and orthogonal complement directions;
6. replace the v2 absolute target-null ceiling with a numerical detectability definition;
7. remove the v2 requirement of three active gates per system;
8. classify gates by recoverability, certified numerical nullity, unresolved status, numerical invalidity, and structural contradiction;
9. retain unresolved gates in all-ten joint intervals without silently setting them to a scientific point value;
10. evaluate the formal behavioral functionals \(A\), \(M\), and \(R\) separately as secondary endpoints;
11. use a newly frozen noun-separated development/confirmation split from never-evaluated v2 confirmation groups;
12. compare the matched estimator with a frozen transport-baseline family.

## 8.4 Why this is not retrospective threshold tuning

The successor does not:

- lower v2’s `20×` SNR threshold on the same outcome;
- lower the three-active-gate requirement;
- relabel v2 null gates;
- reuse v2 development cells as v3 inferential cells;
- select a behavioral functional because it correlated well;
- select a radius because it predicts behavior;
- apply v3 rules retrospectively to claim a v2 success.

Instead:

- the primary target changes to a direct theorem consequence;
- radius is calibrated only against independent numerical AD derivatives;
- the inferential groups are untouched;
- the split is determined by hashes and noun identity;
- all v3 thresholds are frozen before any v3 development response exists.

---

# 9. Frozen v3.0.0 split

## 9.1 Source population

Use only the twelve v2.0.0 confirmation noun-century groups that were never executed.

No v2 development group may enter v3.

## 9.2 Noun-level allocation

Hash each eligible noun using:

```python
sha256(
    f"green-v300-transport-noun-split-20260825|{noun}"
    .encode("utf-8")
).hexdigest()
```

The frozen ordering is:

| Rank | Noun | Rank key |
|---:|---|---|
| 1 | `kingdom` | `2cb603de8d771d6e31da85fcac4a8c92710acaf3f2247cf4a78536ab71bdb421` |
| 2 | `reign` | `444eba1462ed40d15dc1c16c7a6c8546577790ce8dcd4982c2d246f2c29fbe7e` |
| 3 | `siege` | `47348e3ba5e30dca846c17bc4f49a279cf1e4008e9ed23c8bdca07f07802a312` |
| 4 | `warfare` | `9ecb254c9b50e23347e2ff87b974b0276cc0d68ef66309af56d188f25f09211b` |
| 5 | `campaign` | `a61597275b4c915e9c7817db66a23f6b606bc7dbcd90a4ce887f71c817bfc899` |
| 6 | `expedition` | `c321a47e8b081ed358bc3563699b7468de0d7b1e643ee885fa67eb7e9ea514b9` |
| 7 | `treaty` | `fbb0f0ca767e11356284380417c68e28eee578a4e8307418803ba458e4098edb` |

### Development nouns

```text
kingdom
reign
siege
```

Development groups:

```text
kingdom / 12
kingdom / 16
reign   / 12
siege   / 14
siege   / 16
```

With `near` and `far`, this yields 10 cells.

### Confirmation nouns

```text
warfare
campaign
expedition
treaty
```

Confirmation groups:

```text
warfare    / 12
campaign   / 14
campaign   / 16
expedition / 14
expedition / 16
treaty     / 12
treaty     / 16
```

With `near` and `far`, this yields 14 cells.

No noun crosses phases.

## 9.3 Roles and counts

Roles:

```text
transport
joint
```

Per role per cell:

```text
8 records
4 up
4 down
```

Roles must be pair-disjoint within a cell.

Counts:

| Phase | Cells | Transport | Joint | Total |
|---|---:|---:|---:|---:|
| Development | 10 | 80 | 80 | 160 |
| Confirmation | 14 | 112 | 112 | 224 |

Pair allocation salt:

```text
green-v300-transport-pairs-20260825
```

## 9.4 Canonical split payload

Use:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
```

with UTF-8 encoding and no trailing newline.

The literal canonical JSON is:

```json
{"confirmation_groups":[{"century":12,"noun":"warfare"},{"century":14,"noun":"campaign"},{"century":16,"noun":"campaign"},{"century":14,"noun":"expedition"},{"century":16,"noun":"expedition"},{"century":12,"noun":"treaty"},{"century":16,"noun":"treaty"}],"confirmation_nouns":[{"noun":"warfare","rank_key":"9ecb254c9b50e23347e2ff87b974b0276cc0d68ef66309af56d188f25f09211b"},{"noun":"campaign","rank_key":"a61597275b4c915e9c7817db66a23f6b606bc7dbcd90a4ce887f71c817bfc899"},{"noun":"expedition","rank_key":"c321a47e8b081ed358bc3563699b7468de0d7b1e643ee885fa67eb7e9ea514b9"},{"noun":"treaty","rank_key":"fbb0f0ca767e11356284380417c68e28eee578a4e8307418803ba458e4098edb"}],"development_groups":[{"century":12,"noun":"kingdom"},{"century":16,"noun":"kingdom"},{"century":12,"noun":"reign"},{"century":14,"noun":"siege"},{"century":16,"noun":"siege"}],"development_nouns":[{"noun":"kingdom","rank_key":"2cb603de8d771d6e31da85fcac4a8c92710acaf3f2247cf4a78536ab71bdb421"},{"noun":"reign","rank_key":"444eba1462ed40d15dc1c16c7a6c8546577790ce8dcd4982c2d246f2c29fbe7e"},{"noun":"siege","rank_key":"47348e3ba5e30dca846c17bc4f49a279cf1e4008e9ed23c8bdca07f07802a312"}],"distance_bins":["near","far"],"orientations_per_role":{"down":4,"up":4},"records_per_role_per_cell":8,"roles":["transport","joint"],"salt":"green-v300-transport-noun-split-20260825","schema":"green-bridge-v3.0.0-transport-split-v1","source_split":"green-bridge-v2.0.0-unopened-confirmation"}
```

Required SHA-256:

```text
509f791b614db58e0e7b47c1106364ef549c156e2c42a48a51e705a196da0bc7
```

---

# 10. Frozen held-out direction design

## 10.1 Identification directions

Response identification uses only the existing five columns:

\[
Q_j=[q_{j,1},\ldots,q_{j,5}].
\]

No held-out direction may enter:

- finite mixed-derivative identification;
- curvature inversion;
- response-coordinate fitting;
- radius selection.

## 10.2 In-frame held-out coefficients

Use the four orthonormal Helmert vectors:

\[
a_1
=
\frac1{\sqrt5}(1,1,1,1,1),
\]

\[
a_2
=
\frac1{\sqrt2}(1,-1,0,0,0),
\]

\[
a_3
=
\frac1{\sqrt6}(1,1,-2,0,0),
\]

\[
a_4
=
\frac1{\sqrt{12}}(1,1,1,-3,0).
\]

The in-frame held-out directions are:

\[
u_k^{frame}=Q_ja_k,
\qquad
k=1,\ldots,4.
\]

Canonical coefficient payload SHA-256:

```text
1b5cc44b98b74ae7793957d68087a17d8ce9684ebee822b46a67492e4a7892e5
```

## 10.3 Deterministic orthogonal-complement directions

Construct six directions:

\[
n_{j,1},\ldots,n_{j,6}
\]

using this exact algorithm:

1. work in float64;
2. scan canonical standard-basis vectors \(e_0,\ldots,e_{767}\);
3. project each candidate twice against \(Q_j\);
4. project twice against all previously accepted \(n\)-vectors;
5. accept the first candidate with norm greater than \(10^{-12}\);
6. normalize;
7. set the sign so the largest-absolute-value coordinate is positive;
8. continue until six vectors are accepted.

Require:

\[
\|Q_j^\top n_{j,k}\|_\infty\le10^{-12},
\]

\[
\|N_j^\top N_j-I\|_\infty\le10^{-12}.
\]

## 10.4 Mixed and null directions

Mixed directions:

\[
u_k^{mix}
=
\frac{
Q_ja_k+n_{j,k}
}{
\sqrt2
},
\qquad
k=1,\ldots,4.
\]

Pure envelope-null directions:

\[
u_1^{null}=n_{j,5},
\qquad
u_2^{null}=n_{j,6}.
\]

Each gate therefore has ten held-out directions:

```text
4 in-frame
4 mixed
2 pure-null
```

All direction arrays and hashes shall be frozen during prepare before development is authorized.

---

# 11. Outcome-blind radius calibration

## 11.1 Candidate set

Use the exact global multiplier set:

\[
\mathcal R
=
\left\{
1,\frac12,\frac14,\frac18,\frac1{16},\frac1{32},\frac1{64}
\right\}.
\]

Canonical candidate payload hash:

```text
50251164fb42f9ecd97c7725a093ff15084b9f6662b364d3d52be1210c98feb9
```

## 11.2 Calibration panel

Use 40 legacy donor strata:

```text
2 systems × 10 gates × 2 distance bins
```

Select one record per stratum by minimum:

```python
sha256(
    (
        "green-v300-radius-calibration-20260825|"
        f"{pair_digest}|{system}|{gate_slot}|{distance_bin}"
    ).encode("utf-8")
).hexdigest()
```

The panel may read:

```text
pair_digest
system
gate_slot
distance_bin
prompt/token metadata
anchors
frames
```

It may not read:

```text
behavioral
PIE
first_order
single
cell target
cell survival
baseline score
v2 label
v2 performance
```

No v2 development record may enter the calibration panel.

## 11.3 Candidate stencils

For candidate \(\rho\), evaluate:

\[
T_{\rho},
\qquad
T_{\rho/2},
\qquad
T_{\rho/4}.
\]

The immutable scientific candidate estimate is:

\[
\boxed{
R_{\rho,T}^{fine}
=
\frac{
4T_{\rho/4}-T_{\rho/2}
}{
3
}.
}
\]

The coarse diagnostic is:

\[
R_{\rho,T}^{coarse}
=
\frac{
4T_{\rho/2}-T_{\rho}
}{
3
}.
\]

## 11.4 Eligibility rule

For:

\[
T
\in
\{
G,C,J^P,J^C,\Delta H_1,\ldots,\Delta H_5
\},
\]

let:

\[
d_{\rho,T}
=
\|
R_{\rho,T}^{fine}-T_A
\|,
\]

where \(T_A\) is the dual-route AD midpoint.

Let:

\[
b_{\rho,T}
=
r_{A,T}
+
\nu_{\rho,T}.
\]

Candidate \(\rho\) is eligible iff, for every object and all 40 strata:

\[
\boxed{
d_{\rho,T}
\le
\operatorname{up}
\left(
0.10\|T_A\|+b_{\rho,T}
\right).
}
\]

Also require:

- all values finite;
- all endpoint floor assertions pass;
- every AD route passes;
- every exact AD theorem check passes;
- active float32 model remains byte-identical;
- no fallback is used.

The constant `0.10` is an a priori numerical-fidelity requirement for the finite point estimator. It was not derived from v2 behavioral outcomes.

## 11.5 Selection

Select:

\[
\boxed{
\rho_*
=
\max
\{
\rho\in\mathcal R:
\rho\text{ is eligible}
\}.
}
\]

The same \(\rho_*\) applies to:

```text
every gate
every system
every development record
every confirmation record
both h_x and h_z
```

No per-item or per-gate radius adaptation is allowed.

If no candidate is eligible:

```text
PREPARE STOP 08_RADIUS_LOCALITY
```

No development is authorized.

---

# 12. v3.0.0 scientific estimands

## 12.1 Response-only operator

Using the selected radius:

\[
\widehat G_j,
\quad
\widehat C_j,
\quad
\widehat{\Delta H}_{j,i}
\]

are the global-radius fine-Richardson estimates.

When response inversion is admissible:

\[
\widehat A_{j,i}
=
\frac{
\langle\widehat C_j,\widehat{\Delta H}_{j,i}\rangle
}{
\|\widehat C_j\|_2^2
},
\]

\[
\widehat g_j
=
Q_j\widehat A_j,
\]

\[
\boxed{
\widehat{\mathcal P}_j
=
\widehat G_j\widehat g_j^\top.
}
\]

AD and white-box values remain audits and targets. They do not replace the point estimator.

## 12.2 Direct held-out transport target

For held-out direction \(u_k\), compute independent path and control first derivatives:

\[
T_{j,k}^{dir}
=
D_xY_j^P(0,0)[u_k]
-
D_xY_j^C(0,0)[u_k].
\]

The primary executable target is an independently evaluated fine-Richardson direct derivative at \(\rho_*\), certified against dual-route AD.

Prediction:

\[
\widehat T_{j,k}
=
\widehat{\mathcal P}_ju_k.
\]

## 12.3 Prediction bound

Let:

- \(\epsilon_{P,F,j}\) be the response-operator Frobenius uncertainty;
- \(\epsilon_{G,j}\) be response uncertainty;
- \(\epsilon_{\mathrm{env},j}\) be the structural-envelope residual;
- \(B_{dir,j,k}\) be the direct-target certificate radius.

Then:

\[
B_{pred,j,k}
=
\operatorname{up}
\left[
\epsilon_{P,F,j}\|Q_j^\top u_k\|_2
+
(\|\widehat G_j\|_2+\epsilon_{G,j})
\epsilon_{\mathrm{env},j}
\|u_k\|_2
\right],
\]

\[
\boxed{
B_{total,j,k}
=
\operatorname{up}
\left(
B_{pred,j,k}+B_{dir,j,k}
\right).
}
\]

The theorem-compatibility condition is:

\[
\boxed{
\|
\widehat T_{j,k}-T_{j,k}^{dir}
\|_2
\le
B_{total,j,k}.
}
\]

Any valid excess is a `structural-contradiction`.

## 12.4 Gate-level held-out error

For the eight signal-bearing directions:

```text
4 in-frame
4 mixed
```

stack:

\[
\widehat{\mathbf T}_j
\in\mathbb R^{8\times100},
\qquad
\mathbf T_j^{dir}
\in\mathbb R^{8\times100}.
\]

Let:

\[
B_{\mathbf T,j}
=
\sqrt{
\sum_{k=1}^{8}B_{total,j,k}^2
}.
\]

Define:

\[
E_j
=
\frac{
\|
\widehat{\mathbf T}_j-\mathbf T_j^{dir}
\|_F
}{
\max(
\|\mathbf T_j^{dir}\|_F,
B_{\mathbf T,j}
)
}.
\]

Zero rule:

- if both denominator terms are zero and the numerator is zero, \(E_j=0\);
- if both denominator terms are zero and the numerator is positive, \(E_j=\infty\).

## 12.5 Null leakage

For the two pure-null directions:

\[
L_j
=
\frac{
\|
\mathbf T_{j,null}^{dir}
\|_F
}{
\max(
\|\mathbf T_{j,signal}^{dir}\|_F,
B_{\mathbf T,j}
)
}.
\]

The exact envelope theorem predicts negligible gate-mediated transport in the orthogonal complement.

## 12.6 All-ten joint composition

For the frozen physical vector \(v\) and output contrast \(\ell\):

\[
\widehat\theta_{joint}
=
\ell^\top
\sum_{j=1}^{10}
\widehat{\mathcal P}_jv.
\]

Unresolved gates contribute intervals rather than zero scientific points.

The independent target remains:

\[
\theta_{joint}^{target}
=
\ell^\top
D_xY_{\mathrm{joint,bypass-sub}}(0)[v].
\]

Define the joint center error:

\[
E_{joint}
=
\frac{
|
\widehat\theta_{joint}
-
\theta_{joint}^{target}
|
}{
\max(
|\theta_{joint}^{target}|,
B_{joint}
)
}.
\]

---

# 13. v3 gate classes

Every gate-system-item receives exactly one class.

## 13.1 `numerical-invalid`

Use when:

- endpoint equivalence fails;
- active model integrity fails;
- finite or AD tensor is nonfinite;
- AD routes fail;
- radius-calibrated target certificate fails;
- frame or hook contract fails.

These units are never aggregated.

## 13.2 `structural-contradiction`

Use when:

- exact AD matched-bypass factorization fails;
- exact AD direct transport identity fails;
- response estimate exceeds a valid theorem bound;
- shift-null or white-box identity fails;
- pure complement transport exceeds its proof-derived structural bound.

Contradictions may never be converted to unresolved.

## 13.3 `recoverable`

A gate is recoverable iff:

1. no numerical or structural failure occurred;
2. response inverse is admissible;
3. the response operator is finite;
4. curvature relative half-width passes:
   \[
   \frac{\epsilon_C}{\|\widehat C\|_2}\le0.25;
   \]
5. response relative half-width passes:
   \[
   \frac{\epsilon_G}{\|\widehat G\|_2}\le0.25;
   \]
6. operator relative half-width passes:
   \[
   \frac{\epsilon_{P,F}}{\|\widehat P\|_F}\le0.25.
   \]

Zero-denominator rules:

- zero estimate and zero bound: relative width \(0\);
- zero estimate and positive bound: relative width \(\infty\);
- positive estimate and zero bound: relative width \(0\).

The `0.25` threshold is a new precision standard equivalent to a signal-to-uncertainty ratio of at least four. It is not a lowered v2 gate applied to v2 outcomes.

## 13.4 `certified-numerical-null`

Let the exact AD path-operator upper bound be:

\[
U_j^{AD}
=
\operatorname{up}
\left[
(\|G_j^A\|_2+r_{A,G})
(\|g_j^{WB}\|_2+\epsilon_{g,WB})
\right].
\]

Let the direct-transport numerical floor across the ten held-out directions be:

\[
B_{0,j}
=
\sqrt{
\sum_{k=1}^{10}
B_{dir,j,k}^2
}.
\]

The gate is a certified numerical null iff:

\[
\boxed{
U_j^{AD}
\le
B_{0,j}.
}
\]

This replaces the v2 absolute `0.005` ceiling.

## 13.5 `unresolved`

Use when:

- all theorem and numerical checks pass;
- the gate is not a certified numerical null;
- response-only recovery does not meet the `0.25` width standard.

An unresolved gate remains part of:

- detectability analyses;
- all-ten joint uncertainty;
- coverage counts.

It is never silently assigned a causal point estimate.

---

# 14. Admissibility, conditioning, SNR, and interval rules

## 14.1 Gate-direction admissibility

A gate-direction unit is admissible iff:

- gate class is `recoverable` or `certified-numerical-null`;
- direct target is certified;
- no numerical or structural failure exists.

Unresolved units are retained but are not used in point-error summaries.

## 14.2 Record admissibility

A `transport` record is technically admissible iff:

- both `tar` and `pat` systems have all ten gates classified;
- no gate is invalid or contradictory;
- at least 80% of the 20 system-gates are recoverable or certified numerical null;
- all held-out direction targets are finite and certified.

A `joint` record is technically admissible iff:

- all ten gates are accounted for;
- no invalid or contradictory gate exists;
- the joint target is certified;
- the all-ten interval is finite.

## 14.3 Cell survival

A cell survives iff:

```text
technically admissible transport records >= 6 / 8
technically admissible joint records     >= 6 / 8
```

## 14.4 Detectability conditioning

A direct target is nonnull iff:

\[
\frac{
\|\mathbf T_j^{dir}\|_F
}{
B_{\mathbf T,j}
}
\ge4.
\]

A cell is detectability-conditioned iff:

- it has at least one nonnull gate-system unit;
- at least 25% of its nonnull gate-system units are response-recoverable.

Certified numerical nulls are excluded from the denominator.

Unresolved nonnull units remain in the denominator and therefore expose response-identification blindness.

## 14.5 Signed set SNR

For signed joint interval:

\[
I=[\widehat\theta-B,\widehat\theta+B],
\]

define:

\[
\mathrm{SNR}_{set}
=
\begin{cases}
0,&B=0,\ \widehat\theta=0,\\
\infty,&B=0,\ \widehat\theta\ne0,\\
|\widehat\theta|/B,&B>0.
\end{cases}
\]

Do not apply absolute-value interval transformation before computing SNR.

A joint cell is set-SNR-qualified iff:

\[
\mathrm{SNR}_{set}\ge4.
\]

## 14.6 Unresolved interval-mass ratio

For a joint cell:

\[
R_{unresolved}
=
\frac{
B_{unresolved}
}{
\max(
|\theta_{joint}^{target}|,
B_{target}
)
}.
\]

A cell has sufficiently narrow unresolved mass iff:

\[
R_{unresolved}\le0.25.
\]

---

# 15. Frozen baselines and baseline selection

## 15.1 Baseline family

The frozen transport baselines are:

### `zero`

\[
\widehat T^{zero}=0.
\]

### `gate_atom_only`

Use only the fifth, gate-specific structural coordinate:

\[
\widehat g^{atom}
=
\widehat A_5q_5.
\]

### `unmatched_path_mixed`

Replace matched bypass:

\[
\Delta H=H^P-H^C
\]

with:

\[
\Delta H^{unmatched}=H^P.
\]

All other identification steps remain unchanged.

### `raw_path_jacobian`

Predict gate-mediated transport using the full path derivative:

\[
\widehat T^{raw}=J^Pu,
\]

without removing direct bypass.

## 15.2 Diagnostic ceilings

The following are reported but are not eligible baselines:

```text
AD G × white-box g
fine G × white-box g
direct frame Jacobian fitted from target transport
```

They are mechanism-localization ceilings.

## 15.3 Baseline error

For each admissible nonnull unit:

\[
e_{b,u}
=
\frac{
\|
\widehat T_{b,u}-T_u^{dir}
\|_2
}{
\max(
\|T_u^{dir}\|_2,
B_u
)
}.
\]

Group-balanced baseline RMSE is:

\[
\mathrm{RMSE}_b
=
\sqrt{
\frac1{|\mathcal G|}
\sum_{g\in\mathcal G}
\frac1{|g|}
\sum_{u\in g}
e_{b,u}^2
},
\]

where \(\mathcal G\) is the noun-century group set.

## 15.4 Selection and freeze

On development:

```text
best baseline = minimum group-balanced RMSE
tie break = lexicographically smallest baseline name
```

No fitted affine calibration is allowed.

The selected baseline name and all baseline errors are frozen before confirmation.

Confirmation may not reselect a baseline.

## 15.5 Gain

\[
\mathrm{gain}
=
1-
\frac{
\mathrm{RMSE}_{matched}
}{
\mathrm{RMSE}_{best\ baseline}
}.
\]

Zero rule:

- if both RMSE values are zero, gain \(=0\);
- if baseline RMSE is zero and matched RMSE is positive, gain \(=-\infty\).

---

# 16. Exact future development gates

Development is not authorized by this document. The following gates are nevertheless frozen now so that prepare cannot influence them.

## 16.1 Required records

```text
transport records = 80
joint records     = 80
cells             = 10
```

## 16.2 Technical gates

All must pass:

```text
surviving cells >= 8 / 10
surviving near cells >= 4 / 5
surviving far cells >= 4 / 5
all three development nouns represented
numerical-invalid units = 0
structural-contradiction units = 0
resolved gate-system coverage >= 80%
nonnull response-recoverability fraction >= 25%
set-SNR-qualified joint cells >= 6
cells with unresolved-mass ratio <= 0.25 >= 6
every selected gate slot has at least 10 recoverable development units
```

## 16.3 Bound-validity gates

Every admissible recoverable unit must satisfy:

\[
\|
\widehat T-T^{dir}
\|
\le
B_{total}.
\]

Required bound failures:

```text
0
```

## 16.4 Direct transport gates

Across recoverable nonnull units:

```text
median E_j <= 0.10
90th percentile E_j <= 0.25
```

## 16.5 Joint composition gates

Across technically admissible nonnull joint records:

```text
median E_joint <= 0.15
90th percentile E_joint <= 0.30
```

## 16.6 Detectability gate

Compute Spearman correlation across gate-system units:

\[
\rho_{\mathrm{det}}
=
\operatorname{Spearman}
\left(
\log\frac{\|C\|}{\epsilon_C},
-\log(E_j+10^{-12})
\right).
\]

Require:

```text
rho_det >= 0.50
noun-century-cluster bootstrap 95% LCB > 0
```

Bootstrap:

```text
replicates = 100,000
seed = 20260805
cluster = noun-century group
```

## 16.7 Null-leakage gates

```text
median L_j <= 0.05
95th percentile L_j <= 0.10
```

## 16.8 Baseline and robustness gates

```text
matched gain over development-selected best baseline >= 0.20
matched gain over every fixed baseline >= 0.10
group-cluster bootstrap 95% gain LCB > 0
coarse/fine cell Spearman >= 0.90
coarse/fine median symmetric change <= 0.20
```

## 16.9 Development verdicts

### `OPEN_CONFIRMATION`

Only if every technical, theorem, direct-transport, joint, detectability, null-leakage, baseline, and robustness gate passes.

### `POSTER_ONLY`

Only if:

```text
all theorem and numerical gates pass
surviving cells >= 8
median direct error <= 0.15
90th percentile direct error <= 0.35
matched gain >= 0.10
```

but one or more `OPEN_CONFIRMATION` performance gates fail.

Confirmation remains closed.

### `STOP_ORAL`

Any other result.

---

# 17. Exact future confirmation gates

Confirmation is not authorized by this document.

## 17.1 Required records

```text
transport records = 112
joint records     = 112
cells             = 14
```

## 17.2 Technical gates

```text
surviving cells >= 12 / 14
surviving near cells >= 6 / 7
surviving far cells >= 6 / 7
all four confirmation nouns represented
numerical-invalid units = 0
structural-contradiction units = 0
resolved gate-system coverage >= 85%
nonnull response-recoverability fraction >= 30%
set-SNR-qualified joint cells >= 10
cells with unresolved-mass ratio <= 0.20 >= 10
every selected gate slot has at least 16 recoverable confirmation units
```

## 17.3 Oral transport gates

```text
median direct error <= 0.08
90th percentile direct error <= 0.15
median joint error <= 0.10
90th percentile joint error <= 0.20
```

## 17.4 Oral baseline gates

Using only the development-frozen baseline:

```text
matched gain >= 0.25
noun-century-cluster bootstrap 95% gain LCB >= 0.15
near-bin gain >= 0.15
far-bin gain >= 0.15
near-bin gain LCB > 0
far-bin gain LCB > 0
```

## 17.5 Detectability and null gates

```text
detectability Spearman >= 0.60
detectability bootstrap 95% LCB > 0
median null leakage <= 0.03
95th percentile null leakage <= 0.08
```

## 17.6 Radius robustness

```text
coarse/fine cell Spearman >= 0.90
coarse/fine median symmetric change <= 0.20
```

## 17.7 Final outcomes

### Oral-level pass

Every confirmation gate passes.

### Poster Only

No theorem or numerical contradiction exists, technical coverage passes, and matched gain is at least `0.10`, but one or more Oral gates fail.

### Stop

Any theorem contradiction, technical failure, gain below `0.10`, split violation, hash mismatch, or phase violation.

---

# 18. Anti-overfitting and contamination firewall

## 18.1 Existing observations permitted for diagnosis

The following exposed observations may be used only for postmortem diagnosis:

```text
v1.3.6 development rows
v2.0.0 development rows
v2.0.0 prepare audits
v2.0.0 terminal counts
v2.0.0 read-only counterfactual
```

They may be used to:

- identify implementation bugs;
- test exact theorems;
- explain finite-radius error;
- compare aggregation functionals;
- motivate the successor scientific question;
- create clearly labelled postmortem figures.

## 18.2 Existing observations forbidden for protocol selection

They may not be used to choose:

```text
v3 success thresholds
v3 gate slots
v3 held-out directions
v3 split
v3 baseline family
v3 radius by behavioral performance
v3 cell definition
v3 null threshold
v3 development-opening rule
v3 confirmation-opening rule
```

## 18.3 Radius calibration firewall

Radius is selected only using:

- legacy donor records;
- numerical finite-versus-AD error;
- exact theorem checks;
- no behavioral or task target.

The radius panel is disjoint from v3 development and confirmation.

## 18.4 Inferential split firewall

The v3 development and confirmation nouns are disjoint.

No confirmation response, anchor, logit, derivative, cache, or timing may be generated before a future binding decision explicitly authorizes confirmation.

## 18.5 Source-code firewall

The v3 scientific runner shall reject imports or reads from:

```text
analysis/GREEN_V21_POSTMORTEM_20260825/
analysis/GREEN_V200_DEVELOPMENT_TERMINAL_DIAGNOSTIC_20260825.json
analysis/GREEN_V136_TERMINAL_AUDIT_20260825/terminal_admissibility_audit.json
analysis/archive/green_v200_stop_20260825/dev_tensor_scores.parquet
analysis/archive/green_v200_stop_20260825/dev_energy_targets.parquet
```

except inside the dedicated predecessor verifier and postmortem scripts.

Postmortem outputs may not be runtime inputs to v3 development.

---

# 19. Ablations and falsifiers

| Test | Supports | Rejects or weakens |
|---|---|---|
| Exact AD \(J^P-J^C=Gg^\top\) | Exact transport theorem | A failure supports A or a control implementation defect |
| Exact all-ten operator sum vs joint JVP | Additive first-order composition | Failure indicates omitted path, target mismatch, or theorem scope failure |
| Fine response operator vs AD/white-box operator | B if oracle succeeds but response fails | If both fail, B is insufficient |
| Radius ladder | Finite-radius bias if error contracts with \(\rho\) | Stable error at small radii suggests implementation/theorem issue |
| Uncertainty-source decomposition | C if interval width is mostly propagated floor | C weakened if finite-versus-AD discrepancy dominates |
| In-frame vs mixed directions | Envelope correctness if both pass | Mixed-only failure indicates missing ambient component |
| Pure-null complement transport | Exact probe completeness | Leakage beyond bound contradicts structural envelope |
| Matched vs unmatched mixed derivative | Necessity of bypass control | Similar or better unmatched result weakens matched-bypass novelty |
| Gate-atom-only baseline | Necessity of complete frame | Equal performance weakens probe-completeness contribution |
| Raw path Jacobian baseline | Need to separate direct bypass | Equal performance suggests path operator is unnecessary |
| Signed mean vs mean absolute vs RMS | D if only magnitude functionals align | D weakened if all functionals coincide |
| Same-role vs disjoint-role | Sampling mismatch if estimates shift materially | Small shifts reject role sampling as main cause |
| Response error vs curvature detectability | New detectability theorem | No monotonic relationship weakens the successor mechanism |
| Joint interval width vs exact joint error | C if centers are accurate but intervals wide | If centers are poor, point estimation is the issue |
| v1.3.6 vs v2 regime bridge | E if effects differ by estimand/radius | Similar regimes weaken E |
| Future layer/model replication | General mechanism | Single-layer-only behavior weakens Oral claim |

Any exact theorem failure must be treated as a hard falsifier, not repaired through threshold adjustment.

---

# 20. Implementation contract

## 20.1 Branch and worktree

Create a new branch:

```text
codex/green-v300
```

from:

```text
ef09fce529553d5a3d236852a288cde02b88418a
```

The worktree must be under:

```text
/mnt/sdb/ccj/worktrees/idle_1_green_v300
```

Do not modify the v2 branch or official execution artifacts.

## 20.2 Required new files

```text
analysis/GPTPRO_GREEN_V21_POSTMORTEM_DECISION_20260825.md
analysis/GREEN_V21_POSTMORTEM_20260825/analyze_v200_postmortem.py

src/green_bridge_v300_spec.py
src/green_bridge_v300_dataset.py
src/green_bridge_v300_directions.py
src/green_bridge_v300_numerics.py
src/green_bridge_v300_transport.py
src/exp_green_bridge_v300.py
src/analyze_green_bridge_v300.py
src/green_bridge_v300_multigpu_worker.py
src/launch_green_bridge_v300.sh

tests/test_green_bridge_v300_contract.py
```

Historical v2 files shall remain unchanged unless a minimal import-free helper must be copied. Prefer versioned duplication over mutation.

## 20.3 Required functions

### `green_bridge_v300_dataset.py`

```python
build_green_bridge_v300_split()
build_green_bridge_v300_records()
canonical_v300_split_payload()
verify_v300_contamination_firewall()
```

### `green_bridge_v300_directions.py`

```python
helmert_coefficients_v300()
deterministic_complement_v300()
heldout_direction_panel_v300()
direction_design_sha256_v300()
```

### `green_bridge_v300_numerics.py`

```python
response_detectability_v300()
direct_transport_certificate_v300()
joint_composition_certificate_v300()
relative_width_v300()
normalized_transport_error_v300()
signed_set_snr_v300()
radius_candidate_eligibility_v300()
select_global_radius_v300()
```

### `green_bridge_v300_transport.py`

```python
direct_path_control_finite_v300()
direct_path_control_ad_v300()
response_operator_v300()
heldout_transport_prediction_v300()
joint_operator_prediction_v300()
```

### `exp_green_bridge_v300.py`

```python
prepare_v300()
transport_record_v300()
joint_record_v300()
run_split_v300()
verify_v200_terminal_archive_v300()
```

Under this decision:

```python
development_v300()
confirmation_v300()
```

must raise:

```text
UNAUTHORIZED_PHASE_REQUIRES_NEW_GPTPRO_DECISION
```

### `analyze_green_bridge_v300.py`

```python
aggregate_transport_cells_v300()
select_frozen_baseline_v300()
development_decision_v300()
confirmation_decision_v300()
```

The analysis functions may be implemented and tested, but the launcher may not invoke development or confirmation.

## 20.4 Mandatory regression tests

All existing 220 tests must continue to pass.

Add exactly 52 tests.

### Predecessor and immutability — 6

1. `V300PredecessorTests.test_v200_execution_commit_is_exact`
2. `V300PredecessorTests.test_v200_terminal_artifact_hashes_are_verified`
3. `V300PredecessorTests.test_v200_stop_oral_and_confirmation_closed`
4. `V300PredecessorTests.test_v200_root_is_read_only`
5. `V300PredecessorTests.test_v200_development_parquets_are_diagnostic_only`
6. `V300PredecessorTests.test_fixed_rank_donor_pca_remains_terminated`

### Postmortem and firewall — 8

7. `V21PostmortemTests.test_postmortem_reads_only_archived_development`
8. `V21PostmortemTests.test_postmortem_marks_official_verdict_unchanged`
9. `V21PostmortemTests.test_postmortem_forbids_threshold_selection`
10. `V21PostmortemTests.test_confirmation_paths_are_denied`
11. `V21PostmortemTests.test_theorem_checks_read_no_behavioral_fields`
12. `V21PostmortemTests.test_integrity_failure_stops_before_v300`
13. `V21PostmortemTests.test_exact_transport_failure_stops_before_v300`
14. `V21PostmortemTests.test_exact_joint_failure_stops_before_v300`

### Split and records — 8

15. `V300SplitTests.test_literal_split_payload_hash_is_509f791b`
16. `V300SplitTests.test_development_nouns_are_exact`
17. `V300SplitTests.test_confirmation_nouns_are_exact`
18. `V300SplitTests.test_no_noun_crosses_phase`
19. `V300SplitTests.test_v200_development_groups_are_excluded`
20. `V300SplitTests.test_roles_are_transport_and_joint`
21. `V300SplitTests.test_role_pairs_are_disjoint_and_balanced`
22. `V300SplitTests.test_record_counts_are_160_and_224`

### Direction design — 8

23. `V300DirectionTests.test_helmert_coefficients_are_orthonormal`
24. `V300DirectionTests.test_in_frame_directions_are_unit_norm`
25. `V300DirectionTests.test_complement_directions_are_deterministic`
26. `V300DirectionTests.test_complement_is_orthogonal_to_frame`
27. `V300DirectionTests.test_mixed_directions_are_unit_norm`
28. `V300DirectionTests.test_null_directions_are_orthogonal_to_frame`
29. `V300DirectionTests.test_direction_hash_is_repeatable`
30. `V300DirectionTests.test_heldout_directions_never_enter_identification`

### Theory and transport — 8

31. `V300TransportTheoryTests.test_path_minus_control_jacobian_equals_rank_one_operator`
32. `V300TransportTheoryTests.test_matched_bypass_factorization_on_synthetic_map`
33. `V300TransportTheoryTests.test_zero_curvature_is_response_nonidentifiable`
34. `V300TransportTheoryTests.test_detectability_bound_is_monotone_in_curvature`
35. `V300TransportTheoryTests.test_direct_target_is_probe_independent`
36. `V300TransportTheoryTests.test_joint_first_order_composition_is_additive`
37. `V300TransportTheoryTests.test_structural_contradiction_cannot_be_unresolved`
38. `V300TransportTheoryTests.test_ad_is_audit_not_point_estimator`

### Radius calibration — 6

39. `V300RadiusTests.test_candidate_radius_payload_hash_is_exact`
40. `V300RadiusTests.test_largest_eligible_radius_is_selected`
41. `V300RadiusTests.test_calibration_panel_is_behavior_blind`
42. `V300RadiusTests.test_calibration_uses_only_legacy_donors`
43. `V300RadiusTests.test_no_eligible_radius_stops_prepare`
44. `V300RadiusTests.test_selected_radius_is_global_and_frozen`

### Analysis and gates — 5

45. `V300AnalysisTests.test_relative_error_zero_denominator_rules`
46. `V300AnalysisTests.test_recoverable_width_threshold_is_one_quarter`
47. `V300AnalysisTests.test_unresolved_is_not_zeroed_in_joint_interval`
48. `V300AnalysisTests.test_best_baseline_is_group_balanced_and_frozen`
49. `V300AnalysisTests.test_confirmation_cannot_reselect_baseline`

### Launcher and provenance — 3

50. `V300LauncherTests.test_all_runtime_paths_are_under_mnt_sdb`
51. `V300LauncherTests.test_prepare_is_the_only_authorized_phase`
52. `V300LauncherTests.test_phase_all_retry_and_resume_are_forbidden`

Required total:

```text
Ran 272 tests
OK
```

Zero skips.

## 20.5 Required prepare artifacts

The v3 prepare root must contain:

```text
run_ledger.json
predecessor_v200_terminal_manifest.json
postmortem_manifest.json
v300_split.json
v300_record_plan.json
v300_direction_design.json
v300_direction_design.npz
v300_radius_candidate_panel.json
v300_radius_calibration.json
v300_synthetic_theorem_suite.json
v300_model_fingerprint.json
v300_gate04_audit.json
v300_manual_tail_equivalence.json
v300_structural_frame_preflight.json
v300_transport_theorem_preflight.json
v300_joint_composition_preflight.json
v300_operation_counts.json
v300_hardware_plan.json
v300_throughput_preflight.json
prepare_result.json
manifest.json
sha256sums.txt
```

The prepare root must not contain:

```text
dev_transport_scores.parquet
dev_joint_targets.parquet
dev_cells.json
dev_result.json
frozen_analysis.json
confirm_transport_scores.parquet
confirm_joint_targets.parquet
confirm_cells.json
confirm_result.json
```

## 20.6 Prepare gates

Prepare passes only if:

1. v2 archive hashes pass;
2. official v2 verdict and phase state are exact;
3. all twelve read-only postmortem analyses complete;
4. Analyses 04 and 05 have zero theorem failures;
5. no v2 confirmation artifact exists;
6. all 272 tests pass;
7. split hash equals `509f791b...`;
8. coefficient hash equals `1b5cc44b...`;
9. radius candidate hash equals `50251164...`;
10. a global eligible \(\rho_*\) exists;
11. all synthetic theorem tests pass;
12. Gate-04, manual-tail, endpoint, frame, and model-integrity audits pass;
13. held-out directions meet all orthogonality bounds;
14. throughput projects at most 24 eight-GPU wall-clock hours;
15. peak memory is at most 20 GiB per RTX 4090;
16. no lower-precision, selected-projection, reduced-gate, radius, or batching fallback occurs;
17. all artifacts are hashed;
18. no development or confirmation response exists.

A successful result is:

```json
{
  "schema_version": "green-bridge-prepare-v3.0.0",
  "verdict": "PREPARE_PASS",
  "attempt_index": 1,
  "retry_allowed": false,
  "authorized_next_phase": null,
  "development_authorized": false,
  "confirmation_authorized": false,
  "first_failed_gate": null
}
```

---

# 21. GPU, storage, and launcher contract

## 21.1 Storage environment

The launcher must export:

```bash
export GREEN_BASE=/mnt/sdb/ccj
export GREEN_RUNTIME_ROOT=/mnt/sdb/ccj/iclr_1_runs/green_bridge_v300_${EXECUTION_COMMIT}
export HF_HOME=/mnt/sdb/ccj/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/sdb/ccj/cache/huggingface/transformers
export TORCH_HOME=/mnt/sdb/ccj/cache/torch
export XDG_CACHE_HOME=/mnt/sdb/ccj/cache
export PIP_CACHE_DIR=/mnt/sdb/ccj/cache/pip
export TMPDIR=/mnt/sdb/ccj/tmp/green_bridge_v300_${EXECUTION_COMMIT}
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
```

Create all directories before Python starts.

The launcher must reject any of these resolving outside `/mnt/sdb`:

```text
runtime root
output root
HF cache
Torch cache
temporary directory
worker logs
coordinator logs
endpoint ledgers
Parquet staging
```

No substantial data may be written to `/tmp`, `/home`, `/root`, or the root filesystem.

## 21.2 GPU allocation

Prepare:

```text
physical GPU 4
visible as cuda:0
```

Future development, if separately authorized:

```text
physical GPUs 0–7
one deterministic worker per GPU
exact scientific endpoint batch size = 1
role-stratified deterministic assignment
```

No future phase is authorized by this document.

## 21.3 Environment mutation

The launcher may run:

```bash
python -m pip check
```

and exact version/hash validation.

It may not:

```text
pip install
pip upgrade
conda install
download package updates
change Torch build
change TransformerLens source
```

---

# 22. Exact commands

## 22.1 Create the v3 worktree under `/mnt/sdb`

Run from the existing repository:

```bash
set -euo pipefail

SOURCE_REPO="$(git rev-parse --show-toplevel)"

git -C "$SOURCE_REPO" switch codex/green-v200

test "$(
  git -C "$SOURCE_REPO" rev-parse HEAD
)" = \
"ef09fce529553d5a3d236852a288cde02b88418a"

test -z "$(
  git -C "$SOURCE_REPO" status \
    --porcelain=v1 \
    --untracked-files=all
)"

WORKTREE=/mnt/sdb/ccj/worktrees/idle_1_green_v300

test ! -e "$WORKTREE"

mkdir -p /mnt/sdb/ccj/worktrees

git -C "$SOURCE_REPO" worktree add \
  -b codex/green-v300 \
  "$WORKTREE" \
  ef09fce529553d5a3d236852a288cde02b88418a

cd "$WORKTREE"
```

## 22.2 Configure read-only postmortem storage

```bash
export GREEN_POSTMORTEM_ROOT=\
/mnt/sdb/ccj/iclr_1_postmortem/green_v21

export TMPDIR=\
/mnt/sdb/ccj/tmp/green_v21_postmortem

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p \
  "$GREEN_POSTMORTEM_ROOT" \
  "$TMPDIR" \
  analysis/GREEN_V21_POSTMORTEM_20260825
```

## 22.3 Implement and run postmortem analyses

```bash
python \
  analysis/GREEN_V21_POSTMORTEM_20260825/analyze_v200_postmortem.py \
  --archive \
    analysis/archive/green_v200_stop_20260825 \
  --v136-audit \
    analysis/GREEN_V136_TERMINAL_AUDIT_20260825 \
  --output \
    analysis/GREEN_V21_POSTMORTEM_20260825 \
  --scratch \
    "$GREEN_POSTMORTEM_ROOT" \
  --expected-postmortem-commit \
    ef09fce529553d5a3d236852a288cde02b88418a \
  --expected-execution-commit \
    e52e082296c33a10557636706e572147136fce34
```

After completion:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path(
    "analysis/GREEN_V21_POSTMORTEM_20260825"
)

integrity = json.loads(
    (root / "01_integrity_reconstruction.json")
    .read_text(encoding="utf-8")
)
transport = json.loads(
    (root / "04_exact_transport_identity.json")
    .read_text(encoding="utf-8")
)
joint = json.loads(
    (root / "05_exact_joint_composition.json")
    .read_text(encoding="utf-8")
)

assert integrity["all_hashes_passed"] is True
assert integrity["official_verdict_unchanged"] is True
assert integrity["confirmation_data_accessed"] is False

assert transport["route_failures"] == 0
assert transport["theorem_failures"] == 0

assert joint["route_failures"] == 0
assert joint["composition_failures"] == 0
PY
```

If any assertion fails:

```text
STOP AND RETURN TO GPT PRO
```

Do not implement or prepare v3.

## 22.4 Implement v3 and run tests

After the theorem postmortem passes:

```bash
python src/test_green_bridge_contract.py \
  2>&1 |
  tee \
  /mnt/sdb/ccj/logs/green_v300_existing_contract.log

python -m unittest \
  tests.test_green_bridge_v300_contract \
  2>&1 |
  tee \
  /mnt/sdb/ccj/logs/green_v300_new_contract.log
```

The combined authoritative harness must report:

```text
Ran 272 tests
OK
```

Then:

```bash
git diff --check

git status \
  --short \
  --branch
```

## 22.5 Commit the clean implementation

```bash
git add \
  analysis/GPTPRO_GREEN_V21_POSTMORTEM_DECISION_20260825.md \
  analysis/GREEN_V21_POSTMORTEM_20260825 \
  src/green_bridge_v300_spec.py \
  src/green_bridge_v300_dataset.py \
  src/green_bridge_v300_directions.py \
  src/green_bridge_v300_numerics.py \
  src/green_bridge_v300_transport.py \
  src/exp_green_bridge_v300.py \
  src/analyze_green_bridge_v300.py \
  src/green_bridge_v300_multigpu_worker.py \
  src/launch_green_bridge_v300.sh \
  tests/test_green_bridge_v300_contract.py

git diff --cached --check

git commit -m \
  "Add GREEN v3 curvature-detectability transport prepare protocol"

EXECUTION_COMMIT="$(git rev-parse HEAD)"

printf '%s\n' "$EXECUTION_COMMIT"

git merge-base --is-ancestor \
  ef09fce529553d5a3d236852a288cde02b88418a \
  "$EXECUTION_COMMIT"

test -z "$(
  git status --porcelain=v1 --untracked-files=all
)"
```

## 22.6 Launch prepare only

```bash
export EXECUTION_COMMIT="$(git rev-parse HEAD)"

export GREEN_RUNTIME_ROOT=\
/mnt/sdb/ccj/iclr_1_runs/green_bridge_v300_${EXECUTION_COMMIT}

export HF_HOME=/mnt/sdb/ccj/cache/huggingface
export TRANSFORMERS_CACHE=\
/mnt/sdb/ccj/cache/huggingface/transformers
export TORCH_HOME=/mnt/sdb/ccj/cache/torch
export XDG_CACHE_HOME=/mnt/sdb/ccj/cache
export PIP_CACHE_DIR=/mnt/sdb/ccj/cache/pip
export TMPDIR=\
/mnt/sdb/ccj/tmp/green_bridge_v300_${EXECUTION_COMMIT}
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p \
  "$GREEN_RUNTIME_ROOT" \
  "$TMPDIR" \
  /mnt/sdb/ccj/logs

test ! -e \
  "$GREEN_RUNTIME_ROOT/outputs/green_bridge_v300"

bash src/launch_green_bridge_v300.sh \
  4 \
  prepare \
  2>&1 |
  tee \
  /mnt/sdb/ccj/logs/green_bridge_v300_prepare_${EXECUTION_COMMIT}.log
```

No development or confirmation command is authorized.

---

# 23. Explicit STOP conditions

Codex must stop immediately on any of the following:

1. postmortem commit mismatch;
2. official execution commit mismatch;
3. any v2 archive hash mismatch;
4. v2 verdict or phase mismatch;
5. evidence that v2 confirmation was accessed;
6. any read-only postmortem write to the v2 root;
7. any exact AD direct-transport theorem failure;
8. any exact all-ten composition failure;
9. any active-model integrity failure;
10. any use of behavioral or PIE fields in theorem or radius calibration;
11. any v3 split mismatch;
12. any noun crossing development and confirmation;
13. any v2 development group entering v3;
14. any confirmation response being computed;
15. any held-out direction entering response identification;
16. any direction orthogonality failure;
17. any missing or nonfinite AD route;
18. no eligible global radius;
19. any per-item or per-gate radius selection;
20. any estimator selection using behavior;
21. any donor PCA path;
22. any pseudoinverse or ridge path;
23. any point estimator centered on AD or white-box values;
24. any test failure;
25. test count different from 272;
26. dirty worktree at launch;
27. formal root already existing;
28. retry or resume enabled;
29. phase `all` enabled;
30. development or confirmation phase accepted;
31. cache, runtime, logs, or temporary files resolving outside `/mnt/sdb`;
32. memory projection above 20 GiB per GPU;
33. total projected eight-GPU runtime above 24 hours;
34. any package installation during launch;
35. any fallback in precision, gate count, endpoint projection, radius, or batch size;
36. prepare artifact hash mismatch;
37. any development or confirmation artifact in the prepare root;
38. any attempt to reinterpret v2.0.0.

A failed prepare consumes the v3 one-shot. It may not be rerun under the same identity.

---

# 24. Required return bundle before any further execution

After a successful prepare, Codex must stop and return all of the following to GPT Pro:

```text
clean v3 implementation commit
git status proof
combined 272-test log
test-log SHA-256
postmortem_manifest.json
01_integrity_reconstruction.json
04_exact_transport_identity.json
05_exact_joint_composition.json
06_estimator_ladder_summary.json
08_aggregation_functionals.json
11_regime_bridge.json
v300_split.json and SHA-256
v300_record_plan.json and SHA-256
v300_direction_design.json and SHA-256
v300_radius_candidate_panel.json
v300_radius_calibration.json
selected rho*
v300_synthetic_theorem_suite.json
v300_transport_theorem_preflight.json
v300_joint_composition_preflight.json
v300_operation_counts.json
v300_throughput_preflight.json
v300_hardware_plan.json
prepare_result.json
manifest.json
sha256sums.txt verification log
proof that no development artifact exists
proof that no confirmation artifact exists
formal runtime root path
prepare log path and SHA-256
```

GPT Pro must issue a new binding decision before development.

---

# 25. Paper strategy and Oral assessment

## 25.1 Current Oral status

On the evidence available today, the empirical paper is not yet on an Oral trajectory. The exact theory and engineering rigor are strong, but v2.0.0 produced no admissible development cell and its permissive diagnostic performed substantially worse than the baseline.

That negative result must be reported honestly.

The project can return to an Oral trajectory only if v3 establishes a stronger and cleaner mechanism claim than the failed behavioral bridge.

## 25.2 Strongest defensible central claim

The strongest candidate claim is:

> **Gate-mediated residual transport is an exact basis-invariant rank-one operator. Matched-bypass mixed derivatives identify this operator when curvature makes the inverse detectable; exact LayerNorm geometry supplies a complete probe frame. Across held-out directions and unseen semantic groups, recoverability follows a curvature-controlled phase transition, and recoverable operators predict direct and joint causal transport better than unmatched or bypass-confounded alternatives.**

This is theory-first, mechanistic, falsifiable, and more interesting than “our score correlates with behavior.”

## 25.3 Theoretical novelty required

The paper should contain formal statements and proofs for:

1. matched-bypass factorization;
2. ambient operator basis invariance;
3. exact LayerNorm probe completeness;
4. path-minus-control transport identity;
5. curvature-controlled point-identification bound;
6. nonidentifiability when the curvature uncertainty set contains zero;
7. all-ten first-order additivity;
8. error propagation from finite response probes to held-out transport.

## 25.4 Minimum experimental evidence

Before an Oral claim is defensible, the paper needs:

1. zero exact transport-theorem contradictions;
2. zero exact all-ten composition contradictions;
3. outcome-blind global radius selection;
4. strong held-out transport prediction on untouched development and confirmation nouns;
5. clear improvement over every frozen baseline;
6. a curvature-detectability phase transition;
7. complement-direction null transport;
8. robust all-ten joint composition;
9. radius, distance, orientation, and noun robustness;
10. at least one preregistered external replication on another layer or model before submission.

The external replication is not authorized by this decision, but it is likely necessary for Oral-level breadth.

## 25.5 Critical figures

### Figure 1 — theorem geometry

Diagram of:

```text
residual perturbation
    → gate preactivation
    → gate response
    → downstream logits
```

with path, matched control, direct bypass, and rank-one operator.

### Figure 2 — curvature detectability transition

Plot:

\[
\log(\|C\|/\epsilon_C)
\]

against held-out transport error, with theoretical uncertainty envelope.

### Figure 3 — held-out transport

Predicted versus observed direct transport for:

```text
in-frame
mixed
pure-null
```

directions.

### Figure 4 — all-ten composition

Predicted versus independent joint target, with intervals.

### Figure 5 — v2 postmortem

Separate:

\[
|\mathbb E d|,
\quad
\mathbb E|d|,
\quad
\sqrt{\mathbb E[d^2]}
\]

and show how target functional and detectability produced the v2 STOP.

### Figure 6 — replication

Layer/model generalization of transport accuracy and detectability.

## 25.6 Critical tables

1. theorem and implementation audit;
2. response recoverability by gate/layer/system;
3. matched versus frozen baselines;
4. direct and joint transport errors;
5. radius and complement-direction ablations;
6. negative v2 result and immutable protocol status;
7. external replication.

## 25.7 Most likely reviewer objections

### “This is a post hoc pivot after failure.”

Answer:

- v2 is reported as a failure;
- v3 has a new identity;
- all success criteria are frozen before responses;
- development and confirmation nouns are untouched;
- radius calibration is behavior-blind;
- the primary target is an exact theorem consequence.

### “AD makes the result tautological.”

Answer:

- AD is an independent target and numerical audit;
- the scientific point estimator remains finite response-only matched-bypass identification;
- held-out directions never enter the estimator;
- baselines receive the same targets.

### “The signal is numerically tiny.”

Answer:

- the paper explicitly proves and measures detectability;
- null, unresolved, and recoverable regimes are separated;
- claims are restricted to numerically resolved units;
- uncertainty bounds and exact complement tests are reported.

### “The result is only one layer of GPT-2 small.”

Answer:

- an external layer/model replication is required before an Oral claim.

### “Behavior is no longer central.”

Answer:

- the main claim is causal operator identification, not another behavioral correlation;
- behavioral functionals remain secondary and are reported separately;
- the mechanistic theorem is stronger than the failed correlation story.

## 25.8 Abandonment or redirection triggers

GREEN must be abandoned or fundamentally redirected if any of the following occurs:

1. read-only exact direct transport fails;
2. read-only exact all-ten composition fails;
3. no global outcome-blind radius is eligible;
4. v3 development shows structural contradiction;
5. recoverable response operators fail held-out transport;
6. matched control does not outperform unmatched baselines;
7. complement null leakage exceeds its bound;
8. confirmation fails transport or composition;
9. the effect disappears under one external model/layer replication;
10. the detectability relationship is absent after numerical locality is established.

At that point, the appropriate decision would be:

```text
STOP_GREEN_AND_REDIRECT_THEORY
```

not another threshold amendment.

---

# 26. Immediate dependency-ordered actions

1. **Freeze the current state.** Verify branch `codex/green-v200`, postmortem commit `ef09fce...`, official execution commit `e52e082...`, archive hashes, and absence of confirmation artifacts.

2. **Create the `/mnt/sdb` v3 worktree.** Branch `codex/green-v300` from the exact postmortem commit. Do not modify v2 source or artifacts.

3. **Implement the twelve read-only postmortem analyses.** All scratch, logs, decoded Parquets, and temporary files must remain under `/mnt/sdb`.

4. **Run Analysis 01.**  
   **STOP AND RETURN TO GPT PRO** if any integrity count, hash, verdict, or phase state differs.

5. **Run Analyses 02 and 03.** Reconstruct the complete gate-certificate and uncertainty-source distributions.

6. **Run Analysis 04.**  
   **STOP AND RETURN TO GPT PRO** if any exact direct-transport theorem residual exceeds its certified bound.

7. **Run Analysis 05.**  
   **STOP AND RETURN TO GPT PRO** if any exact all-ten composition residual exceeds its certified bound.

8. **Run Analyses 06–12.** Produce the estimator ladder, null-mass audit, functional audit, role audit, SNR proof, regime bridge, and reporting audit. Mark every output read-only and non-protocol.

9. **Freeze the postmortem bundle.** Write `postmortem_manifest.json` with every source and output hash.

10. **Implement versioned v3 source files only after Steps 1–9 pass.** Do not alter the v2 runner or official artifacts.

11. **Implement the exact split, held-out directions, transport target, detectability classes, global radius calibration, baselines, future gates, and storage firewall in this decision.**

12. **Add all 52 v3 tests.** Preserve all 220 previous tests.

13. **Run exactly 272 tests.**  
    **STOP AND RETURN TO GPT PRO** if any test fails, is skipped, or the count differs.

14. **Commit the clean v3 implementation.** Record the commit and verify that it descends from `ef09fce...`.

15. **Launch v3 prepare exactly once on physical GPU 4.** No development or confirmation response may be computed.

16. **Verify the prepare root.** Check every artifact and hash, selected global radius, split, direction design, theorem suite, memory, runtime, and absence of development/confirmation files.

17. **STOP AND RETURN TO GPT PRO.** Provide the full return bundle in Section 24. Do not run development.

AUTHORIZE_NEW_VERSION_PREPARE_ONLY

