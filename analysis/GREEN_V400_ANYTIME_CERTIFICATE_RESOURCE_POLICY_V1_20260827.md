# GREEN v4 budget-monotone anytime certificate — theory and resource-policy candidate

Date: 2026-08-27  
Scope: outcome-blind design; no real response, P13 result, development label, or confirmation content was opened.

## 1. Decision

Do not replace the infeasible `max_depth=24,max_cells=262144` lock with an arbitrary smaller cap while retaining the old all-tolerances-or-fail interpretation. Replace the computational theorem with a **budget-monotone anytime certificate** whose soundness is independent of the available budget.

The scientific estimand, four-branch matched-bypass functional, exact dyadic control, 384/512-bit rounding audit, signed endpoint residual, P13 threshold, and `UNRESOLVED` treatment remain unchanged. A finite budget changes only whether the valid enclosure is narrow enough to decide the unchanged gate.

## 2. Objects

For radius `h`, let `P_b` be the complete dyadic leaf partition after at most `b` active leaves. Every leaf `J` has a certified interval

`Q(J) superseteq { Psi''(t) : t in J }`.

Let `w_h(J)` be the exact nonnegative rational signed-curvature kernel weight on one side of zero. The raw one-sided remainder enclosure is

`K_raw(P_b) = sum_{J in P_b} w_h(J) Q(J)`,

computed with outward MPFR arithmetic and the already implemented weight/summation accounting.

Let `B_+` and `B_-` be the independent direct endpoint/slope residual intervals. They are evaluated independently of the partition.

## 3. Monotone certificate recurrence

The initial signed enclosure is

`K_0 = K_raw(P_0)`.

After replacing one leaf by its two exact dyadic children, compute a new sound raw enclosure `K_raw(P_{b+1})` and define

`K_{b+1} = K_b intersect K_raw(P_{b+1})`.

The intersection must be nonempty. An empty intersection is an implementation failure, never a scientific result.

The endpoint remainder enclosure at budget `b` is

`R_{b,+} = B_+ intersect K_{b,+}`,

`R_{b,-} = B_- intersect K_{b,-}`.

The witness interval is then computed by the unchanged signed-secant identity from `R_{b,+}` and `R_{b,-}`.

This explicit intersection recurrence is important: ordinary interval reevaluation is sound but finite-precision summation can prevent syntactic nesting. Intersecting two independently sound enclosures restores a machine-checkable monotonicity invariant without selecting a favorable empirical value.

## 4. Theorem candidate

### Theorem 1 — finite-budget soundness

Assume:

1. every primitive interval-jet operation outwardly encloses its real value and first two derivatives;
2. every partition is a complete disjoint cover split exactly at zero;
3. curvature weights are nonnegative exact rationals and are evaluated outwardly;
4. direct endpoint/slope and partitioned-curvature enclosures are independently sound;
5. graph reductions preserve exact algebraic identity;
6. 512-bit audit quantities nest inside their 384-bit official counterparts before a result is reported.

Then for every finite leaf budget `b >= 2`, the reported official interval contains the exact frozen Joint Witness quantity. Therefore a success declared because the full official interval satisfies the unchanged P13 predicate is a valid success for every budget. Resource exhaustion cannot create a false positive.

### Proof sketch

Primitive soundness and composition give `Q(J)` for each leaf. Nonnegative exact integration weights preserve inclusion, and outward summation gives a sound enclosure of the integral remainder on the complete partition. The intersection of two enclosures of the same real quantity remains an enclosure when nonempty. The direct endpoint/slope residual therefore intersects soundly with signed curvature. The central-secant algebra is exact, with outward interval arithmetic preserving inclusion. Finally, the 512-inside-384 audit checks the implementation of the official enclosure without replacing it. No step depends on the leaf budget except enclosure width.

### Theorem 2 — budget monotonicity

Under Theorem 1, define each refined certificate by intersection with the preceding certificate. Then

`I_{b+1} subseteq I_b`

for every completed refinement. Consequently:

- once an unchanged interval predicate is certified, additional budget cannot revoke it;
- an unresolved row may become resolved under more budget;
- a resolved row cannot become unresolved merely because more cells were evaluated;
- the budget-resolution curve is an algorithmic completeness curve, not a changing estimand.

The same recurrence is applied separately to 384 and 512 bits. Cross-precision nesting remains mandatory at every reportable checkpoint.

## 5. Deterministic resource semantics

The binding resource unit should be **completed native cell evaluations**, not wall time. Wall time depends on contention and is retained only as an operational safety observation.

For each row/radius/precision:

1. evaluate the two initial half-cells;
2. retain the existing exact priority `w_h(J) * width(Q(J))`;
3. break ties by exact lower endpoint and then depth;
4. split one selected leaf into two children;
5. count both completed child evaluations;
6. update the monotone curvature and witness intersections;
7. continue until the predeclared leaf/evaluation budget is reached;
8. evaluate the 512-bit audit on the exact final official partition;
9. report `CERTIFIED` only if every binding predicate and precision-nesting check passes;
10. otherwise report `UNRESOLVED_RESOURCE`, never success by midpoint or rounding.

Mathematically, the absolute/relative width tolerances are sufficient sharpness conditions rather than requirements for enclosure soundness. Operationally, however, the current v4 primary outcome treats them as frozen prerequisites. The conservative integration therefore keeps them mandatory at the official budget `B*`: checkpoints below `B*` prove soundness and resumability but are provisional and cannot upgrade the primary result. Making the tolerances optional for official acceptance is a stronger protocol candidate that must be frozen before outcomes because it changes operational completeness and therefore can change the primary `Y` even though it does not change the estimand.

To prevent outcome-dependent compute allocation, every row in the same preregistered static complexity stratum receives the same maximum evaluation budget. Scientific threshold status is applied only after the final predeclared checkpoint. A real row may not receive more official work because its current witness, sign, P13 status, or interval width looks promising.

## 6. Semantics-preserving acceleration

The following optimizations do not change the partition mathematics or certificate result:

1. evaluate the two initial half-cells concurrently;
2. evaluate the two children of one already selected parent concurrently, then insert them in deterministic endpoint order;
3. evaluate all 512-bit audit cells concurrently only after the official partition is frozen;
4. evaluate the fixed `{-h,0,+h}` endpoint points concurrently within one precision;
5. run independent rows concurrently in isolated processes;
6. persist 384/512 native contexts across all radii for one row;
7. memoize cell results by `(row identity, precision, exact lower, exact upper, backend identity)`;
8. reuse only exact matching cells and the center endpoint across nested dyadic radii, attaching the current depth and recomputing priority for the current `h`;
9. serialize the priority queue, current partition, monotone intersections, and accounting after every completed split for crash-safe resumption;
10. use content hashes, not process-local object identity, for every cache key.

The following are not silently authorized as equivalent optimizations:

- splitting the top `k` leaves as a batch rather than the single frozen priority order;
- choosing budgets from observed response magnitude or P13 status;
- pruning cells with empirical samples or gradients;
- omitting 512-bit audit because 384-bit appears decisive;
- changing precision, tolerances, radii, hooks, contrast, or scientific thresholds;
- treating a wall-clock timeout as a certificate success.

Splitting the top `k` queued parents is not schedule-equivalent: children of the first split may outrank the previously second parent. Safe parallelism is therefore restricted to siblings, fixed endpoints, frozen-partition audit cells, and independent rows.

## 7. Feasible lock calibration

The measured actual-shape bounded run used six 384-bit native evaluations and took 134.2749 s including context startup/teardown. Its process-tree peak sampled RSS was 180,832 KiB. These are observations, not upper bounds.

A production cap must be selected before real outcomes by:

1. running the theorem fixtures and closed synthetic actual-shape stores at candidate leaf budgets `{4,8,16,32}`;
2. measuring monotone width reduction, exact operation counts, 384/512 nesting, wall time, and process-tree RSS;
3. choosing the largest uniform budget that fits the declared machine-hour and memory envelope with a preregistered engineering guardband;
4. freezing the cap, concurrency, retry, crash-resume, and invalid/unresolved rules by hash;
5. never increasing the cap after inspecting real resolution or P13 counts.

The existing `262144`-leaf value is retained only as historical provenance; it is not computationally credible at the observed per-cell cost and must not be represented as an executable production plan.

Changing `B*`, the split order, the precision ladder, or the tightening rule does not change the real-valued Joint Witness, but it can change which rows resolve and therefore changes the operational completeness of the Boundary Transition outcome. Such choices must be frozen once, outcome-blind, before official real-row execution. After that freeze, larger-budget continuations are audit-only and cannot upgrade official `Y`.

### 7.1 Exact work formula

With `R` radii and `L_r` final official leaves at radius `r`, the current non-deduplicated algorithm requires

`N_384 = sum_r (2 L_r + 1)`

and

`N_512 = sum_r (L_r + 3)`.

Here official work is `2L_r-2` adaptive cells plus three endpoint/center passes, while audit work is `L_r` frozen-partition cells plus three endpoint/center passes. Thus

`N_total = 3 sum_r L_r + 4R`.

For the 17 frozen radii, even the minimal `L_r=2` needs 170 complete native passes. Any purported whole-row cap below 170 passes is internally inconsistent unless exact-domain reuse is implemented and separately accounted.

### 7.2 Provisional calibration target, not yet a final lock

Define one formal TensorProgram evaluation pass as one admitted 81-node native dispatch for one exact domain at one precision. Charge the pass before execution; failure, timeout, or process death does not refund it. Also record the frozen arithmetic-taxonomy count of 352,275,450 directed primitives per pass, while explicitly not presenting that arithmetic count as total machine work.

A numerics audit proposed a calibration candidate `L_r <= 14` for every radius. Before memoization this implies `N_384=493`, `N_512=289`, and 782 total passes for 17 radii. Using deliberately conservative provisional accounting tokens of 90 seconds per 384-bit pass and 100 seconds per 512-bit pass yields 73,270 token-seconds. This is suitable only as a target for additional outcome-blind calibration; the token weights came from too few actual-shape observations to be called a production bound.

The final lock requires at least 30 cold-process samples per precision across endpoint, center, positive/negative, and deep dyadic domains; a complete 17-radius dry orchestration; cgroup OOM and timeout fault injection; and a frozen machine/concurrency manifest. A Linux `/proc` sampler remains observational. A hard memory policy requires cgroup-v2 `memory.max` plus `memory.events`, and a hard time policy requires an external monotonic supervisor. Partial child artifacts must never be publishable as success.

### 7.3 Phase-major failure semantics

The safest execution order is:

1. complete every 384-bit official partition and endpoint computation in frozen radius order without inspecting sign or P13;
2. if any official radius hits depth, leaf, pass, memory, or time policy, publish only `RESOURCE_INCONCLUSIVE` and do not start 512-bit work;
3. after all official partitions freeze, replay their exact cells/endpoints at 512 bits, permitting parallel evaluation but canonical commit order;
4. require every nesting and same-quantity intersection check before emitting `INTERVAL_COMPUTED`;
5. classify against the scientific gate only in the subsequent frozen decision stage.

This phase-major order gives a global official-precision short circuit and avoids spending audit work on a row that can no longer produce a complete official artifact. It changes orchestration rather than the mathematical interval, but bit-identical artifact parity against radius-major execution must be tested before adoption.

## 8. Paper-level contribution candidate

The stronger theoretical statement is not merely “we used interval arithmetic.” It is:

> A relational Transformer causal certificate can be made budget-monotone and anytime: every finite checkpoint is sound, refinement can only tighten the certified causal claim, and the resource-resolution frontier measures deterministic mechanistic detectability without changing the estimand.

This preserves the Oral-level main line and may strengthen it by formally separating three notions that ordinary activation-patching evidence conflates:

1. the causal effect is absent;
2. the effect exists but the current proof enclosure is unresolved;
3. the effect is certified at a declared computational budget.

Novelty against current certified-robustness, validated-numerics, and mechanistic-interpretability literature must be searched and audited separately before this is stated as a new theorem contribution.

## 9. Implementation gates

Before any real row is opened:

1. add a `MonotoneAnytimeCertificateState` schema with immutable cell/cache identities;
2. implement nonempty intersection recurrence for both signed curvature sides and the final witness;
3. add exact completed-evaluation accounting and crash-safe checkpoints;
4. prove deterministic replay gives byte-identical artifacts;
5. add adversarial tests where raw interval widths fail to nest but monotone intersections remain sound;
6. add tests that larger budgets cannot widen any reportable interval;
7. add tests that resource exhaustion never returns success without a full valid interval predicate;
8. execute the `{4,8,16,32}` synthetic calibration only;
9. freeze the selected production resource manifest;
10. retain real certificate/development/confirmation access as disabled until all gates pass.
