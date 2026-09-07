# GREEN v4.1 LunaMax master execution plan

## Mission

Continue `GREEN-V410-SFC-JWTEC-20260830/A1` as an execution and engineering
agent. The scientific design is closed. Follow the binding decision document
`analysis/GPTPRO_GREEN_V4_SUCCESSOR_CERTIFICATE_PROTOCOL_DECISION_20260830.md`.
Never optimize the design using real values. The target is a valid prospective
successor confirmation, not a favorable result at the cost of protocol drift.

## Current checkpoint

The only formal P13 root is:

```text
/mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal
```

Its frozen runtime closure passes. Formal response capture is complete for both
tasks (108/108 sites each). No graph materialization, MPFR direction
certification, P13 aggregation, confirmation prediction, confirmation endpoint,
or final analysis is running or complete.

Use `analysis/GREEN_V410_P13_LUNAMAX_HANDOFF_20260906.md` for exact P13 commands.
The implementation includes a `--plan-only` supervisor mode; use it before each
launch to verify the work-unit count without starting children.

## Gate dependency graph

```text
T0–T4 PASS
   |
   v
T5a P13 graphs (864 IOI + 864 GT)
   |
   v
T5b P13 certificates (864 IOI + 864 GT)
   |
   v
T5c frozen two-task P13 receipt
   | PASS only
   v
T6 confirmation non-endpoint authorization and adoption receipts
   |
   v
T7 prediction + Grant + GT clean validity + replay + graph + certificate
   |
   v
T8 immutable five-state site-status manifest
   |
   v
T9 endpoint-transition receipt
   |
   v
T10 physically separated sealed endpoint execution
   |
   v
T11 one-row-per-site merge and one-shot analyzer
   |
   v
T12 canonical scientific verdict
```

No arrow may be bypassed. A STOP receipt terminates A1.

## T5 execution plan

1. Run the outcome-blind status command and both materialization supervisors in
   plan-only mode. Expected planned work is 16 shards per task.
2. Run materialization. Require 864 identity-valid graphs per task and both
   immutable materialization completion receipts.
3. Run both certificate supervisors in plan-only mode. Expected work is 864
   directions per task on a fresh run.
4. Start 32 certificate children per task only if current host memory and CPU
   headroom support the calibrated worst case. Concurrency is not a scientific
   parameter, but any already-started child remains bound by its hard resource
   limits.
5. Monitor counts only. Do not inspect partial contraction values.
6. Require 864 valid direction records per task and both certificate completion
   receipts. Run the finalizer once.
7. If the two-task receipt is not PASS, stop A1. Do not build confirmation.

The representative R4 calibration projects hours per direction, not seconds.
With 64 total single-thread certificate workers, T5b should be treated as a
multi-day run; actual completion depends strongly on layer mix and shared-host
load. Do not promise a fixed wall-clock deadline.

## T6–T12 engineering readiness

The binding scientific rules and schemas exist, and the v4.0 parent execution
primitives exist, but the repository does not yet contain a complete v4.1
confirmation coordinator. Therefore, after a P13 PASS, first complete and test
the following pure engineering modules before opening any confirmation model
output:

1. `green_v410_confirmation_prepare.py`
   - validates the exact P13 PASS and T3 seal receipts;
   - creates byte-identical adoption receipts for parent confirmation site,
     direction, prediction, Grant, replay, and sealed endpoint job manifests;
   - builds the two certificate queues with exactly 1,152 IOI sites/9,216
     directions and 3,456 GT sites/up to 27,648 directions;
   - builds 384 GT prompt clean-validity jobs;
   - emits the non-endpoint authorization queue;
   - contains no endpoint payload or endpoint path.
2. `green_v410_confirmation_nonendpoint_supervisor.py`
   - wraps the frozen v4.0 prediction, Grant, and replay primitives;
   - produces v4.1 receipts and strict GT clean-validity receipts;
   - captures response centers, materializes full downstream-cone graphs, and
     invokes the same L4/384/512 certificate worker;
   - emits task-precondition INVALID site packets for clean-invalid GT prompts
     without direction work.
3. `green_v410_site_finalize.py`
   - accepts exactly eight complete direction certificates for every task-valid
     site and the exact global P13 PASS receipt;
   - invokes `green_v410_classifier.classify_site` without recomputing or
     inferring statuses;
   - emits immutable site packets and one status manifest over the entire parent
     population, keeping all five states in the denominator.
4. `green_v410_endpoint_transition.py`
   - validates T3, P13, all non-endpoint receipts, replays, site packets, and the
     status manifest;
   - emits the only receipt permitted to unlock endpoint material;
   - runs in a process that has no endpoint key or payload.
5. `green_v410_endpoint_supervisor.py`
   - runs only after the transition receipt;
   - receives no GREEN directions, interval widths, P13 values, or status values;
   - adopts and executes the sealed parent endpoint jobs in a physically separate
     environment and output root.
6. `green_v410_merge_analyze.py`
   - enforces exactly one row per immutable parent site and the exact columns in
     Section 12.10;
   - never recomputes status;
   - wraps the inherited analyzer once and writes the canonical final receipt.

Until these modules, their unit tests, endpoint-firewall tests, seeded-conflict
tests, and synthetic end-to-end test all pass, T6 must remain closed. Building
and testing them with synthetic or public prepare manifests is authorized;
opening confirmation model-derived values is not.

## Confirmation invariants

- IOI confirmation queue: exactly 1,152 sites and 9,216 directions.
- GT confirmation queue: exactly 3,456 sites; 384 clean-validity prompt jobs;
  task-valid sites use eight directions and clean-invalid sites short-circuit.
- Queue sort key: `(task, phase, prompt_row_id, layer, site_row_id)`.
- Fixed radii: `1, 1/2, 1/4`; fixed L4; official/audit precision 384/512.
- Branch order: `PAT_J, PAT_B, TAR_J, TAR_B`; no cross-branch common-subexpression
  sharing.
- Site classifier consumes P13 only as the global PASS receipt.
- Status domain is exactly CERTIFIED_POSITIVE, CERTIFIED_NEGATIVE, UNRESOLVED,
  INVALID, RESOURCE_INCONCLUSIVE. `CERTIFIED_NULL` is forbidden.
- GT clean validity is strict: correct probability mass must be greater than
  incorrect mass; equality is false.
- Endpoint material is absent from every pre-transition command, environment,
  queue, log, cache, temporary path, and receipt.
- Final analysis keeps warning, unresolved, resource, and method-invalid rows in
  the primary denominator as frozen.

## Hard-stop report format

On any hard failure, stop the affected phase and report:

```text
protocol and attempt
gate/stage/task
exact command
PID and exit code
first canonical failure code
traceback tail and log path
expected/observed artifact counts
runtime/source/backend/queue closure status
whether any final artifact was committed
whether any endpoint material was accessible
```

Do not propose changed thresholds, samples, directions, radii, budgets, status
mapping, or fallback universes in the same attempt.

## Server boundaries

- SSH: `ccj@10.10.217.244`.
- Use `/mnt/sdb`, never the root disk, for large work.
- GPUs 0–3 are forbidden for this project. P13 graph/certificate work is CPU
  work; do not reserve GPUs for it.
- GPUs 4–7 may be occupied by unrelated tasks. Never terminate or alter them.
- Never modify RLS_CVPR2027 or other project processes.
- Never push to GitHub unless the user explicitly asks.

## Completion definition

The handoff is complete only when either:

1. a canonical STOP receipt is produced at the first failed gate and all later
   gates remain unopened; or
2. T12 emits the canonical final scientific verdict after the one-shot analyzer.

P13 completion alone is not paper completion. It is the prerequisite that
decides whether the untouched confirmation may legally begin.
