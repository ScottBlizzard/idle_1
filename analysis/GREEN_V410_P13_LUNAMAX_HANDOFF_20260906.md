# GREEN v4.1 P13 — LunaMax execution handoff

Date: 2026-09-06 (Asia/Shanghai)

## 1. Scope and authority

This handoff is execution-only. Do not redesign the estimand, sample, directions,
radii, branch order, thresholds, precision, leaf budget, task metric, graph
semantics, queue order, model revision, output roots, or receipt logic. Do not
push to GitHub. Do not use GPUs 0–3. Do not stop, modify, or inspect other users'
processes. All large artifacts must remain under `/mnt/sdb/ccj`.

The formal runtime is:

```text
source root: /mnt/sdb/ccj/green_v410_resource_successor_r4/source
formal run:  /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal
backend:     /mnt/sdb/ccj/green_v410_resource_successor_r4/runtime/lib/libgreen_v400_mpfr_backend.so
python:      /home/ccj/miniconda3/envs/green_bridge_20260805/bin/python
model:       /mnt/sdb/ccj/iclr_1_runs/green_v400_model_manifest_precision_smoke_20260829_v1/analysis/model_manifest.json
prepare:     /mnt/sdb/ccj/green_v410_resource_successor_r4/source/analysis/GREEN_V410_P13_QUALIFICATION_PREPARE_20260831
```

Only `p13_A1_formal` is eligible for aggregation. The directories `p13_A1`,
`p13_A1_hookabi_fixed`, and `p13_engineering_preflight` are retained diagnostic
roots and must never be read by a formal finalizer.

## 2. Already complete

- R4 resource selection: PASS, fixed leaf budget L4.
- T3 confirmation seal: PASS.
- Two immutable P13 queues: 108 sites and 864 directions per task.
- Runtime source bundle and compiled backend are hash-bound into both queues.
- Formal response capture: IOI 108/108 and Greater-Than 108/108, with one shard
  completion receipt per task.
- No materialization or certificate supervisor is currently running.
- No formal graph, direction certificate, P13 contraction record, task receipt,
  or global P13 result has been generated yet.

Frozen identities:

```text
source bundle: 847698face1c118232b5917fb3a7d4912dc3047a41b44210d1dc4a0418006f66
backend:       d859f4e107c81b103336df7776b812ba6a283dccb8cdd047344a3b534c3d98e2
IOI queue:     f7ee1456f8daf298d6119bf862a4434ddce406c6ae9db073306a072f787d87c5
GT queue:      eb0e1fbdc7d02d93e978fc94aa748c28c5f928b8e5f83d541a4d13eb55595491
```

## 3. Mandatory preflight

Run this before every stage and after any SSH interruption. It checks identities
and counts but does not print scientific values.

```bash
cd /mnt/sdb/ccj/green_v410_resource_successor_r4/source
PYTHONPATH="$PWD/src:$PWD:/mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages" \
/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python \
  analysis/green_v410_p13_status.py \
  --source-root "$PWD" \
  --run-root /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal \
  --compiled-backend /mnt/sdb/ccj/green_v410_resource_successor_r4/runtime/lib/libgreen_v400_mpfr_backend.so
```

Expected initial state:

```text
runtime_closure=PASS
ioi: captures=108, graphs=0, certificates=0
greater_than: captures=108, graphs=0, certificates=0
```

Any runtime-closure failure is a hard stop. Do not repair a frozen source file in
place and do not regenerate a queue without explicit scientific authorization.

## 4. T5a — graph materialization

Graph materialization is CPU and storage work; it does not need a GPU. A
one-site engineering measurement produced eight graphs in about one minute and
about 3.5 GiB. The complete two-task graph set is expected to occupy roughly
0.75 TiB. Verify at least 1 TiB free on `/mnt/sdb` before launch.

Run one supervisor per task. Sixteen total workers is a reasonable starting
point for the current server: eight workers per task, with 16 deterministic
shards per task. Concurrency is orchestration-only and may be reduced if other
users need CPU or disk bandwidth; do not change any scientific parameter.

```bash
cd /mnt/sdb/ccj/green_v410_resource_successor_r4/source
COMMON="--source-root $PWD --run-root /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal --prepare-root $PWD/analysis/GREEN_V410_P13_QUALIFICATION_PREPARE_20260831 --model-manifest /mnt/sdb/ccj/iclr_1_runs/green_v400_model_manifest_precision_smoke_20260829_v1/analysis/model_manifest.json --compiled-backend /mnt/sdb/ccj/green_v410_resource_successor_r4/runtime/lib/libgreen_v400_mpfr_backend.so --python /home/ccj/miniconda3/envs/green_bridge_20260805/bin/python"
mkdir -p /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/logs/supervisors

nohup /home/ccj/miniconda3/envs/green_bridge_20260805/bin/python \
  analysis/green_v410_p13_supervisor.py --stage materialize --task ioi \
  $COMMON --parallelism 8 --materialize-shards 16 \
  > /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/logs/supervisors/materialize_ioi.log 2>&1 < /dev/null &

nohup /home/ccj/miniconda3/envs/green_bridge_20260805/bin/python \
  analysis/green_v410_p13_supervisor.py --stage materialize --task greater_than \
  $COMMON --parallelism 8 --materialize-shards 16 \
  > /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/logs/supervisors/materialize_greater_than.log 2>&1 < /dev/null &
```

Do not start certification until status reports `graphs=864` for both tasks and
both `status/materialize_<task>_completion.json` receipts exist.

## 5. T5b — fixed-budget direction certification

Certification is CPU/MPFR work and does not need GPUs. The R4 calibration bound
permits at most 64 GiB and 23 h 50 min per direction; representative memory was
about 2.2–10.3 GiB depending on task and layer. Start with 32 workers per task
(64 total). This keeps the worst observed aggregate below approximately 0.7 TiB
on a 1.4 TiB host and uses at most one quarter of the 256 logical CPUs if each
worker remains single-threaded. Reduce concurrency if host pressure rises; never
alter L4, 384/512-bit precision, the three radii, or any queue record.

```bash
cd /mnt/sdb/ccj/green_v410_resource_successor_r4/source
COMMON="--source-root $PWD --run-root /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal --prepare-root $PWD/analysis/GREEN_V410_P13_QUALIFICATION_PREPARE_20260831 --model-manifest /mnt/sdb/ccj/iclr_1_runs/green_v400_model_manifest_precision_smoke_20260829_v1/analysis/model_manifest.json --compiled-backend /mnt/sdb/ccj/green_v410_resource_successor_r4/runtime/lib/libgreen_v400_mpfr_backend.so --python /home/ccj/miniconda3/envs/green_bridge_20260805/bin/python"

nohup /home/ccj/miniconda3/envs/green_bridge_20260805/bin/python \
  analysis/green_v410_p13_supervisor.py --stage certificate --task ioi \
  $COMMON --parallelism 32 \
  > /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/logs/supervisors/certificate_ioi.log 2>&1 < /dev/null &

nohup /home/ccj/miniconda3/envs/green_bridge_20260805/bin/python \
  analysis/green_v410_p13_supervisor.py --stage certificate --task greater_than \
  $COMMON --parallelism 32 \
  > /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/logs/supervisors/certificate_greater_than.log 2>&1 < /dev/null &
```

The supervisor holds an exclusive per-stage/per-task lock, skips only
identity-valid existing records, and uses atomic no-clobber worker outputs.
Certificate workers are never automatically relaunched: the frozen protocol
allows a process relaunch only with a byte-validated checkpoint and charged
counters, and the current worker intentionally implements no such checkpoint.
Therefore any certificate child failure—including OOM, process loss, wall
ceiling, graph/hash/schema/numerical/AD error—is a hard stop. Do not inspect
contraction values while the fleet is incomplete.

## 6. Monitoring and recovery

Use the status command in Section 3. Mutable supervisor state is under
`p13_A1_formal/status`; per-unit logs are under
`p13_A1_formal/logs/materialize` and `p13_A1_formal/logs/certificate`.

After SSH loss, first check whether a supervisor process still exists. Do not
launch a duplicate: the lock will reject it. If the supervisor itself is gone,
rerun the exact same command. Existing valid no-clobber artifacts are skipped;
missing work is resumed. Never delete a conflicting final artifact. A conflict
is a hard stop for audit.

Memory pressure detected before launching certificate children may be handled by
waiting or by starting the fleet at lower *concurrency only*. Once a certificate
child has failed, do not relaunch it without the protocol-required checkpoint.
Do not change worker arguments or scientific fields.
Never terminate unrelated processes, including RLS_CVPR2027 or work under other
project roots.

## 7. T5c — terminal aggregation

Run only after both tasks report exactly 864 valid certificate records and both
certificate completion receipts exist:

```bash
cd /mnt/sdb/ccj/green_v410_resource_successor_r4/source
PYTHONPATH="$PWD/src:$PWD:/mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages" \
/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python \
  analysis/green_v410_p13_finalize.py \
  --ioi-queue /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/queues/ioi_p13_queue.json \
  --gt-queue /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/queues/greater_than_p13_queue.json \
  --certificate-root /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/certificates \
  --output-root /mnt/sdb/ccj/green_v410_resource_successor_r4/p13_A1_formal/final
```

Interpretation is binding:

- `PASS`: both task medians are at most 0.20 and both nearest-rank p90 values are
  at most 0.50, with all 864 records per task (1,728 total) complete as specified.
- `STOP_P13_QUALIFICATION_FAILED`: stop. Do not open confirmation, tune anything,
  replace the reserve, or modify the protocol under this attempt.

The finalizer writes task site-ratio manifests, task receipts, and exactly one
two-task receipt. Only the two-task receipt is permitted to authorize the next
confirmation phase.

## 8. LunaMax operating instruction

Execute the handoff exactly as written. Make no scientific choices. Before each
stage, run the outcome-blind status check and verify no duplicate supervisor is
active. Report counts, runtime/resource failures, and completion, but do not
summarize or optimize based on interim P13 values. If a hard error occurs, stop
and return the exact command, log path, traceback tail, artifact counts, and
runtime-closure status. If both task certificate fleets finish, run the frozen
finalizer once and report the canonical two-task terminal state and receipt path.
