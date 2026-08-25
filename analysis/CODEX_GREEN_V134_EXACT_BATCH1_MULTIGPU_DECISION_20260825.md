# GREEN v1.3.4 exact-batch-one multi-GPU execution decision

Date: 2026-08-25

v1.3.4 is a fresh one-shot engineering identity. It preserves every scientific
object and threshold from v1.3.3, but evaluates the active manual tail with the
same batch-one operation graph as the frozen full-hook reference.

```text
schema_version = green-bridge-v1.3.4
protocol_run_id = green-bridge-v1.3.4-one-shot
attempt_index = 1
retry_allowed = false
output_root = outputs/green_bridge_v134
manual_tail_batch_size = 1
worker_physical_gpus = [0,1,2,3,4,5,6,7]
```

No output recentering, cross-shape comparison, tolerance relaxation, theory
change, data change, or analysis change is allowed. Prepare must show bitwise
agreement between batch-one manual and independent full-hook endpoints for the
center and nonzero path/control panel, plus deterministic repeat behavior.

Development and confirmation records are sorted and deterministically assigned
round-robin across eight isolated worker processes. Each worker exposes exactly
one physical GPU, uses its own durable endpoint ledger and artifacts, and writes
a hashed completion manifest. The coordinator launches every worker once,
requires zero exit status and exact record coverage, then merges only committed
artifacts in pair-digest order. A failed worker is terminal; no shard is retried.

The scientific payload hash must remain
`60ca5e9e221064f288a1993ee3cbf42e99330bbf6f9008946a25556438cbc3d3`.

