# GREEN v4.1 resource fail-fast corrigendum — 2026-09-01

## Disposition

The original strict-serial calibration supervisor is stopped. Its smallest-budget
`ioi:layer0/affine` worker published a complete, hash-valid, outcome-free official
artifact before process termination. The artifact reports deterministic replay,
passing theorem checks, `4,302,881` graph nodes, and peak process-tree RSS of
`57,530,040,320` bytes.

The graph exceeds the frozen `2,000,000`-node ceiling. In addition, the frozen
`5/4` memory guardband produces a guarded value of `71,912,550,400` bytes, above
the frozen `68,719,476,736`-byte ceiling. In binding check order, the smallest
leaf budget therefore first fails `MAX_GRAPH_NODES_EXCEEDED`; guarded memory is a
second independent resource violation.

## Correction

The smallest leaf budget is now an explicit prefix-admission gate for structural
resource failures. If L4 exceeds maximum depth, maximum graph nodes, or guarded
memory on any validated outcome-free calibration record, the supervisor publishes
`STOP_RESOURCE_LOCK_INFEASIBLE` and schedules no larger candidate.

This is conservative and cannot improve a scientific result. Larger schedules use
the same TensorProgram and evaluator and retain a superset of live leaf state; an
L4 structural/memory violation cannot be repaired by adding leaves. Wall-clock
measurements are deliberately excluded from single-record fail-fast because they
can fluctuate between cold processes.

## Scientific invariants

This corrigendum does not change the model, checkpoint, tasks, prompts, sites,
directions, radii, precisions, branch semantics, threshold, P13 rule, confirmation
split, endpoint, baselines, seeds, graph identity, or output paths. It consumes no
scientific or endpoint outcome and cannot authorize confirmation.

## Current attempt

`GREEN-V410-SFC-JWTEC-20260830/A1` is resource-infeasible under its frozen backend
and resource lock. Its completed raw artifacts remain immutable diagnostic evidence.
No remaining job in the 240-item queue is scientifically authorized.

The trigger record carries artifact self-hash
`75cd802d9de12933458b210aca9149e4803e93e6853e47672a5f098369e4941a`.
The terminal STOP receipt carries self-hash
`c1a5051f8b6bd2527abcfd84fa07036269a0af194c1368534bf661a440515fe8`.

Further execution requires a resource-only successor with a hash-distinct backend
and a fresh synthetic calibration. It may adopt the unchanged scientific protocol
and sealed confirmation universe only after the ordinary seal audit passes.
