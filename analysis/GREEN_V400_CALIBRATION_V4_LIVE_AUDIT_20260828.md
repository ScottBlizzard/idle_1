# GREEN v4 actual-shape budget calibration — live audit record

Date: 2026-08-28

## Scope

This record covers the outcome-blind, closed-synthetic actual-shape budget
calibration rooted at
`/mnt/sdb/ccj/iclr_1_runs/green_v400_actual_shape_budget_calibration_20260828_v4`.
It is not a real certificate, development result, confirmation result, or
scientific outcome. The frozen manifest explicitly forbids scientific-threshold
application and exposes only selector-safe resource artifacts to the budget
selector.

## Frozen identity

- repository commit: `1381040df9aa600088fa0faad0c19f3f978ec804`
- manifest semantic hash:
  `20d06140ebdd63793a442bfa6afc26fa686090ed91c6e58707073b050518eb7b`
- continuation report semantic hash:
  `33bcba7d90ddb6e5c02b24b4402e8fcafcddd268b5d50d6fdbeabd0b97e8a330`
- continuation external resource-report semantic hash:
  `16ffc2ca153982f9e066b31bf4d367f12d88af6a3ea1e00e1c0d29354f7c69c2`

The repository and all source/artifact hashes remain frozen for the lifetime of
this run. Later local hardening commits are intentionally not synchronized to
the server until the run terminates.

An offline Git bundle was used to create a separate validation clone under
`/mnt/sdb`; the active frozen repository was not modified. At local commit
`b8868ff`, the two directly affected test modules passed 33/33 tests and the
complete repository suite passed 634/634 tests in the server dependency
environment. Both validation commands hid all GPUs.

## Continuation-to-L32 gate

The mandatory continuation track at radius `2^-14` completed successfully.

- external process status: `COMPLETED`, exit code 0
- external elapsed time: `4542.947343621985` seconds
- sampled peak process-tree RSS: `265818112` bytes
- sampled peak process-tree swap: `0` bytes
- charged native calls: 65 at 384 bits, followed by 65 at 512 bits
- total charged native calls: 130
- stderr size: 0 bytes
- complete 384-then-512 history: true
- same priority path: true
- 512-bit recurrence uses official 384-bit intervals: false
- all independent nesting checks passed: true
- checkpoint final-leaf counts: 4, 8, 16, 32
- all endpoint and cell value/first/second nesting checks passed at every
  checkpoint
- report contains scientific outcome: false
- scientific threshold applied: false
- selector may read the numerics report: false

This closes the implementation-consistency gate that had previously been
invalidated by constructive intersection with the official intervals. The
512-bit path now replays the complete frozen official split history while
maintaining an independent recurrence, and containment is tested only after
the independent values exist.

## Standalone budget track

After accepting the continuation terminal, the frozen driver automatically
started the four fresh-process, all-17-radius jobs in the predeclared order:
`L4`, `L8`, `L16`, `L32`. Each budget executes all 384-bit work before any
512-bit work. The no-cache maximum charged-call counts are respectively 306,
578, 1122, and 2210.

The first `L4` job was active when this live record was created. No budget has
yet been selected. Selection remains restricted to the selector-safe resource
records and the predeclared largest-candidate-with-4/5-guardband rule.

## Shared-server allocation

This calibration is CPU-only and hides every GPU. Any later authorized GPU
work may expose only physical GPUs 4, 5, 6, and 7. Physical GPUs 0, 1, 2, and 3
are reserved for the collaborator and must not be used or managed by this
project.
