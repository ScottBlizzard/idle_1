# GREEN v4.1 resource successor R1 — 2026-09-01

## Scope and status

This is an outcome-blind resource/executor successor to the stopped
`GREEN-V410-SFC-JWTEC-20260830/A1` calibration. It does not read qualification,
confirmation, response, derivative-sign, P13, or endpoint material. The model,
checkpoint, synthetic fixtures, seeds, tasks, branch semantics, interval
arithmetic, precisions, radii, leaf candidates, theorem checks, and scientific
decision rules are unchanged.

R1 is not yet a production resource manifest. It must first pass exact-output
parity, 384/512-bit probes, deterministic replay, and the closed synthetic
calibration under a fresh output root.

## Corrected graph-resource accounting

The A1 worker serialized `dependent_scalar_outputs_total` as `graph_nodes`. That
quantity is cumulative work over the whole TensorProgram, not simultaneously
live graph storage. For the six frozen actual-shape bundles it is:

| Profile | Tensor-SSA nodes | Cumulative dependent scalar outputs | Root-only peak live dependent scalar outputs |
|---|---:|---:|---:|
| greater_than:layer0 | 547 | 1,660,641 | 38,402 |
| greater_than:layer4 | 355 | 1,107,681 | 38,402 |
| greater_than:layer8 | 163 | 554,721 | 38,402 |
| ioi:layer0 | 547 | 4,302,881 | 99,842 |
| ioi:layer4 | 355 | 2,865,185 | 99,842 |
| ioi:layer8 | 163 | 1,427,489 | 99,842 |

R1 applies the frozen `2,000,000` ceiling to the mechanically recomputed
`root_only_peak_live_dependent_scalar_outputs_v1` metric. It continues to record
the cumulative total separately and never discards it. The metric implements
topological SSA last-use accounting, retains the four branch roots and final
output, and takes the peak after allocating each output but before releasing its
last-consumed parents.

## Executor correction and invalidated early diagnostics

R1:

1. enables the compiled native-resident buffer path for v4.1 unpacked,
   hash-validated tensor closures;
2. releases owned intermediate buffers immediately after their final SSA use;
3. caches interval-independent rows by the closed program/node/precision/row
   identity without exporting MPFR state to Python;
4. retains cache-owned buffers across cells and releases them when the evaluator
   closes;
5. commits executor and v2 resource-config identities once at artifact
   boundaries; runtime progress tracing never serializes node values.

The compiled MPFR backend library remains byte-identical to A1. R1 changes
orchestration, representation, cache identity, and liveness only.

The first resident probes exposed two executor defects before any formal
successor launch:

1. the native static-row cache identity omitted the TensorProgram node semantic
   identity, permitting unrelated PAT/TAR and dynamic/zero-anchor nodes with the
   same kernel constants to collide; and
2. a native-pointer-to-Python-row memoization key survived native allocation
   reuse and could return a stale exported row.

Both paths are removed or closed in the working successor.  The corrected
resident cache now uses `green-v400-static-row-lineage-v1`: a recursively hashed
identity over the exact source tensor row, value-affecting operator attributes,
tensor constants, and parent-row lineages.  Dynamic rows have no cache identity.
This safely recognizes duplicated shared/zero-anchor rows while keeping PAT and
TAR distinct when their controlled-hook tensors differ.

On the actual `ioi:layer8` graph, 1,026 rowwise outputs are statically cacheable;
763 have unique value lineages and 263 are provably reusable.  Native 384/512-bit
small-graph tests pass exact five-root parity, cross-cell reuse, and cache-identity
rejection.  The final-candidate combined native executor and v4.1 resource suite
passes on the server (`43 passed`, zero failures).  On 2026-09-02 the
actual-shape conservative and lineage-v2 paths completed all 163
`ioi:layer8` nodes and agreed exactly on all five scalar roots.  The affine
resource fixture yields zero at those roots; agreement with the conservative
value-keyed path shows that this is not lineage-cache aliasing.

Every probe in the
table below predates completion of that correction and returned the exact hash
of five all-zero jets.  Those rows are retained only as invalidated engineering
diagnostics; they are **not** correctness, parity, or scientific evidence and
must not be cited as such.

## Outcome-free probe evidence

All probes use backend SHA-256
`d859f4e107c81b103336df7776b812ba6a283dccb8cdd047344a3b534c3d98e2`.
Times include one exact full-cone cell; RSS is process `ru_maxrss`.

| Profile | Precision | Invalidated mode | Cell time(s) | Max RSS KiB | Native exports (Jet count) | Invalid all-zero SHA-256 |
|---|---:|---|---:|---:|---:|---|
| ioi:layer8 | 80 | R1 before redundant-export removal | 254.968 | 1,719,840 | 324,106 | `7c47ca57564b8e2d6b7cd20705f0f243fa2a977c9ac05f8bf0c17aa84fd399c0` |
| ioi:layer8 | 80 | R1 corrected cold | 205.504 | 1,134,588 | 3,072 | `7c47ca57564b8e2d6b7cd20705f0f243fa2a977c9ac05f8bf0c17aa84fd399c0` |
| ioi:layer8 | 80 | R1 corrected, two-cell run | 274.663 cold; 143.336 hot | 1,165,324 | 3,072 per cell | identical for both cells |
| ioi:layer0 | 80 | R1 corrected cold | 784.763 | 4,902,056 | 3,072 | `7c47ca57564b8e2d6b7cd20705f0f243fa2a977c9ac05f8bf0c17aa84fd399c0` |
| ioi:layer8 | 384 | R1 corrected cold | 350.319 | 1,532,336 | 3,072 | `7c47ca57564b8e2d6b7cd20705f0f243fa2a977c9ac05f8bf0c17aa84fd399c0` |

The apparent speed of these invalid probes was partly caused by the incorrect
cross-node cache reuse.  Their timings therefore do not predict the corrected
executor.  A temporary post-fix diagnostic introduced a separate engineering
failure: it converted exact MPFR endpoints to numerator/denominator integers
after selected nodes.  One tiny endpoint has a huge binary exponent, so
formatting its denominator consumed CPU indefinitely even though node 51 had
already completed.  The diagnostic was removed.  Progress now records only
node ordinal, provenance, kernel, and elapsed time.

## Post-fix full-root admission evidence — 2026-09-02

All runs completed all 163 nodes with exit code zero.  No node value was
serialized by the progress logger.

| Path | Cell time (s) | Peak RSS (KiB) | Node 51 time (s) | Result |
|---|---:|---:|---:|---|
| lineage-v2 final candidate | 337.338 | 1,217,204 | 0.007 | exact five-root agreement |
| lineage-v2 without external cross-cell cache | 251.823 | 1,216,852 | normal | identical five roots |
| conservative exact-value-keyed cache | 364.859 | 1,309,400 | 0.047 | identical five roots |

The worst-shape `ioi:layer0` 384-bit probe then completed all 547 nodes in
1,536.642 seconds with peak RSS 8,095,040 KiB.  This is below the frozen
85,800-second direction wall and 68,719,476,736-byte memory ceilings.  The
corresponding 512-bit scaling probe completed in 1,650.894 seconds with peak
RSS 9,312,372 KiB, also below both ceilings.  The fresh R2 source closure passed
its supervisor/finalizer smoke suite, and the 240-job strict-serial formal queue
started under a fresh no-clobber raw root on 2026-09-02.

The two stalled traced successor processes and the impractically slow
non-resident Python/JSON reference were terminated.  Small-graph non-resident,
resident, and cache-reuse parity remains covered by the passing regression
suite.  The actual-shape conservative path is the independent full-root
engineering baseline for this admission gate.

## Remaining admission gates

- Relevant local regression suites must remain green.
- The strict-serial formal successor calibration must complete, finalize its
  candidate receipts, and publish one admitted resource manifest.
