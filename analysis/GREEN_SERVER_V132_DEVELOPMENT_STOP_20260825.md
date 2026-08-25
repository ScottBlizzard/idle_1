# GREEN v1.3.2 development terminal record

Date: 2026-08-25

## Outcome

The unique v1.3.2 prepare passed. The unique development execution completed
all frozen raw endpoint batches and wrote the development parquet files, then
the analyzer raised `ValueError: insufficient surviving development cells`.
Confirmation was never opened or started. The v1.3.2 identity is terminal and
must not be rerun.

## Frozen observations

- Development cells: 16 total, 0 surviving.
- Every cell retained 8 admissible energy records and 0 admissible tensor
  records.
- Tensor records: 128 total, 0 admissible.
- Duplicate model noise: 96/96 errors exactly zero; `epsilon_y_dev=1e-7`.
- Every tensor record was rejected before gate classification because both the
  target and patched systems reported `active_gates=0`.
- Center errors were shape-quantized and far above the frozen center gate:
  target max-absolute range `[1.678466796875e-4, 3.35693359375e-4]`, patched
  range `[1.678466796875e-4, 4.119873046875e-4]`, versus threshold `2e-5`.

## Root cause

The scientific anchor is captured at batch size one. v1.3.2 evaluates the
manual tail in a fixed padded batch of 512. At zero intervention the manual
tail recomputes block-10 and downstream GEMMs at the new matrix shape. CUDA
selects shape-dependent GEMM kernels/accumulation orders, so the absolute
fixed-shape zero endpoint is not bitwise equal to the frozen batch-one anchor.

The v1.3.2 prepare gate proved only fixed-shape repeat, padding-content, and
wrapper/direct equivalence. It did not compare the fixed-shape zero endpoint
to the frozen anchor. This omission explains the complete tensor attrition.
It does not constitute evidence against the matched-bypass theory or the
frozen scientific hypothesis.

## Immutable hashes

- `dev_cells.json`: `1294a76d6d79c81f240c20c4257aa6b0fe76457d46b30cfc5d5699e27759ae1f`
- `dev_tensor_scores.parquet`: `9a81230ab4979c4460f9bd6d8ff59a48d41b813bd878b59d65725cece4c936d6`
- `dev_energy_targets.parquet`: `3faefcbd96503da9cb83b181336d9f2a914434affc561200e57a227329f00c44`
- `endpoint_ledger.jsonl`: `78defc4809b0e8dc260fc885c9cd92dd2e15055ba6f9cff5be3149b0b2dc8788`
- `run_ledger.json`: `e0eea0bdd595498bfccb8376e5ff45a6df07c37ade1c5ba0746dddebdc785df0`

The complete output tree is hard-link archived on the data disk at
`/mnt/sdb/ccj/iclr_1_runs/green_bridge_v132_terminal_archive_20260825`.
