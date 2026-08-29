# GREEN v4.0 development execution status — 2026-08-29

This note records execution state and integrity evidence only. It does not
report or interpret any scientific endpoint outcome.

## Authorization boundary

- IOI activated plan SHA-256:
  `79d3d80b154266d634013c2b2c38d4c9a35be529c4c2560e7713e116fb5383fe`
- Greater-Than activated plan SHA-256:
  `25795204637313ee509cc3ecaf03b874be986a198d172042797be6b790e42067`
- Development authorization SHA-256:
  `8c7e38d2aa5ea753052f59226c23802b9af6f3871a0d7b5a74e35e006aed6cc5`
- Both activated plans have `development_authorized=true`,
  `confirmation_authorized=false`, and `plan_gate=DEVELOPMENT_ONLY_AUTHORIZED`.
- Independent recomputation validated each child against its sealed v8 parent,
  the authorization sidecar, and the parent file hash. All 25 bound source
  hashes matched the server validation checkout.

## Completed execution

The IOI development Grant route completed all nine planned cohort jobs on
physical GPUs 4–7. Its four completion receipts are:

- shard 0: `21dc1acc1565245e4adbdfc1bc140a89c1a4bfb837ece2278fb410dffc687db6`
- shard 1: `21d1a6193c1c3e6e120d9a753afb29257684a1b711de23ee24fa9ff9095e25c1`
- shard 2: `e0e0178baf5962bf37b186f575a66b9bfcf92b0397af51e1aa1bf295279e8a8b`
- shard 3: `eb5fc90654f62134fc5dedfc47da46b3875fe57305123d2c974af9b61494c6a7`

Every receipt records complete artifact coverage, typed artifact validation,
per-job model-session binding, unchanged model weights, no residual hooks or
gradients, and no endpoint authorization.

## Running execution

The IOI development prediction route started as four persistent-model shards:

- shard 0 maps to physical GPU 4;
- shard 1 maps to physical GPU 5;
- shard 2 maps to physical GPU 6;
- shard 3 maps to physical GPU 7.

Each shard has 288 jobs. The frozen numerical parameters are IG steps 65,
MS-HVP segments 8, and response chunk size 16. Each process sees only
`cuda:0`; cross-route model sharing is forbidden. Results, logs, temporary
files, caches, and supervisor state are all under `/mnt/sdb/ccj/iclr_1_runs`.

At 2026-08-29T18:13:56+08:00, the four shard artifact counts were
107/109/108/106, or 430/1152 total. All four workers remained alive, with
approximately 7.6 GiB device memory per GPU and 93–100% utilization.

## Automatic development-only continuation

Three fail-closed supervisors are active:

1. The batch supervisor waits for all four IOI prediction completion receipts,
   then runs GT Grant and GT prediction, one four-GPU fleet at a time.
2. The replay supervisor validates both protocols' batch artifacts into a
   hash-chained phase ledger, runs independent fresh-process A/B numerical
   replays, and issues typed per-layer receipts only if every replay is stable.
3. The endpoint supervisor waits for both replay assemblies, prepares per-job
   endpoint authorizations from the prediction, Grant, replay, universe, and
   ledger bindings, then executes development endpoints.

The supervisors never change numerical parameters, never overwrite an
artifact, never use GPUs 0–3, and never start confirmation. A missing receipt,
worker failure, replay instability, hash mismatch, or authorization mismatch
stops the chain while retaining completed artifacts for audited resume.

## Verification and local history

- Focused activation/batch tests: 32 passed locally.
- Batch-ledger tests: 15 passed locally and on the server runtime.
- Replay receipt assembly tests: 14 passed locally and on the server runtime.
- Endpoint authorization tests: 14 passed locally and on the server runtime.
- Most recent complete server regression before execution: 783 passed,
  51 skipped.
- Local commits (not pushed):
  - `6aab583` — activate audited GREEN development batches
  - `0e402be` — orchestrate development fleets without confirmation
  - `4c58dd6` — derive phase ledger from validated batches
  - `1f62947` — validate and schedule independent replay fleets
  - `2a1bc12` — prepare and schedule development endpoint fleets
