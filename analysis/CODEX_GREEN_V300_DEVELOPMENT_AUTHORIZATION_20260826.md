# GREEN v3.0.0 Development Authorization — 2026-08-26

## Decision

Authorize the single frozen GREEN v3.0.0 **development** phase.

```text
authorization_id = CODEX-GREEN-V300-DEVELOPMENT-v1-20260826
authorized_phase = development
confirmation_authorized = false
```

This authorization follows the formal `PREPARE_PASS` and the user's explicit
instruction to continue after reviewing the numerical correction.

## Scientific judgment

The isolated `float64_response_only` finite evaluator is accepted as a
numerical-stability correction. It evaluates the same frozen, float32-quantized
parameters with higher arithmetic precision and does not change the model,
estimand, split, radius candidates, selected radius, thresholds, or response
estimator information set. The point estimator consumes finite function values
only. AD targets and audits run on a second isolated model copy.

The original float32 diagnostic is retained. The correction was frozen before
any v3 development or confirmation response was opened, and all eligible
radii—not a selected favorable radius—passed after the correction.

## Binding execution scope

- Read exactly the 80 prepared development transport records and 80 prepared
  development joint records.
- Use physical GPUs 0–7, one deterministic worker per GPU.
- Use the formal globally selected radius and frozen `epsilon_y`.
- Preserve the finite-response/AD-copy separation.
- Write all substantial runtime data under `/mnt/sdb`.
- Do not read, materialize, schedule, or score any confirmation record.
- Do not authorize confirmation automatically, even if development returns
  `OPEN_CONFIRMATION`.
- Do not retry, resume, change thresholds, reselect the radius, or introduce a
  fallback after development starts.

## Required terminal artifacts

```text
dev_transport_scores.parquet
dev_joint_targets.parquet
dev_cells.json
dev_result.json
frozen_analysis.json
run_ledger.json
sha256sums.txt
```

The terminal development verdict is one of `OPEN_CONFIRMATION`, `POSTER_ONLY`,
or `STOP_ORAL`. Confirmation remains sealed under every development verdict.
