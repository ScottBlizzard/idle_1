# GREEN Server Gate-08 Stop Report — 2026-08-05

## Status

- Binding decision implemented: `analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md`
- Execution commit: `5083774e03b99c9958312c6686cf3ead40c3c115`
- Decision base commit: `0c81e054ec2c4e79879652c19e5c126e3d0e6409`
- Hardware: physical NVIDIA RTX 4090 GPU 4, exposed as `cuda:0`
- Phase executed: `prepare` only
- Terminal verdict: `STOP`
- First failed gate: `08_BASIS_SPECTRUM`
- Terminal detail: `sigma4/sigma5=1.04, sigma4/sigma1=0.5501`
- Development responses observed: false
- Confirmation responses observed: false
- Confirmation remained locked: true

The amended cross-implementation preflight passed. The run subsequently stopped at the first previously unobserved donor-basis gate. No development or confirmation phase was started.

## Provenance and preservation

The old protocol-v1 Gate-04 stop was preserved at:

```text
outputs/green_bridge_gate04_stop_0c81e05_20260805
```

Its archived hashes are:

```text
058eb05837951043cc567d4ce5e3bdd487f82780c251b6b56636302fd57daf6f  result.json
cf0c48257892809f580c299ea03236d7e8ea3de417e8e8b7c975df104ed8f853  manifest.json
```

The server contract suite passed all 71 tests before the amended prepare run. The local complete regression set passed 104 tests: 71 GREEN contract tests, 19 mixed-path identification tests, seven interventional-response tests, and seven cross-fit validity tests.

## Amended Gate-04 result

The exact holdout panel was donor `pair_digest` ranks `16:32`, disjoint from the original ranks `0:16`. It contained 16 pairs and 32 prompts in clean-then-corrupt order. The ordered prompt-key hash was:

```text
619d21c10d4f30e6ce2597c3ba4df1de72cf0cb4f6cce322d82c2d3ec62803ce
```

The Hugging Face backend was explicitly `eager`, batch size was one, and cache use was disabled. The zero-tolerance converted-weight audit passed with zero mismatches.

| Metric | Observed maximum | Maximum threshold | Observed RMS | RMS threshold | Result |
|---|---:|---:|---:|---:|---|
| Raw 100-year logits | `1.068115234375e-4` | `3.0e-4` | `3.124910032944241e-5` | `7.5e-5` | pass |
| Centered 100-year logits | `4.9591064453125e-5` | `2.5e-4` | `1.149335430902504e-5` | `6.0e-5` | pass |
| Greater-Than task margin | `9.78552776834239e-6` | `2.0e-4` | `3.832678665827661e-6` | `5.0e-5` | pass |
| Block-10 final-position `resid_mid` | `6.4849853515625e-5` | `1.0e-4` | `3.66433163993219e-6` | `2.0e-5` | pass |
| Ten selected block-10 `hook_pre` coordinates | `3.337860107421875e-6` | `5.0e-4` | `1.010086597483797e-6` | `1.0e-4` | pass |
| Ten selected block-10 `hook_post` coordinates | `3.337860107421875e-6` | `5.0e-4` | `8.878869914757482e-7` | `1.0e-4` | pass |

The same-TransformerLens block-8 no-op patch audit also passed:

```text
n = 32
maximum absolute error = 0.0
binding threshold = 2.0e-5
```

The HF-versus-TransformerLens errors were recorded as reporting-only and did not enter `epsilon_y`.

## Gate-08 stop

After Gate 04 and the no-op gate passed, the runner evaluated all 1,024 donor pairs and formed the four-dimensional residual basis from the frozen 512 basis-donor chords. The frozen admissibility conditions were:

```text
sigma4 / sigma5 >= 1.10
sigma4 / sigma1 >= 1.0e-4
```

The observed values were:

```text
sigma4 / sigma5 = 1.04   # fail
sigma4 / sigma1 = 0.5501 # pass
```

Thus the failure is not rank collapse or a small fourth singular direction. The first four singular values remain comparable to the leading direction, but the fourth and fifth directions are not separated by the preregistered spectral-gap threshold. The frozen four-dimensional subspace is therefore not uniquely isolated under the current donor-basis contract.

The runner stopped before leave-one-noun basis stability, radius construction, manual-tail audit, development duplicate-noise audit, any development tensor/target response, or any confirmation access.

## Artifact audit

Current stopped-run hashes:

```text
7d52411b487f7e85f0dc539c760541d16bf5c9b756da75490edd8b9ad5ad7f90  outputs/green_bridge/result.json
baff192581726f4cae8f23418df5600ccb0fff549b0c81edff8c2c1f95d914df  outputs/green_bridge/manifest.json
49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99  outputs/green_bridge/hook_audit.json
845cb7746be048dacbcb6c841e45d29e3d51d7e7632074e08b63c92dea5d8fb8  logs/green_bridge_prepare_gate04_v2.log
```

Files present when stopped:

```text
development_splits.json
first_order_directions.npy
hook_audit.json
manifest.json
model_fingerprint.json
result.json
splits.json
```

The following did not exist:

```text
donor_basis.npz
radii.json
tail_audit.json
noise_audit_dev.json
dev_tensor_scores.parquet
dev_energy_targets.parquet
dev_result.json
frozen_analysis.json
confirm_tensor_scores.parquet
confirm_energy_targets.parquet
```

## Clean-worktree caveat

The execution commit itself was a clean descendant of `0c81e05`, and no tracked file was modified. However, the offline Git transport bundle remained as an untracked file inside the isolated worktree when `write_initial_manifest` ran, so:

```text
repository_dirty_at_launch = true
```

The bundle was moved out of the worktree immediately after the stop, and `git status --porcelain` is now empty. This transport-only condition cannot explain the spectral result, but any authorized future run must begin only after checking the full worktree, including untracked files, is clean. The current stopped run must retain this caveat permanently.

## Binding boundary

No rerun or amendment was attempted. The frozen bridge document states that a failed donor or numerical gate terminates the oral line without post hoc changes to the basis dimension, donors, thresholds, radii, gates, corruption, target, or analysis. The Gate-04 decision did not authorize a Gate-08 change.

GPT Pro is therefore required before any further model execution. The required decision is whether to:

1. terminate this four-dimensional bridge line;
2. authorize a scientifically independent, preregistered redesign that preserves the matched-bypass theorem and ICLR-oral-level claim while resolving the empirically non-isolated fourth/fifth donor directions; or
3. identify a conformance defect in the current basis construction that can be repaired without adapting to development or confirmation outcomes.

No scientific response beyond donor-only preflight information is available for making that decision.
