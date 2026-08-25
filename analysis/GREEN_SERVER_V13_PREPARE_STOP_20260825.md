# GREEN v1.3 Server Prepare STOP — 2026-08-25

## Status

- Protocol: `green-bridge-v1.3`
- Protocol ID: `structural-envelope-matched-bypass-v1`
- Server repository: `/home/ccj/workspace_1/idle_1_green_bridge`
- Execution commit: `ed4b3b4c55ba2c7acfda1291b4814957ce90c845`
- Branch: `main`
- Repository status at launch: clean
- Physical GPU: 4 (`CUDA_VISIBLE_DEVICES=4`, process-visible `cuda:0`)
- Attempt index: 1
- Retry allowed: false
- Verdict: `STOP`
- First failed gate: `06_MANUAL_TAIL`
- Prepare timestamp: `2026-08-25T02:26:32Z`
- STOP timestamp: `2026-08-25T02:26:40Z`

The one-shot prepare was claimed exactly once. It must not be rerun without a
new binding GPTPro decision.

## Exact failure

The manual block-10-to-output tail was compared with the independent full-hook
implementation on five frozen conditions:

```text
conditions:
  [path, path, path, path, control]

absolute maximum errors:
  [
    7.62939453125e-05,
    6.103515625e-05,
    6.103515625e-05,
    7.62939453125e-05,
    6.103515625e-05
  ]

max_abs:
  7.62939453125e-05

frozen max_abs threshold:
  2.0e-05

derivative relative errors:
  [
    25.89502372509329,
    1.3051775251256144,
    0.00038560278974042017,
    0.00030073659422462503,
    0.00032303475823304003
  ]

max_derivative_relative:
  25.89502372509329

frozen derivative-relative threshold:
  1.0e-04
```

The terminal result was written before the process exited. No endpoint retry
was attempted.

## Gates passed before STOP

### Repository and one-run gates

- `main` branch: passed
- reviewed commit ancestor: passed
- clean worktree at launch: passed
- active output root absent before launch: passed
- attempt index exactly one: passed
- retry disabled: passed

### CPU contract

The frozen server environment passed all contract tests immediately before
prepare:

```text
Ran 124 tests in 3.074s
OK
```

### Gate-04 and no-op audit

- HF/TL audit: passed
- ordered prompt SHA-256:
  `619d21c10d4f30e6ce2597c3ba4df1de72cf0cb4f6cce322d82c2d3ec62803ce`
- HF/TL error enters `epsilon_y`: false
- same-TL no-op maximum absolute error: `0.0`
- `hook_audit.json` SHA-256:
  `49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99`

### Structural-envelope preflight

- common frame dimension: 4
- gate frame dimension: 5
- all-gate frame dimension: 14
- maximum orthogonality error: `5.773159728050814e-15`
- maximum raw-atom residual: `1.2494914337557025e-15`
- maximum analytic-gradient envelope residual: `2.162901736726871e-15`
- maximum formula/autograd absolute error: `2.7755575615628914e-17`
- maximum formula/autograd relative error: `2.545170364968639e-16`
- maximum shift-null metric: `1.2018516789897275e-17`
- repeated frames bitwise equal: true
- structural frame verdict: passed

## Phase firewall

- `development_started`: false
- `confirmation_started`: false
- `frozen_analysis.json`: absent
- development score artifacts: absent
- confirmation inputs and score artifacts: absent
- no development or confirmation scientific response was observed

## Frozen hashes

```text
result.json
  6f61c77b262eee821970dc19ff98f3baaf78e0aa9a65135bed343ed54ac7445d

run_ledger.json
  a4c21ea2bea3e42de13bd7789a17db849290556147250ba6f284b3aefa51172c

hook_audit.json
  49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99

structural_frame_preflight.json
  e0f65f22d29fb8db891094c407c25234f7f8f9f19738d4edaf3fb2ed5a19a05a

first_order_coefficients.npy
  d9305194f8d026ddde1a1d9084dd74409eae21e25b0b7600ca51f8887ff7b926

splits.json
  0490113fbfe66bcab1fba924896f832fac4668f2566402aa0107ed4fa43ed0ca

development_splits.json
  7fb05a1bf83d0083c622630694df09485dbaf18f4caaf6f5614200e0d8d2baf0

model_fingerprint.json
  fb9bd5a686d1bb09fa31c4cc308ff51f26c1d64075feb57d5a330db8fcaa6cb0

gate04_legacy_panel.json
  646d2ebcf1229645c83ebadea7f39d782e12152a8248dbd122f8c11e58c83df1

/tmp/green_bridge_v13_prepare.log
  28c2788da0477b5c95c4498d70a9a4183f2188419dbca6e5f4725b60d5dc8e52
```

## Decision required from GPTPro

The executor requests a binding determination on whether the observed
manual-tail mismatch is:

1. a terminal v1.3 implementation failure with no further run authorized; or
2. a specifically identifiable technical-equivalence defect for which GPTPro
   authorizes an exact code correction and a newly versioned, explicitly fresh
   attempt.

No threshold relaxation, endpoint retry, alternative precision, scientific
design change, development access, or confirmation access has been performed
or is proposed by the executor.
