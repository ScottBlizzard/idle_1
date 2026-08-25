# GREEN v1.3.1 server prepare report — terminal technical STOP

Date: 2026-08-25
Repository: `/mnt/sdb/ccj/iclr_1_runs/idle_1_green_bridge_v131`
Execution commit: `67bd92c72057db48642280dd28ef2fa9b03c0cac`
Physical device: RTX 4090 GPU 4, exposed as `cuda:0`
Protocol run: `green-bridge-v1.3.1-one-shot`, attempt 1, retry false

## Terminal result

The unique v1.3.1 prepare command terminated at:

```text
first_failed_gate = 06E_BATCH_SHAPE_EQUIVALENCE
manual batch = 512
comparison = batched manual output versus concatenated batch-one manual outputs
max_abs = 0.000213623046875
rms = 0.00008455902711485236
threshold = 0.00002
```

No development or confirmation response was accessed.

Frozen terminal hashes:

```text
result.json
e911860ea406e6b38d7dc475dffd500dde68044185c11e0bc7be605f899ebbbf

run_ledger.json
a65eb0aed2611996ecd7074512450efe1e598f392127c398d28d53bfe02bb47b
```

The terminal root was copied without mutation to:

```text
/mnt/sdb/ccj/iclr_1_runs/green_bridge_v131_terminal_archive_20260825
```

## Gates passed before the STOP

The v1.3 legacy discrepancy was reproduced exactly:

```python
[
    7.62939453125e-05,
    6.103515625e-05,
    6.103515625e-05,
    7.62939453125e-05,
    6.103515625e-05,
]
```

All five stage traces had no divergence through the corrected full-vocabulary
TransformerLens endpoint. The five original raw conditions and all twelve
signed derivative endpoints had manual/full maximum absolute error `0.0`.
All six true central finite-difference derivative audits had error `0.0`.
All five independent joint path-target endpoints had error `0.0`.

Therefore the v1.3.1 STOP is a batch-operation-graph audit failure, not a
manual-tail correction failure and not a scientific failure of the structural
matched-bypass hypothesis.

## Independent batch diagnostic

Artifact:

```text
analysis/GREEN_V131_BATCH_SHAPE_DIAGNOSTIC_20260825.json
666a20604fa4b123732bd68a15681fa7a16cafeef8edc2b61544fd911567d07d
```

Every cross-shape comparison for batch sizes 2 through 512 exceeded `2e-5`.
This establishes that the failure is not specific to batch 512. CUDA selects
shape-dependent GEMM implementations, so changing the leading matrix dimension
changes float32 accumulation.

At fixed batch shape 512:

- repeated execution was bitwise identical;
- changing all 511 padding rows left the audited row bitwise identical;
- peak allocated memory was 2,763,337,728 bytes;
- projected 1,609,824 tail evaluations took approximately 66 seconds.

# VERDICT — V1.3.1 STOP IS IMMUTABLE; ALL ENDPOINT AND DERIVATIVE REPAIRS PASSED; FAILURE IS CONFINED TO AN INVALID CROSS-BATCH-SHAPE NUMERICAL REQUIREMENT
