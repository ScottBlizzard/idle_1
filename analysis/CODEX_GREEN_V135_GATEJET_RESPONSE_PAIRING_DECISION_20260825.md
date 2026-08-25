# GREEN v1.3.5 GateJet response-pairing decision

## Terminal predecessor

The unique v1.3.4 development attempt terminated at `11_MULTIGPU_WORKER` before
any worker completed a record. All eight independently launched workers raised
the same exception:

```text
AttributeError: 'GateIdentification' object has no attribute 'G'
```

The terminal result and ledger are preserved under
`analysis/GREEN_V134_TERMINAL_ARCHIVE_20260825/`. The v1.3.4 identity is not
reopened or retried.

## Root cause

`identify_gate(jet)` returns a `GateIdentification` containing `A`, `P`, and
`D`; the observed response vector `G` remains on the corresponding `GateJet`.
The physical operator contraction is the frozen expression

```text
contrast.T @ G @ (A.T @ Q.T @ physical_v)
```

The v1.3.4 implementation incorrectly requested `estimate.G`. The correct
interface pairs each identification with the response from the same numerical
scale: `(rich_id, rich.G)`, `(full_id, full.G)`, and `(half_id, half.G)`.

## Authorized executable difference

v1.3.5 changes only protocol identity, predecessor archival metadata, and this
GateIdentification/GateJet interface pairing. It does not change datasets,
splits, prompts, interventions, radii, estimators, thresholds, stopping rules,
random seeds, exact batch-one execution, or deterministic eight-GPU sharding.
