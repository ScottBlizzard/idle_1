# GREEN v2.0.0 development terminal report — 2026-08-25

## Executive decision

The corrected `green-bridge-v2.0.0-one-shot` completed prepare and development exactly once and terminated with the immutable official verdict:

```text
STOP_ORAL
first_failed_gate = 12_DEVELOPMENT_SURVIVAL
n_surviving_cells = 0
n_conditioned_cells = 0
n_snr_cells = 0
```

Confirmation did not open and must not be started. This is now a scientific-design decision point for GPT Pro, not an engineering retry point.

## Identity and provenance

- Binding specification: `analysis/GPTPRO_GREEN_V200_CORRIGENDUM_DECISION_20260825.md`
- Binding source SHA-256: `44e577ea72cde985de4e06bc51d90f1c85e2dc315a9de77299862d40df8069c0`
- Official execution commit: `e52e082296c33a10557636706e572147136fce34`
- Official server root: `/mnt/sdb/ccj/iclr_1_runs/idle_1_green_bridge_v200_f99626f/outputs/green_bridge_v200`
- Prepare claim: `2026-08-25T12:41:01Z`
- Development claim: `2026-08-25T12:48:56Z`
- Corrected split SHA-256: `0873915c966bef8f54b83d4151a9d7c75b577da5dfc17ee093b9f5c58a9590f7`
- Runtime: eight RTX 4090 GPUs, frozen `torch 2.7.1+cu126`, all run/cache/log paths under `/mnt/sdb`

The first launcher attempt rejected an inexact Torch version string before Python prepare and before creation of the formal root. It did not claim or consume the one-shot. Commit `e52e082...` made the launcher validate the exact CUDA build; the successful prepare below is the first and only official claim.

## Prepare result

Prepare passed all required preflight checks:

- exact test count: 220/220 passed, zero skips;
- corrected split and predecessor-object checks passed;
- 40/40 AD strata completed with zero misses, route failures, endpoint failures, or structural failures;
- three-scale numerical diagnostic passed;
- actual eight-record throughput preflight passed;
- 160 AD gate-system certificates and 320 GateJet routes executed;
- projected total runtime: 1,367.95 seconds;
- peak allocation: 7.4576 GiB;
- no precision, radius, gate-reduction, or projection fallback;
- active scientific model remained byte-identical, with float32 scientific parameters and an isolated float64 AD clone.

After prepare, `result.json` was absent and development/confirmation were still unopened, as required.

## Official development result

All eight workers completed without failure and produced exactly 64 tensor records and 64 energy records. Every worker reported an unchanged active scientific model. The merge hashes match the archived Parquet files.

The eight development cells each had eight admissible energy observations and zero set-admissible tensor observations. Therefore all eight cells failed survival before baseline selection or performance analysis:

| Quantity | Result |
|---|---:|
| Development cells | 8 |
| Cells with `n_energy = 8` | 8 |
| Cells with `n_tensor > 0` | 0 |
| Surviving cells | 0 |
| Conditioned cells | 0 |
| Set-SNR cells | 0 |
| Tensor records set-admissible | 0/64 |
| Tensor records point-complete | 55/64 |
| Energy records admissible | 64/64 |

The official result consequently contains no baseline calibration or mixed-predictor RMSE. Those fields are correctly `null`; they were never reached by the binding decision procedure.

## Read-only root-cause audit

This audit only aggregates already-materialized development certificates. It does not mutate artifacts, change any classifier, reopen the run, or produce another official verdict.

Across 64 tensor records, two systems, and ten gates (1,280 gate-system classifications):

| Certificate class / failure | Count |
|---|---:|
| `active-identified` | 7 |
| `certified-target-null` | 1,262 |
| `unresolved-bounded` | 11 |
| numerical invalid | 0 |
| structural contradiction | 0 |
| AD route failure | 0 |
| AD theorem failure | 0 |
| white-box coordinate failure | 0 |

Among the 128 system-level records, 121 had zero active gates and seven had one active gate. None reached the binding requirement of at least three active gates, so no tensor record could become set-admissible.

For inverse points that did not become active, the observed active-condition failure counts were dominated by curvature SNR (1,163), followed by response SNR (22), with zero curvature/response materiality failures. Across the 1,273 classifications that expose a finite `null_bound` (including certified-null and unresolved-bounded cases), those bounds ranged from `1.483457e-08` to `0.006831456`, with median `0.0001819194`.

This rules out the tempting engineering explanations: the terminal result is not caused by a broken AD route, a failed enclosure theorem, structural contradictions, invalid numerics, missing workers, or mutation of the active model. Under the corrected proof-derived bounds, almost all gate-system effects are certified null and the rest are too sparse to satisfy set identification.

## Deliberately permissive counterfactual diagnostic

To determine whether the failure was only the `>=3 active gates` rule, a strictly in-memory diagnostic temporarily treated every point-complete tensor interval as admissible. This is not part of v2.0.0 and cannot alter its verdict.

Under that deliberately permissive bypass:

- all 8/8 development cells would survive and condition;
- every cell's interval SNR is exactly 1, so 0/8 meet the binding SNR threshold of 3;
- mixed midpoint RMSE is `0.004746674288442684`;
- mixed worst-case RMSE is `0.010643813119989697`;
- best baseline LOOCV RMSE is `0.0008111058675662651`;
- robust relative gain is `-12.122594158918629`.

Thus simply weakening the active-gate count would not rescue the preregistered development claim. The certified intervals remain too wide relative to their cell targets and the interval predictor is substantially worse than the development baseline in this diagnostic.

## What is and is not concluded

The official conclusion is narrow but serious: the v2.0.0 operationalization of the matched-bypass theory does not survive its corrected development split. The result does not by itself identify which of the following explanations is correct:

1. the matched-bypass theory fails on this regime;
2. the theorem is sound, but the contribution target or active-set gate is mismatched to the scientific signal;
3. the interval composition is proof-valid but too conservative for the current aggregation scale;
4. the target scale or cell aggregation discards the signal seen in earlier experiments;
5. the earlier strong results measured a related but materially different phenomenon.

Those alternatives require a new, explicitly versioned scientific protocol. They cannot be resolved by threshold tuning on this observed development set.

## Binding non-actions

- Do not launch confirmation.
- Do not retry or relabel the v2.0.0 one-shot.
- Do not modify the formal server root.
- Do not lower thresholds, reduce the active-gate count, or redefine cells using these development outcomes.
- Do not present the permissive counterfactual as a protocol result.

## Evidence map

The selected terminal evidence is archived in `analysis/archive/green_v200_stop_20260825/`. All 28 copied official artifacts were verified byte-for-byte against the official `sha256sums.txt`. The archive includes the terminal and phase results, manifest and ledger, split, preflights, AD audits, model-integrity audit, cell decision, merged development tables, noise/radius/structural audits, and operation counts.

Machine-readable postmortem aggregates and their interpretation boundary are in `analysis/GREEN_V200_DEVELOPMENT_TERMINAL_DIAGNOSTIC_20260825.json`.
