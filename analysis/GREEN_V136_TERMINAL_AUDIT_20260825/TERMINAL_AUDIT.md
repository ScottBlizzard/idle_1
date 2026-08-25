# GREEN v1.3.6 terminal admissibility audit

## Frozen outcome

- Development verdict: `STOP_ORAL`; the outer runner stopped at `12_DEVELOPMENT_SURVIVAL`.
- Tensor records: 128 produced, 0 admissible.
- Energy records: 128 produced, 128 admissible.
- Cells: 16 evaluated, 0 survived.
- Confirmation remained closed.

## Localization

The execution pipeline completed. The terminal failure is localized to the mixed-tensor gate certification: every energy row passed, while no tensor row had both tar and pat mixed systems admissible.

### tar

- System-admissible items: 0/128.
- All-valid systems: 0/128.
- Gate labels across 1280 gate audits: `{'invalid': 566, 'active-identified': 708, 'certified-target-null': 6}`.
- Active-gate histogram: `{2: 1, 3: 4, 4: 21, 5: 36, 6: 41, 7: 18, 8: 5, 9: 2}`.
- Invalid-gate histogram: `{1: 3, 2: 4, 3: 20, 4: 41, 5: 34, 6: 22, 7: 3, 8: 1}`.

### pat

- System-admissible items: 2/128.
- All-valid systems: 2/128.
- Gate labels across 1280 gate audits: `{'invalid': 571, 'active-identified': 702, 'certified-target-null': 7}`.
- Active-gate histogram: `{2: 2, 3: 9, 4: 21, 5: 35, 6: 33, 7: 16, 8: 8, 9: 2, 10: 2}`.
- Invalid-gate histogram: `{0: 2, 1: 2, 2: 8, 3: 17, 4: 34, 5: 34, 6: 22, 7: 8, 8: 1}`.

## Persisted active-criterion failures among invalid gates

A gate can fail multiple criteria, so counts are not mutually exclusive.

- `whitebox_agreement`: 851
- `factorization_residual`: 823
- `curvature_snr`: 229
- `tensor_snr_unserialized_inferred`: 74
- `tensor_symmetric_change`: 62
- `gate_response_snr`: 30
- `richardson_change`: 11
- `tensor_cosine`: 4

`tensor_snr_unserialized_inferred` is inferred only when every serialized active criterion and the shift check pass; the P-norm itself was not written into the parquet audit.

The dominant pair is factorization residual plus white-box agreement (418 gates with exactly that pair; 635 gates with at least both). Their invalid-gate medians are near but above the frozen cutoffs: factorization is about 0.185–0.190 versus 0.15, and relative white-box error is about 0.065–0.072 versus 0.05. Exact distributions normalized against the thresholds are serialized in `terminal_admissibility_audit.json`.

## Non-confirmatory signal diagnostic

Although inadmissible rows cannot support the registered claim, the raw PIE baseline has item-level Spearman `0.613` and 16-cell mean Spearman `0.962` with the behavioral target. This is evidence that the run contains structured signal, but it is not a valid estimate of confirmatory performance.

## Interpretation boundary

This is a scientific/protocol STOP, not a worker crash or missing-data event. Any change to gate thresholds, null certification, completeness, minimum active gates, or tensor SNR would alter the frozen scientific protocol and therefore requires an explicit theory-level decision before another confirmatory run. Raw inadmissible correlations in the JSON are diagnostic only.
