# GREEN v3.0.0 Development Execution Incident — 2026-08-26

## Incident

The first development launcher invocation used commit:

```text
2ef281794d08933ec686e624f2b01252b3b8ace1
```

All eight transport workers completed their numerical loops, then stopped
before writing any Parquet, JSON result, aggregate, threshold verdict, or
scientific metric to disk. Every worker raised the identical engineering
exception:

```text
DEVELOPMENT_ACTIVE_MODEL_INTEGRITY_FAILURE
```

## Root cause

The worker read `finite_tail.active_model_unchanged` and
`ad_tail.active_model_unchanged` *inside* their context-manager block. These
fields are populated by the context managers' `__exit__` methods and therefore
were still `None`. The guard interpreted `None` as failure even though the
before/after active-model hashes had not changed.

The correction moves the two reads to immediately after context-manager exit.
It does not alter the model, data, record assignment, finite estimator, AD
target, selected radius, uncertainty construction, thresholds, baselines, or
analysis.

## Evidence boundary

- No worker result file exists for the incident invocation.
- No development Parquet exists for the incident invocation.
- No `dev_result.json`, `dev_cells.json`, or `frozen_analysis.json` exists.
- No scientific score or threshold verdict was printed to the logs.
- Confirmation remained sealed.
- The original worker directories and logs are retained under `/mnt/sdb`.

The active phase ledger remains truthful: development started. Recovery is
recorded as an execution-finalization recovery within that already-started
phase, not presented as an untouched first observation. The recovery may reuse
only the same frozen inputs and must not change any scientific choice.
