# GREEN v1.3.6 terminal scientific handoff for GPTPro — 2026-08-25

## Requested role

Act as the theory and scientific-protocol decision maker. The local executor has completed the authorized engineering corrections, prepare audits, and full development computation. Do not treat this as another ordinary implementation bug unless the repository evidence establishes one. Produce a binding, executable next-step decision that preserves or strengthens the ICLR-Oral-level theoretical contribution.

## Scientific invariant that must not be downgraded

The central claim remains:

> Matched-bypass derivatives identify a basis-invariant ambient rank-one path operator, made probe-complete by the exact LayerNorm structural envelope.

Fixed-rank donor PCA remains terminated. Do not restore it. Do not replace the central estimator by PIE or another baseline merely because an unregistered diagnostic looks strong. Any new protocol must preserve the distinction between the theory-derived mixed estimator, baselines, development, and unopened confirmation.

## Execution lineage

The reviewed v1.3 decision authorized a pre-scientific endpoint correction without changing the scientific payload. Subsequent versioned identities corrected newly exposed implementation or executable-shape defects while keeping the payload hash fixed:

| Identity | Outcome | Classification |
| --- | --- | --- |
| v1.3.1 | prepare STOP | exact endpoint exposed batch-shape dependence |
| v1.3.2 | prepare pass; development 0/16 | fixed-512 endpoint was not the exact batch-one endpoint |
| v1.3.3 | prepare STOP | output recentering did not establish equivalence |
| v1.3.4 | prepare pass; worker failure | `GateIdentification.G` did not exist |
| v1.3.5 | prepare pass; worker failure | `GateIdentification.D` orientation mismatch |
| v1.3.6 | prepare pass; development completed | genuine frozen development STOP at survival |

The v1.3.6 source fixes are at commits:

- `5d46840dc3baf0bb16db6833db6b3073c47c1912` — direct-bypass orientation fix.
- `e4624c92f31062a7f70699a2670cd1e6243e96ce` — self-contained predecessor archive verification.

The scientific payload hash remained:

`60ca5e9e221064f288a1993ee3cbf42e99330bbf6f9008946a25556438cbc3d3`

The v1.3.6 spec hash recorded in the development result is:

`cb771c59e91b4fc553ef73a1c7a116ec0ee55f499ce46a2f91e4c600cd8bd41d`

## v1.3.6 mechanical execution evidence

- Prepare passed all frozen hashes and exact batch-one bitwise equivalence checks.
- All 168 contract tests passed locally and on the server.
- Development ran on eight physical GPUs with exact endpoint batch size one.
- All eight workers completed, each producing 32 records and a 64-line endpoint ledger.
- Aggregate output: 128 tensor rows and 128 energy rows.
- Peak allocated memory per worker: 1,865,645,056 bytes.
- Worker result hashes matched the merged parquet artifacts.
- Confirmation was never opened.

Merged artifact hashes:

- `dev_tensor_scores.parquet`: `660788dde8bc5df1d057db31b4dc1065b222ac7777efc0e4c6220e09f1ed81ff`
- `dev_energy_targets.parquet`: `23a99b6998ec2c51184ae26b8f86a7656247ff2091e251752c1fccd06295e593`

Frozen development outcome:

- outer stop gate: `12_DEVELOPMENT_SURVIVAL`
- development verdict: `STOP_ORAL`
- surviving cells: 0/16
- conditioned cells: 0
- SNR cells: 0
- mixed RMSE, best baseline, and relative gain: undefined because no cell survived

## Exact terminal localization

This is not missing data. All raw records were produced:

- energy: 128/128 admissible;
- tensor: 0/128 admissible;
- every cell: 0 admissible tensor rows and 8 admissible energy rows.

For each tensor item, both `tar` and `pat` mixed systems must be admissible. A mixed system is admissible only if all ten selected gates are non-invalid, at least three are active-identified, and the common-frame bypass disagreement is at most 0.15.

Across 1,280 gate audits per system:

| System | active-identified | certified-target-null | invalid | all-valid items | admissible systems |
| --- | ---: | ---: | ---: | ---: | ---: |
| tar | 708 | 6 | 566 | 0/128 | 0/128 |
| pat | 702 | 7 | 571 | 2/128 | 2/128 |

Thus many individual gates are identified, but the ten-gate completeness conjunction eliminates every tar item and therefore every tensor row.

Among 1,137 invalid gate audits, persisted active-criterion failures are non-exclusive:

| Failure | Count |
| --- | ---: |
| white-box agreement | 851 |
| factorization residual | 823 |
| curvature SNR | 229 |
| inferred tensor SNR, whose P-norm was not serialized | 74 |
| tensor symmetric change | 62 |
| gate-response SNR | 30 |
| Richardson change | 11 |
| tensor cosine | 4 |

Factorization residual and white-box agreement fail together in 635 invalid gates. Their invalid-gate medians are close to but above the frozen cutoffs:

- factorization residual: median about 0.185–0.190 versus threshold 0.15;
- relative white-box error: median about 0.065–0.072 versus threshold 0.05.

The tensor cosine and Richardson-stability criteria usually pass. The exact distributions, per-item summaries, source hashes, and failure combinations are in `analysis/GREEN_V136_TERMINAL_AUDIT_20260825/terminal_admissibility_audit.json`; the reproducer is `analyze_terminal.py`.

## Non-confirmatory diagnostic only

All tensor rows are inadmissible, so the following cannot support the registered claim. Nevertheless, the raw rows contain structured signal:

- raw PIE versus behavioral target, item-level Spearman: 0.613;
- raw PIE versus behavioral target, 16-cell mean Spearman: 0.962;
- raw first-order versus behavioral target, 16-cell mean Spearman: 0.876.

This makes a blanket “no signal” interpretation implausible. It also creates a serious outcome-adaptive-design risk: thresholds or estimator definitions must not be changed merely to recover these correlations.

## Decision questions

Please inspect the code and all evidence, then decide:

1. Is the v1.3.6 development STOP a valid scientific failure of the frozen protocol, or does the evidence reveal a remaining mathematical/estimand mismatch in factorization or white-box comparison?
2. Is requiring every one of ten gates to be either active-identified or certified-null theoretically necessary for the basis-invariant operator claim, or can a proof-valid partial-identification/aggregation rule preserve the theorem without outcome-driven threshold relaxation?
3. Are the 0.15 factorization and 0.05 white-box cutoffs derivable from the numerical error model used here? If not, specify a principled calibration or confidence-bound construction using only design/noise quantities and no behavioral targets.
4. Should the next scientific identity use an improved estimator, stencil, radius schedule, robust gate aggregation, or explicit partial-identification interval? Give the theorem-level justification and exact frozen protocol.
5. How should the strong inadmissible PIE diagnostic be handled without demoting the theory-derived estimator or contaminating confirmation?
6. Is any new GPU development run scientifically authorized? If yes, define the new identity, immutable predecessor handling, exact implementation changes, tests, audits, thresholds, phase locks, and exact commands. Confirmation must remain forbidden unless the newly frozen development gate opens it.

## Required output

Write one integrated Markdown decision document at:

`analysis/GPTPRO_GREEN_V136_TERMINAL_DECISION_20260825.md`

The document must include:

- a binding classification of the v1.3.6 outcome;
- a theorem/estimand audit, not only high-level advice;
- exact code-level changes if any;
- a non-outcome-adaptive justification for every scientific change;
- protocol/version identity and predecessor immutability rules;
- exact unit, regression, prepare, development, and confirmation gates;
- exact executable commands;
- explicit STOP conditions;
- an executor checklist;
- a final one-line binding verdict.

If no scientifically defensible next run exists, say so directly and explain how the paper should retain its theoretical height without overstating experimental support.
