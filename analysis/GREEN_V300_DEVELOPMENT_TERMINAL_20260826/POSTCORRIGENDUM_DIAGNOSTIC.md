# GREEN v3.0.0 Post-Corrigendum Read-Only Diagnostic

This diagnostic is non-protocol. Confirmation was neither inspected nor started.

## Terminal outcome

- Verdict: `POSTER_ONLY`
- Failed gates: `detectability_lcb, detectability_rho, set_snr_cells`
- All other frozen development gates passed.

## Joint certificate

- The point estimator matches the AD joint target extremely closely: median
  absolute center/target error `9.12123e-09`.
- Nevertheless, all `80` of 80 record-level
  certified intervals cross zero.
- Median record-level set SNR is `0.102241`; the maximum
  is `0.652`; 0/80 reach 4.
- The median bound is `9.79144` times the
  absolute target, and the worst ratio is `503.108`.
- Consequently, 0/10 cells reach the frozen set-SNR threshold.

The corrected projection/envelope contraction changed a joint bound by at most
`1.23046e-07` relatively. This is expected because
the frozen physical joint direction lies almost entirely in each selected probe
frame; it proves the remaining width is not caused by that implementation omission.

## Radius stability

Restoring the inherited v2 denominator floor changed the coarse/fine median
symmetric change from `0.936902` to
`0.00182003`, which passes the frozen gate.

## Detectability saturation

- Gate-system classes: `{"recoverable": 1599, "unresolved": 1}`.
- Curvature SNR range: `8.99797` to
  `6384.8`.
- Direct-error median: `3.00547e-06`; p90:
  `1.10275e-05`.
- Fraction with direct error at most 1e-4: `0.9925`.
- Frozen detectability Spearman: `0.0161269`;
  cluster-bootstrap LCB: `-0.0457267`.

The observed panel is almost wholly recoverable and already at negligible error.
It does not span the recovery boundary needed to demonstrate a monotone
identifiability/error transition.

## Scientific decision point

The remaining failures require a new theory/protocol decision, not a small code
repair: either derive a valid correlation-aware joint certificate that avoids
worst-case per-gate triangle composition, and/or design a fresh outcome-blind
boundary-spanning development panel. The sealed v3 confirmation must not be opened
under the current `POSTER_ONLY` verdict.
