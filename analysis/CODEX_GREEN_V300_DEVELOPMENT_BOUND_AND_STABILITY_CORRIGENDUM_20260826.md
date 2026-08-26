# GREEN v3.0.0 Development Bound and Stability Corrigendum — 2026-08-26

## Status

This is a post-terminal implementation corrigendum, not a new scientific
protocol and not an authorization to inspect or run confirmation data.
The original `POSTER_ONLY` result produced by merge commit
`2f8e1cf6b061a554e7afc584589ccf5529cff08d` must be independently archived
before corrected official paths are written.

## Error 1: recoverable joint-set contraction

The first development implementation bounded a recoverable scalar contraction
by

```text
||contrast|| * ||direction|| * epsilon_P_F.
```

That expression dropped both the projection of the direction into the probe
frame and the frozen response-envelope term.  The v3 theorem requires

```text
||contrast|| * [
  epsilon_P_F * ||Q^T direction||
  + (||G|| + epsilon_G) * epsilon_env * ||direction||
].
```

This is a direct implementation correction to the already frozen bound.  It
does not change a radius, threshold, record, target, point estimator, baseline,
or gate classification.  Because the required per-record probe-frame values
were not saved in the merged artifact, only the 80 development joint records
must be recomputed.  The completed 80-record transport role is reused byte for
byte and revalidated by the existing merge hashes.

## Error 2: coarse/fine radius stability denominator

The first v3 merge divided the coarse/fine error difference by the larger of
the two errors.  The inherited v2 confirmatory definition instead uses the
mean absolute error with a frozen `0.05` scale floor:

```text
abs(coarse - fine) / max((abs(coarse) + abs(fine)) / 2, 0.05).
```

The missing floor made two scientifically negligible errors appear unstable
solely because both were close to zero.  Restoring the inherited definition is
a deterministic re-merge of unchanged transport values.

## Firewall and provenance

- The initial terminal evidence is copied to a separate archive before any
  official file is replaced.
- Only development joint records are recomputed.
- Confirmation remains sealed and unauthorized.
- The detectability statistic, bootstrap, threshold, and underlying transport
  values are not changed by this corrigendum.
- Any remaining detectability failure is therefore treated as a genuine
  scientific decision point, not an engineering defect to tune away.

