# CPU Theory-Gate Execution Record

Date: 5 August 2026

Authority: `analysis/GPTPRO_THEORY_PACKAGE_20260805.md`

Theory status: **AMBER**

## Scope

This record covers only the local work authorized by the binding theory package.
It does not claim that the ASG-RDAG theorem applies to arbitrary transformer
block outputs, and it does not authorize the blocked Greater-Than GPU design.

## Implemented modules

### Restricted structural estimator

`src/mixed_path_identification.py` implements:

- central first-order responses with probe-specific radii;
- four-corner mixed responses in the exact `++ - +- - -+ + --` order;
- full-rank Cartesian and arbitrary-paired Kronecker designs;
- SVD/least-squares tensor recovery without ridge fallback;
- known-curvature correction `P = rho * H`;
- the ASG inverse `G = J2`, `D = J1 - sum_j P_j`;
- active-channel recovery of detailed `A` and `C` factors;
- factorization residuals for the separable-gate necessary condition;
- absolute local path effects and output contrasts;
- Frobenius-compatible structural response energy;
- probe moment, rank, eigenvalue, condition-number, and effective-rank diagnostics;
- bounded-i.i.d. and without-replacement covariance radii;
- Hoeffding-Serfling finite-population radii;
- deterministic center-replay checks;
- hard failures for rank deficiency and zero/sub-threshold curvature.

The module contains no transformer hooks and cannot be invoked as evidence that
the current Greater-Than blocks instantiate the theorem.

### Analytic tests

`src/test_mixed_path_identification.py` contains all 18 tests required by the
theory package plus one additional positive test for complete non-Cartesian
paired-Kronecker recovery.

All 19 pass locally in float64:

1. quadratic ASG exact recovery;
2. non-coordinate full-rank recovery;
3. active-edge inverse;
4. product energy identity and theorem bounds;
5. first-order cancellation requiring mixed responses;
6. rank-deficient first-block refusal;
7. full marginal span with incomplete paired Kronecker design;
8. complete paired-Kronecker recovery;
9. unknown curvature-ratio non-identification;
10. zero-curvature refusal;
11. omitted-bypass impossibility;
12. central-first `r^2` scaling;
13. mixed four-point `r^2+t^2` scaling;
14. output-dimension energy scaling;
15. exact without-replacement Serfling enumeration;
16. corner-sign order;
17. variable-radius denominators;
18. deterministic center replay;
19. no ridge fallback.

### Synthetic runner

`src/run_mixed_path_synthetic.py` runs the analytic ASG model without transformer
libraries or a GPU and writes a structured JSON artifact.

Five deterministic seeds passed:

| Seed | Overall maximum tensor error | Energy-identity error |
|---:|---:|---:|
| 0 | `1.7763568394002505e-15` | `2.220446049250313e-16` |
| 1 | `2.6645352591003757e-15` | `2.220446049250313e-16` |
| 2 | `1.3322676295501878e-15` | `2.220446049250313e-16` |
| 3 | `2.6645352591003757e-15` | `4.440892098500626e-16` |
| 4 | `1.7763568394002505e-15` | `4.440892098500626e-16` |

These are numerical implementation checks, not independent scientific
replications.

## First-round implementation repairs

### Vector-output IRS convention

`src/interventional_response.py` now sums squared discrepancies over the output
axis and averages over probes/items. Replicating a scalar output `k` times now
multiplies squared response energy by `k`, matching the Frobenius convention in
the theory package. The normalized ratio is unchanged when numerator and target
energy are replicated together. Scalar-output historical GPT-2 results are
unchanged.

The comparison object now reports:

- per-item target RMS;
- the per-item normalization-floor mask;
- the fraction of items whose normalizer hits the floor.

Two regression tests cover output-dimension scaling and floor reporting.

### Explicit composite-conformal data roles

`src/validity_crossfit.py` now permits explicit, disjoint normalization and
final-calibration arrays and requires a declared `target_law` label. Diagnostics
record whether explicit splits were used and the exact normalization/calibration
sizes. A regression test verifies the final conformal grid is determined only by
the final-calibration size.

This API permits induced-endpoint-law calibration. It does not make an induced
endpoint exchangeable with natural clean examples, and it does not license
manifold or naturalness claims.

## Verification commands

The local bundled Python runtime was used with `PYTHONPATH=src`.

```text
python -m compileall -q \
  src/mixed_path_identification.py \
  src/test_mixed_path_identification.py \
  src/run_mixed_path_synthetic.py

python src/test_interventional_response.py
python src/test_validity_crossfit.py
python src/test_mixed_path_identification.py

python src/run_mixed_path_synthetic.py --seed 0
python src/run_mixed_path_synthetic.py --seed 1
python src/run_mixed_path_synthetic.py --seed 2
python src/run_mixed_path_synthetic.py --seed 3
python src/run_mixed_path_synthetic.py --seed 4
```

At the time of this record:

- 7 interventional-response tests pass;
- 7 validity/conformal tests pass;
- 19 mixed-path tests pass;
- all three new modules compile;
- all five analytic runner seeds pass.

## Remaining blocker

The CPU implementation validates the restricted theorem's algebra. It cannot
resolve the Greater-Than structural-inverse lemma.

The currently proposed raw blocks still fail:

- two-block topological order;
- gate-input intervention semantics;
- known curvature correction at the hook;
- complete mixed-cut isolation;
- feasible raw-space probe completeness;
- identification of an independent absolute path target from the measured total
  response.

No server or GPU execution is scientifically authorized until GPT Pro either
provides a topology-correct transformer instantiation with a proved structural
inverse or issues a final RED decision.

## Local gate decision

**CPU IMPLEMENTATION: PASS.**

**CURRENT GREATER-THAN SERVER PROGRAM: BLOCKED BY THEORY.**

**NEXT ACTION: GPT Pro green-bridge adjudication using
`GPTPRO_GREEN_BRIDGE_PROMPT_20260805.md`.**
