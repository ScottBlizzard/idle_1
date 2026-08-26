# GREEN v3.0.0 Float64 Response-Only Numerical Corrigendum

Date: 2026-08-26  
Authority: Codex numerical implementation correction  
Scope: prepare-only legacy-donor radius calibration and the frozen future
finite-response point estimator

## Decision

GREEN v3.0.0 evaluates its finite-response point estimator on an isolated
float64 copy of the frozen model tail. The active float32 model remains
byte-identical. A second, distinct float64 copy supplies the dual-route AD
target and exact-theorem audit.

The point estimator uses only finite function values at the preregistered
stencil endpoints. It does not call `jacfwd`, `jacrev`, JVP, VJP, white-box
gradients, or any automatic derivative. AD values remain excluded from the
point estimate and are used only by the already frozen numerical-fidelity and
theorem gates.

## Triggering evidence

The complete float32 legacy-only dry run produced:

- 0/40 AD route failures;
- 0/40 exact direct-transport theorem failures;
- nearly universal success for `G`, `J_path`, and `J_control`;
- failure concentrated in the second-order `C` and `delta_H` stencils;
- worsening finite-versus-AD error as the radius decreased.

This signature is catastrophic subtraction of quantized float32 endpoints,
not a failure of the matched-bypass identity or response-only estimand.

The independent float64 response-only probe at commit `393d2e3` produced:

| Global rho | Eligible objects | Maximum difference / ceiling |
|---:|---:|---:|
| 1 | 360 / 360 | 0.002004559655602945 |
| 1/2 | 360 / 360 | 0.00009254117208877623 |
| 1/4 | 360 / 360 | 0.0000028691715818839074 |
| 1/8 | 360 / 360 | 0.0000009219794400428948 |

The largest eligible global radius is therefore `rho*=1`, exactly as required
by the frozen largest-eligible selection rule. The same probe recorded:

- zero direct-transport theorem failures;
- zero all-ten joint-composition failures;
- peak allocated memory `7.303274154663086 GiB`;
- projected prepare + development + confirmation time
  `2113.7324653687306 seconds`, below the 24-hour ceiling.

## Invariants

This correction changes no model weights, mathematical coefficient, radius
candidate, dataset record, split, gate, direction, threshold, success rule,
baseline, or phase authorization. It is a strict numerical precision upgrade
whose purpose is to make the already specified finite-response estimator
faithfully approximate the local response of the frozen model.

No v3 development or confirmation anchor, logit, derivative, cache, response,
or timing measurement was generated while selecting this correction.
