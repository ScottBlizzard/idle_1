# GREEN v3.0.0 Formal Prepare Summary

Date: 2026-08-26  
Execution commit: `602e7c9c5e43f1aecf1485d4ad2fc6574b2fdaa1`  
Server worktree: `/mnt/sdb/ccj/worktrees/idle_1_green_v300_impl`  
Formal root: `/mnt/sdb/ccj/iclr_1_runs/green_bridge_v300_602e7c9c5e43f1aecf1485d4ad2fc6574b2fdaa1/outputs/green_bridge_v300`

## Result

The unique formal GREEN v3.0.0 prepare completed successfully:

```text
verdict: PREPARE_PASS
formal_one_shot: true
attempt_index: 1
retry_allowed: false
selected_global_radius: 1.0
development_started: false
confirmation_started: false
```

No development or confirmation response, anchor, derivative, cache, timing, or
inferential artifact was generated.

## Contract and numerical results

- Exact combined regression suite: `272/272`, zero skips.
- Canonical split hash passed.
- Canonical coefficient and radius payload hashes passed.
- Radius calibration used 40 outcome-blind legacy-donor strata.
- Finite-response point estimator: `float64_response_only`.
- Finite point estimates and AD audits used distinct isolated model-tail copies.
- Automatic derivatives were not used by the point estimator.
- Selected global radius: `rho*=1`, by the frozen largest-eligible rule.
- At `rho=1`, all `360/360` object-stratum checks passed.
- Maximum finite-versus-AD difference / eligibility ceiling at `rho=1`:
  `0.002004559655602945`.
- Exact direct-transport routes: zero failures.
- Exact direct-transport theorems: zero failures.
- Exact all-ten joint-composition routes: zero failures.
- Exact all-ten joint-composition checks: zero failures.
- Direction frame/complement maximum error remained below `1e-12`.
- Peak allocated GPU memory: `7.696776390075684 GiB`.
- Projected prepare + development + confirmation time:
  `2023.380114857573 seconds`.

## Numerical correction evidence

The pre-formal float32 legacy-only diagnostic had zero AD/theorem failures but
failed the global radius gate in the second-order `C` and `delta_H` stencils.
Errors worsened at smaller radii, identifying endpoint cancellation after
float32 quantization. `G`, `J_path`, and `J_control` remained accurate.

The float64 response-only correction changed no model weight, mathematical
coefficient, candidate radius, split, record, gate, direction, threshold,
success rule, baseline, or phase authorization. It restored the intended
finite-response estimator without making AD the point estimator. The complete
decision and evidence are recorded in
`analysis/CODEX_GREEN_V300_FLOAT64_RESPONSE_CORRIGENDUM_20260826.md`.

## Integrity

All 21 non-self-referential formal artifacts pass `sha256sums.txt` locally and
on the server. The formal root contains 22 required protocol files.

Key hashes:

```text
manifest.json
57dac7258feafdfd1aa7dab12c19168c08d9549d2de2fa3b75ec57031d42f40a

prepare_result.json
31abb5532852d339eca14b2812fe9868328ba87dedb5181d4c3f7afbd497e3f0

sha256sums.txt
8c3be74d19c43fce4ee2ff9f4816d69a0310109f2ae771482761a871c183b552

green_v300_combined_272_tests.log
04a991586c9fb648803b966ad752ba93afcd060b5cc504463fad0138491b0b91
```

## Current authorization state

The prior decision authorizes prepare only. This result does not itself
authorize development or confirmation. The complete formal return bundle must
now be reviewed before any development response is computed.
