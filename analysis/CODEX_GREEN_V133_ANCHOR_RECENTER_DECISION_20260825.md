# GREEN v1.3.3 frozen-anchor recentering execution decision

Date: 2026-08-25

## Scope

v1.3.3 is a fresh, one-shot technical correction to the v1.3.2 fixed-shape
manual-tail endpoint. It does not change the paper's theory, structural
estimand, data split, model, sites, selected gates, frames, radii, target
vectors, coefficients, thresholds, calibration, or confirmation rules.

## Identity

```text
schema_version = green-bridge-v1.3.3
protocol_id = structural-envelope-matched-bypass-v1.3.3
protocol_run_id = green-bridge-v1.3.3-one-shot
attempt_index = 1
retry_allowed = false
output_root = outputs/green_bridge_v133
```

## Endpoint correction

For a frozen batch-one anchor `y0`, let `F512(delta,z)` be the manual tail
evaluated with the frozen padded operation shape of 512. The active executable
endpoint is

```text
F133(delta,z) = y0 + F512(delta,z) - F512(0,0).
```

In exact arithmetic `F512(0,0)=y0`, so this is the same mathematical function.
The subtraction removes only the constant absolute offset caused by the CUDA
operation-shape implementation, while all finite perturbation increments stay
inside one deterministic fixed graph. The final partial chunk remains padded
to 512 and sliced back to its declared rows.

## New binding prepare gates

Before development, v1.3.3 must establish all of the following:

1. the fixed-shape zero endpoint equals the frozen batch-one anchor bitwise;
2. repeated fixed-shape calls are bitwise identical;
3. a zero row is invariant to the content of padding rows bitwise;
4. padded wrapper output equals direct fixed-shape output bitwise;
5. recentered fixed-shape center and nonzero path/control endpoints remain
   within the unchanged `2e-5` raw-logit tolerance of the independent full
   batch-one hook reference;
6. peak memory and projected runtime remain below their frozen caps;
7. the scientific payload hash remains exactly the v1.3/v1.3.1/v1.3.2 hash.

No cross-shape bitwise requirement is reintroduced and no numerical threshold
is relaxed.

## Terminal behavior

The analyzer now records insufficient survival as the already frozen
`STOP_ORAL` decision before attempting calibration. This only converts the
v1.3.2 unhandled exception into the intended terminal gate; it does not alter
the decision rule. v1.3.3 prepare, development, and confirmation are each
single-use. Any scientific STOP is terminal for this identity.

