# GREEN v1.3.2 fixed-operation-graph execution decision

Date: 2026-08-25

## Classification

The v1.3.1 STOP remains immutable. It is not retried and its output root is not
modified. The failure occurred after all endpoint, stagewise, raw-logit,
derivative, and independent-target equivalence assertions passed exactly.

The failed assertion compared distinct CUDA operation graphs: a full-vocabulary
unembedding with leading batch dimensions 512 and 1. Such cross-shape equality
is not an executable-equivalence property. The diagnostic panel shows the same
effect for every batch size greater than one.

## Authorized v1.3.2 correction

One new protocol identity is authorized:

```text
schema_version = green-bridge-v1.3.2
protocol_id = structural-envelope-matched-bypass-v1.3.2
protocol_run_id = green-bridge-v1.3.2-one-shot
attempt_index = 1
retry_allowed = false
output_root = outputs/green_bridge_v132
```

The scientific tail shall use one fixed executable shape:

```text
manual tail batch size = 512
final partial batch = pad to 512, execute, then slice to the declared rows
full-hook executable-equivalence batch size = 1
target/JVP batch size = 1
```

Padding rows may not affect declared rows. Prepare must require:

1. fixed-shape repeated calls are bitwise identical;
2. changing padding content leaves declared rows bitwise identical;
3. the padded wrapper is bitwise identical to direct execution at shape 512;
4. peak allocated memory is at most 20 GB;
5. projected runtime is at most 24 GPU hours;
6. all v1.3.1 predecessor hashes and its terminal phase state are exact.

Cross-shape raw-logit equality is recorded as not applicable. It is not assigned
a relaxed threshold and is not replaced by centered logits, deltas, or margins.

## Scientific invariance

The following remain byte-level or value-level unchanged:

- structural-envelope ambient rank-one estimand;
- selected gates, sites, anchors, frames, radii, and directions;
- raw-logit endpoint threshold `2e-5`;
- derivative relative threshold `1e-4`;
- true central finite differences and near-zero absolute bounds;
- independent path target and residual-bypass subtraction;
- development/confirmation split, gates, bootstrap, and forward counts;
- float32, TF32-disabled deterministic CUDA environment;
- permanent termination of donor PCA, rank search, and learned alignment.

The fixed-batch wrapper changes only how already-declared independent tail rows
are packed into an executable GEMM. It does not change an input, response,
estimator, threshold, or statistical decision.

## Execution rule

Prepare, development, and confirmation may each be issued at most once. A phase
may continue only after mechanical verification of the preceding phase. Any
v1.3.2 STOP is terminal for this identity.

# BINDING EXECUTOR VERDICT — ONE FIXED-SHAPE V1.3.2 ATTEMPT AUTHORIZED; THEORY AND SCIENTIFIC DESIGN UNCHANGED
