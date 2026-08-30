# GREEN v4.0.0 Outcome-Blind Certificate Protocol Audit — 2026-08-30

## Purpose and audit boundary

This document records an independent, outcome-blind audit of whether the remaining GREEN v4.0.0 certificate path can be completed as a purely mechanical implementation under the already frozen contract.

The audit used commit `6aab5837bf3950b86b166ef938097e9739810d9b` and earlier materials only. This cutoff predates the formal development supervisors and result/status artifacts. The auditors did not inspect confirmation outcomes, did not use development outcomes to choose a rule, and did not modify any scientific parameter, seed, identity, queue, threshold, frozen plan, or output path.

Three independent audits examined:

1. recovery of the intended mathematical status mapping;
2. protocol and reviewer defensibility of an in-place implementation;
3. the executable code path from the frozen SFC universe to a certificate decision.

## Unified verdict

`IMPOSSIBLE_UNDER_FROZEN_CONTRACT`, with `SCIENTIFIC_AMBIGUITY`.

The remaining work is not merely missing orchestration. The repository contains enough material to compute Joint Witness interval quantities in synthetic/preparation settings, but it does not contain a unique, executable, frozen rule that maps the current SFC sites to the final GREEN status consumed by the shared analyzer. Implementing one of several plausible mappings now would constitute post-development scientific design.

## What is mechanically recoverable

The earlier materials specify a core Joint Witness scalar construction, including the fixed branch order, exact dyadic radii, precision policy, partitioning, and tolerances. The central scalar can be recovered as

```text
Psi(t) = C_PAT,J(t) - C_PAT,B(t) - C_TAR,J(t) + C_TAR,B(t).
```

The numerical certificate engine can produce interval records and distinguish at least:

- `INTERVAL_COMPUTED`;
- `RESOURCE_INCONCLUSIVE`;
- hard invalid input or execution failures.

Some downstream policies are also recoverable in prose: an interval touching the P13 boundary is unresolved, resource exhaustion is resource-inconclusive, hard invalidity is invalid, and only a final certified-positive state may count as accepted.

These facts are insufficient to produce the required row-level final classification.

## Missing scientific decisions

### 1. P13 is not an executable per-site predicate

The original explicit P13 definition in the v3.0.0 decision is a donor-cohort contraction feasibility gate:

```text
median(B_JW / B_box) <= 0.20
p90(B_JW / B_box) <= 0.50
```

The current v4.0.0 shared analyzer instead requires one final `green_status` for each SFC site. No frozen rule converts the donor-level aggregate P13 gate into a per-site status classifier.

The v4.0.0 protocol lock does not hash an actual mathematical predicate. Its `p13_definition_sha256` binds a placeholder equivalent to:

```json
{"policy": "unchanged; not executed"}
```

Therefore it does not determine a numerical boundary, sign orientation, prerequisite set, equality policy, or row-level decision.

### 2. The interval-to-status adapter is absent

The shared analyzer consumes a status in a final domain such as:

```text
CERTIFIED_POSITIVE
CERTIFIED_NEGATIVE
UNRESOLVED
INVALID
RESOURCE_INCONCLUSIVE
```

The certificate engine does not emit these states. It emits an interval computation state. No frozen classifier defines how the interval, P13, validity flags, branch contrast, and prerequisites jointly determine the final status. The documentary status domain is also not fully consistent: one corrigendum conditionally mentions `CERTIFIED_NULL`, while the shared decision schema does not define a corresponding final route.

### 3. Eight directions do not have a frozen row-level aggregation rule

Each SFC site is precommitted with eight GREEN directions. The existing certificate formulation takes one `physical_direction` and one scalar parameter `t`. The frozen materials do not specify whether the site decision is:

- one certificate per direction;
- an intersection or union across directions;
- a worst-case decision;
- a vote or count rule;
- a separate multivariate construction.

All of these choices can change coverage, denominators, and success rates. There is no unique mechanical choice.

### 4. Radius and perturbation identities are not bound

The older `CertificatePlan` contains 17 radii. The SFC direction payload uses `direction_norm = 0.001`. No frozen crosswalk binds the old radius schedule to the current SFC direction amplitudes and site decisions.

### 5. The old certificate row identity is not the SFC site identity

The older certificate row hash is based on fields such as phase, noun, century, distance, role, orientation, labels, pair key, and prompts. It describes an older Greater-Than prompt-pair universe and contains no layer/site coordinate.

The SFC site identity is based on:

```text
site_row_id = hash(prompt_row_id, layer, hook)
```

It spans layer-specific IOI and Greater-Than sites. No frozen crosswalk maps the old certificate rows to the new SFC sites.

### 6. The causal graph semantics are not a mechanical rebind

The checked-in GPT-2 TensorProgram was prepared for fixed later blocks and identifies injection with the final causal token. The current SFC patches layers 0–8 at earlier IOI/Greater-Than token positions. A valid certificate would require a complete downstream causal-cone builder/materializer for each patch site. That builder is not present, and choosing the current branch/control/contrast semantics is not determined by the old program.

## Missing production path and authorization

The frozen SFC execution plan contains queues for prediction, Grant cohorts, numerical replay, and endpoints. It contains no certificate queue, runner, packet, or production source binding. The formal worker similarly supports only prediction, grant, replay, and endpoint modes.

The code deliberately prevents treating the preparation path as production:

- `CertificatePlan.__post_init__` rejects `execution_authorized = true`;
- `JointWitnessRowSpec` allows only `formal_prepare_pool` or `synthetic` splits;
- the real certificate serializer rejects with `REAL_CERTIFICATE_SERIALIZATION_UNAUTHORIZED`;
- `CertificateResourceLock` rejects production authorization, while the supervisor requires it for production execution;
- the activated child plan must remain an exact mechanical derivation of the sealed parent, and the authorization does not permit adding queues, schemas, or source identities.

These engineering components can be built only after their scientific inputs and a successor authorization are fixed. They cannot be inserted into the existing frozen protocol without changing its contract.

## Why an internal majority vote is insufficient

Independent agents can verify that the definition is missing; they cannot create the missing estimand and classifier by consensus. Selecting a mapping after development has been opened would affect coverage, denominators, and the success decision. Labeling that choice as an implementation detail would not make it prospective or reviewer-defensible.

## Binding decisions requested from GPT Pro

GPT Pro should issue a machine-readable successor-protocol decision addressing all of the following.

### A. Scientific estimand and certificate semantics

1. Define precisely what one SFC site certificate asserts about restoration/transport.
2. Define the scalar or multivariate object certified for each of the eight directions.
3. Define branch, control, target, pattern, and contrast semantics for IOI and Greater-Than.
4. Define how the perturbation amplitude/radius schedule binds to `direction_norm = 0.001` and any additional radii.
5. Define the full downstream causal-cone graph from layer 0–8 and earlier patch positions to the evaluated output.

### B. Executable P13 and final status mapping

Provide a total deterministic function of the form:

```text
classify_site(
    site_identity,
    direction_certificate_records,
    p13_specification,
    validity_flags,
    resource_flags
) -> {
    CERTIFIED_POSITIVE,
    CERTIFIED_NEGATIVE,
    UNRESOLVED,
    INVALID,
    RESOURCE_INCONCLUSIVE
}
```

It must specify:

- the exact P13 quantity and numerical thresholds;
- sign orientation and equality/boundary handling;
- prerequisites and invalidity rules;
- aggregation across eight directions and all radii;
- aggregation across multiple records if applicable;
- the status domain, including whether `CERTIFIED_NULL` exists;
- denominator, coverage, and acceptance rules;
- the treatment of clean-task-invalid Greater-Than rows.

### C. Protocol versioning and confirmation eligibility

1. Decide whether this must be a new protocol/queue/schema/decision-rule version rather than an in-place v4.0.0 patch.
2. Decide whether the currently untouched confirmation split remains scientifically eligible after the successor rule is frozen.
3. If it is not eligible, specify the required new untouched cohort/reserve or independent rerun.
4. Specify which existing development outputs may be used only diagnostically and which, if any, may be reused mechanically.

### D. Production authorization and implementation contract

Define the new identities and frozen artifacts required for:

- the SFC-site-to-certificate-row builder;
- executable graph manifests;
- certificate queues and workers;
- real-row serialization;
- resource ceilings and deterministic retry policy;
- no-clobber atomic outputs and validation receipts;
- the shared analyzer input schema;
- independent confirmation execution and firewalling.

## Requested final deliverable

Please write one binding Markdown decision document into `analysis/` in the repository. It should include:

1. a clear `GO`, `CONDITIONAL_GO`, `REDESIGN`, or `STOP` verdict;
2. an explicit ruling on whether the ambiguity is scientific or purely engineering;
3. the complete mathematical and machine-readable successor specification;
4. the protocol-version and confirmation-eligibility ruling;
5. a closed implementation checklist that Codex can execute without making new scientific choices;
6. explicit forbidden actions and stop conditions;
7. a short reviewer-facing justification explaining why the successor protocol remains prospective and non-opportunistic.

Until such a decision is frozen, the correct action is to preserve all existing artifacts, keep confirmation locked, and not launch certificate production.
