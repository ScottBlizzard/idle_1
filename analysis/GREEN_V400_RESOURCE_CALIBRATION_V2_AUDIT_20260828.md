# GREEN v4 outcome-blind resource calibration v2 audit

Date: 2026-08-28

Repository commit: `b2e7fed5867fb3d247cd112b637296f29bd83a11`

Scope: synthetic/resource engineering only. No real certificate, development outcome, confirmation content, P13 threshold result, or response Jet payload was opened.

## Decision

The shared-host engineering path is **GO** for synthetic resource calibration. The evidence does not authorize a real certificate or any scientific decision. Formal hard-memory authorization remains unavailable on this school host because delegated cgroup-v2 `memory.max`/`memory.events` control is not available without host administration.

## Regression and state closure

- The production compiled backend completed all `611` collected Linux tests at commit `b2e7fed` with no skip or failure.
- `MonotoneAnytimeCertificateState` now separates raw and tightened enclosures, keeps logical/admitted/completed/cache-hit accounting distinct, applies nonempty monotone intersections, serializes with a parent hash chain, and rejects post-construction nested mutation.
- A supervisor reconciliation transition records partially completed or failed admitted work without refunding it and leaves a half-partition uncommitted.
- The frozen resource reason is consistently `MAX_FINAL_LEAVES_PER_RADIUS_REACHED`.

## Sanitizer evidence

The strict two-family summary is `PASS_SANITIZER_AUDIT_SUMMARY` with semantic hash
`28987cf99da2e5bdd316bea7cdb4cd67f5e5b26875322f622e8e392c806947cf`.

- ASan+UBSan close-first: exit 0, no signal, expected PASS status, clean diagnostic log.
- Native-only TSan close-first: exit 0, no signal, empty TSan diagnostic log, `PASS_CLOSE_FIRST_PRELOCK_WAITER`; sealed semantic hash `879f68d9344d77d806b2b309046d55711570f58517798d14ecb5c00ec338189d`.
- The earlier Python/ctypes TSan attempt stopped making CPU progress and was removed by its 30-minute external limit. It is retained as **INCONCLUSIVE**, not PASS. Replacing that process boundary with a TSan-instrumented C++ harness isolated the audit from Python, NumPy, and ctypes.
- TSan covers the two repository C++ translation units and the executed close-first interleaving. It does not instrument Python, libstdc++, MPFR, or GMP and does not prove unexecuted interleavings.

Artifacts:

- `GREEN_V400_SANITIZER_AUDIT_SUMMARY_V1_20260828.json`, file SHA-256 `ef425a9478567adc35643e18ea45b8b192b476a3b976d3587117d689076fbf2f`.
- `GREEN_V400_NATIVE_CLOSE_FIRST_TSAN_SEALED_V1_20260828.json`, file SHA-256 `e38efd59cb97813418c8069af56bff10702285b80a9a23f0635d4968ee2842d4`.

## Cold-process native matrix

`PASS_NATIVE_COLD_MATRIX` completed 60 distinct Linux process identities: 30 at 384 bits followed by 30 at 512 bits. Each precision covered the seven frozen exact-domain classes; each process opened one context, made one physical 81-node dispatch, retained only five root hashes, and ran with GPU visibility disabled.

| Precision | Samples | Median dispatch | Maximum dispatch | Maximum total | Maximum peak RSS |
|---|---:|---:|---:|---:|---:|
| 384 bit | 30 | 22.1598 s | 37.6662 s | 52.9968 s | 179,844 KiB |
| 512 bit | 30 | 26.2166 s | 33.5837 s | 50.8062 s | 197,888 KiB |

Report semantic hash: `5c121ebf2be881aa1a35303640c6af12a09e5f46c0d0aa5537d582f6bf3f9cee`.

Artifact file SHA-256: `db78b5b5745749a066c598cd5d4a0981b732268e4de98995118f656e00bad107`.

These are shared-host observations, not formal wall-time or memory bounds.

## Seventeen-radius dry orchestration

The exact radius order was `2^0, 2^-1, ..., 2^-16`, with no center reuse and no memoization. Every pass was charged before execution and committed canonically by manifest ordinal.

### Binding semantic rehearsal

- Status: `PASS_BINDING_SEMANTIC_DRY_ORCHESTRATION`.
- 384-bit phase: 493/493 charged and completed.
- 512-bit phase: 289/289 charged and completed only after the complete 384-bit phase.
- Total: 782/782.
- Report semantic hash: `9a6bfb1e63e6a2137d86fdfb1b3acba55dba8dc107b9a0348bcc5c76be56fd48`.
- Artifact file SHA-256: `a4dd5176527dfa81e3d0abc3986d60e46b654f0c42fbe8c19b031776c5594df4`.

### Nonbinding worst-case L14 resource stress

- Status: `PASS_NONBINDING_WORST_CASE_L14_RESOURCE_STRESS`.
- Explicitly `nonbinding=true` and `scientific_certificate_authorized=false`.
- 493/493 official-precision passes and 289/289 audit-precision passes completed.
- Report semantic hash: `4c07050f8b71c8b02657f57f354f4286044ed0bd9f1004c589677f3b387b50ac`.
- Artifact file SHA-256: `906245b033868303fa8804291e714a3b8a91363be3fef4a4d7aeb98d454c3c49`.

### Binding failure short circuit

An injected failure at manifest ordinal 5 returned `RESOURCE_INCONCLUSIVE`. Four-worker admission overlap produced nine charged passes, eight completed passes, and one failed charged pass. No refund occurred, and the 512-bit launch count was exactly zero.

- Report semantic hash: `a310811698e20bf9503cd01e73d44ec9c3d37686f1bbfbee5384f8655aa5b45d`.
- Artifact file SHA-256: `2e679b44d4acec0b59e4ff7757f06856931cbdc41293124170c3b5f42438de07`.

## Remaining boundary

The next authorized work is synthetic-only budget calibration and resource-manifest selection. A real certificate/development/confirmation run remains disabled. On this host, RLIMIT_AS, process-tree observation, external monotonic deadlines, descendant cleanup, and fail-closed publication are available; cgroup-v2 hard-memory evidence is not. This host limitation changes resource authorization, not the Joint Witness estimand or the anytime soundness theorem.
