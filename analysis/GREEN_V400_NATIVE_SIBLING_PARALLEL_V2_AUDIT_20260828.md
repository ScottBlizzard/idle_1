# GREEN v4.0 native sibling parallel audit v2

Date: 2026-08-28

## Decision

`PASS_PHYSICAL_SIBLING_PARALLEL` for the closed, outcome-blind 384-bit adaptive audit only.

The v1 report established canonical two-worker scheduling but did not establish physical overlap. Its six dispatches took 147.6284 seconds and reached 295,668 KiB sampled tree RSS because `green_v400_native_precision_context_dispatch_cell_v1` retained the global context-registry mutex for the complete 81-event dispatch. The v1 status must therefore be read as a structural scheduling pass, not as evidence of concurrent native execution.

The corrected runtime now uses the registry mutex only to acquire a `shared_ptr` to a context. A per-context execution mutex preserves same-context serialization and close safety, while distinct contexts execute independently. The dispatch kernels share only the read-only plan mapping; all resident Jet2 buffers remain context-owned.

## Hard gates observed

- Clean repository commit: `2ba755f9e0a15d8006b9014fa8ee2633dd9cc068`.
- Report semantic hash: `44fc4d0622438a65eb91272332ee3dbf3bf1b2768edcf1d749d96ba93ca756db`.
- MPFR TLS build option: enabled; shared cache: disabled.
- Concurrent audit dispatch entries: 6; terminal active count: 0; peak global active count: 2.
- Per-context entries: 3 and 3; terminal active counts: 0 and 0; peak per-context active counts: 1 and 1.
- After canonical native JSON decoding, every finite value/first/second Jet2 endpoint exactly equals its same-precision sequential baseline: 6 of 6 dispatches. The machine-readable field retains the legacy `bit_exact` label, but this audit did not compare raw JSON bytes.
- Canonical adaptive pair rounds: 3; result commit order remains input order.
- Complete audit wall time, including the six-dispatch sequential baseline and context setup/teardown: 238.4912 seconds.
- Peak sampled process-tree RSS: 314,792 KiB. This is an observation, not a production upper bound.

The machine-readable evidence is `GREEN_V400_NATIVE_SIBLING_PARALLEL_V2_AUDIT_20260828.json`. It retains domains, scheduling/resource counters, equality booleans, and provenance only; it does not retain response Jet2 payloads or apply a scientific threshold.

## Claim boundary

This result authorizes the engineering statement that the frozen initial negative/positive pair can execute concurrently and, after initialization, the two children of exactly one already-selected heap parent can execute simultaneously on two independent native contexts without changing any decoded finite Jet2 endpoint or canonical commit order. The v2 JSON's shorter `claim_scope` phrase “one frozen heap parent per round” is interpreted with this explicit initial-pair exception. It does not authorize popping multiple heap parents per wave, changing adaptive selection, changing certificate semantics, or running real scientific outcomes. The backend hash identifies the loaded binary; it is not a reproducible-build receipt for the compiler and linker toolchain.

The separate context-race v1 audit closes same-context contention, active-dispatch close waiting, post-close stale-handle rejection, and concurrent double-close linearization on the actual plan. Before production execution, the remaining concurrency-specific checks are mixed 384/512 exact-repeat and nesting under overlap, TSan/ASan stress over broader interleavings, and integration with the external resource supervisor. These are engineering gates and do not require a new scientific design decision.
