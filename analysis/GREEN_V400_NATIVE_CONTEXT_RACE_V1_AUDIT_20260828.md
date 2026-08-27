# GREEN v4.0 native context race audit v1

Date: 2026-08-28

## Decision

`PASS_CONTEXT_LIFECYCLE_RACES` for the closed, outcome-blind native context API.

On clean commit `4c2fe95a86b2e7cd5f0e99259e7bf970de0feaca`, the actual 384-bit plan passed the following gates:

- Two simultaneous calls on one context produced two completed entries but a global and per-context peak active count of one. Their decoded payloads were exactly equal and both retained the frozen 81-event trace.
- A close initiated only after native metrics observed one active dispatch returned status 0 after waiting 21.6920 seconds. The active dispatch completed with its valid trace; the terminal active count was zero.
- After that close returned, context info, projection info, projection export, dispatch, and a second close all returned stale-handle status 2.
- Two simultaneous closes on a fresh context linearized as statuses `[0,2]`.
- MPFR TLS was enabled and shared cache was disabled.

The complete audit took 90.0379 seconds and reached 248,208 KiB sampled process-tree RSS. These are observations, not production limits. The machine-readable report semantic hash is `ce9cfc9df67d5d3d07322653533ef29bb9b94445e45f2da42c500c9aa00295b4`.

## Claim boundary

The audit closes the concrete same-context contention, active-dispatch close, post-close rejection, and double-close linearization cases. It retains only trace/equality/status/resource evidence and no response Jet2 payload.

The separate close-first hook audit deterministically closes the ordering in which dispatch has copied the context `shared_ptr`, close marks it inactive before it acquires the execution mutex, and the old waiter must return status 2. This is still not a sanitizer proof over every possible interleaving. TSan/ASan stress and an independent external supervisor remain required engineering checks before production execution.
