# GREEN v4.0 native close-first hook audit v1

Date: 2026-08-28

## Decision

`PASS_CLOSE_FIRST_PRELOCK_WAITER` for the deterministic audit-only interleaving.

An audit-only backend built from clean commit `3829cb89d4acaea47c680a450fb23cb61fe3ae79` paused a dispatch after it copied the context `shared_ptr` but before it acquired the per-context execution mutex. While the dispatch remained paused, close removed the handle from the registry, acquired the execution mutex, marked the context inactive, and returned status 0. Only then was the dispatch released; it acquired the mutex, observed inactive state, and returned stale-handle status 2 without entering the native dispatch body. Entry, active, and peak dispatch counts therefore all remained zero. A post-close info call also returned status 2.

The audit took 9.0283 seconds and reached 91,460 KiB sampled process-tree RSS. Its semantic hash is `fef813531628bf936c97139d7ba5c03f07f73f40582638ae4f8e7f9308d62067`.

## Claim boundary

The barrier and its three control symbols exist only under `GREEN_V400_NATIVE_AUDIT_TEST_HOOKS`; the production backend build does not contain them. This closes the precise close-first/pre-lock-waiter ordering identified by independent review. It does not replace TSan, ASan/LSan, or broader randomized stress.
