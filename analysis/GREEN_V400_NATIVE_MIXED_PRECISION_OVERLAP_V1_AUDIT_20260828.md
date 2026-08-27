# GREEN v4.0 native mixed-precision overlap audit v1

Date: 2026-08-28

## Decision

`PASS_MIXED_PRECISION_OVERLAP` as an outcome-blind thread-isolation stress test. Mixed-precision overlap remains prohibited in production; the binding phase order remains `ALL_384_THEN_REPLAY_SAME_PARTITION_512`.

On clean commit `04238979a1c5e6ac0661614a1a9472d27fc559fa`, one 384-bit and one 512-bit dispatch of the same closed dyadic domain executed on independent native contexts with global peak active count two and per-context peaks one. Each concurrent decoded payload exactly equaled its own sequential repeat. All 15 root/component interval checks—five native roots times value, first derivative, and second derivative—satisfied 512-inside-384 nesting, and every concurrent and sequential trace matched the frozen 81-event vector.

The complete audit took 96.2199 seconds and reached 369,744 KiB sampled process-tree RSS. The machine-readable report semantic hash is `d4af407632b649bdb145d39ccf284d8c775847fdf9072a0a0ac0e1adee6f7905`.

## Claim boundary

This closes the narrow MPFR/context thread-isolation question under mixed precision. It does not change the official-first phase order, authorize production mixed-precision scheduling, expose a scientific outcome, or establish a production resource bound.
