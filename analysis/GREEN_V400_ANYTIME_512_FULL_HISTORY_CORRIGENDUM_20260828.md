# GREEN v4 anytime 512-bit full-history audit corrigendum

Date: 2026-08-28  
Scope: outcome-blind synthetic resource calibration only  
Scientific outcomes opened: no

## Decision

The provisional `N_512 = L + 3` formula audits only the final frozen partition.
It is insufficient for the stronger policy claim that the monotone anytime
recurrence is replayed independently at 384 and 512 bits.

The official checkpoint after `k` splits intersects the raw enclosures from
every partition on the split path.  Therefore an independent precision audit
must evaluate every unique cell in that path at 512 bits, reconstruct every
partition in the same canonical order, run the monotone recurrence using only
512-bit quantities, and only then check containment inside the corresponding
384-bit checkpoint.  Intersecting a 512-bit result with the 384-bit monotone
interval before the containment check is forbidden because it makes nesting
true by construction.

For a final partition with `L` leaves and no cache reuse:

- historical unique cells at each precision: `2L - 2`;
- endpoint/center passes at each precision: `3`;
- `N_384 = 2L + 1`;
- `N_512 = 2L + 1`;
- `N_total = 4L + 2` per radius.

For 17 radii, the worst-case candidate counts are:

| final-leaf budget | 384 | 512 | total |
|---:|---:|---:|---:|
| 4 | 153 | 153 | 306 |
| 8 | 289 | 289 | 578 |
| 16 | 561 | 561 | 1122 |
| 32 | 1105 | 1105 | 2210 |

The earlier 782-pass `L=14` dry orchestration remains evidence for final-
partition replay and phase ordering only.  It is not evidence of a separate
512-bit anytime recurrence.  The corrected no-cache `L=14` count is 986.

This correction changes resource accounting and audit implementation, not the
Joint Witness estimand, scientific thresholds, radii, split priority, or real-
row authorization.  Real certificate/development/confirmation execution
remains disabled.
