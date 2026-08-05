# P0 within-site, temporally eligible mechanism audit

Runs: 3; prompt-site observations: 2160.
Every patch is at the IO position and occurs before both measured GPT-2 Name Mover Heads.

| Layer | Mean R | Mean NMH recovery | Corrupt overlap | Matched overlap | Mean within-layer rho (corrupt/matched) |
|--:|--:|--:|--:|--:|:--:|
| 0 | 1.027 | 1.021 | 0.486 | 0.490 | 0.135/0.178 |
| 1 | 1.067 | 1.022 | 0.490 | 0.487 | 0.133/0.194 |
| 2 | 1.042 | 0.959 | 0.488 | 0.492 | 0.106/0.229 |
| 3 | 1.006 | 0.724 | 0.493 | 0.487 | 0.096/0.193 |
| 4 | 0.884 | 0.336 | 0.493 | 0.495 | 0.186/0.171 |
| 5 | 0.869 | 0.265 | 0.500 | 0.498 | 0.201/0.169 |
| 6 | 0.861 | 0.271 | 0.510 | 0.507 | 0.236/0.218 |
| 7 | 0.858 | 0.270 | 0.529 | 0.512 | 0.232/0.249 |
| 8 | 0.854 | 0.292 | 0.529 | 0.515 | 0.239/0.237 |

## Decision

- Stable layers with mean R > 0.8 and mean R−A > 0.4 in every seed: `[4, 5, 6, 7, 8]`.
- Supports behavioral/mechanism separation: **True**.
- Supports low overlap as necessary for bypass: **False**.
- Pooled fixed-effect residual rho: corrupt=0.043, matched=0.102.
- Reason: Divergence layers have mean overlap near the held-out reference baseline; overlap is at most a weak within-site correlate, not a necessary condition.

The defensible claim is therefore `R does not imply A`. These runs do not support the stronger claim that low IVS is the cause, or a necessary signature, of mechanism bypass.
