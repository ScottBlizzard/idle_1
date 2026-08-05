# P0 Interventional Response Signature: GPT-2 aggregate

Runs: 3; prompt-layer rows: 2160.

| Layer | Mean R (min) | Mean NMH (max) | IRS normalized RMSE | IRS cosine | Center p | Endpoint accept (min) |
|--:|:--:|:--:|--:|--:|--:|:--:|
| 0 | 1.017 (1.011) | 1.021 (1.033) | 0.089 | 0.996 | 0.517 | 0.999 (0.998) |
| 1 | 1.037 (1.028) | 1.009 (1.022) | 0.140 | 0.990 | 0.518 | 0.999 (0.998) |
| 2 | 1.017 (1.011) | 0.958 (0.969) | 0.266 | 0.959 | 0.529 | 0.995 (0.986) |
| 3 | 0.978 (0.974) | 0.711 (0.726) | 0.534 | 0.838 | 0.529 | 1.000 (1.000) |
| 4 | 0.847 (0.830) | 0.304 (0.314) | 0.778 | 0.618 | 0.529 | 0.999 (0.997) |
| 5 | 0.825 (0.809) | 0.234 (0.259) | 0.957 | 0.404 | 0.528 | 0.999 (0.997) |
| 6 | 0.821 (0.802) | 0.246 (0.267) | 0.931 | 0.432 | 0.526 | 0.998 (0.995) |
| 7 | 0.819 (0.800) | 0.247 (0.267) | 0.797 | 0.583 | 0.530 | 0.997 (0.997) |
| 8 | 0.813 (0.794) | 0.269 (0.288) | 0.784 | 0.582 | 0.542 | 0.998 (0.997) |

## Decision-facing summary

- Stable high-R/low-A/admissible layers: `[4, 5, 6, 7]`.
- Mean IRS, stable divergence: 0.8657970593853681.
- Mean IRS, aligned layers: 0.16483962497891566.
- Layer/seed fixed-effect residual Spearman IRS vs NMH: -0.762.
- Prompt fixed-effect residual Spearman IRS vs NMH: -0.060.

A negative IRS-vs-NMH association is supportive only when high-R/low-A layers retain target-reference endpoint acceptance. IRS remains a local functional witness, not structural circuit identification.
