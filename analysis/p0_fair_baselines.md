# P0 fair baseline audit

This benchmark tests distribution classification only; labels are not mechanism bypass or causal-validity ground truth.

| Method | Overall AUROC | IO-position AUROC | Last-position AUROC |
|:--|--:|--:|--:|
| `crossfit_ivs` | 0.578 | 0.477 | 0.946 |
| `components_logistic` | 0.570 | 0.457 | 0.984 |
| `components_mlp` | 0.530 | 0.417 | 0.944 |
| `activations_logistic` | 0.539 | 0.413 | 1.000 |
| `activations_mlp` | 0.577 | 0.462 | 1.000 |

## Decision

- IVS has a clear advantage under the matched protocol: **False**.
- Best supervised method: `activations_mlp` (mean AUROC 0.577).
- The old MLP result supports a high-dimensional-overfit claim: **False**.
- Reason: With standardized features, identical splits, and equal tuning, supervised baselines are competitive overall and perfectly separate the last-position shift in activation space.

The method-level averages mix two regimes: all detectors are near chance at IO sites, while all detect the last-position clean/corrupt shift. This benchmark therefore does not establish causal validity or mechanism bypass.
