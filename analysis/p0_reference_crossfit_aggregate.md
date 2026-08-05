# P0 reference-distribution and cross-fit audit

Runs: 3 seeds; retained site observations: 42.

| Reference | Mean overlap-z | 2.5%–97.5% | Below historical 0.3 | Max |recon-z| | Max floor span |
|:--|--:|:--:|--:|--:|--:|
| `corrupt_observational` | 0.418 | 0.140–0.519 | 9/42 | 4.94 | 0 |
| `clean_source` | 0.503 | 0.435–0.525 | 0/42 | 0.306 | 0 |
| `mixture` | 0.481 | 0.357–0.542 | 0/42 | 0.834 | 0 |
| `matched_semantic_counterfactual` | 0.492 | 0.457–0.529 | 0/42 | 0.406 | 0 |

## Decision

- Million-scale reconstruction gap survives: **False**.
- Historical low-overlap labels are invariant to reference: **False**.
- The defensible estimand at this stage is **conditional target-reference overlap**, not general causal validity.

## Reference sensitivity

- `corrupt_vs_clean_source`: mean paired change=0.085; historical labels changed=9; Spearman rho=0.296.
- `corrupt_vs_mixture`: mean paired change=0.063; historical labels changed=9; Spearman rho=0.359.
- `corrupt_vs_matched_semantic_counterfactual`: mean paired change=0.073; historical labels changed=9; Spearman rho=-0.291.

The historical 0.3 cutoff is shown only for continuity. It was not re-tuned on these runs and must not be interpreted as a preregistered causal-validity boundary.
