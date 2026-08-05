# P0 NMH timing and site-confound audit

This analysis re-evaluates the existing NMH experiment without treating pooled AUROC as independent evidence. A residual-stream `resid_post` patch at layer L occurs after attention at layer L and can affect only NMH heads in strictly later layers.

| Model | Pooled AUROC | All-heads-downstream AUROC | Informative exact site strata | Weighted within-site AUROC | Stratified permutation p | Adjusted residual rho |
|:--|--:|--:|--:|--:|--:|--:|
| gpt2 | 0.894 | n/a | 0/14 | n/a | 1.000 | -0.017 |
| gpt2-medium | 0.778 | 0.714 | 1/36 | 0.500 | 1.000 | 0.060 |

## Interpretation rule

The pooled AUROC is considered timing/site-confounded unless the association survives both temporally eligible subsets and exact layer-position comparisons. An unavailable AUROC is itself informative when a matched subset contains only one mechanism class.

## gpt2

- `all_heads_downstream`: n=36, bypass=0, mean IVS=0.3475236346135465, mean NMH=0.594939806809028.
- `partial_heads_downstream`: n=8, bypass=4, mean IVS=0.17208133111485677, mean NMH=0.22652459144592285.
- `no_heads_downstream`: n=12, bypass=12, mean IVS=0.11423614875072731, mean NMH=0.0.

## gpt2-medium

- `all_heads_downstream`: n=92, bypass=9, mean IVS=0.30255387093935554, mean NMH=0.6622967337683329.
- `partial_heads_downstream`: n=16, bypass=12, mean IVS=0.17143583500535592, mean NMH=0.18077455634738726.
- `no_heads_downstream`: n=17, bypass=17, mean IVS=0.10647192504355026, mean NMH=0.0.
