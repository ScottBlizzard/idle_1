# P0 IRS frozen stress-test summary

Probe robustness pass: **False**.

| Interpolation | Probes | Rank vs primary | Rank vs NMH | Min support | Pass |
|--:|--:|--:|--:|--:|:--:|
| 0.10 | 4 | 0.967 | -0.950 | 0.969 | True |
| 0.10 | 8 | 0.983 | -0.983 | 0.966 | True |
| 0.10 | 16 | 0.983 | -0.983 | 0.968 | True |
| 0.25 | 4 | 0.967 | -0.950 | 0.997 | True |
| 0.25 | 8 | 1.000 | -0.967 | 0.997 | True |
| 0.25 | 16 | 1.000 | -0.967 | 0.998 | True |
| 0.50 | 4 | 0.817 | -0.850 | 0.978 | False |
| 0.50 | 8 | 0.783 | -0.833 | 0.975 | False |
| 0.50 | 16 | 0.767 | -0.850 | 0.975 | False |

## Corruption stability

- IRS/single cross-corruption rank: 0.783/0.817.
- IRS/single relative L2 change: 0.804/0.858.
- Relative-change ratio IRS/single: 0.937.
- Minimum endpoint acceptance IRS/single: 0.997/0.613.
- Stability non-inferior: **True**.
- Clear stability superiority: **False**.

## Frozen novelty decision

Oral-level method novelty established by these two tests: **False**.
