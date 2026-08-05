# Known-ground-truth synthetic cross-fit audit

| Seed | Trained independently | acc off/on | z AUROC | conformal AUROC | Clean/Donor conformal |
|--:|:--:|:--:|--:|--:|:--:|
| 0 | False | 1.000/1.000 | 0.9988 | 0.9994 | 0.813/0.005 |
| 1 | True | 1.000/1.000 | 0.9999 | 1.0000 | 0.430/0.002 |
| 2 | True | 1.000/1.000 | 1.0000 | 0.9993 | 0.720/0.002 |
| 3 | True | 1.000/1.000 | 1.0000 | 1.0000 | 0.573/0.002 |

## Decision

- Robust across model seeds: **True**.
- Mean/min AUROC: 0.9997/0.9988.
- Mean clean/donor ECDF overlap: 0.651/0.007.
- Composite conformal robust: **True**.
- Mean/min conformal AUROC: 0.9997/0.9993.
- Mean clean/donor conformal overlap: 0.634/0.003.
- Million-scale z is safe evidence: **False**.
- Defensible claim: When the target support is fixed by task design, cross-fitted empirical overlap separates on-support clean donors from off-support gate-on donors.
