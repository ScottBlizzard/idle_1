# GREEN v2.0.0 implementation status and pre-launch blockers — 2026-08-25

## Scope and contamination status

- Binding decision: `GPTPRO_GREEN_V136_TERMINAL_DECISION_20260825.md`.
- Reviewed base: `3bdeac04a16724461f266705ef250a6357ced1cf`.
- Implementation branch: `codex/green-v200`.
- The official `outputs/green_bridge_v200` root has **not** been created.
- No v2.0.0 development or confirmation response has been evaluated.
- The server diagnostic below used one legacy donor record, outside the new
  v2.0.0 inferential population and outside the frozen 40-stratum AD panel.
- The immutable v1.3.6 output remains at
  `/mnt/sdb/ccj/iclr_1_runs/idle_1_green_bridge_v136/outputs/green_bridge_v136`.

## Locally completed implementation

The branch implements the authorized v2.0.0 identity, three fixed scales,
fine-Richardson point estimator, propagated compatibility bounds, four gate
classes, all-ten interval accounting, interval cell aggregation, robust
development/confirmation analysis, fresh group-level split construction,
versioned multigpu workers, an eight-GPU launcher, predecessor verification,
prepare/development/confirmation phase wiring, and the prepare-only float64 AD
module.

The 168 historical tests and 32 named v2.0.0 tests are present, for exactly 200
tests. After strengthening the split-hash test to hash the actual canonical
payload rather than merely checking the expected constant, 199 tests pass and
the one split-hash test fails for the authoritative mismatch below. There are
no skips, expected failures, or threshold monkeypatches.

## Blocker 1 — canonical split payload does not hash to the frozen digest

The natural payload implementing the exact field list in Section 10.4 is:

```json
{
  "schema": "green-bridge-v2.0.0-resplit-v1",
  "salt": "green-v200-resplit-20260825",
  "source_split": "green-bridge-v1.3.6-confirmation",
  "development_groups": [
    {"noun": "dynasty", "century": 16, "rank_key": "066e4d0fbd2636a5de7c5587fea60ba6d83c2173fd8a1a3b9598806973ed2596"},
    {"noun": "dynasty", "century": 12, "rank_key": "0fe1884c7d56deb8cdcb34d7b4eea65b398a9fdf03fc638f8bbc4a422c6ff6b6"},
    {"noun": "reign", "century": 14, "rank_key": "169c8a45b7aea24c90ce94ecefc84aa0588e34831b5074b9a57a4c5380373b51"},
    {"noun": "warfare", "century": 14, "rank_key": "36d63a7d439059d0877995705989132fddb35d1cfc9381be110925cacc8776c4"}
  ],
  "confirmation_groups": [
    {"noun": "treaty", "century": 12, "rank_key": "5419d9cb8844c61db83ae2eae7243dbd16a9c2bf5ee7967401eafd7f70f2475a"},
    {"noun": "warfare", "century": 12, "rank_key": "57822f1c018d9552848007996257f81da49ebef54f6e4559dc84fe13312ed2b4"},
    {"noun": "expedition", "century": 14, "rank_key": "5f5f6555263c3ee9052d9f1240096f6004091201fdd805b4ede1769481fcc321"},
    {"noun": "kingdom", "century": 12, "rank_key": "6c27075f448a87bd7bdb373924e72caba1816a5033625dbe92d1c59d1977dae8"},
    {"noun": "treaty", "century": 16, "rank_key": "6c8d9da9bd864657ac675f5b68f65e22f44ac97b0ccec1a4acfa8987a513fb77"},
    {"noun": "kingdom", "century": 16, "rank_key": "8571c8283f76806da63c769868b6a34448f6f02ae86d57f8a13db6597cecde00"},
    {"noun": "campaign", "century": 14, "rank_key": "9942a20d23a6fb97e7f33390172c7049ebe78341bde1b047a28ae64e997d431b"},
    {"noun": "siege", "century": 16, "rank_key": "a19e2bc49bf4b522ae28f500cd6596f5c492e8f817008f0c5985341e55c45741"},
    {"noun": "reign", "century": 12, "rank_key": "aa4cd1c743ab745f1278738367fcdd5a3937d36082f92b10aa3144c990001af4"},
    {"noun": "siege", "century": 14, "rank_key": "c39c88f5f37a424b7196cf99a4d34062f6150740ede6ab7c97abdd36d2d76d01"},
    {"noun": "campaign", "century": 16, "rank_key": "e1d35b6e9b3ec70687d8ed270afec74fc565c633b0d0a43011d507323df4f939"},
    {"noun": "expedition", "century": 16, "rank_key": "f7fcfdc5e4306cca1d4b0309c086dc0d6e033b72ce1236d8bb6d1986c362351f"}
  ],
  "distance_bins": ["near", "far"],
  "roles": ["tensor", "energy"],
  "records_per_role_per_cell": 8
}
```

Using the mandated serialization

```python
json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

produces:

```text
0873915c966bef8f54b83d4151a9d7c75b577da5dfc17ee093b9f5c58a9590f7
```

The decision instead freezes:

```text
f012a286801bc3e3e937b390f0a62d7e92f8d5a21ba59d7e53478ae911e72cfc
```

The document specifies the top-level field names but not the exact JSON value
shape for group rows, `source_split`, distance bins, or roles. More than 90,000
plausible schema-preserving representations were checked without finding the
frozen digest. The group allocation itself is unambiguous and validates to 8
development cells (64 tensor and 64 energy records) and 24 confirmation cells
(192 tensor and 192 energy records), with zero overlap with the old development
groups.

Required Pro action: provide the literal canonical JSON payload whose digest is
`f012...`, or issue a hash corrigendum to `087391...` while preserving the exact
group allocation.

## Blocker 2 — prescribed float32 Richardson balls fail a non-inferential smoke case

One legacy donor item, target system, gate slot 0 was evaluated at the frozen
base/half/quarter radii with `epsilon_y = 1e-7`. This is an implementation smoke
case, not evidence from the new development or confirmation split.

Coarse/fine ball-overlap ratios were:

| Object | center distance / summed radii |
|---|---:|
| G | 2.3718 |
| C | 3.0029 |
| J | 2.6480 |
| delta-H rows | 2.7523 to 3.1550 |

Every ratio exceeds one, so every dyadic overlap check fails under the literal
Section 11 implementation.

The independent float64 AD value was also outside both prescribed balls:

| Object | coarse AD distance | coarse bound | fine AD distance | fine bound |
|---|---:|---:|---:|---:|
| G | 0.00221915 | 0.00092192 | 0.00380546 | 0.00113385 |
| C | 0.07930465 | 0.01851914 | 0.26483988 | 0.06437722 |
| J | 0.00198734 | 0.00057760 | 0.00375826 | 0.00105197 |

For delta-H, four of five coarse rows and all five fine rows missed. Fine-row AD
distances ranged from `0.01142` to `0.03914`, while bounds ranged from `0.00337`
to `0.01036`.

This suggests that duplicate-output reproducibility with a floor of `1e-7`
does not bound float32 finite-difference quantization/truncation at the quarter
radius. The immutable fine estimator can be farther from AD than the coarse
estimator.

Required Pro action: decide whether this prepare STOP is scientifically
intended, or authorize a proof-derived numerical correction. A correction must
not be empirical pass-rate tuning. Plausible categories for Pro to assess are a
formal float32 roundoff/ULP term, a revised theorem-backed Richardson remainder
bound, or a different prepare-only enclosure construction that retains the
frozen scientific point estimator.

## AD route implementation finding (resolved locally)

Both AD routes execute successfully. Peak allocated memory was approximately
3.78 GB after forward-over-forward and 8.24 GB after reverse-over-forward, below
the 20 GB ceiling.

TransformerLens consults `model.cfg.dtype` for internal tensors. Casting only
parameters to float64 left this config at float32 and caused route discrepancies
around `1e-7`. Setting the prepare-only local-tail config dtype to float64, then
restoring it before scientific execution, reduced route differences to:

```text
G          1.56e-14
C          1.76e-15
J          8.15e-15
H_path     4.78e-16
H_control  2.54e-16
```

These satisfy the implemented outward-rounded route discrepancy bounds. This
is an engineering correction within the already-authorized float64 prepare-only
audit and does not change the float32 scientific estimator.

## Launch status

The official one-shot must not be launched until both blockers receive a
binding clarification/corrigendum. Starting now would deterministically stop at
the split firewall, before the scientific prepare audit, and would consume the
single authorized attempt without producing development evidence.
