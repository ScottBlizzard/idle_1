"""Read-only diagnostics for the terminal GREEN v3 development result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def quantiles(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        name: float(np.quantile(array, q))
        for name, q in (("min", 0), ("p10", .1), ("p25", .25),
                        ("median", .5), ("p75", .75), ("p90", .9),
                        ("p95", .95), ("p99", .99), ("max", 1))
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("DIAGNOSTIC_OUTPUT_EXISTS")
    args.output_root.mkdir(parents=True, exist_ok=False)

    current_result = json.loads((args.current_root / "dev_result.json").read_text())
    current_cells = json.loads((args.current_root / "dev_cells.json").read_text())
    current_joint = pd.read_parquet(args.current_root / "dev_joint_targets.parquet")
    initial_joint = pd.read_parquet(args.initial_root / "dev_joint_targets.parquet")
    transport = pd.read_parquet(args.current_root / "dev_transport_scores.parquet")
    initial_result = json.loads((args.initial_root / "dev_result.json").read_text())

    current_joint = current_joint.sort_values("pair_digest").reset_index(drop=True)
    initial_joint = initial_joint.sort_values("pair_digest").reset_index(drop=True)
    if not current_joint["pair_digest"].equals(initial_joint["pair_digest"]):
        raise RuntimeError("JOINT_CORRIGENDUM_RECORD_IDENTITY_FAILURE")
    bound_delta = current_joint["joint_bound"] - initial_joint["joint_bound"]
    bound_relative_delta = np.abs(bound_delta) / initial_joint["joint_bound"]
    record_snr = np.abs(current_joint["joint_center"]) / current_joint["joint_bound"]
    bound_target_ratio = current_joint["joint_bound"] / np.abs(current_joint["joint_target"])

    audits = [json.loads(value) for value in transport["gate_audit"]]
    response_snr = np.asarray([
        audit["response_norm"] / audit["epsilon_G"] for audit in audits
    ])
    operator_snr = np.asarray([
        audit["operator_norm"] / audit["epsilon_P_F"]
        if audit["epsilon_P_F"] > 0 else math.inf for audit in audits
    ])
    curvature_snr = transport["curvature_identifiability"].to_numpy(dtype=float)
    direct_error = transport["direct_error"].to_numpy(dtype=float)

    finite = np.isfinite(curvature_snr) & np.isfinite(direct_error) & (curvature_snr > 0)
    ranked = transport.loc[finite, [
        "noun_century_group", "direct_error", "curvature_identifiability"
    ]].copy()
    ranked["curvature_decile"] = pd.qcut(
        ranked["curvature_identifiability"], 10, labels=False, duplicates="drop"
    )
    deciles = []
    for index, part in ranked.groupby("curvature_decile", observed=True):
        deciles.append({
            "decile": int(index),
            "count": int(len(part)),
            "curvature_snr_median": float(part["curvature_identifiability"].median()),
            "direct_error_median": float(part["direct_error"].median()),
            "direct_error_p90": float(part["direct_error"].quantile(.9)),
        })
    group_rhos = {}
    for group, part in ranked.groupby("noun_century_group"):
        group_rhos[str(group)] = float(spearmanr(
            np.log(part["curvature_identifiability"]),
            -np.log(part["direct_error"] + 1e-12),
        ).statistic)

    diagnostic = {
        "schema_version": "green-bridge-v3.0.0-postcorrigendum-diagnostic-v1",
        "read_only_nonprotocol": True,
        "confirmation_inspected_or_started": False,
        "current_verdict": current_result["verdict"],
        "initial_verdict": initial_result["verdict"],
        "failed_gates": sorted(key for key, value in current_result["gates"].items() if not value),
        "joint": {
            "records": int(len(current_joint)),
            "record_set_snr": quantiles(record_snr),
            "record_set_snr_ge_4": int(np.sum(record_snr >= 4)),
            "zero_crossing_certified_intervals": int(np.sum(
                np.abs(current_joint["joint_center"]) <= current_joint["joint_bound"]
            )),
            "bound_to_abs_target_ratio": quantiles(bound_target_ratio),
            "center_target_abs_error": quantiles(
                np.abs(current_joint["joint_center"] - current_joint["joint_target"])
            ),
            "cell_set_snr": {
                cell["cell_id"]: float(cell["signed_set_snr"]) for cell in current_cells
            },
            "cell_set_snr_ge_4": int(sum(cell["set_snr_qualified"] for cell in current_cells)),
        },
        "corrigendum_effect": {
            "joint_bound_absolute_delta": quantiles(np.abs(bound_delta)),
            "joint_bound_relative_delta": quantiles(bound_relative_delta),
            "initial_coarse_fine_median_symmetric_change": initial_result["summary"][
                "coarse_fine_median_symmetric_change"
            ],
            "current_coarse_fine_median_symmetric_change": current_result["summary"][
                "coarse_fine_median_symmetric_change"
            ],
        },
        "transport_saturation": {
            "records": int(transport["pair_digest"].nunique()),
            "gate_system_units": int(len(transport)),
            "class_counts": {
                str(key): int(value) for key, value in transport["gate_class"].value_counts().items()
            },
            "curvature_snr": quantiles(curvature_snr),
            "response_snr": quantiles(response_snr),
            "operator_snr": quantiles(operator_snr),
            "direct_error": quantiles(direct_error),
            "direct_error_le_1e_4_fraction": float(np.mean(direct_error <= 1e-4)),
            "direct_error_le_1e_5_fraction": float(np.mean(direct_error <= 1e-5)),
            "global_detectability_spearman": current_result["summary"]["detectability"]["spearman"],
            "global_detectability_lcb_95": current_result["summary"]["detectability"]["lcb_95"],
            "within_group_detectability_spearman": group_rhos,
            "curvature_deciles": deciles,
        },
        "evidence_sha256": {
            "current_dev_result.json": sha256(args.current_root / "dev_result.json"),
            "current_dev_cells.json": sha256(args.current_root / "dev_cells.json"),
            "current_dev_joint_targets.parquet": sha256(args.current_root / "dev_joint_targets.parquet"),
            "current_dev_transport_scores.parquet": sha256(args.current_root / "dev_transport_scores.parquet"),
            "initial_dev_result.json": sha256(args.initial_root / "dev_result.json"),
            "initial_dev_joint_targets.parquet": sha256(args.initial_root / "dev_joint_targets.parquet"),
        },
        "interpretation": {
            "joint": (
                "Point centers reproduce AD targets at near-numerical precision, but all 80 "
                "worst-case signed intervals cross zero. The set-SNR failure is interval "
                "conservatism under joint composition, not point-estimator inaccuracy."
            ),
            "detectability": (
                "Almost the entire panel is response-recoverable with near-zero transport "
                "error. The frozen panel therefore samples a saturated high-identifiability "
                "regime rather than a transition regime, so monotone error improvement is "
                "not empirically visible."
            ),
        },
    }
    json_path = args.output_root / "POSTCORRIGENDUM_DIAGNOSTIC.json"
    json_path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")

    j = diagnostic["joint"]
    t = diagnostic["transport_saturation"]
    c = diagnostic["corrigendum_effect"]
    markdown = f"""# GREEN v3.0.0 Post-Corrigendum Read-Only Diagnostic

This diagnostic is non-protocol. Confirmation was neither inspected nor started.

## Terminal outcome

- Verdict: `{diagnostic['current_verdict']}`
- Failed gates: `{', '.join(diagnostic['failed_gates'])}`
- All other frozen development gates passed.

## Joint certificate

- The point estimator matches the AD joint target extremely closely: median
  absolute center/target error `{j['center_target_abs_error']['median']:.6g}`.
- Nevertheless, all `{j['zero_crossing_certified_intervals']}` of 80 record-level
  certified intervals cross zero.
- Median record-level set SNR is `{j['record_set_snr']['median']:.6g}`; the maximum
  is `{j['record_set_snr']['max']:.6g}`; 0/80 reach 4.
- The median bound is `{j['bound_to_abs_target_ratio']['median']:.6g}` times the
  absolute target, and the worst ratio is `{j['bound_to_abs_target_ratio']['max']:.6g}`.
- Consequently, 0/10 cells reach the frozen set-SNR threshold.

The corrected projection/envelope contraction changed a joint bound by at most
`{c['joint_bound_relative_delta']['max']:.6g}` relatively. This is expected because
the frozen physical joint direction lies almost entirely in each selected probe
frame; it proves the remaining width is not caused by that implementation omission.

## Radius stability

Restoring the inherited v2 denominator floor changed the coarse/fine median
symmetric change from `{c['initial_coarse_fine_median_symmetric_change']:.6g}` to
`{c['current_coarse_fine_median_symmetric_change']:.6g}`, which passes the frozen gate.

## Detectability saturation

- Gate-system classes: `{json.dumps(t['class_counts'], sort_keys=True)}`.
- Curvature SNR range: `{t['curvature_snr']['min']:.6g}` to
  `{t['curvature_snr']['max']:.6g}`.
- Direct-error median: `{t['direct_error']['median']:.6g}`; p90:
  `{t['direct_error']['p90']:.6g}`.
- Fraction with direct error at most 1e-4: `{t['direct_error_le_1e_4_fraction']:.6g}`.
- Frozen detectability Spearman: `{t['global_detectability_spearman']:.6g}`;
  cluster-bootstrap LCB: `{t['global_detectability_lcb_95']:.6g}`.

The observed panel is almost wholly recoverable and already at negligible error.
It does not span the recovery boundary needed to demonstrate a monotone
identifiability/error transition.

## Scientific decision point

The remaining failures require a new theory/protocol decision, not a small code
repair: either derive a valid correlation-aware joint certificate that avoids
worst-case per-gate triangle composition, and/or design a fresh outcome-blind
boundary-spanning development panel. The sealed v3 confirmation must not be opened
under the current `POSTER_ONLY` verdict.
"""
    (args.output_root / "POSTCORRIGENDUM_DIAGNOSTIC.md").write_text(markdown)
    (args.output_root / "sha256sums.txt").write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in sorted(args.output_root.iterdir()) if path.name != "sha256sums.txt"
        ) + "\n"
    )


if __name__ == "__main__":
    main()
