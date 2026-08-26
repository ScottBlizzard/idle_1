"""Merge and score the immutable eight-GPU GREEN v3 development run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analyze_green_bridge_v300 import (
    BASELINES,
    group_balanced_rmse_v300,
    matched_gain_v300,
    select_frozen_baseline_v300,
)
from green_bridge_spec import sha256_file
from green_bridge_v300_development import DEVELOPMENT_AUTHORIZATION_ID, write_json
from green_bridge_v300_numerics import signed_set_snr_v300
from green_bridge_v300_spec import THRESHOLDS


BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260805


def _read_worker_tables(worker_root: Path):
    transports, joints, records, results = [], [], [], []
    for worker in range(8):
        for role in ("transport", "joint"):
            root = worker_root / f"worker_{worker:02d}" / role
            result = json.loads((root / "worker_result.json").read_text(encoding="utf-8"))
            if result.get("authorization_id") != DEVELOPMENT_AUTHORIZATION_ID:
                raise RuntimeError("DEVELOPMENT_AUTHORIZATION_ID_MISMATCH")
            if result.get("worker_index") != worker or result.get("role") != role:
                raise RuntimeError("DEVELOPMENT_WORKER_IDENTITY_MISMATCH")
            if result.get("records") != 10 or not result.get("active_model_unchanged"):
                raise RuntimeError("DEVELOPMENT_WORKER_COMPLETENESS_FAILURE")
            paths = {
                "transport": root / "transport_rows.parquet",
                "joint": root / "joint_rows.parquet",
                "record": root / "record_rows.parquet",
            }
            for name, key in (
                ("transport", "transport_rows_sha256"),
                ("joint", "joint_rows_sha256"),
                ("record", "record_rows_sha256"),
            ):
                if sha256_file(paths[name]) != result[key]:
                    raise RuntimeError(f"DEVELOPMENT_WORKER_HASH_FAILURE:{worker}:{role}:{name}")
            transport = pd.read_parquet(paths["transport"])
            joint = pd.read_parquet(paths["joint"])
            record = pd.read_parquet(paths["record"])
            if len(transport):
                transports.append(transport)
            if len(joint):
                joints.append(joint)
            records.append(record)
            results.append(result)
    transport = pd.concat(transports, ignore_index=True)
    joint = pd.concat(joints, ignore_index=True)
    record = pd.concat(records, ignore_index=True)
    if len(transport) != 1600 or len(joint) != 80 or len(record) != 160:
        raise RuntimeError(
            f"DEVELOPMENT_MERGE_COUNTS:{len(transport)}:{len(joint)}:{len(record)}"
        )
    if transport["pair_digest"].nunique() != 80 or joint["pair_digest"].nunique() != 80:
        raise RuntimeError("DEVELOPMENT_UNIQUE_RECORD_COUNT_FAILURE")
    if record["pair_digest"].nunique() != 160:
        raise RuntimeError("DEVELOPMENT_ROLE_PAIR_DISJOINTNESS_FAILURE")
    return transport, joint, record, results


def _cluster_bootstrap_spearman(rows: pd.DataFrame) -> dict:
    finite = rows[
        np.isfinite(rows["curvature_identifiability"])
        & np.isfinite(rows["direct_error"])
        & (rows["curvature_identifiability"] > 0)
    ].copy()
    finite["x"] = np.log(finite["curvature_identifiability"].astype(float))
    finite["y"] = -np.log(finite["direct_error"].astype(float) + 1.0e-12)
    rho = float(spearmanr(finite["x"], finite["y"]).statistic) if len(finite) >= 2 else math.nan
    finite["rx"] = rankdata(finite["x"], method="average")
    finite["ry"] = rankdata(finite["y"], method="average")
    groups = sorted(finite["noun_century_group"].unique())
    stats = []
    for group in groups:
        part = finite[finite["noun_century_group"] == group]
        x, y = part["rx"].to_numpy(), part["ry"].to_numpy()
        stats.append((len(part), x.sum(), y.sum(), np.square(x).sum(),
                      np.square(y).sum(), (x * y).sum()))
    values = np.asarray(stats, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
    counts = np.zeros((BOOTSTRAP_REPLICATES, len(groups)), dtype=np.float64)
    for index in range(len(groups)):
        counts[:, index] = np.sum(draws == index, axis=1)
    totals = counts @ values
    n, sx, sy, sx2, sy2, sxy = totals.T
    numerator = n * sxy - sx * sy
    denominator = np.sqrt(np.maximum(n * sx2 - sx * sx, 0)
                          * np.maximum(n * sy2 - sy * sy, 0))
    boot = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)
    return {
        "spearman": rho,
        "cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cluster_bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_count": len(groups),
        "lcb_95": float(np.nanpercentile(boot, 2.5)),
        "finite_units": len(finite),
    }


def _cluster_bootstrap_gain(rows: pd.DataFrame, baseline: str) -> dict:
    groups = sorted(rows["noun_century_group"].unique())
    matched_mse, baseline_mse = [], []
    for group in groups:
        part = rows[rows["noun_century_group"] == group]
        matched_mse.append(float(np.mean(np.square(part["error_matched"].astype(float)))))
        baseline_mse.append(float(np.mean(np.square(part[f"error_{baseline}"].astype(float)))))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
    matched = np.sqrt(np.mean(np.asarray(matched_mse)[draws], axis=1))
    base = np.sqrt(np.mean(np.asarray(baseline_mse)[draws], axis=1))
    gain = np.where(base == 0, np.where(matched == 0, 0.0, -np.inf), 1.0 - matched / base)
    return {
        "baseline": baseline,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "lcb_95": float(np.percentile(gain, 2.5)),
        "median": float(np.median(gain)),
    }


def _cell_rows(transport: pd.DataFrame, joint: pd.DataFrame,
               records: pd.DataFrame) -> list[dict]:
    result = []
    for cell_id in sorted(records["cell_id"].unique()):
        rec = records[records["cell_id"] == cell_id]
        t_rec = rec[rec["role"] == "transport"]
        j_rec = rec[rec["role"] == "joint"]
        t = transport[transport["cell_id"] == cell_id]
        j = joint[joint["cell_id"] == cell_id]
        admissible_joint = j[j["technically_admissible"]]
        center = float(admissible_joint["joint_center"].mean()) if len(admissible_joint) else 0.0
        bound = float(admissible_joint["joint_bound"].mean()) if len(admissible_joint) else math.inf
        target = float(admissible_joint["joint_target"].mean()) if len(admissible_joint) else 0.0
        target_bound = float(admissible_joint["target_bound"].mean()) if len(admissible_joint) else math.inf
        unresolved = float(admissible_joint["unresolved_bound"].mean()) if len(admissible_joint) else math.inf
        target_scale = max(abs(target), target_bound)
        unresolved_ratio = (0.0 if unresolved == 0 else math.inf) if target_scale == 0 else unresolved / target_scale
        nonnull = t[t["nonnull"]]
        recoverable_nonnull = nonnull[nonnull["gate_class"] == "recoverable"]
        result.append({
            "cell_id": cell_id,
            "distance_bin": str(rec["distance_bin"].iloc[0]),
            "noun": str(rec["noun"].iloc[0]),
            "century": int(rec["century"].iloc[0]),
            "transport_admissible_records": int(t_rec["technically_admissible"].sum()),
            "joint_admissible_records": int(j_rec["technically_admissible"].sum()),
            "survived": bool(t_rec["technically_admissible"].sum() >= 6
                             and j_rec["technically_admissible"].sum() >= 6),
            "nonnull_gate_system_units": len(nonnull),
            "recoverable_nonnull_fraction": (
                len(recoverable_nonnull) / len(nonnull) if len(nonnull) else 0.0
            ),
            "joint_center": center, "joint_bound": bound,
            "joint_target": target, "target_bound": target_bound,
            "signed_set_snr": signed_set_snr_v300(center, bound),
            "set_snr_qualified": signed_set_snr_v300(center, bound) >= 4.0,
            "unresolved_bound": unresolved,
            "unresolved_mass_ratio": unresolved_ratio,
            "unresolved_mass_narrow": unresolved_ratio <= 0.25,
        })
    return result


def merge_development_v300(worker_root: Path, output_root: Path) -> dict:
    transport, joint, records, worker_results = _read_worker_tables(worker_root)
    transport = transport.sort_values(
        ["cell_id", "pair_digest", "system", "gate_slot"]
    ).reset_index(drop=True)
    joint = joint.sort_values(["cell_id", "pair_digest"]).reset_index(drop=True)
    records = records.sort_values(["role", "cell_id", "pair_digest"]).reset_index(drop=True)
    cells = _cell_rows(transport, joint, records)
    cell_table = pd.DataFrame(cells)

    scoring = transport[
        (transport["gate_class"] == "recoverable")
        & transport["nonnull"]
        & transport["technically_admissible"]
    ].copy()
    finite_scoring = scoring.copy()
    for field in ["error_matched"] + [f"error_{name}" for name in BASELINES]:
        finite_scoring = finite_scoring[np.isfinite(finite_scoring[field])]
    frozen = select_frozen_baseline_v300(finite_scoring)
    matched_rmse = group_balanced_rmse_v300(finite_scoring, "error_matched")
    baseline_gains = {
        name: matched_gain_v300(matched_rmse, score)
        for name, score in frozen["group_balanced_rmse"].items()
    }
    selected = frozen["selected_baseline"]
    detectability = _cluster_bootstrap_spearman(
        transport[transport["nonnull"] & transport["technically_admissible"]]
    )
    gain_bootstrap = _cluster_bootstrap_gain(finite_scoring, selected)

    direct_errors = scoring["direct_error"].astype(float).to_numpy()
    admissible_joint = joint[joint["technically_admissible"] & joint["nonnull"]]
    joint_errors = admissible_joint["joint_error"].astype(float).to_numpy()
    null_leakage = transport[
        transport["technically_admissible"]
    ]["null_leakage"].astype(float).to_numpy()
    cell_fine = transport.groupby("cell_id")["direct_error"].mean()
    cell_coarse = transport.groupby("cell_id")["coarse_direct_error"].mean()
    common = cell_fine.index.intersection(cell_coarse.index)
    coarse_fine_spearman = float(spearmanr(cell_fine.loc[common], cell_coarse.loc[common]).statistic)
    symmetric = np.abs(cell_fine.loc[common] - cell_coarse.loc[common]) / np.maximum(
        np.maximum(np.abs(cell_fine.loc[common]), np.abs(cell_coarse.loc[common])),
        np.finfo(float).tiny,
    )
    gate_slot_counts = (
        transport[transport["gate_class"] == "recoverable"]
        .groupby("gate_slot").size().reindex(range(10), fill_value=0)
    )
    invalid = int(records["numerical_invalid_units"].sum())
    contradictions = int(records["structural_contradiction_units"].sum())
    total_gate_systems = 20 * len(records)
    resolved = int(records["resolved_gate_systems"].sum())
    nonnull = transport[transport["nonnull"]]
    recoverable_nonnull = nonnull[nonnull["gate_class"] == "recoverable"]
    surviving = cell_table[cell_table["survived"]]
    summary = {
        "schema_version": "green-bridge-v3.0.0-development-summary-v1",
        "authorization_id": DEVELOPMENT_AUTHORIZATION_ID,
        "transport_records": int(records[records["role"] == "transport"]["pair_digest"].nunique()),
        "joint_records": int(records[records["role"] == "joint"]["pair_digest"].nunique()),
        "cells": len(cell_table),
        "surviving_cells": len(surviving),
        "surviving_near_cells": int((surviving["distance_bin"] == "near").sum()),
        "surviving_far_cells": int((surviving["distance_bin"] == "far").sum()),
        "development_nouns_represented": int(records["noun"].nunique()),
        "numerical_invalid_units": invalid,
        "structural_contradiction_units": contradictions,
        "resolved_coverage": resolved / total_gate_systems,
        "nonnull_recoverability_fraction": (
            len(recoverable_nonnull) / len(nonnull) if len(nonnull) else 0.0
        ),
        "set_snr_qualified_cells": int(cell_table["set_snr_qualified"].sum()),
        "unresolved_mass_narrow_cells": int(cell_table["unresolved_mass_narrow"].sum()),
        "recoverable_units_by_gate_slot": {str(k): int(v) for k, v in gate_slot_counts.items()},
        "bound_failures": int((~transport["bound_valid"]).sum()),
        "direct_error_median": float(np.median(direct_errors)) if len(direct_errors) else math.inf,
        "direct_error_p90": float(np.percentile(direct_errors, 90)) if len(direct_errors) else math.inf,
        "joint_error_median": float(np.median(joint_errors)) if len(joint_errors) else math.inf,
        "joint_error_p90": float(np.percentile(joint_errors, 90)) if len(joint_errors) else math.inf,
        "detectability": detectability,
        "null_leakage_median": float(np.median(null_leakage)) if len(null_leakage) else math.inf,
        "null_leakage_p95": float(np.percentile(null_leakage, 95)) if len(null_leakage) else math.inf,
        "matched_group_balanced_rmse": matched_rmse,
        "frozen_baseline": frozen,
        "matched_gain_by_baseline": baseline_gains,
        "matched_best_baseline_gain": baseline_gains[selected],
        "gain_bootstrap": gain_bootstrap,
        "coarse_fine_cell_spearman": coarse_fine_spearman,
        "coarse_fine_median_symmetric_change": float(np.median(symmetric)),
    }
    gates = {
        "record_counts": summary["transport_records"] == 80 and summary["joint_records"] == 80,
        "surviving_cells": summary["surviving_cells"] >= 8,
        "surviving_near": summary["surviving_near_cells"] >= 4,
        "surviving_far": summary["surviving_far_cells"] >= 4,
        "all_nouns": summary["development_nouns_represented"] == 3,
        "no_invalid": invalid == 0,
        "no_contradiction": contradictions == 0,
        "coverage": summary["resolved_coverage"] >= 0.80,
        "recoverability": summary["nonnull_recoverability_fraction"] >= 0.25,
        "set_snr_cells": summary["set_snr_qualified_cells"] >= 6,
        "unresolved_mass": summary["unresolved_mass_narrow_cells"] >= 6,
        "gate_slot_coverage": bool((gate_slot_counts >= 10).all()),
        "bounds": summary["bound_failures"] == 0,
        "direct_median": summary["direct_error_median"] <= THRESHOLDS.development_direct_median_max,
        "direct_p90": summary["direct_error_p90"] <= THRESHOLDS.development_direct_p90_max,
        "joint_median": summary["joint_error_median"] <= THRESHOLDS.development_joint_median_max,
        "joint_p90": summary["joint_error_p90"] <= THRESHOLDS.development_joint_p90_max,
        "detectability_rho": detectability["spearman"] >= THRESHOLDS.development_detectability_spearman_min,
        "detectability_lcb": detectability["lcb_95"] > 0,
        "null_median": summary["null_leakage_median"] <= THRESHOLDS.development_null_leakage_median_max,
        "null_p95": summary["null_leakage_p95"] <= THRESHOLDS.development_null_leakage_p95_max,
        "best_baseline_gain": summary["matched_best_baseline_gain"] >= THRESHOLDS.development_best_baseline_gain_min,
        "every_baseline_gain": min(baseline_gains.values()) >= THRESHOLDS.development_every_baseline_gain_min,
        "gain_lcb": gain_bootstrap["lcb_95"] > 0,
        "coarse_fine_spearman": coarse_fine_spearman >= THRESHOLDS.coarse_fine_spearman_min,
        "coarse_fine_change": summary["coarse_fine_median_symmetric_change"] <= THRESHOLDS.coarse_fine_symmetric_change_max,
    }
    if all(gates.values()):
        verdict = "OPEN_CONFIRMATION"
    elif (gates["no_invalid"] and gates["no_contradiction"]
          and summary["surviving_cells"] >= 8
          and summary["direct_error_median"] <= 0.15
          and summary["direct_error_p90"] <= 0.35
          and summary["matched_best_baseline_gain"] >= 0.10):
        verdict = "POSTER_ONLY"
    else:
        verdict = "STOP_ORAL"
    result = {
        "schema_version": "green-bridge-v3.0.0-development-result-v1",
        "verdict": verdict, "gates": gates, "summary": summary,
        "development_started": True, "development_completed": True,
        "confirmation_started": False, "confirmation_authorized": False,
        "worker_results": worker_results,
    }

    transport.to_parquet(output_root / "dev_transport_scores.parquet", index=False)
    joint.to_parquet(output_root / "dev_joint_targets.parquet", index=False)
    cell_table.to_json(output_root / "dev_cells.json", orient="records", indent=2)
    write_json(output_root / "frozen_analysis.json", frozen | {
        "authorization_id": DEVELOPMENT_AUTHORIZATION_ID,
        "selected_global_radius": worker_results[0]["selected_global_radius"],
    })
    write_json(output_root / "dev_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = merge_development_v300(args.worker_root, args.output_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
