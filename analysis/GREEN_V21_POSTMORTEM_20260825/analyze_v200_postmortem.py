"""Read-only GREEN v2.0.0 postmortem required before v3 prepare.

This script never writes to the official v2 root or to the committed archive.
CPU reconstruction is complete from the selected archive.  Exact AD transport,
joint-composition, and signed-field recomputation are merged from immutable GPU
shards produced by ``green_bridge_v300_postmortem_worker.py``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


POSTMORTEM_COMMIT = "ef09fce529553d5a3d236852a288cde02b88418a"
EXECUTION_COMMIT = "e52e082296c33a10557636706e572147136fce34"
OFFICIAL_VERDICT = "STOP_ORAL"
FIRST_FAILED_GATE = "12_DEVELOPMENT_SURVIVAL"
EPSILON_Y = 1.0e-7
FINE_H_Z = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    """Normalize NumPy scalars emitted by pandas before canonical JSON output."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_sha256s(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line)
        if not match:
            raise ValueError(f"invalid checksum line: {line!r}")
        rows[match.group(2)] = match.group(1)
    return rows


def source_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in paths if path.is_file()}


def common(schema: str, script: Path, sources: dict[str, str]) -> dict:
    return {
        "postmortem_schema": schema,
        "postmortem_commit": POSTMORTEM_COMMIT,
        "official_execution_commit": EXECUTION_COMMIT,
        "official_verdict": OFFICIAL_VERDICT,
        "official_verdict_unchanged": True,
        "confirmation_data_accessed": False,
        "usable_for_threshold_selection": False,
        "source_artifact_sha256": dict(sorted(sources.items())),
        "analysis_script_sha256": sha256_file(script),
    }


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))])
    if not len(array):
        return {name: None for name in ("min", "p25", "median", "p75", "p90", "p95", "p99", "max")}
    return {name: float(np.quantile(array, q)) for name, q in (
        ("min", 0), ("p25", .25), ("median", .5), ("p75", .75),
        ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1),
    )}


def _decode(value):
    return json.loads(value) if isinstance(value, str) else value


def integrity_reconstruction(archive: Path, formal_root: Path | None, output: Path, script: Path) -> dict:
    expected = parse_sha256s(archive / "sha256sums.txt")
    checked, missing, mismatches = {}, [], {}
    for name, digest in expected.items():
        candidates = [archive / name]
        if formal_root is not None:
            candidates.append(formal_root / name)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            missing.append(name)
            continue
        actual = sha256_file(path)
        checked[name] = actual
        if actual != digest:
            mismatches[name] = {"expected": digest, "actual": actual, "path": str(path)}
    tensor = pd.read_parquet(archive / "dev_tensor_scores.parquet")
    energy = pd.read_parquet(archive / "dev_energy_targets.parquet")
    gates = [gate for value in tensor["mixed_audit"] for system in _decode(value).values() for gate in system["gates"]]
    labels = Counter(gate["label"] for gate in gates)
    result = json.loads((archive / "result.json").read_text(encoding="utf-8"))
    dev_result = json.loads((archive / "dev_result.json").read_text(encoding="utf-8"))
    cells = json.loads((archive / "dev_cells.json").read_text(encoding="utf-8"))
    ledger = json.loads((archive / "run_ledger.json").read_text(encoding="utf-8"))
    confirmation_names = (
        "confirm_transport_scores.parquet", "confirm_joint_targets.parquet",
        "confirm_cells.json", "confirm_result.json", "confirmation_anchor_cache.pt",
        "confirm_tensor_scores.parquet", "confirm_energy_targets.parquet",
    )
    confirmation_present = [name for name in confirmation_names
                            if (archive / name).exists() or (formal_root is not None and (formal_root / name).exists())]
    exact = (
        not mismatches and len(tensor) == 64 and len(energy) == 64
        and len(cells["cells"]) == 8 and len(gates) == 1280
        and sum(labels.values()) == 1280
        and result["verdict"] == dev_result["verdict"] == OFFICIAL_VERDICT
        and result["first_failed_gate"] == FIRST_FAILED_GATE
        and not ledger["confirmation_started"] and not confirmation_present
    )
    if missing and formal_root is not None:
        exact = False
    payload = common("green-v21-postmortem-integrity-v1", script, {
        "sha256sums.txt": sha256_file(archive / "sha256sums.txt"),
        "result.json": sha256_file(archive / "result.json"),
        "dev_tensor_scores.parquet": sha256_file(archive / "dev_tensor_scores.parquet"),
        "dev_energy_targets.parquet": sha256_file(archive / "dev_energy_targets.parquet"),
    }) | {
        "all_hashes_passed": not mismatches and (not missing if formal_root is not None else True),
        "selected_archive_hashes_passed": not mismatches,
        "checked_artifact_count": len(checked),
        "missing_from_selected_archive_or_formal_root": missing,
        "hash_mismatches": mismatches,
        "tensor_rows": len(tensor), "energy_rows": len(energy), "cells": len(cells["cells"]),
        "gate_system_rows": len(gates), "class_counts": dict(sorted(labels.items())),
        "verdict": result["verdict"], "first_failed_gate": result["first_failed_gate"],
        "confirmation_open": False, "confirmation_started": ledger["confirmation_started"],
        "confirmation_artifacts_present": confirmation_present,
        "integrity_reconstruction_passed": exact,
    }
    write_json(output / "01_integrity_reconstruction.json", payload)
    if not exact:
        raise RuntimeError("STOP: postmortem integrity reconstruction failed")
    return payload


def gate_certificate_decomposition(archive: Path, output: Path, script: Path) -> tuple[pd.DataFrame, dict]:
    tensor = pd.read_parquet(archive / "dev_tensor_scores.parquet")
    rows = []
    fields = (
        "gate_slot", "gate_index", "label", "reason", "curvature_norm", "gate_response_norm",
        "epsilon_C", "epsilon_G", "epsilon_J", "inverse_lower_bound", "inverse_admissible",
        "epsilon_P_F", "null_bound", "unresolved_bound", "dyadic_overlap", "ad_route_passed",
        "whitebox_coordinate_error", "contribution_center", "contribution_error",
        "contribution_lower", "contribution_upper",
    )
    for _, item in tensor.iterrows():
        audit = _decode(item["mixed_audit"])
        for system_name, system in audit.items():
            for gate in system["gates"]:
                row = {"pair_digest": item["pair_digest"], "cell_id": item["cell_id"],
                       "distance_bin": item["distance_bin"], "orientation": item["orientation"],
                       "system": system_name}
                for name in fields:
                    row[name] = gate.get(name)
                row["epsilon_delta_H"] = canonical_json(gate.get("epsilon_delta_H"))
                row["epsilon_A"] = canonical_json(gate.get("epsilon_A"))
                row["ad_matched_bypass"] = canonical_json(gate.get("ad_matched_bypass"))
                for name in ("factorization", "whitebox", "whitebox_factorization", "shift"):
                    row[name] = canonical_json(gate.get(name))
                epsilon_c = gate.get("epsilon_C")
                epsilon_g = gate.get("epsilon_G")
                row["s_C"] = (None if epsilon_c is None else
                              (math.inf if epsilon_c == 0 and gate.get("curvature_norm", 0) > 0 else
                               0.0 if epsilon_c == 0 else gate["curvature_norm"] / epsilon_c))
                row["s_G"] = (None if epsilon_g is None else
                              (math.inf if epsilon_g == 0 and gate.get("gate_response_norm", 0) > 0 else
                               0.0 if epsilon_g == 0 else gate["gate_response_norm"] / epsilon_g))
                a = gate.get("A")
                eps_p = gate.get("epsilon_P_F")
                operator_norm = None if a is None or gate.get("gate_response_norm") is None else float(np.linalg.norm(a)) * gate["gate_response_norm"]
                row["operator_norm_reconstructed"] = operator_norm
                row["s_P"] = None if operator_norm is None or eps_p in (None, 0) else operator_norm / eps_p
                row["field_not_applicable_reason"] = None if a is not None else f"classifier branch {gate.get('reason')} did not serialize A"
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "02_gate_certificate_rows.parquet", index=False)
    group_columns = ["gate_slot", "gate_index", "system", "distance_bin", "orientation", "cell_id", "label", "reason"]
    grouped = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        grouped.append(dict(zip(group_columns, keys)) | {
            "count": len(group), "s_C": quantiles(group["s_C"]),
            "s_G": quantiles(group["s_G"]), "s_P": quantiles(group["s_P"]),
        })
    summary = common("green-v21-gate-certificate-summary-v1", script, {
        "dev_tensor_scores.parquet": sha256_file(archive / "dev_tensor_scores.parquet")
    }) | {
        "rows": len(frame), "label_counts": frame["label"].value_counts().sort_index().to_dict(),
        "reason_counts": frame["reason"].value_counts().sort_index().to_dict(),
        "active_gate_histogram": dict(sorted(Counter(
            system["active_gates"] for value in tensor["mixed_audit"] for system in _decode(value).values()
        ).items())),
        "summary_groups": grouped,
    }
    write_json(output / "02_gate_certificate_summary.json", summary)
    return frame, summary


def uncertainty_decomposition(gates: pd.DataFrame, archive: Path, output: Path, script: Path) -> tuple[pd.DataFrame, dict]:
    tensor = pd.read_parquet(
        archive / "dev_tensor_scores.parquet",
        columns=["pair_digest", "residual_radius"],
    ).set_index("pair_digest")
    rows = []
    for _, gate in gates.iterrows():
        h_x = float(tensor.loc[gate["pair_digest"], "residual_radius"]) * 0.5
        raw_source = tensor_gate_lookup(
            archive, gate["pair_digest"], gate["system"], int(gate["gate_slot"])
        )
        objects = [
            ("G", gate["epsilon_G"], 10 * 3 * EPSILON_Y / FINE_H_Z,
             _decode(gate["ad_matched_bypass"]), raw_source),
            ("C", gate["epsilon_C"], 10 * 64 * EPSILON_Y / (3 * FINE_H_Z**2),
             _decode(gate["ad_matched_bypass"]), raw_source),
            ("J", gate["epsilon_J"], math.sqrt(500) * 3 * EPSILON_Y / h_x,
             _decode(gate["ad_matched_bypass"]), raw_source),
        ]
        route_map = {"G": "ad_route_radius_G", "C": "ad_route_radius_C", "J": "ad_route_radius_J"}
        for name, total, endpoint, _, raw in objects:
            if total is None:
                continue
            route = float(raw[route_map[name]])
            fd = max(0.0, float(total) - route - endpoint)
            denom = float(total)
            rows.append({"pair_digest": gate["pair_digest"], "system": gate["system"],
                         "gate_slot": int(gate["gate_slot"]), "object": name,
                         "finite_radius": fd, "route": route, "endpoint": endpoint,
                         "epsilon": denom, "fraction_fd": fd / denom if denom else 0,
                         "fraction_route": route / denom if denom else 0,
                         "fraction_endpoint": endpoint / denom if denom else 0})
        eps_h = _decode(gate["epsilon_delta_H"])
        route_h = raw_source.get("ad_route_radius_delta_H") or []
        endpoint_h = 20 * 17 * EPSILON_Y / (3 * h_x * FINE_H_Z)
        for index, total in enumerate(eps_h or []):
            route = float(route_h[index]); fd = max(0.0, float(total) - route - endpoint_h)
            rows.append({"pair_digest": gate["pair_digest"], "system": gate["system"],
                         "gate_slot": int(gate["gate_slot"]), "object": f"delta_H_{index}",
                         "finite_radius": fd, "route": route, "endpoint": endpoint_h,
                         "epsilon": float(total), "fraction_fd": fd / total if total else 0,
                         "fraction_route": route / total if total else 0,
                         "fraction_endpoint": endpoint_h / total if total else 0})
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "03_uncertainty_source_rows.parquet", index=False)
    summary = common("green-v21-uncertainty-source-summary-v1", script, {
        "dev_tensor_scores.parquet": sha256_file(archive / "dev_tensor_scores.parquet"),
        "noise_audit_dev.json": sha256_file(archive / "noise_audit_dev.json"),
    }) | {
        "rows": len(frame),
        "by_object": {name: {field: quantiles(group[field]) for field in
                              ("finite_radius", "route", "endpoint", "fraction_fd", "fraction_route", "fraction_endpoint")}
                      for name, group in frame.groupby("object")},
        "global_fraction_quantiles": {field: quantiles(frame[field]) for field in
                                      ("fraction_fd", "fraction_route", "fraction_endpoint")},
    }
    write_json(output / "03_uncertainty_source_summary.json", summary)
    return frame, summary


_TENSOR_AUDIT_CACHE: dict[str, dict[str, Any]] = {}


def tensor_gate_lookup(archive: Path, digest: str, system: str, gate_slot: int) -> dict:
    archive_key = str(archive.resolve())
    if archive_key not in _TENSOR_AUDIT_CACHE:
        tensor = pd.read_parquet(archive / "dev_tensor_scores.parquet", columns=["pair_digest", "mixed_audit"])
        _TENSOR_AUDIT_CACHE[archive_key] = {
            row.pair_digest: _decode(row.mixed_audit) for row in tensor.itertuples(index=False)
        }
    return _TENSOR_AUDIT_CACHE[archive_key][digest][system]["gates"][gate_slot]


def set_snr_geometry(archive: Path, output: Path, script: Path) -> dict:
    tensor = pd.read_parquet(archive / "dev_tensor_scores.parquet")
    rows = []
    for cell_id, group in tensor[tensor["point_complete"]].groupby("cell_id"):
        tar = (float(group["theta_tar_lower"].mean()), float(group["theta_tar_upper"].mean()))
        pat = (float(group["theta_pat_lower"].mean()), float(group["theta_pat_upper"].mean()))
        signed = (pat[0] - tar[1], pat[1] - tar[0])
        upper = max(abs(signed[0]), abs(signed[1]))
        absolute = (0.0, upper) if signed[0] <= 0 <= signed[1] else tuple(sorted((abs(signed[0]), abs(signed[1]))))
        midpoint = sum(absolute) / 2; half = (absolute[1] - absolute[0]) / 2
        snr = midpoint / half if half else (math.inf if midpoint else 0.0)
        rows.append({"cell_id": cell_id, "n": len(group), "tar_interval": tar,
                     "pat_interval": pat, "signed_interval": signed,
                     "crosses_zero": signed[0] <= 0 <= signed[1],
                     "absolute_interval": absolute, "midpoint": midpoint,
                     "half_width": half, "snr": snr})
    payload = common("green-v21-set-snr-geometry-v1", script, {
        "dev_tensor_scores.parquet": sha256_file(archive / "dev_tensor_scores.parquet")
    }) | {"cells": rows, "all_nonempty_zero_crossings_have_snr_one": all(
        row["crosses_zero"] and abs(row["snr"] - 1) <= 1e-15 for row in rows if row["half_width"] > 0
    )}
    write_json(output / "10_set_snr_geometry.json", payload)
    return payload


def aggregation_functionals(archive: Path, output: Path, script: Path, gpu_rows: pd.DataFrame | None) -> dict:
    tensor = pd.read_parquet(archive / "dev_tensor_scores.parquet")[
        ["pair_digest", "cell_id", "distance_bin", "orientation"]
    ]
    energy = pd.read_parquet(archive / "dev_energy_targets.parquet")
    rows = []
    for _, row in energy.iterrows():
        systems = _decode(row["systems"])
        rows.append({"pair_digest": row["pair_digest"], "cell_id": row["cell_id"],
                     "distance_bin": row["distance_bin"], "orientation": row["orientation"],
                     "role": "energy", "source": "independent_finite_energy",
                     "signed": float(systems["pat"]["full"] - systems["tar"]["full"]),
                     "magnitude_only": False})
    if gpu_rows is not None and len(gpu_rows):
        source_columns = {
            "behavioral": "signed_behavioral", "pie": "signed_pie",
            "single": "signed_single", "matched_bypass": "signed_matched_bypass",
            "independent_joint_ad": "signed_independent_joint_target",
        }
        for row in gpu_rows.itertuples(index=False):
            for source, name in source_columns.items():
                value = getattr(row, name)
                rows.append({
                    "pair_digest": row.pair_digest, "cell_id": row.cell_id,
                    "distance_bin": row.distance_bin, "orientation": row.orientation,
                    "role": "tensor", "source": source, "signed": float(value),
                    "magnitude_only": False,
                })
    frame = pd.DataFrame(rows)
    summaries = []
    for (cell, source), group in frame.groupby(["cell_id", "source"]):
        signed = group["signed"].astype(float).to_numpy()
        A = abs(float(np.mean(signed)))
        M = float(np.mean(np.abs(signed)))
        R = float(np.sqrt(np.mean(signed**2)))
        summaries.append({
            "cell_id": cell, "source": source, "n": len(group),
            "distance_bin": group["distance_bin"].iloc[0],
            "A_abs_signed_mean": A, "M_mean_absolute": M, "R_rms": R,
            "cancellation_ratio": A / max(M, 1e-12),
            "sign_stability_fraction": max(
                float(np.mean(signed >= 0)), float(np.mean(signed <= 0))
            ),
        })
    cell_frame = pd.DataFrame(summaries)

    def grouped_functionals(columns: list[str]) -> list[dict]:
        result = []
        for keys, group in frame.groupby(columns, sort=True):
            values = group["signed"].astype(float).to_numpy()
            A = abs(float(np.mean(values))); M = float(np.mean(np.abs(values)))
            result.append(dict(zip(columns, keys if isinstance(keys, tuple) else (keys,))) | {
                "n": len(values), "A_abs_signed_mean": A, "M_mean_absolute": M,
                "R_rms": float(np.sqrt(np.mean(values**2))),
                "cancellation_ratio": A / max(M, 1e-12),
                "sign_stability_fraction": max(
                    float(np.mean(values >= 0)), float(np.mean(values <= 0))
                ),
            })
        return result

    comparisons = []
    sources = sorted(cell_frame["source"].unique())
    for functional in ("A_abs_signed_mean", "M_mean_absolute", "R_rms"):
        pivot = cell_frame.pivot(index="cell_id", columns="source", values=functional)
        for left_index, left in enumerate(sources):
            for right in sources[left_index + 1:]:
                pair = pivot[[left, right]].dropna()
                x = pair[left].to_numpy(dtype=float); y = pair[right].to_numpy(dtype=float)
                pearson = float(np.corrcoef(x, y)[0, 1]) if len(pair) >= 2 and np.std(x) and np.std(y) else None
                spearman = float(pair[left].corr(pair[right], method="spearman")) if len(pair) >= 2 else None
                comparisons.append({
                    "left": left, "right": right, "functional": functional, "cells": len(pair),
                    "pearson": pearson, "spearman": spearman,
                    "rmse": float(np.sqrt(np.mean((x - y) ** 2))) if len(pair) else None,
                })
    frame.to_parquet(output / "08_aggregation_functionals.parquet", index=False)
    payload = common("green-v21-aggregation-functionals-v1", script, {
        "dev_tensor_scores.parquet": sha256_file(archive / "dev_tensor_scores.parquet"),
        "dev_energy_targets.parquet": sha256_file(archive / "dev_energy_targets.parquet"),
    }) | {
        "items": len(frame), "cells": summaries, "pairwise_cell_metrics": comparisons,
        "near_far_summaries": grouped_functionals(["source", "distance_bin"]),
        "orientation_summaries": grouped_functionals(["source", "orientation"]),
        "signed_sources": sources,
        "signed_behavioral_pie_single_complete": bool(
            gpu_rows is not None and len(gpu_rows)
            and gpu_rows[["signed_behavioral", "signed_pie", "signed_single"]].notna().all().all()
        ),
        "complete": set(sources) == {
            "behavioral", "pie", "single", "matched_bypass",
            "independent_joint_ad", "independent_finite_energy",
        },
    }
    write_json(output / "08_aggregation_functionals.json", payload)
    return payload


def role_sampling_audit(archive: Path, output: Path, script: Path, gpu_rows: pd.DataFrame | None) -> dict:
    energy = pd.read_parquet(archive / "dev_energy_targets.parquet")
    energy = energy.copy()
    energy["signed"] = energy["systems"].map(
        lambda value: float(_decode(value)["pat"]["full"] - _decode(value)["tar"]["full"])
    )
    replicates, seed = 100_000, 20260825
    rng = np.random.default_rng(seed)
    cells = []
    if gpu_rows is not None and "signed_independent_joint_target" in gpu_rows:
        for cell in sorted(energy["cell_id"].unique()):
            official_rows = energy[energy["cell_id"] == cell]
            same_rows = gpu_rows[gpu_rows["cell_id"] == cell]
            official = official_rows["signed"].to_numpy(dtype=float)
            same = same_rows["signed_independent_joint_target"].to_numpy(dtype=float)
            official_boot = official[rng.integers(0, len(official), size=(replicates, len(official)))].mean(axis=1)
            same_boot = same[rng.integers(0, len(same), size=(replicates, len(same)))].mean(axis=1)
            shifts = same_boot - official_boot

            def orientation_balanced(frame: pd.DataFrame, value: str) -> float:
                return float(frame.groupby("orientation")[value].mean().mean())

            official_mean = float(np.mean(official)); same_mean = float(np.mean(same))
            cells.append({
                "cell_id": cell, "noun_century_cluster": "|".join(cell.split("|")[:2]),
                "distance_bin": cell.split("|")[-1],
                "n_official": len(official), "n_same_role": len(same),
                "official_disjoint_signed_mean": official_mean,
                "same_role_signed_mean": same_mean,
                "pooled_role_signed_mean": float(np.mean(np.concatenate((official, same)))),
                "official_orientation_balanced_mean": orientation_balanced(official_rows, "signed"),
                "same_role_orientation_balanced_mean": orientation_balanced(
                    same_rows, "signed_independent_joint_target"
                ),
                "role_shift": same_mean - official_mean,
                "role_shift_item_bootstrap_interval": [
                    float(np.quantile(shifts, 0.025)), float(np.quantile(shifts, 0.975))
                ],
                "official_disjoint_standard_error": float(np.std(official, ddof=1) / math.sqrt(len(official))),
                "same_role_standard_error": float(np.std(same, ddof=1) / math.sqrt(len(same))),
            })
    cluster_rows = []
    for cluster in sorted({row["noun_century_cluster"] for row in cells}):
        selected = [row for row in cells if row["noun_century_cluster"] == cluster]
        cluster_rows.append({
            "noun_century_cluster": cluster, "cells": len(selected),
            "official_disjoint_signed_mean": float(np.mean([row["official_disjoint_signed_mean"] for row in selected])),
            "same_role_signed_mean": float(np.mean([row["same_role_signed_mean"] for row in selected])),
            "role_shift": float(np.mean([row["role_shift"] for row in selected])),
        })
    cluster_shifts = np.asarray([row["role_shift"] for row in cluster_rows], dtype=float)
    if len(cluster_shifts):
        cluster_boot = cluster_shifts[
            rng.integers(0, len(cluster_shifts), size=(replicates, len(cluster_shifts)))
        ].mean(axis=1)
        overall_cluster_interval = [
            float(np.quantile(cluster_boot, 0.025)), float(np.quantile(cluster_boot, 0.975))
        ]
    else:
        overall_cluster_interval = [None, None]
    distance_rows = []
    for distance in sorted({row["distance_bin"] for row in cells}):
        selected = [row for row in cells if row["distance_bin"] == distance]
        distance_rows.append({
            "distance_bin": distance,
            "official_disjoint_signed_mean": float(np.mean([row["official_disjoint_signed_mean"] for row in selected])),
            "same_role_signed_mean": float(np.mean([row["same_role_signed_mean"] for row in selected])),
            "pooled_role_signed_mean": float(np.mean([row["pooled_role_signed_mean"] for row in selected])),
            "role_shift": float(np.mean([row["role_shift"] for row in selected])),
        })
    payload = common("green-v21-role-sampling-audit-v1", script, {
        "dev_energy_targets.parquet": sha256_file(archive / "dev_energy_targets.parquet")
    }) | {
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "cluster": "noun-century group", "cells": cells,
        "noun_century_cluster_summaries": cluster_rows,
        "overall_role_shift_cluster_bootstrap_interval": overall_cluster_interval,
        "distance_stratified_summaries": distance_rows,
        "same_role_reconstruction_complete": bool(gpu_rows is not None and len(gpu_rows)),
        "pooled_role_complete": bool(cells), "orientation_balanced_complete": bool(cells),
        "distance_stratified_complete": bool(distance_rows),
        "complete": len(cells) == 8 and len(cluster_rows) == 4 and len(distance_rows) == 2,
    }
    write_json(output / "09_role_sampling_audit.json", payload)
    return payload


def regime_bridge(archive: Path, v136: Path, output: Path, script: Path) -> dict:
    rows = []
    version_details = {}
    for version, root in (("v136", v136), ("v200", archive)):
        tensor = pd.read_parquet(root / "dev_tensor_scores.parquet")
        energy = pd.read_parquet(root / "dev_energy_targets.parquet")
        energy = energy.copy()
        energy["signed_target"] = energy["systems"].map(
            lambda value: float(_decode(value)["pat"]["full"] - _decode(value)["tar"]["full"])
        )
        energy_by_cell = energy.groupby("cell_id")["signed_target"].mean().to_dict()
        global_labels, curvature, response, factorization = Counter(), [], [], []
        for _, item in tensor.iterrows():
            audit = _decode(item["mixed_audit"])
            gates = [gate for system in audit.values() for gate in system["gates"]]
            labels = Counter(gate["label"] for gate in gates); global_labels.update(labels)
            curvature.extend(float(gate["curvature_norm"]) for gate in gates if gate.get("curvature_norm") is not None)
            response.extend(float(gate["gate_response_norm"]) for gate in gates if gate.get("gate_response_norm") is not None)
            for gate in gates:
                if version == "v136" and gate.get("factorization_residual") is not None:
                    factorization.append(float(gate["factorization_residual"]))
                elif gate.get("factorization"):
                    factorization.append(float(gate["factorization"]["max_ratio"]))
            if version == "v136":
                signed_matched = float(item["theta_pat"] - item["theta_tar"])
            else:
                signed_matched = float(item["theta_pat_center"] - item["theta_tar_center"])
            rows.append({
                "version": version, "pair_digest": item["pair_digest"],
                "cell_id": item["cell_id"], "distance_bin": item["distance_bin"],
                "orientation": item["orientation"], "role": "tensor",
                "behavioral": float(item["behavioral"]), "pie": float(item["pie"]),
                "single": float(item["single"]), "signed_matched_bypass": signed_matched,
                "independent_energy_cell_target": float(energy_by_cell[item["cell_id"]]),
                "residual_radius": float(item["residual_radius"]),
                "admissible": bool(item["admissible"]),
                "gate_label_counts": canonical_json(dict(sorted(labels.items()))),
                "curvature_median": float(np.median([
                    gate["curvature_norm"] for gate in gates if gate.get("curvature_norm") is not None
                ])),
                "gate_response_median": float(np.median([
                    gate["gate_response_norm"] for gate in gates if gate.get("gate_response_norm") is not None
                ])),
            })
        functionals = []
        for cell, group in tensor.groupby("cell_id", sort=True):
            for source in ("behavioral", "pie", "single"):
                values = group[source].astype(float).to_numpy()
                functionals.append({
                    "cell_id": cell, "source": source, "signed_available": False,
                    "A_abs_signed_mean": None, "M_mean_absolute": float(np.mean(values)),
                    "R_rms": float(np.sqrt(np.mean(values**2))),
                })
            matched_name = ("theta_pat", "theta_tar") if version == "v136" else (
                "theta_pat_center", "theta_tar_center"
            )
            signed = (group[matched_name[0]] - group[matched_name[1]]).astype(float).to_numpy()
            functionals.append({
                "cell_id": cell, "source": "matched_bypass", "signed_available": True,
                "A_abs_signed_mean": abs(float(np.mean(signed))),
                "M_mean_absolute": float(np.mean(np.abs(signed))),
                "R_rms": float(np.sqrt(np.mean(signed**2))),
            })
        version_details[version] = {
            "tensor_role_items": len(tensor), "energy_role_items": len(energy),
            "systems_per_tensor_item": 2,
            "cell_definitions": sorted(tensor["cell_id"].unique().tolist()),
            "orientation_values": sorted(tensor["orientation"].unique().tolist()),
            "distance_values": sorted(tensor["distance_bin"].unique().tolist()),
            "gate_label_counts": dict(sorted(global_labels.items())),
            "curvature_norm": quantiles(curvature),
            "gate_response_norm": quantiles(response),
            "factorization_metric": quantiles(factorization),
            "cell_functionals": functionals,
            "independent_energy_target": quantiles(energy["signed_target"].abs()),
        }
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "11_regime_bridge_rows.parquet", index=False)
    payload = common("green-v21-regime-bridge-v1", script, {
        "v136_dev_tensor_scores.parquet": sha256_file(v136 / "dev_tensor_scores.parquet"),
        "v200_dev_tensor_scores.parquet": sha256_file(archive / "dev_tensor_scores.parquet"),
        "v136_dev_energy_targets.parquet": sha256_file(v136 / "dev_energy_targets.parquet"),
        "v200_dev_energy_targets.parquet": sha256_file(archive / "dev_energy_targets.parquet"),
    }) | {
        "versions": {
            version: version_details[version] | {
                "rows": len(group), "admissible": int(group["admissible"].sum()),
                "behavioral": quantiles(group["behavioral"]),
                "pie": quantiles(group["pie"]), "single": quantiles(group["single"]),
                "residual_radius": quantiles(group["residual_radius"]),
            }
            for version, group in frame.groupby("version")
        },
        "complete": set(frame["version"]) == {"v136", "v200"},
    }
    write_json(output / "11_regime_bridge.json", payload)
    return payload


def reporting_consistency(archive: Path, output: Path, script: Path) -> dict:
    operation = json.loads((archive / "operation_counts_v200.json").read_text(encoding="utf-8"))
    throughput = json.loads((archive / "throughput_preflight.json").read_text(encoding="utf-8"))
    counts = operation["counts"]
    checks = {
        "prepare_panel_certificates_40": counts.get("prepare_ad_gate_system_certificates") == 40,
        "prepare_panel_routes_80": counts.get("prepare_ad_gatejet_routes") == 80,
        "throughput_certificates_160": throughput.get("ad_gate_system_certificates_executed") == 160,
        "throughput_routes_320": throughput.get("ad_gatejet_routes_executed") == 320,
        "development_certificates_1280": counts.get("development_ad_gate_system_certificates") == 1280,
        "development_routes_2560": counts.get("development_ad_gatejet_routes") == 2560,
        "throughput_embedded_counts_match": throughput.get("operation_counts") == counts,
        "throughput_preflight_passed": throughput.get("passed") is True,
        "timing_artifacts_positive": all(
            float(throughput.get(name, 0)) > 0 for name in (
                "prepare_seconds", "development_seconds", "confirmation_seconds",
                "finite_difference_seconds", "ad_certification_seconds", "total_seconds",
            )
        ),
    }
    payload = common("green-v21-reporting-consistency-v1", script, {
        "operation_counts_v200.json": sha256_file(archive / "operation_counts_v200.json"),
        "throughput_preflight.json": sha256_file(archive / "throughput_preflight.json"),
    }) | {
        "checks": checks, "passed": all(checks.values()), "complete": all(checks.values()),
        "counts": {
            "frozen_prepare_panel": {"certificates": 40, "routes": 80},
            "throughput_preflight": {"certificates": 160, "routes": 320},
            "development": {"certificates": 1280, "routes": 2560},
        },
        "timing_seconds": {
            name: throughput[name] for name in (
                "prepare_seconds", "development_seconds", "confirmation_seconds",
                "finite_difference_seconds", "ad_certification_seconds", "total_seconds",
            )
        },
        "clarification": (
            "40/80 is the frozen theorem panel; 160/320 is the eight-record throughput "
            "workload; 1280/2560 is full development."
        ),
    }
    write_json(output / "12_reporting_consistency.json", payload)
    return payload


def load_gpu_rows(gpu_shards: Path | None) -> pd.DataFrame | None:
    return load_gpu_table(gpu_shards, "postmortem_rows.parquet")


def load_gpu_table(gpu_shards: Path | None, name: str) -> pd.DataFrame | None:
    if gpu_shards is None or not gpu_shards.exists():
        return None
    paths = sorted(gpu_shards.glob(f"worker_*/{name}"))
    if not paths:
        return None
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def merge_exact_gpu(gpu_shards: Path | None, output: Path, script: Path) -> tuple[dict | None, dict | None, pd.DataFrame | None]:
    rows = load_gpu_rows(gpu_shards)
    if rows is None:
        return None, None, None
    transport_rows = load_gpu_table(gpu_shards, "transport_rows.parquet")
    joint_rows = load_gpu_table(gpu_shards, "joint_rows.parquet")
    if transport_rows is None or joint_rows is None:
        raise RuntimeError("STOP: detailed exact transport/joint GPU evidence missing")
    transport_rows.to_parquet(output / "04_exact_transport_identity.parquet", index=False)
    joint_rows.to_parquet(output / "05_exact_joint_composition.parquet", index=False)
    route_failures = int((~transport_rows["route_passed"].astype(bool)).sum())
    theorem_failures = int((~transport_rows["theorem_passed"].astype(bool)).sum())
    joint_route_failures = int((~joint_rows["route_passed"].astype(bool)).sum())
    composition_failures = int((~joint_rows["composition_passed"].astype(bool)).sum())
    source = {
        str(path.relative_to(gpu_shards)): sha256_file(path)
        for pattern in ("postmortem_rows.parquet", "transport_rows.parquet", "joint_rows.parquet")
        for path in sorted(gpu_shards.glob(f"worker_*/{pattern}"))
    }
    expected_transport_rows = len(rows) * 2 * 10 * 15
    expected_joint_rows = len(rows) * 3
    transport = common("green-v21-exact-transport-identity-v1", script, source) | {
        "rows": len(transport_rows), "expected_rows": expected_transport_rows,
        "aggregation_unit": "gate-system-item-direction",
        "direction_class_counts": transport_rows["direction_class"].value_counts().sort_index().to_dict(),
        "route_failures": route_failures,
        "theorem_failures": theorem_failures,
        "max_residual_to_bound_ratio": float(transport_rows["residual_to_bound"].max()),
        "active_model_unchanged": bool(transport_rows["active_model_unchanged"].all()),
        "complete": len(transport_rows) == expected_transport_rows,
    }
    joint = common("green-v21-exact-joint-composition-v1", script, source) | {
        "rows": len(joint_rows), "expected_rows": expected_joint_rows,
        "aggregation_units": ["item-system", "item-level pat-minus-tar"],
        "system_counts": joint_rows["system"].value_counts().sort_index().to_dict(),
        "route_failures": joint_route_failures,
        "composition_failures": composition_failures,
        "max_residual_to_bound_ratio": float(joint_rows["residual_to_bound"].max()),
        "active_model_unchanged": bool(joint_rows["active_model_unchanged"].all()),
        "complete": len(joint_rows) == expected_joint_rows,
    }
    write_json(output / "04_exact_transport_identity.json", transport)
    write_json(output / "05_exact_joint_composition.json", joint)
    if (route_failures or theorem_failures or joint_route_failures or composition_failures
            or not transport["complete"] or not joint["complete"]
            or not rows["active_model_unchanged"].all()):
        raise RuntimeError("STOP: exact transport or joint theorem postmortem failed")
    return transport, joint, rows


def estimator_ladder(
    archive: Path, output: Path, script: Path, gpu_shards: Path | None
) -> dict | None:
    gates = load_gpu_table(gpu_shards, "ladder_gate_rows.parquet")
    joint = load_gpu_table(gpu_shards, "joint_rows.parquet")
    if gates is None or joint is None:
        return None
    estimators = (
        "fine_response", "coarse_response", "ad_response_whitebox",
        "fine_G_whitebox_g", "active_only_v2",
        "all_gate_response_where_invertible", "zero_centered_v2",
    )
    rows = []

    def add_row(*, unit: str, estimator: str, target: str, estimate: float,
                target_value: float, **identity: Any) -> None:
        error = abs(float(estimate) - float(target_value))
        rows.append(identity | {
            "aggregation_unit": unit, "estimator": estimator, "target": target,
            "estimate": float(estimate), "target_value": float(target_value),
            "absolute_error": error,
            "relative_error": error / max(abs(float(target_value)), 1e-12),
        })

    for row in gates.itertuples(index=False):
        identity = {
            "pair_digest": row.pair_digest, "cell_id": row.cell_id,
            "distance_bin": row.distance_bin, "orientation": row.orientation,
            "system": row.system, "gate_slot": int(row.gate_slot),
        }
        for estimator in estimators:
            value = getattr(row, estimator)
            if pd.notna(value):
                add_row(unit="gate-system-item", estimator=estimator,
                        target="exact_direct_transport", estimate=value,
                        target_value=row.exact_direct, **identity)

    item_values = []
    item_keys = ["pair_digest", "cell_id", "distance_bin", "orientation", "system"]
    for keys, group in gates.groupby(item_keys, sort=True):
        entry = dict(zip(item_keys, keys))
        for estimator in estimators:
            values = group[estimator].dropna().astype(float)
            entry[estimator] = float(values.sum()) if len(values) else math.nan
            entry[f"{estimator}_gate_count"] = int(len(values))
        entry["exact_direct_transport"] = float(group["exact_direct"].sum())
        item_values.append(entry)
    items = pd.DataFrame(item_values)
    joint_system = joint[joint["system"].isin(("tar", "pat"))].set_index(["pair_digest", "system"])
    for row in items.itertuples(index=False):
        target = float(joint_system.loc[(row.pair_digest, row.system), "independent_target"])
        for estimator in estimators:
            value = getattr(row, estimator)
            if pd.notna(value):
                add_row(
                    unit="item-system", estimator=estimator,
                    target="independent_joint_ad", estimate=value, target_value=target,
                    pair_digest=row.pair_digest, cell_id=row.cell_id,
                    distance_bin=row.distance_bin, orientation=row.orientation,
                    system=row.system, gate_slot=None,
                )

    contrast_rows = []
    for (digest, cell, distance, orientation), group in items.groupby(
        ["pair_digest", "cell_id", "distance_bin", "orientation"], sort=True
    ):
        systems = group.set_index("system")
        entry = {"pair_digest": digest, "cell_id": cell,
                 "distance_bin": distance, "orientation": orientation}
        for estimator in estimators:
            entry[estimator] = float(systems.loc["pat", estimator] - systems.loc["tar", estimator])
        contrast_rows.append(entry)
    contrasts = pd.DataFrame(contrast_rows)
    joint_contrast = joint[joint["system"] == "pat_minus_tar"].set_index("pair_digest")
    for row in contrasts.itertuples(index=False):
        target = float(joint_contrast.loc[row.pair_digest, "independent_target"])
        for estimator in estimators:
            add_row(
                unit="item-pat-minus-tar", estimator=estimator,
                target="independent_joint_ad", estimate=getattr(row, estimator),
                target_value=target, pair_digest=row.pair_digest, cell_id=row.cell_id,
                distance_bin=row.distance_bin, orientation=row.orientation,
                system="pat_minus_tar", gate_slot=None,
            )

    energy = pd.read_parquet(archive / "dev_energy_targets.parquet")
    energy["signed_target"] = energy["systems"].map(
        lambda value: float(_decode(value)["pat"]["full"] - _decode(value)["tar"]["full"])
    )
    energy_cell = energy.groupby("cell_id")["signed_target"].mean().to_dict()
    for (cell, distance), group in contrasts.groupby(["cell_id", "distance_bin"], sort=True):
        for estimator in estimators:
            add_row(
                unit="cell", estimator=estimator, target="independent_finite_energy",
                estimate=float(group[estimator].mean()), target_value=float(energy_cell[cell]),
                pair_digest=None, cell_id=cell, distance_bin=distance,
                orientation="aggregated", system="pat_minus_tar", gate_slot=None,
            )

    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "06_estimator_ladder_rows.parquet", index=False)
    metrics = {}
    for (estimator, target, unit), group in frame.groupby(
        ["estimator", "target", "aggregation_unit"], sort=True
    ):
        metrics[f"{estimator}|{target}|{unit}"] = {
            "rows": len(group), "absolute_error": quantiles(group["absolute_error"]),
            "relative_error": quantiles(group["relative_error"]),
        }
    source = {
        str(path.relative_to(gpu_shards)): sha256_file(path)
        for path in sorted(gpu_shards.glob("worker_*/ladder_gate_rows.parquet"))
    } | {
        "dev_energy_targets.parquet": sha256_file(archive / "dev_energy_targets.parquet"),
        "dev_cells.json": sha256_file(archive / "dev_cells.json"),
    }
    formal_cells = json.loads((archive / "dev_cells.json").read_text(encoding="utf-8"))["cells"]
    available_formal_targets = sum("target" in cell for cell in formal_cells)
    payload = common("green-v21-estimator-ladder-summary-v1", script, source) | {
        "rows": len(frame), "estimators": list(estimators), "metrics": metrics,
        "fine_invertible_gate_system_items": int(gates["fine_invertible"].sum()),
        "coarse_invertible_gate_system_items": int(gates["coarse_invertible"].sum()),
        "formal_v2_cell_targets_available": available_formal_targets,
        "formal_v2_cell_target_note": (
            "No formal v2 cell target was emitted because development survival failed; "
            "the independent finite-energy cell target is reported without altering that STOP."
        ),
        "complete": set(frame["estimator"]) == set(estimators)
                    and set(frame["target"]) == {"exact_direct_transport", "independent_joint_ad", "independent_finite_energy"},
        "selection_for_v3_forbidden": True,
    }
    write_json(output / "06_estimator_ladder_summary.json", payload)
    return payload


def null_unresolved_mass(archive: Path, output: Path, script: Path, gpu_rows: pd.DataFrame | None) -> dict | None:
    if gpu_rows is None:
        return None
    tensor = pd.read_parquet(archive / "dev_tensor_scores.parquet")
    rows = []
    for _, item in tensor.iterrows():
        audit = _decode(item["mixed_audit"])
        exact = gpu_rows[gpu_rows["pair_digest"] == item["pair_digest"]]
        exact_systems = _decode(exact.iloc[0]["system_audit"]) if len(exact) else {}
        for system_name, system in audit.items():
            target = abs(float(exact_systems.get(system_name, {}).get("joint_target", 0.0)))
            bound = float(exact_systems.get(system_name, {}).get("joint_bound", 0.0))
            denominator = max(target, bound)
            null = sum(float(gate.get("null_bound", 0)) for gate in system["gates"] if gate["label"] == "certified-target-null")
            unresolved = sum(float(gate.get("unresolved_bound", 0)) for gate in system["gates"] if gate["label"] == "unresolved-bounded")
            active = abs(sum(float(gate.get("contribution_center", 0)) for gate in system["gates"] if gate["label"] == "active-identified"))
            ratio = lambda value: (0 if value == 0 else math.inf) if denominator == 0 else value / denominator
            rows.append({"pair_digest": item["pair_digest"], "cell_id": item["cell_id"], "system": system_name,
                         "U_null": null, "U_unresolved": unresolved, "A_active": active,
                         "exact_joint_scale": target, "joint_bound": bound,
                         "R_null": ratio(null), "R_unresolved": ratio(unresolved)})
    frame = pd.DataFrame(rows); frame.to_parquet(output / "07_null_unresolved_mass.parquet", index=False)
    payload = common("green-v21-null-unresolved-mass-v1", script, {
        "dev_tensor_scores.parquet": sha256_file(archive / "dev_tensor_scores.parquet")
    }) | {"rows": len(frame), "R_null": quantiles(frame["R_null"]),
          "R_unresolved": quantiles(frame["R_unresolved"]),
          "complete": len(frame) == len(tensor) * 2}
    write_json(output / "07_null_unresolved_mass.json", payload)
    return payload


def postmortem_manifest(output: Path, script: Path) -> dict:
    names = [f"{index:02d}_" for index in range(1, 13)]
    files = [path for path in output.iterdir() if path.is_file() and path.name != "postmortem_manifest.json"]
    coverage = {prefix: any(path.name.startswith(prefix) for path in files) for prefix in names}
    semantic = {}
    if all(coverage.values()):
        documents = {
            index: json.loads(next(output.glob(f"{index:02d}_*.json")).read_text(encoding="utf-8"))
            for index in range(1, 13)
        }
        semantic = {
            "01_integrity": documents[1].get("integrity_reconstruction_passed") is True,
            "02_gate_rows": documents[2].get("rows") == 1280,
            "03_uncertainty_rows": documents[3].get("rows", 0) > 0,
            "04_exact_transport": documents[4].get("complete") is True
                                  and documents[4].get("theorem_failures") == 0,
            "05_exact_joint": documents[5].get("complete") is True
                              and documents[5].get("composition_failures") == 0,
            "06_estimator_ladder": documents[6].get("complete") is True,
            "07_null_mass": documents[7].get("complete") is True,
            "08_functionals": documents[8].get("complete") is True,
            "09_role_sampling": documents[9].get("complete") is True,
            "10_snr_geometry": documents[10].get("all_nonempty_zero_crossings_have_snr_one") is True,
            "11_regime_bridge": documents[11].get("complete") is True,
            "12_reporting": documents[12].get("passed") is True,
        }
    all_complete = all(coverage.values()) and bool(semantic) and all(semantic.values())
    payload = common("green-v21-postmortem-manifest-v1", script, {}) | {
        "analysis_coverage": coverage, "semantic_completion_checks": semantic,
        "all_twelve_complete": all_complete,
        "artifact_sha256": {path.name: sha256_file(path) for path in sorted(files)},
    }
    write_json(output / "postmortem_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--v136-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--formal-v200-root", type=Path)
    parser.add_argument("--gpu-shards", type=Path)
    parser.add_argument("--expected-postmortem-commit", required=True)
    parser.add_argument("--expected-execution-commit", required=True)
    args = parser.parse_args()
    if args.expected_postmortem_commit != POSTMORTEM_COMMIT or args.expected_execution_commit != EXECUTION_COMMIT:
        raise RuntimeError("STOP: commit identity mismatch")
    if args.output.resolve() == args.archive.resolve():
        raise RuntimeError("postmortem output may not overwrite the archive")
    if args.formal_v200_root is not None and args.output.resolve() == args.formal_v200_root.resolve():
        raise RuntimeError("postmortem output may not overwrite the formal v2 root")
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    integrity_reconstruction(args.archive, args.formal_v200_root, args.output, script)
    gates, _ = gate_certificate_decomposition(args.archive, args.output, script)
    uncertainty_decomposition(gates, args.archive, args.output, script)
    transport, joint, gpu_rows = merge_exact_gpu(args.gpu_shards, args.output, script)
    estimator_ladder(args.archive, args.output, script, args.gpu_shards)
    null_unresolved_mass(args.archive, args.output, script, gpu_rows)
    aggregation_functionals(args.archive, args.output, script, gpu_rows)
    role_sampling_audit(args.archive, args.output, script, gpu_rows)
    set_snr_geometry(args.archive, args.output, script)
    regime_bridge(args.archive, args.v136_audit, args.output, script)
    reporting_consistency(args.archive, args.output, script)
    manifest = postmortem_manifest(args.output, script)
    if not manifest["all_twelve_complete"]:
        missing = [name for name, passed in manifest["analysis_coverage"].items() if not passed]
        failed = [name for name, passed in manifest["semantic_completion_checks"].items() if not passed]
        print(json.dumps({"status": "POSTMORTEM_INCOMPLETE", "missing": missing,
                          "failed_semantic_checks": failed}, sort_keys=True))
        raise SystemExit(3)
    print(json.dumps({"status": "POSTMORTEM_PASS", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
