"""Cell-level frozen analysis for the GPT-2 green-bridge experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from green_bridge_spec import OUTPUT_ROOT, THRESHOLDS, frozen_spec_hash, write_json_atomic


BASELINES = ("behavioral", "single", "first_order", "pie")


def _cells(payload: dict) -> list[dict]:
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError("result payload must contain a cells list")
    return [cell for cell in cells if cell.get("survived", False)]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def fit_nonnegative_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Exact two-variable NNLS for columns ``[1,x]`` without SciPy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 1:
        raise ValueError("x and y must be equal nonempty vectors")
    design = np.column_stack([np.ones(len(x)), x])
    candidates: list[tuple[float, float]] = [(0.0, 0.0)]
    unconstrained = np.linalg.lstsq(design, y, rcond=None)[0]
    if np.all(unconstrained >= 0):
        candidates.append((float(unconstrained[0]), float(unconstrained[1])))
    candidates.append((max(float(y.mean()), 0.0), 0.0))
    denom = float(x @ x)
    candidates.append((0.0, max(float(x @ y / denom), 0.0) if denom > 0 else 0.0))
    return min(candidates, key=lambda ab: np.sum((y - ab[0] - ab[1] * x) ** 2))


def calibrate_development(cells: list[dict]) -> dict:
    if len(cells) < THRESHOLDS.development_cells_min:
        raise ValueError("insufficient surviving development cells")
    y = np.array([cell["target"] for cell in cells], dtype=np.float64)
    result = {}
    for baseline in BASELINES:
        x = np.array([cell["baselines"][baseline] for cell in cells], dtype=np.float64)
        loocv = np.empty(len(cells), dtype=np.float64)
        for heldout in range(len(cells)):
            mask = np.arange(len(cells)) != heldout
            alpha, beta = fit_nonnegative_affine(x[mask], y[mask])
            loocv[heldout] = alpha + beta * x[heldout]
        alpha, beta = fit_nonnegative_affine(x, y)
        result[baseline] = {
            "alpha": alpha,
            "beta": beta,
            "loocv_prediction": loocv.tolist(),
            "loocv_rmse": rmse(y, loocv),
        }
    return result


def development_decision(payload: dict) -> dict:
    cells = _cells(payload)
    conditioned = sum(cell.get("conditioned", False) for cell in cells)
    snr_count = sum(
        cell.get("snr", 0.0) >= THRESHOLDS.development_snr_min
        for cell in cells
    )
    if len(cells) < THRESHOLDS.development_cells_min:
        return {
            "phase": "development",
            "verdict": "STOP_ORAL",
            "n_surviving_cells": len(cells),
            "n_conditioned_cells": conditioned,
            "n_snr_cells": snr_count,
            "mixed_rmse": None,
            "best_baseline": None,
            "best_baseline_loocv_rmse": None,
            "relative_gain": None,
            "baseline_calibration": {},
            "spec_sha256": frozen_spec_hash(),
        }
    calibration = calibrate_development(cells)
    y = np.array([cell["target"] for cell in cells], dtype=np.float64)
    mixed = np.array([cell["mixed"] for cell in cells], dtype=np.float64)
    mixed_rmse = rmse(y, mixed)
    best_name = min(BASELINES, key=lambda name: calibration[name]["loocv_rmse"])
    best_rmse = calibration[best_name]["loocv_rmse"]
    gain = 1.0 - mixed_rmse / best_rmse if best_rmse > 0 else float("-inf")
    gates_pass = (
        len(cells) >= THRESHOLDS.development_cells_min
        and conditioned >= THRESHOLDS.development_cells_min
        and snr_count >= THRESHOLDS.development_snr_cells_min
    )
    if not gates_pass or gain < THRESHOLDS.development_stop_below:
        verdict = "STOP_ORAL"
    elif gain < THRESHOLDS.confirmation_open_gain_min:
        verdict = "POSTER_ONLY"
    else:
        verdict = "OPEN_CONFIRMATION"
    return {
        "phase": "development",
        "verdict": verdict,
        "n_surviving_cells": len(cells),
        "n_conditioned_cells": conditioned,
        "n_snr_cells": snr_count,
        "mixed_rmse": mixed_rmse,
        "best_baseline": best_name,
        "best_baseline_loocv_rmse": best_rmse,
        "relative_gain": gain,
        "baseline_calibration": calibration,
        "spec_sha256": frozen_spec_hash(),
    }


def freeze_confirmation(development: dict, path: Path) -> dict:
    result = development_decision(development)
    if result["verdict"] != "OPEN_CONFIRMATION":
        raise PermissionError(f"confirmation cannot open after {result['verdict']}")
    frozen = {
        "schema_version": "green-bridge-frozen-analysis-v1",
        "spec_sha256": frozen_spec_hash(),
        "baseline_calibration": {
            name: {
                "alpha": result["baseline_calibration"][name]["alpha"],
                "beta": result["baseline_calibration"][name]["beta"],
            }
            for name in BASELINES
        },
        "bootstrap_replicates": 100_000,
        "bootstrap_seed": 20260805,
        "confirmation_retries": 0,
        "development_summary": result,
    }
    write_json_atomic(path, frozen)
    return frozen


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    x = _rank_average(np.asarray(list(a), dtype=np.float64))
    y = _rank_average(np.asarray(list(b), dtype=np.float64))
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both classes")
    ranks = _rank_average(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _stratified_indices(cells: list[dict], rng: np.random.Generator) -> np.ndarray:
    parts = []
    for bin_name in ("near", "far"):
        indices = np.array([i for i, cell in enumerate(cells) if cell["distance_bin"] == bin_name])
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(parts)


def confirmation_decision(confirm_payload: dict, frozen: dict) -> dict:
    cells = _cells(confirm_payload)
    if len(cells) < THRESHOLDS.confirmation_technical_min:
        return {"phase": "confirmation", "verdict": "FAIL_SURVIVAL", "n_cells": len(cells)}
    y = np.array([cell["target"] for cell in cells], dtype=np.float64)
    mixed = np.array([cell["mixed"] for cell in cells], dtype=np.float64)
    predictions = {}
    for name in BASELINES:
        calibration = frozen["baseline_calibration"][name]
        raw = np.array([cell["baselines"][name] for cell in cells], dtype=np.float64)
        predictions[name] = calibration["alpha"] + calibration["beta"] * raw
    baseline_rmse = {name: rmse(y, prediction) for name, prediction in predictions.items()}
    best_name = min(BASELINES, key=baseline_rmse.get)
    best_rmse = baseline_rmse[best_name]
    mixed_rmse = rmse(y, mixed)
    relative_gain = 1.0 - mixed_rmse / best_rmse
    absolute_gain = best_rmse - mixed_rmse

    rng = np.random.Generator(np.random.PCG64(int(frozen["bootstrap_seed"])))
    replicates = int(frozen["bootstrap_replicates"])
    bootstrap_gain = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        idx = _stratified_indices(cells, rng)
        mixed_r = rmse(y[idx], mixed[idx])
        best_r = min(rmse(y[idx], predictions[name][idx]) for name in BASELINES)
        bootstrap_gain[replicate] = 1.0 - mixed_r / best_r if best_r > 0 else -np.inf
    relative_lcb = float(np.percentile(bootstrap_gain, 2.5))

    per_bin = {}
    for bin_name in ("near", "far"):
        idx = np.array([i for i, cell in enumerate(cells) if cell["distance_bin"] == bin_name])
        mixed_b = rmse(y[idx], mixed[idx])
        best_b = min(rmse(y[idx], predictions[name][idx]) for name in BASELINES)
        gains = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            sample = rng.choice(idx, size=len(idx), replace=True)
            mr = rmse(y[sample], mixed[sample])
            br = min(rmse(y[sample], predictions[name][sample]) for name in BASELINES)
            gains[replicate] = 1.0 - mr / br if br > 0 else -np.inf
        per_bin[bin_name] = {
            "n": len(idx),
            "relative_gain": 1.0 - mixed_b / best_b,
            "relative_gain_lcb": float(np.percentile(gains, 2.5)),
            "absolute_gain": best_b - mixed_b,
        }

    cancellation = [
        (i, cell)
        for i, cell in enumerate(cells)
        if cell["cancellation_dx"] * cell["cancellation_dz"] < 0
        and min(abs(cell["cancellation_dx"]), abs(cell["cancellation_dz"]))
        >= THRESHOLDS.cancellation_main_effect_min
    ]
    cancellation_result = {"n": len(cancellation), "valid": False}
    if cancellation:
        idx = np.array([row[0] for row in cancellation])
        labels = y[idx] >= THRESHOLDS.cancellation_target_threshold
        bin_counts = {
            bin_name: sum(cell["distance_bin"] == bin_name for _, cell in cancellation)
            for bin_name in ("near", "far")
        }
        balanced = (
            len(idx) >= THRESHOLDS.cancellation_size_min
            and labels.sum() >= THRESHOLDS.cancellation_class_min
            and (~labels).sum() >= THRESHOLDS.cancellation_class_min
            and min(bin_counts.values()) >= THRESHOLDS.cancellation_bin_min
        )
        if balanced:
            point_auc = auroc(labels, mixed[idx])
            auc_boot = np.empty(replicates, dtype=np.float64)
            strata = {}
            for bin_name in ("near", "far"):
                for label in (False, True):
                    group = idx[
                        np.array([
                            cells[i]["distance_bin"] == bin_name and bool(y[i] >= THRESHOLDS.cancellation_target_threshold) == label
                            for i in idx
                        ])
                    ]
                    strata[bin_name, label] = group
            for replicate in range(replicates):
                sample = np.concatenate([
                    rng.choice(group, size=len(group), replace=True)
                    for group in strata.values() if len(group)
                ])
                auc_boot[replicate] = auroc(
                    y[sample] >= THRESHOLDS.cancellation_target_threshold, mixed[sample]
                )
            cancellation_result = {
                "n": len(idx), "valid": True, "auroc": point_auc,
                "auroc_lcb": float(np.percentile(auc_boot, 2.5)),
                "bin_counts": bin_counts,
            }

    full = np.array([cell["mixed_full"] for cell in cells], dtype=np.float64)
    half = np.array([cell["mixed_half"] for cell in cells], dtype=np.float64)
    symmetric = np.abs(full - half) / np.maximum((np.abs(full) + np.abs(half)) / 2.0, 0.05)
    radius = {
        "spearman": spearman(full, half),
        "median_change": float(np.median(symmetric)),
        "bins": {},
    }
    for bin_name in ("near", "far"):
        idx = np.array([i for i, cell in enumerate(cells) if cell["distance_bin"] == bin_name])
        radius["bins"][bin_name] = {
            "spearman": spearman(full[idx], half[idx]),
            "median_change": float(np.median(symmetric[idx])),
        }

    conditioned = sum(cell.get("conditioned", False) for cell in cells)
    success = (
        len(cells) >= THRESHOLDS.confirmation_oral_min
        and conditioned >= THRESHOLDS.confirmation_oral_min
        and all(sum(cell["distance_bin"] == b for cell in cells) >= THRESHOLDS.cells_per_bin_min for b in ("near", "far"))
        and relative_gain >= THRESHOLDS.confirmation_relative_gain_min
        and relative_lcb >= THRESHOLDS.confirmation_relative_lcb_min
        and absolute_gain >= THRESHOLDS.confirmation_absolute_gain_min
        and all(
            row["relative_gain"] >= THRESHOLDS.per_bin_relative_gain_min
            and row["relative_gain_lcb"] > 0
            and row["absolute_gain"] >= THRESHOLDS.per_bin_absolute_gain_min
            for row in per_bin.values()
        )
        and cancellation_result.get("valid", False)
        and cancellation_result.get("auroc", 0) >= THRESHOLDS.cancellation_auroc_min
        and cancellation_result.get("auroc_lcb", 0) >= THRESHOLDS.cancellation_auroc_lcb_min
        and radius["spearman"] >= THRESHOLDS.half_radius_spearman_min
        and radius["median_change"] <= THRESHOLDS.half_radius_change_max
        and all(
            row["spearman"] >= THRESHOLDS.half_radius_spearman_min
            and row["median_change"] <= THRESHOLDS.half_radius_change_max
            for row in radius["bins"].values()
        )
    )
    return {
        "phase": "confirmation",
        "verdict": "ORAL_RESULT_PASS" if success else "ORAL_RESULT_FAIL",
        "n_cells": len(cells),
        "n_conditioned": conditioned,
        "mixed_rmse": mixed_rmse,
        "baseline_rmse": baseline_rmse,
        "best_baseline": best_name,
        "relative_gain": relative_gain,
        "relative_gain_lcb": relative_lcb,
        "absolute_gain": absolute_gain,
        "per_bin": per_bin,
        "cancellation": cancellation_result,
        "half_radius": radius,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=OUTPUT_ROOT / "frozen_analysis.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.phase == "development":
        result = development_decision(payload)
    else:
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
        result = confirmation_decision(payload, frozen)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
