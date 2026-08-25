"""Frozen analysis rules for future GREEN v3 development and confirmation."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Mapping

import numpy as np

from green_bridge_v300_numerics import signed_set_snr_v300
from green_bridge_v300_spec import THRESHOLDS


BASELINES = ("gate_atom_only", "raw_path_jacobian", "unmatched_path_mixed", "zero")


def _records(rows) -> list[dict]:
    if hasattr(rows, "to_dict"):
        return list(rows.to_dict("records"))
    return [dict(row) for row in rows]


def group_balanced_rmse_v300(rows: Iterable[Mapping], value_field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["noun_century_group"])].append(float(row[value_field]) ** 2)
    if not grouped:
        return math.inf
    return float(math.sqrt(np.mean([np.mean(values) for values in grouped.values()])))


def select_frozen_baseline_v300(rows) -> dict:
    records = _records(rows)
    scores = {
        baseline: group_balanced_rmse_v300(records, f"error_{baseline}")
        for baseline in BASELINES
    }
    selected = min(scores, key=lambda name: (scores[name], name))
    return {
        "schema_version": "green-bridge-v3.0.0-frozen-baseline-v1",
        "selected_baseline": selected,
        "group_balanced_rmse": dict(sorted(scores.items())),
        "selection_rule": "minimum group-balanced RMSE; lexicographic tie break",
        "fitted_affine_calibration": False,
        "confirmation_reselection_allowed": False,
    }


def matched_gain_v300(matched_rmse: float, baseline_rmse: float) -> float:
    matched_rmse, baseline_rmse = float(matched_rmse), float(baseline_rmse)
    if baseline_rmse == 0:
        return 0.0 if matched_rmse == 0 else -math.inf
    return 1.0 - matched_rmse / baseline_rmse


def aggregate_transport_cells_v300(transport_rows, joint_rows) -> list[dict]:
    transport = _records(transport_rows); joint = _records(joint_rows)
    cells = sorted({str(row["cell_id"]) for row in transport + joint})
    result = []
    for cell in cells:
        t = [row for row in transport if row["cell_id"] == cell]
        j = [row for row in joint if row["cell_id"] == cell]
        admissible_t = [row for row in t if row.get("technically_admissible", False)]
        admissible_j = [row for row in j if row.get("technically_admissible", False)]
        nonnull = [row for row in t if row.get("nonnull", False)]
        recoverable_nonnull = [row for row in nonnull if row.get("gate_class") == "recoverable"]
        joint_center = float(np.mean([row.get("joint_center", 0.0) for row in admissible_j])) if admissible_j else 0.0
        joint_bound = float(np.mean([row.get("joint_bound", math.inf) for row in admissible_j])) if admissible_j else math.inf
        unresolved = float(np.mean([row.get("unresolved_bound", math.inf) for row in admissible_j])) if admissible_j else math.inf
        target_scale = float(np.mean([
            max(abs(float(row.get("joint_target", 0.0))), float(row.get("target_bound", 0.0)))
            for row in admissible_j
        ])) if admissible_j else 0.0
        unresolved_ratio = ((0.0 if unresolved == 0 else math.inf) if target_scale == 0
                            else unresolved / target_scale)
        result.append({
            "cell_id": cell,
            "transport_admissible": len(admissible_t), "joint_admissible": len(admissible_j),
            "survived": len(admissible_t) >= 6 and len(admissible_j) >= 6,
            "nonnull_gate_system_units": len(nonnull),
            "recoverable_nonnull_fraction": (
                len(recoverable_nonnull) / len(nonnull) if nonnull else 0.0
            ),
            "detectability_conditioned": bool(
                nonnull and len(recoverable_nonnull) / len(nonnull) >= 0.25
            ),
            "joint_center": joint_center, "joint_bound": joint_bound,
            "signed_set_snr": signed_set_snr_v300(joint_center, joint_bound),
            "set_snr_qualified": signed_set_snr_v300(joint_center, joint_bound) >= 4.0,
            "unresolved_bound": unresolved,
            "unresolved_mass_ratio": unresolved_ratio,
            "unresolved_mass_narrow": unresolved_ratio <= 0.25,
        })
    return result


def development_decision_v300(summary: Mapping) -> dict:
    hard = {
        "surviving_cells": int(summary.get("surviving_cells", 0)) >= 8,
        "surviving_near": int(summary.get("surviving_near_cells", 0)) >= 4,
        "surviving_far": int(summary.get("surviving_far_cells", 0)) >= 4,
        "all_nouns": int(summary.get("development_nouns_represented", 0)) == 3,
        "no_invalid": int(summary.get("numerical_invalid_units", 1)) == 0,
        "no_contradiction": int(summary.get("structural_contradiction_units", 1)) == 0,
        "coverage": float(summary.get("resolved_coverage", 0)) >= 0.80,
        "recoverability": float(summary.get("nonnull_recoverability_fraction", 0)) >= 0.25,
        "direct_median": float(summary.get("direct_error_median", math.inf)) <= 0.10,
        "direct_p90": float(summary.get("direct_error_p90", math.inf)) <= 0.25,
        "joint_median": float(summary.get("joint_error_median", math.inf)) <= 0.15,
        "joint_p90": float(summary.get("joint_error_p90", math.inf)) <= 0.30,
        "gain": float(summary.get("matched_best_baseline_gain", -math.inf)) >= 0.20,
    }
    if all(hard.values()):
        verdict = "OPEN_CONFIRMATION"
    elif (hard["no_invalid"] and hard["no_contradiction"]
          and int(summary.get("surviving_cells", 0)) >= 8
          and float(summary.get("direct_error_median", math.inf)) <= 0.15
          and float(summary.get("direct_error_p90", math.inf)) <= 0.35
          and float(summary.get("matched_best_baseline_gain", -math.inf)) >= 0.10):
        verdict = "POSTER_ONLY"
    else:
        verdict = "STOP_ORAL"
    return {"verdict": verdict, "gates": hard, "confirmation_authorized": False}


def confirmation_decision_v300(summary: Mapping, frozen_baseline: Mapping) -> dict:
    if summary.get("selected_baseline") not in (None, frozen_baseline["selected_baseline"]):
        raise RuntimeError("CONFIRMATION_BASELINE_RESELECTION_FORBIDDEN")
    checks = {
        "surviving_cells": int(summary.get("surviving_cells", 0)) >= 12,
        "no_invalid": int(summary.get("numerical_invalid_units", 1)) == 0,
        "no_contradiction": int(summary.get("structural_contradiction_units", 1)) == 0,
        "direct_median": float(summary.get("direct_error_median", math.inf)) <= 0.08,
        "direct_p90": float(summary.get("direct_error_p90", math.inf)) <= 0.15,
        "joint_median": float(summary.get("joint_error_median", math.inf)) <= 0.10,
        "joint_p90": float(summary.get("joint_error_p90", math.inf)) <= 0.20,
        "gain": float(summary.get("matched_gain", -math.inf)) >= 0.25,
    }
    if all(checks.values()):
        verdict = "ORAL_LEVEL_PASS"
    elif checks["no_invalid"] and checks["no_contradiction"] and float(summary.get("matched_gain", -math.inf)) >= 0.10:
        verdict = "POSTER_ONLY"
    else:
        verdict = "STOP"
    return {"verdict": verdict, "gates": checks,
            "selected_baseline": frozen_baseline["selected_baseline"]}
