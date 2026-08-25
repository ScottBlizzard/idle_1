"""Numerical contracts and interval rules for GREEN v3.0.0."""
from __future__ import annotations

import math
from typing import Iterable
import numpy as np

from green_bridge_numerics import round_up
from green_bridge_v300_spec import RECOVERABLE_RELATIVE_WIDTH_MAX


def relative_width_v300(estimate: float, bound: float) -> float:
    estimate, bound = abs(float(estimate)), float(bound)
    if bound < 0 or not math.isfinite(bound):
        return math.inf
    if estimate == 0:
        return 0.0 if bound == 0 else math.inf
    return bound / estimate


def response_detectability_v300(curvature_norm: float, epsilon_c: float,
                                response_norm: float, epsilon_g: float,
                                operator_norm: float, epsilon_p: float) -> dict:
    widths = {
        "curvature": relative_width_v300(curvature_norm, epsilon_c),
        "response": relative_width_v300(response_norm, epsilon_g),
        "operator": relative_width_v300(operator_norm, epsilon_p),
    }
    return {
        "relative_width": widths,
        "recoverable": all(value <= RECOVERABLE_RELATIVE_WIDTH_MAX for value in widths.values()),
    }


def normalized_transport_error_v300(prediction, target, bound: float) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    numerator = float(np.linalg.norm(prediction - target))
    target_norm = float(np.linalg.norm(target))
    denominator = max(target_norm, float(bound))
    if denominator == 0:
        return 0.0 if numerator == 0 else math.inf
    return numerator / denominator


def signed_set_snr_v300(center: float, half_width: float) -> float:
    center, half_width = float(center), float(half_width)
    if half_width < 0 or not math.isfinite(half_width):
        return 0.0
    if half_width == 0:
        return 0.0 if center == 0 else math.inf
    return abs(center) / half_width


def direct_transport_certificate_v300(operator_error_fro: float, projected_direction_norm: float,
                                      gate_response_norm: float, epsilon_g: float,
                                      envelope_residual: float, direction_norm: float,
                                      direct_target_radius: float) -> float:
    terms = (
        float(operator_error_fro) * float(projected_direction_norm)
        + (float(gate_response_norm) + float(epsilon_g))
        * float(envelope_residual) * float(direction_norm)
        + float(direct_target_radius)
    )
    return round_up(terms)


def joint_composition_certificate_v300(per_gate_bounds: Iterable[float], target_bound: float) -> float:
    return round_up(math.fsum(float(value) for value in per_gate_bounds) + float(target_bound))


def radius_candidate_eligibility_v300(rows: Iterable[dict], relative_fidelity: float = 0.10) -> bool:
    for row in rows:
        estimate = np.asarray(row["fine"], dtype=np.float64)
        ad = np.asarray(row["ad_midpoint"], dtype=np.float64)
        route = float(row["ad_route_radius"])
        endpoint = float(row["endpoint_radius"])
        if not (np.all(np.isfinite(estimate)) and np.all(np.isfinite(ad))
                and math.isfinite(route) and math.isfinite(endpoint)):
            return False
        difference = float(np.linalg.norm(estimate - ad))
        ceiling = round_up(relative_fidelity * float(np.linalg.norm(ad)) + route + endpoint)
        if difference > ceiling:
            return False
        if not row.get("ad_route_passed", False) or not row.get("theorem_passed", False):
            return False
        if not row.get("endpoint_floor_passed", False) or row.get("fallback_used", False):
            return False
    return True


def select_global_radius_v300(candidate_rows: dict[float, list[dict]]) -> float:
    eligible = [float(radius) for radius, rows in candidate_rows.items()
                if radius_candidate_eligibility_v300(rows)]
    if not eligible:
        raise RuntimeError("PREPARE STOP 08_RADIUS_LOCALITY")
    return max(eligible)


def classify_gate_v300(*, numerical_valid: bool, structural_valid: bool,
                       recoverable: bool, exact_operator_upper: float,
                       direct_numerical_floor: float) -> str:
    if not numerical_valid:
        return "numerical-invalid"
    if not structural_valid:
        return "structural-contradiction"
    if recoverable:
        return "recoverable"
    if float(exact_operator_upper) <= float(direct_numerical_floor):
        return "certified-numerical-null"
    return "unresolved"
