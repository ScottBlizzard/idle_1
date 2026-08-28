"""Independent held-out transport endpoint for the GREEN challenge."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from green_v400_endpoint_firewall import seal_endpoint_packet
from green_v400_response_baselines import exact_finite_response


def split_conformal_upper_tail_p(
    calibration_scores: Iterable[float], query_score: float
) -> float:
    """Conservative split-conformal p-value for unusually large error."""

    calibration = tuple(float(value) for value in calibration_scores)
    if not calibration:
        raise ValueError("endpoint calibration scores must be nonempty")
    if not math.isfinite(query_score) or not all(math.isfinite(v) for v in calibration):
        raise ValueError("endpoint scores must be finite")
    return (1.0 + sum(value >= query_score for value in calibration)) / (
        len(calibration) + 1.0
    )


def compute_heldout_transport_endpoint(
    *,
    protocol_id: str,
    row_id: str,
    prediction_commitment: dict[str, Any],
    target_response: Any,
    patched_response: Any,
    center: torch.Tensor,
    endpoint_directions: torch.Tensor,
    endpoint_calibration_scores: Iterable[float],
    failure_alpha: float = 0.05,
    normalization_floor: float = 1e-12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a sealed exact-response panel without reading predictions."""

    if not (0.0 < failure_alpha < 1.0):
        raise ValueError("failure_alpha must lie strictly between zero and one")
    if normalization_floor <= 0:
        raise ValueError("normalization_floor must be positive")

    target = exact_finite_response(target_response, center, endpoint_directions)
    patched = exact_finite_response(patched_response, center, endpoint_directions)
    discrepancy = patched - target
    error = torch.sqrt(torch.mean(discrepancy.square()))
    target_scale = torch.sqrt(torch.mean(target.square()))
    normalized_error = error / torch.clamp(target_scale, min=normalization_floor)
    error_value = float(error.detach().cpu())
    normalized_value = float(normalized_error.detach().cpu())
    if not math.isfinite(error_value) or not math.isfinite(normalized_value):
        raise ValueError("held-out endpoint produced a non-finite error")

    conformal_p = split_conformal_upper_tail_p(
        endpoint_calibration_scores, normalized_value
    )
    packet = {
        "schema_version": "green-v400-sfc-endpoint-packet-v1",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "route": "endpoint",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "endpoint_direction_count_private": int(endpoint_directions.shape[0]),
        "endpoint_target_effects_private": [
            float(value) for value in target.detach().cpu()
        ],
        "endpoint_patched_effects_private": [
            float(value) for value in patched.detach().cpu()
        ],
        "endpoint_discrepancies_private": [
            float(value) for value in discrepancy.detach().cpu()
        ],
        "heldout_transport_error_private": error_value,
        "heldout_transport_normalized_error_private": normalized_value,
        "endpoint_conformal_upper_tail_p_private": conformal_p,
        "endpoint_failure_alpha_private": failure_alpha,
        "endpoint_failure_label_private": conformal_p <= failure_alpha,
    }
    return packet, seal_endpoint_packet(packet, prediction_commitment)

