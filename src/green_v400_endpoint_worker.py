"""Independent held-out transport endpoint for the GREEN challenge."""

from __future__ import annotations

import math
from typing import Any

import torch

from green_v400_direction_binding import verify_direction_binding
from green_v400_endpoint_firewall import seal_endpoint_packet
from green_v400_execution_receipts import (
    validate_endpoint_authorization_receipt,
    validate_runtime_input_receipt,
)
from green_v400_response_baselines import exact_finite_response


TRANSPORT_FAILURE_THRESHOLD = 0.20
NORMALIZATION_FLOOR = 1e-12


def _source_sha256() -> str:
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def compute_heldout_transport_endpoint(
    *,
    protocol_id: str,
    row_id: str,
    prediction_commitment: dict[str, Any],
    endpoint_authorization_receipt: dict[str, Any],
    runtime_input_receipt: dict[str, Any],
    response_precision_receipt: dict[str, Any],
    target_response: Any,
    patched_response: Any,
    center: torch.Tensor,
    endpoint_directions: torch.Tensor,
    endpoint_direction_binding: dict[str, Any],
    expected_endpoint_direction_binding_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a sealed response panel after a numerical-stability gate."""

    verify_direction_binding(
        tensor=endpoint_directions,
        binding=endpoint_direction_binding,
        expected_binding_sha256=expected_endpoint_direction_binding_sha256,
        protocol_id=protocol_id,
        row_id=row_id,
        panel_kind="endpoint",
    )
    validate_endpoint_authorization_receipt(
        receipt=endpoint_authorization_receipt,
        protocol_id=protocol_id,
        row_id=row_id,
        prediction_commitment=prediction_commitment,
        endpoint_direction_binding_sha256=expected_endpoint_direction_binding_sha256,
        endpoint_worker_source_sha256=_source_sha256(),
    )
    validate_runtime_input_receipt(
        receipt=runtime_input_receipt,
        authorization=endpoint_authorization_receipt,
        center=center,
        response_precision_receipt=response_precision_receipt,
    )

    target = exact_finite_response(target_response, center, endpoint_directions)
    patched = exact_finite_response(patched_response, center, endpoint_directions)
    discrepancy = patched - target
    error = torch.sqrt(torch.mean(discrepancy.square()))
    target_scale = torch.sqrt(torch.mean(target.square()))
    patched_scale = torch.sqrt(torch.mean(patched.square()))
    symmetric_scale = torch.maximum(target_scale, patched_scale)
    normalized_error = error / torch.clamp(symmetric_scale, min=NORMALIZATION_FLOOR)
    error_value = float(error.detach().cpu())
    normalized_value = float(normalized_error.detach().cpu())
    if not math.isfinite(error_value) or not math.isfinite(normalized_value):
        raise ValueError("held-out endpoint produced a non-finite error")

    packet = {
        "schema_version": "green-v400-sfc-endpoint-packet-v2",
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
        "endpoint_status_private": "VALID",
        "heldout_transport_target_rms_private": float(target_scale.detach().cpu()),
        "heldout_transport_patched_rms_private": float(patched_scale.detach().cpu()),
        "heldout_transport_symmetric_scale_private": float(symmetric_scale.detach().cpu()),
        "heldout_transport_symmetric_normalized_error_private": normalized_value,
        "endpoint_transport_failure_threshold_private": TRANSPORT_FAILURE_THRESHOLD,
        "endpoint_normalization_floor_private": NORMALIZATION_FLOOR,
        "endpoint_failure_label_private": normalized_value > TRANSPORT_FAILURE_THRESHOLD,
        "endpoint_failure_label_role_private": "secondary_effect_size_label",
        "numerical_replay_layer_receipt_sha256_private": endpoint_authorization_receipt[
            "numerical_replay_layer_receipt_sha256"
        ],
        "endpoint_direction_binding_sha256_private": expected_endpoint_direction_binding_sha256,
        "endpoint_authorization_receipt_sha256_private": endpoint_authorization_receipt[
            "receipt_sha256"
        ],
        "decision_spec_sha256_private": endpoint_authorization_receipt[
            "decision_spec_sha256"
        ],
        "runtime_input_receipt_sha256_private": runtime_input_receipt[
            "receipt_sha256"
        ],
        "response_precision_receipt_sha256_private": response_precision_receipt[
            "receipt_sha256"
        ],
        "scientific_null_distribution_claimed_private": False,
        "scientific_outcome_evaluated_private": True,
    }
    return packet, seal_endpoint_packet(packet, prediction_commitment)
