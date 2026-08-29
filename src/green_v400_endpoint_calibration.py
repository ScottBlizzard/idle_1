"""Endpoint-only target--target numerical replay gate.

The replay pair is a reproducibility check, not a scientific null sample.  In
particular, replay disagreement must never be converted into a p-value or used
to choose the transport-failure effect-size threshold.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from green_v400_direction_binding import verify_direction_binding
from green_v400_endpoint_firewall import seal_endpoint_numerical_replay_packet
from green_v400_response_baselines import exact_finite_response
from green_v400_response_precision import verify_precision_receipt


def _hex_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value


def compute_target_replay_packet(
    *,
    protocol_id: str,
    row_id: str,
    target_response: Any,
    center: torch.Tensor,
    endpoint_directions: torch.Tensor,
    endpoint_direction_binding: dict[str, Any],
    expected_endpoint_direction_binding_sha256: str,
    replay_id: str,
    worker_instance_id: str,
    model_manifest_sha256: str,
    response_precision_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one target replay in a prediction-inaccessible worker."""

    if replay_id not in {"A", "B"}:
        raise ValueError("replay_id must be A or B")
    _hex_digest(
        expected_endpoint_direction_binding_sha256,
        "endpoint direction binding commitment",
    )
    _hex_digest(worker_instance_id, "worker instance identifier")
    _hex_digest(model_manifest_sha256, "model manifest hash")
    verify_precision_receipt(response_precision_receipt, model_manifest_sha256)
    if center.dtype != torch.float64:
        raise ValueError("target numerical replay requires a float64 center")
    verify_direction_binding(
        tensor=endpoint_directions,
        binding=endpoint_direction_binding,
        expected_binding_sha256=expected_endpoint_direction_binding_sha256,
        protocol_id=protocol_id,
        row_id=row_id,
        panel_kind="endpoint",
    )
    effects = exact_finite_response(target_response, center, endpoint_directions)
    values = [float(value) for value in effects.detach().cpu()]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("target replay effects must be nonempty and finite")
    packet = {
        "schema_version": "green-v400-sfc-target-replay-v1",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "route": "endpoint_numerical_replay",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "calibration_kind": "target_target_replay",
        "replay_id_private": replay_id,
        "worker_instance_id_private": worker_instance_id,
        "endpoint_direction_binding_sha256_private": expected_endpoint_direction_binding_sha256,
        "endpoint_direction_count_private": int(endpoint_directions.shape[0]),
        "endpoint_replay_effects_private": values,
        "model_manifest_sha256_private": model_manifest_sha256,
        "response_precision_receipt_sha256_private": response_precision_receipt[
            "receipt_sha256"
        ],
        "response_evaluation_dtype_private": "float64",
    }
    return packet, seal_endpoint_numerical_replay_packet(packet)


def merge_target_replay_stability(
    replay_a: dict[str, Any],
    commitment_a: dict[str, Any],
    replay_b: dict[str, Any],
    commitment_b: dict[str, Any],
    *,
    absolute_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a numerical-stability gate from independently committed replays."""

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("replay tolerances must be nonnegative")
    if absolute_tolerance == 0 and relative_tolerance == 0:
        raise ValueError("at least one replay tolerance must be positive")
    for packet, commitment in (
        (replay_a, commitment_a),
        (replay_b, commitment_b),
    ):
        if seal_endpoint_numerical_replay_packet(packet) != commitment:
            raise ValueError("target replay packet or commitment changed")
        if packet.get("calibration_kind") != "target_target_replay":
            raise ValueError("merge requires target replay packets")
    for field in (
        "protocol_id",
        "row_id",
        "endpoint_direction_binding_sha256_private",
        "endpoint_direction_count_private",
        "model_manifest_sha256_private",
        "response_precision_receipt_sha256_private",
        "response_evaluation_dtype_private",
    ):
        if replay_a.get(field) != replay_b.get(field):
            raise ValueError(f"target replays disagree on {field}")
    if {replay_a.get("replay_id_private"), replay_b.get("replay_id_private")} != {
        "A",
        "B",
    }:
        raise ValueError("target replays must contain exactly A and B")
    if replay_a.get("worker_instance_id_private") == replay_b.get(
        "worker_instance_id_private"
    ):
        raise ValueError("target replays must come from distinct worker instances")

    effects_a = torch.as_tensor(
        replay_a.get("endpoint_replay_effects_private", []), dtype=torch.float64
    )
    effects_b = torch.as_tensor(
        replay_b.get("endpoint_replay_effects_private", []), dtype=torch.float64
    )
    if effects_a.ndim != 1 or effects_a.shape != effects_b.shape or effects_a.numel() == 0:
        raise ValueError("target replay effects must have the same nonempty shape")
    if not torch.isfinite(effects_a).all() or not torch.isfinite(effects_b).all():
        raise ValueError("target replay effects must be finite")
    difference = effects_b - effects_a
    rms_difference = float(torch.sqrt(torch.mean(difference.square())).cpu())
    max_absolute_difference = float(torch.max(torch.abs(difference)).cpu())
    replay_scale = float(
        torch.max(torch.stack((torch.max(torch.abs(effects_a)), torch.max(torch.abs(effects_b))))).cpu()
    )
    allowed_max_absolute_difference = absolute_tolerance + relative_tolerance * replay_scale
    if not all(
        math.isfinite(value)
        for value in (
            rms_difference,
            max_absolute_difference,
            replay_scale,
            allowed_max_absolute_difference,
        )
    ):
        raise ValueError("target replay stability statistic is non-finite")

    packet = {
        "schema_version": "green-v400-sfc-target-replay-stability-v1",
        "protocol_id": replay_a["protocol_id"],
        "row_id": replay_a["row_id"],
        "route": "endpoint_numerical_replay",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "calibration_kind": "target_target_numerical_replay_gate",
        "endpoint_direction_binding_sha256_private": replay_a[
            "endpoint_direction_binding_sha256_private"
        ],
        "replay_a_packet_sha256_private": commitment_a[
            "endpoint_numerical_replay_packet_sha256"
        ],
        "replay_b_packet_sha256_private": commitment_b[
            "endpoint_numerical_replay_packet_sha256"
        ],
        "worker_instances_distinct_private": True,
        "replay_rms_difference_private": rms_difference,
        "replay_max_absolute_difference_private": max_absolute_difference,
        "replay_effect_scale_private": replay_scale,
        "replay_absolute_tolerance_private": absolute_tolerance,
        "replay_relative_tolerance_private": relative_tolerance,
        "replay_allowed_max_absolute_difference_private": allowed_max_absolute_difference,
        "numerical_replay_stable_private": (
            max_absolute_difference <= allowed_max_absolute_difference
        ),
        "scientific_null_distribution_claimed": False,
        "defines_transport_failure_label": False,
    }
    return packet, seal_endpoint_numerical_replay_packet(packet)
