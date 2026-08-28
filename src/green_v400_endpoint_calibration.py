"""Endpoint-only target--target replay calibration for silent failure."""

from __future__ import annotations

import math
from typing import Any

import torch

from green_v400_endpoint_firewall import seal_endpoint_calibration_packet
from green_v400_response_baselines import exact_finite_response


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
    endpoint_direction_commitment: str,
    replay_id: str,
    worker_instance_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one target replay in a prediction-inaccessible worker."""

    if replay_id not in {"A", "B"}:
        raise ValueError("replay_id must be A or B")
    _hex_digest(endpoint_direction_commitment, "endpoint direction commitment")
    _hex_digest(worker_instance_id, "worker instance identifier")
    effects = exact_finite_response(target_response, center, endpoint_directions)
    values = [float(value) for value in effects.detach().cpu()]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("target replay effects must be nonempty and finite")
    packet = {
        "schema_version": "green-v400-sfc-target-replay-v1",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "route": "endpoint_calibration",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "calibration_kind": "target_target_replay",
        "replay_id_private": replay_id,
        "worker_instance_id_private": worker_instance_id,
        "endpoint_direction_commitment_private": endpoint_direction_commitment,
        "endpoint_direction_count_private": int(endpoint_directions.shape[0]),
        "endpoint_replay_effects_private": values,
    }
    return packet, seal_endpoint_calibration_packet(packet)


def merge_target_replay_score(
    replay_a: dict[str, Any],
    commitment_a: dict[str, Any],
    replay_b: dict[str, Any],
    commitment_b: dict[str, Any],
    *,
    normalization_floor: float = 1e-12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one calibration score from two independently committed replays."""

    if normalization_floor <= 0:
        raise ValueError("normalization_floor must be positive")
    for packet, commitment in (
        (replay_a, commitment_a),
        (replay_b, commitment_b),
    ):
        if seal_endpoint_calibration_packet(packet) != commitment:
            raise ValueError("target replay packet or commitment changed")
        if packet.get("calibration_kind") != "target_target_replay":
            raise ValueError("merge requires target replay packets")
    for field in (
        "protocol_id",
        "row_id",
        "endpoint_direction_commitment_private",
        "endpoint_direction_count_private",
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
    error = torch.sqrt(torch.mean((effects_b - effects_a).square()))
    scale = torch.sqrt(torch.mean(effects_a.square()))
    score = float((error / torch.clamp(scale, min=normalization_floor)).cpu())
    if not math.isfinite(score):
        raise ValueError("target replay calibration score is non-finite")

    packet = {
        "schema_version": "green-v400-sfc-target-replay-score-v1",
        "protocol_id": replay_a["protocol_id"],
        "row_id": replay_a["row_id"],
        "route": "endpoint_calibration",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "calibration_kind": "target_target_replay_score",
        "endpoint_direction_commitment_private": replay_a[
            "endpoint_direction_commitment_private"
        ],
        "replay_a_packet_sha256_private": commitment_a[
            "endpoint_calibration_packet_sha256"
        ],
        "replay_b_packet_sha256_private": commitment_b[
            "endpoint_calibration_packet_sha256"
        ],
        "worker_instances_distinct_private": True,
        "endpoint_calibration_score_private": score,
        "normalization_floor_private": normalization_floor,
    }
    return packet, seal_endpoint_calibration_packet(packet)
