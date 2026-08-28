"""Outcome-isolated prediction worker core for the silent-failure challenge."""

from __future__ import annotations

import math
from typing import Any

import torch

from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_response_baselines import (
    compare_batched_response_fields,
    compare_response_fields,
)


BASELINE_METHODS = (
    "exact",
    "first_order",
    "integrated_gradients",
    "hvp",
)


def _tensor_values(value: torch.Tensor) -> list[float]:
    result = [float(item) for item in value.detach().cpu().reshape(-1)]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("prediction baseline produced non-finite values")
    return result


def compute_response_baseline_packet(
    *,
    protocol_id: str,
    row_id: str,
    target_response: Any,
    patched_response: Any,
    center: torch.Tensor,
    green_directions: torch.Tensor,
    ordinary_restoration: float,
    integrated_gradients_steps: int = 65,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute and commit non-held-out prediction baselines for one row.

    `green_directions` is the public prediction panel. The function has no
    endpoint-panel parameter and the packet firewall rejects endpoint fields.
    """

    if not math.isfinite(ordinary_restoration):
        raise ValueError("ordinary restoration must be finite")
    if integrated_gradients_steps < 2:
        raise ValueError("integrated gradients requires at least two steps")

    baselines: dict[str, Any] = {}
    batched = bool(
        getattr(target_response, "supports_batch", False)
        and getattr(patched_response, "supports_batch", False)
    )
    for method in BASELINE_METHODS:
        compare = compare_batched_response_fields if batched else compare_response_fields
        comparison = compare(
            method,
            target_response,
            patched_response,
            center,
            green_directions,
            integrated_gradients_steps=integrated_gradients_steps,
        )
        baselines[method] = {
            "target_effects": _tensor_values(comparison.target_effects),
            "patched_effects": _tensor_values(comparison.patched_effects),
            "discrepancies": _tensor_values(comparison.discrepancies),
            "rmse": comparison.rmse,
            "normalized_rmse": comparison.normalized_rmse,
        }

    exact = baselines["exact"]
    direction_count = int(green_directions.shape[0])
    analytic_features = {
        "direction_count": direction_count,
        "raw_mismatch_rmse": exact["rmse"],
        "raw_normalized_mismatch": exact["normalized_rmse"],
        "sqrt_n_scaled_mismatch": math.sqrt(direction_count)
        * exact["normalized_rmse"],
    }
    packet = {
        "schema_version": "green-v400-sfc-prediction-packet-v1",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "ordinary_restoration": float(ordinary_restoration),
        "response_baselines": baselines,
        "raw_snr_analytic_features": analytic_features,
        "integrated_gradients_steps": integrated_gradients_steps,
        "response_batching": batched,
    }
    return packet, seal_prediction_packet(packet)
