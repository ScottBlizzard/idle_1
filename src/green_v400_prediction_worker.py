"""Outcome-isolated prediction worker core for the silent-failure challenge."""

from __future__ import annotations

import math
from statistics import NormalDist
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


def compute_raw_snr_analytic_power(
    target_effects: torch.Tensor,
    discrepancies: torch.Tensor,
    *,
    alpha: float = 0.05,
    normalization_floor: float = 1e-12,
) -> dict[str, Any]:
    """Compute a frozen raw-effect SNR and Gaussian power surrogate.

    The calculation is deliberately labelled a surrogate: it treats the RMS
    target response as a fixed unit scale and the panel directions as
    independent Gaussian observations. It is a prediction baseline, not a
    calibrated hypothesis test and not evidence for a GREEN claim.
    """

    if target_effects.ndim != 1 or discrepancies.ndim != 1:
        raise ValueError("raw SNR inputs must be one-dimensional")
    if target_effects.shape != discrepancies.shape or target_effects.numel() == 0:
        raise ValueError("raw SNR inputs must have the same nonempty shape")
    if not target_effects.is_floating_point() or not discrepancies.is_floating_point():
        raise ValueError("raw SNR inputs must be floating point")
    if not torch.isfinite(target_effects).all() or not torch.isfinite(discrepancies).all():
        raise ValueError("raw SNR inputs must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if normalization_floor <= 0:
        raise ValueError("normalization_floor must be positive")

    mismatch_rms = float(torch.sqrt(torch.mean(discrepancies.double().square())).cpu())
    target_rms = float(torch.sqrt(torch.mean(target_effects.double().square())).cpu())
    scale = max(target_rms, normalization_floor)
    raw_snr = mismatch_rms / scale
    direction_count = int(target_effects.numel())
    noncentrality = math.sqrt(direction_count) * raw_snr
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / 2.0)
    power = normal.cdf(-critical - noncentrality) + 1.0 - normal.cdf(
        critical - noncentrality
    )
    values = (mismatch_rms, target_rms, raw_snr, noncentrality, critical, power)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("raw SNR analytic power produced a non-finite value")
    return {
        "direction_count": direction_count,
        "raw_mismatch_rmse": mismatch_rms,
        "target_response_rms": target_rms,
        "normalization_floor": normalization_floor,
        "raw_snr": raw_snr,
        "sqrt_n_scaled_snr": noncentrality,
        "alpha_two_sided": alpha,
        "standard_normal_critical_value": critical,
        "gaussian_location_surrogate_power": min(1.0, max(0.0, power)),
        "assumption": "independent_direction_gaussian_location_surrogate_with_target_rms_unit_scale",
        "inferential_test_claimed": False,
    }


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
    analytic_features = compute_raw_snr_analytic_power(
        torch.as_tensor(exact["target_effects"], dtype=torch.float64),
        torch.as_tensor(exact["discrepancies"], dtype=torch.float64),
    )
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
