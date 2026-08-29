"""Outcome-isolated prediction worker core for the silent-failure challenge."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import torch

from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_four_branch_baseline import (
    empirical_four_branch_interaction_response,
    empirical_four_branch_interaction_response_batched,
)
from green_v400_response_baselines import (
    compare_batched_response_fields,
    compare_response_fields,
)


BASELINE_METHODS = (
    "finite_activation_patching",
    "first_order_attribution",
    "integrated_gradients",
    "single_point_hvp",
    "ms_hvp",
    "empirical_four_branch_interaction",
)

_RESPONSE_METHODS = {
    "finite_activation_patching": "exact",
    "first_order_attribution": "first_order",
    "integrated_gradients": "integrated_gradients",
    "single_point_hvp": "hvp",
    "ms_hvp": "ms_hvp",
}


def compute_normalized_mismatch_surrogate(
    target_effects: torch.Tensor,
    discrepancies: torch.Tensor,
    *,
    alpha: float = 0.05,
    normalization_floor: float = 1e-12,
) -> dict[str, Any]:
    """Describe normalized finite-response mismatch and a Gaussian surrogate.

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
    normalized_mismatch = mismatch_rms / scale
    direction_count = int(target_effects.numel())
    noncentrality = math.sqrt(direction_count) * normalized_mismatch
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / 2.0)
    power = normal.cdf(-critical - noncentrality) + 1.0 - normal.cdf(
        critical - noncentrality
    )
    values = (
        mismatch_rms,
        target_rms,
        normalized_mismatch,
        noncentrality,
        critical,
        power,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("raw SNR analytic power produced a non-finite value")
    return {
        "direction_count": direction_count,
        "raw_mismatch_rmse": mismatch_rms,
        "target_response_rms": target_rms,
        "normalization_floor": normalization_floor,
        "normalized_finite_response_mismatch": normalized_mismatch,
        "sqrt_n_scaled_normalized_mismatch": noncentrality,
        "alpha_two_sided": alpha,
        "standard_normal_critical_value": critical,
        "gaussian_location_surrogate_power": min(1.0, max(0.0, power)),
        "assumption": "independent_direction_gaussian_location_surrogate_with_target_rms_unit_scale",
        "inferential_test_claimed": False,
        "independent_baseline_claimed": False,
    }


def compute_raw_snr_analytic_power(
    target_effects: torch.Tensor,
    discrepancies: torch.Tensor,
    *,
    alpha: float = 0.05,
    normalization_floor: float = 1e-12,
) -> dict[str, Any]:
    """Compatibility alias; new packets use the non-inferential name."""

    return compute_normalized_mismatch_surrogate(
        target_effects,
        discrepancies,
        alpha=alpha,
        normalization_floor=normalization_floor,
    )


def _tensor_values(value: torch.Tensor) -> list[float]:
    result = [float(item) for item in value.detach().cpu().reshape(-1)]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("prediction baseline produced non-finite values")
    return result


def compute_ordinary_restoration(
    clean_score: torch.Tensor,
    corrupt_score: torch.Tensor,
    patched_score: torch.Tensor,
    *,
    denominator_floor: float = 1e-12,
) -> float:
    """Compute the conventional clean-minus-corrupt normalized restoration."""

    values = []
    for name, value in (
        ("clean", clean_score),
        ("corrupt", corrupt_score),
        ("patched", patched_score),
    ):
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"{name} score must be one scalar tensor")
        scalar = float(value.detach().double().cpu().reshape(()))
        if not math.isfinite(scalar):
            raise ValueError(f"{name} score must be finite")
        values.append(scalar)
    if denominator_floor <= 0:
        raise ValueError("ordinary restoration denominator floor must be positive")
    clean, corrupt, patched = values
    denominator = clean - corrupt
    if abs(denominator) < denominator_floor:
        raise ValueError("ordinary restoration clean-minus-corrupt denominator is degenerate")
    restoration = (patched - corrupt) / denominator
    if not math.isfinite(restoration):
        raise ValueError("ordinary restoration is non-finite")
    return restoration


def compute_response_baseline_packet(
    *,
    protocol_id: str,
    row_id: str,
    target_response: Any,
    patched_response: Any,
    four_branch_responses: dict[str, Any],
    center: torch.Tensor,
    green_directions: torch.Tensor,
    ordinary_restoration: float,
    integrated_gradients_steps: int = 65,
    ms_hvp_segments: int = 8,
    response_batch_chunk_size: int = 32,
    formal_execution_binding: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute and commit non-held-out prediction baselines for one row.

    `green_directions` is the public prediction panel. The function has no
    endpoint-panel parameter and the packet firewall rejects endpoint fields.
    """

    if not math.isfinite(ordinary_restoration):
        raise ValueError("ordinary restoration must be finite")
    if integrated_gradients_steps < 2:
        raise ValueError("integrated gradients requires at least two steps")
    if ms_hvp_segments < 2:
        raise ValueError("MS-HVP requires at least two path segments")
    if response_batch_chunk_size <= 0:
        raise ValueError("response batch chunk size must be positive")

    baselines: dict[str, Any] = {}
    batched = bool(
        getattr(target_response, "supports_batch", False)
        and getattr(patched_response, "supports_batch", False)
    )
    for public_name, method in _RESPONSE_METHODS.items():
        compare = compare_batched_response_fields if batched else compare_response_fields
        compare_kwargs = {
            "integrated_gradients_steps": integrated_gradients_steps,
            "ms_hvp_segments": ms_hvp_segments,
        }
        if batched:
            compare_kwargs["batch_chunk_size"] = response_batch_chunk_size
        comparison = compare(
            method,
            target_response,
            patched_response,
            center,
            green_directions,
            **compare_kwargs,
        )
        baselines[public_name] = {
            "target_effects": _tensor_values(comparison.target_effects),
            "patched_effects": _tensor_values(comparison.patched_effects),
            "discrepancies": _tensor_values(comparison.discrepancies),
            "rmse": comparison.rmse,
            "normalized_rmse": comparison.normalized_rmse,
            "diagnostics": comparison.diagnostics,
        }

    four_branch_batched = bool(
        four_branch_responses
        and all(
            getattr(response, "supports_batch", False)
            for response in four_branch_responses.values()
        )
    )
    four_branch_compare = (
        empirical_four_branch_interaction_response_batched
        if four_branch_batched
        else empirical_four_branch_interaction_response
    )
    interaction = four_branch_compare(
        four_branch_responses, center, green_directions
    )
    finite = baselines["finite_activation_patching"]
    target_scale = math.sqrt(
        sum(value * value for value in finite["target_effects"])
        / len(finite["target_effects"])
    )
    patched_scale = math.sqrt(
        sum(value * value for value in finite["patched_effects"])
        / len(finite["patched_effects"])
    )
    symmetric_scale = max(target_scale, patched_scale, 1e-12)
    baselines["empirical_four_branch_interaction"] = {
        "psi_at_center": float(interaction.psi_at_center.cpu()),
        "psi_effects": _tensor_values(interaction.psi_effects),
        "rms_effect": interaction.rms_effect,
        "symmetric_finite_response_scale": symmetric_scale,
        "normalized_risk_score": interaction.rms_effect / symmetric_scale,
        "diagnostics": {
            "branch_order": list(interaction.branch_order),
            "branch_weights": list(interaction.branch_weights),
            "point_sampling_only": interaction.point_sampling_only,
            "certificate_claimed": interaction.certificate_claimed,
            "response_batching": four_branch_batched,
        },
    }
    exact = baselines["finite_activation_patching"]
    mismatch_description = compute_normalized_mismatch_surrogate(
        torch.as_tensor(exact["target_effects"], dtype=torch.float64),
        torch.as_tensor(exact["discrepancies"], dtype=torch.float64),
    )
    packet = {
        "schema_version": "green-v400-sfc-prediction-packet-v2",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "ordinary_restoration": float(ordinary_restoration),
        "response_baselines": baselines,
        "normalized_mismatch_description": mismatch_description,
        "integrated_gradients_steps": integrated_gradients_steps,
        "ms_hvp_segments": ms_hvp_segments,
        "response_batch_chunk_size": response_batch_chunk_size,
        "response_batching": batched,
    }
    if formal_execution_binding is not None:
        packet["formal_execution_binding"] = formal_execution_binding
    return packet, seal_prediction_packet(packet)
