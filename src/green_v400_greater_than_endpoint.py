"""Independent layer-10 MLP endpoint for the Greater-Than replication."""

from __future__ import annotations

import math
from typing import Any

import torch

from green_v400_endpoint_firewall import seal_endpoint_packet
from green_v400_greater_than_response_adapter import (
    GreaterThanInterventionSite,
    capture_resid_post_center,
    greater_than_contrast,
)


ENDPOINT_LAYER = 10
ENDPOINT_HOOK = "blocks.10.hook_mlp_out"


def _unembedding_readout(model: Any, site: GreaterThanInterventionSite) -> torch.Tensor:
    matrix = getattr(model, "W_U", None)
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise ValueError("model.W_U must be a rank-two tensor")
    suffix_ids = torch.as_tensor(
        site.suffix_token_ids, dtype=torch.long, device=matrix.device
    )
    if int(suffix_ids.max()) >= matrix.shape[1]:
        raise ValueError("suffix token identifier lies outside model.W_U")
    selected = matrix.index_select(1, suffix_ids)
    contrast = greater_than_contrast(
        site.clean_suffix, dtype=selected.dtype, device=selected.device
    )
    readout = selected @ contrast
    if readout.ndim != 1 or not torch.isfinite(readout).all():
        raise ValueError("Greater-Than unembedding readout is invalid")
    return readout


def _mlp10_projection(
    model: Any,
    tokens: torch.Tensor,
    readout: torch.Tensor,
    resid_hook: tuple[str, Any] | None = None,
) -> float:
    captured: list[torch.Tensor] = []

    def capture(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        if activation.ndim != 3 or activation.shape[0] != 1:
            raise ValueError("MLP endpoint activation must have shape [1, seq, d_model]")
        captured.append(activation[0, -1, :].detach().clone())
        return activation

    hooks = [] if resid_hook is None else [resid_hook]
    hooks.append((ENDPOINT_HOOK, capture))
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)
    if len(captured) != 1:
        raise RuntimeError("layer-10 MLP endpoint hook must fire exactly once")
    vector = captured[0]
    if vector.shape != readout.shape:
        raise ValueError("MLP endpoint width does not match unembedding readout")
    value = float((vector.to(readout) * readout).sum().cpu())
    if not math.isfinite(value):
        raise ValueError("MLP endpoint projection is non-finite")
    return value


def compute_greater_than_mlp_endpoint(
    *,
    protocol_id: str,
    row_id: str,
    prediction_commitment: dict[str, Any],
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    site: GreaterThanInterventionSite,
    endpoint_calibration_denominator: float,
    denominator_floor: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure structural recovery without exposing the endpoint to prediction."""

    if site.layer >= ENDPOINT_LAYER:
        raise ValueError("the layer-10 MLP endpoint must be strictly downstream")
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("clean and corrupt tokens must have equal shape")
    if not math.isfinite(endpoint_calibration_denominator):
        raise ValueError("endpoint calibration denominator must be finite")
    if denominator_floor <= 0:
        raise ValueError("denominator_floor must be positive")
    if abs(endpoint_calibration_denominator) < denominator_floor:
        raise ValueError("endpoint calibration denominator is degenerate")

    center = capture_resid_post_center(model, clean_tokens, site)
    readout = _unembedding_readout(model, site)

    def patch(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        result = activation.clone()
        result[0, site.position, :] = center.to(
            dtype=activation.dtype, device=activation.device
        )
        return result

    clean_projection = _mlp10_projection(model, clean_tokens, readout)
    corrupt_projection = _mlp10_projection(model, corrupt_tokens, readout)
    patched_projection = _mlp10_projection(
        model,
        corrupt_tokens,
        readout,
        resid_hook=(site.hook_name, patch),
    )
    recovery = (
        patched_projection - corrupt_projection
    ) / endpoint_calibration_denominator
    if not math.isfinite(recovery):
        raise ValueError("Greater-Than structural recovery is non-finite")

    packet = {
        "schema_version": "green-v400-sfc-greater-than-mlp-endpoint-v1",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "route": "endpoint",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "endpoint_hook_private": ENDPOINT_HOOK,
        "endpoint_temporally_eligible_private": True,
        "endpoint_clean_projection_private": clean_projection,
        "endpoint_corrupt_projection_private": corrupt_projection,
        "endpoint_patched_projection_private": patched_projection,
        "greater_than_mlp_recovery_private": recovery,
        "endpoint_denominator_source_private": "endpoint_calibration_only",
    }
    return packet, seal_endpoint_packet(packet, prediction_commitment)
