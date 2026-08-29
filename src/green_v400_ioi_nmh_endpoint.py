"""Independent downstream Name Mover Head endpoint for IOI."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from green_v400_endpoint_firewall import seal_endpoint_packet
from green_v400_ioi_response_adapter import (
    IOIInterventionSite,
    capture_resid_post_center,
)


DEFAULT_NMH_HEADS = ((9, 9), (10, 0))
DENOMINATOR_FLOOR = 1e-6


def _mean_nmh_attention(
    model: Any,
    tokens: torch.Tensor,
    io_position: int,
    heads: tuple[tuple[int, int], ...],
    resid_hook: tuple[str, Any] | None = None,
) -> float:
    captured: dict[tuple[int, int], torch.Tensor] = {}

    def make_capture(layer: int, head: int):
        def capture(pattern: torch.Tensor, hook: Any) -> torch.Tensor:
            if pattern.ndim != 4:
                raise ValueError("attention pattern must have shape [batch, head, query, key]")
            captured[(layer, head)] = pattern[0, head, -1, io_position].detach()
            return pattern

        return capture

    hooks = [
        (f"blocks.{layer}.attn.hook_pattern", make_capture(layer, head))
        for layer, head in heads
    ]
    if resid_hook is not None:
        hooks.append(resid_hook)
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)
    if set(captured) != set(heads):
        raise RuntimeError("every declared NMH hook must fire exactly once")
    value = torch.stack([captured[head] for head in heads]).mean()
    result = float(value.cpu())
    if not math.isfinite(result):
        raise ValueError("NMH attention endpoint is non-finite")
    return result


def compute_ioi_nmh_endpoint(
    *,
    protocol_id: str,
    row_id: str,
    prediction_commitment: dict[str, Any],
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    site: IOIInterventionSite,
    heads: Iterable[tuple[int, int]] = DEFAULT_NMH_HEADS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute a task-specific structural endpoint without prediction access."""

    declared_heads = tuple((int(layer), int(head)) for layer, head in heads)
    if not declared_heads or len(declared_heads) != len(set(declared_heads)):
        raise ValueError("NMH heads must be nonempty and unique")
    if any(layer <= site.layer for layer, _ in declared_heads):
        raise ValueError("every NMH head must be strictly downstream of the patch")
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("clean and corrupt tokens must have equal shape")
    clean_center = capture_resid_post_center(model, clean_tokens, site)

    def patch(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        result = activation.clone()
        result[0, site.position, :] = clean_center.to(
            dtype=activation.dtype, device=activation.device
        )
        return result

    clean_attention = _mean_nmh_attention(
        model, clean_tokens, site.position, declared_heads
    )
    corrupt_attention = _mean_nmh_attention(
        model, corrupt_tokens, site.position, declared_heads
    )
    patched_attention = _mean_nmh_attention(
        model,
        corrupt_tokens,
        site.position,
        declared_heads,
        resid_hook=(site.hook_name, patch),
    )
    denominator = clean_attention - corrupt_attention
    if not math.isfinite(denominator) or abs(denominator) < DENOMINATOR_FLOOR:
        raise ValueError("internally computed clean-minus-corrupt denominator is degenerate")
    recovery = (patched_attention - corrupt_attention) / denominator
    if not math.isfinite(recovery):
        raise ValueError("NMH recovery is non-finite")

    packet = {
        "schema_version": "green-v400-sfc-ioi-nmh-endpoint-v1",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "route": "endpoint",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "endpoint_nmh_heads_private": [list(head) for head in declared_heads],
        "endpoint_nmh_temporally_eligible_private": True,
        "endpoint_nmh_clean_attention_private": clean_attention,
        "endpoint_nmh_corrupt_attention_private": corrupt_attention,
        "endpoint_nmh_patched_attention_private": patched_attention,
        "nmh_recovery_private": recovery,
        "endpoint_denominator_private": denominator,
        "endpoint_denominator_floor_private": DENOMINATOR_FLOOR,
        "endpoint_denominator_source_private": "internally_computed_clean_minus_corrupt_attention",
    }
    return packet, seal_endpoint_packet(packet, prediction_commitment)
