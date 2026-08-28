"""Prediction-route serialization for the Grant divergence cohort baseline."""

from __future__ import annotations

from typing import Any

import torch

from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_grant_divergence import SinkhornLoss, grant_divergence_panel


def compute_grant_divergence_prediction_packet(
    *,
    protocol_id: str,
    cohort_id: str,
    natural_states: torch.Tensor,
    intervened_states: torch.Tensor,
    seed: int,
    sample_size: int = 5000,
    sinkhorn_loss: SinkhornLoss | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute and commit one development-only cohort prediction.

    `cohort_id` occupies the firewall's row identity slot and must be a stable
    64-character digest of the ordered development cohort manifest.
    """

    panel = grant_divergence_panel(
        natural_states,
        intervened_states,
        seed=seed,
        sample_size=sample_size,
        sinkhorn_loss=sinkhorn_loss,
    )
    packet = {
        "schema_version": "green-v400-grant-divergence-prediction-v1",
        "protocol_id": protocol_id,
        "row_id": cohort_id,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "scope": "development_cohort_only",
        "grant_style_divergence": panel.to_dict(),
        "source_repository_commit": (
            "f2548d2ea9b4f4b87a87ba5d53db43838d15c521"
        ),
    }
    return packet, seal_prediction_packet(packet)
