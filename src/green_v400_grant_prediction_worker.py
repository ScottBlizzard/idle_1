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
    phase: str,
    diagnostic_label: str,
    natural_states: torch.Tensor,
    intervened_states: torch.Tensor,
    unpatched_corrupt_states: torch.Tensor,
    seed: int,
    sample_size: int = 5000,
    sinkhorn_loss: SinkhornLoss | None = None,
    formal_execution_binding: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute and commit one development-only cohort prediction.

    `cohort_id` occupies the firewall's row identity slot and must be a stable
    64-character digest of the ordered development cohort manifest.
    """

    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("protocol_id must be nonempty")
    if not isinstance(cohort_id, str) or len(cohort_id) != 64:
        raise ValueError("cohort_id must be a 64-character hexadecimal digest")
    try:
        int(cohort_id, 16)
    except ValueError as error:
        raise ValueError("cohort_id must be hexadecimal") from error
    if phase not in {"development", "confirmation"}:
        raise ValueError("Grant phase must be development or confirmation")
    if diagnostic_label != "grant_style_downstream_contextual_divergence_extension":
        raise ValueError("Grant diagnostic label is not the frozen extension")

    panel = grant_divergence_panel(
        natural_states,
        intervened_states,
        seed=seed,
        sample_size=sample_size,
        sinkhorn_loss=sinkhorn_loss,
    )
    contextual_control = grant_divergence_panel(
        natural_states,
        unpatched_corrupt_states,
        seed=seed,
        sample_size=sample_size,
        sinkhorn_loss=sinkhorn_loss,
    )
    panel_values = panel.to_dict()
    control_values = contextual_control.to_dict()
    packet = {
        "schema_version": "green-v400-grant-divergence-prediction-v2",
        "protocol_id": protocol_id,
        "row_id": cohort_id,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "scope": f"{phase}_phase_by_layer_cohort_only",
        "phase": phase,
        "diagnostic_label": diagnostic_label,
        "grant_style_divergence": {
            "patched_vs_clean": panel_values,
            "unpatched_corrupt_vs_clean_control": control_values,
            "excess_normalized_sinkhorn": panel_values["emd"]
            - panel_values["base_emd"],
            "control_excess_normalized_sinkhorn": control_values["emd"]
            - control_values["base_emd"],
            "negative_excess_values_preserved": True,
        },
        "source_repository_commit": (
            "f2548d2ea9b4f4b87a87ba5d53db43838d15c521"
        ),
    }
    if formal_execution_binding is not None:
        packet["formal_execution_binding"] = formal_execution_binding
    return packet, seal_prediction_packet(packet)
