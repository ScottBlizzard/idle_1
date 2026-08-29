from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_grant_prediction_worker import (
    compute_grant_divergence_prediction_packet,
)


PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
COHORT_ID = "4d" * 32


def test_grant_worker_serializes_and_commits_without_endpoint_fields():
    generator = torch.Generator().manual_seed(4)
    natural = torch.randn(12, 4, generator=generator)
    intervened = natural + 0.1 * torch.randn(12, 4, generator=generator)
    fake = lambda x, y: (x - y).square().mean()
    packet, commitment = compute_grant_divergence_prediction_packet(
        protocol_id=PROTOCOL,
        cohort_id=COHORT_ID,
        phase="development",
        diagnostic_label="grant_style_downstream_contextual_divergence_extension",
        natural_states=natural,
        intervened_states=intervened,
        unpatched_corrupt_states=natural + 0.2,
        seed=88,
        sample_size=12,
        sinkhorn_loss=fake,
    )
    assert packet["scope"] == "development_phase_by_layer_cohort_only"
    assert packet["phase"] == "development"
    assert packet["grant_style_divergence"]["patched_vs_clean"]["sample_size"] == 12
    assert "unpatched_corrupt_vs_clean_control" in packet["grant_style_divergence"]
    assert commitment == seal_prediction_packet(packet)


def test_grant_worker_rejects_non_digest_cohort_identity():
    states = torch.randn(8, 3)
    with pytest.raises(ValueError, match="64-character"):
        compute_grant_divergence_prediction_packet(
            protocol_id=PROTOCOL,
            cohort_id="development",
            phase="development",
            diagnostic_label="grant_style_downstream_contextual_divergence_extension",
            natural_states=states,
            intervened_states=states,
            unpatched_corrupt_states=states,
            seed=1,
            sinkhorn_loss=lambda x, y: torch.tensor(0.0),
        )


def test_grant_worker_rejects_non_hexadecimal_cohort_identity_before_compute():
    states = torch.randn(8, 3)
    with pytest.raises(ValueError, match="hexadecimal"):
        compute_grant_divergence_prediction_packet(
            protocol_id=PROTOCOL,
            cohort_id="z" * 64,
            phase="development",
            diagnostic_label="grant_style_downstream_contextual_divergence_extension",
            natural_states=states,
            intervened_states=states,
            unpatched_corrupt_states=states,
            seed=1,
            sinkhorn_loss=lambda x, y: torch.tensor(0.0),
        )
