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
        natural_states=natural,
        intervened_states=intervened,
        seed=88,
        sample_size=12,
        sinkhorn_loss=fake,
    )
    assert packet["scope"] == "development_cohort_only"
    assert packet["grant_style_divergence"]["sample_size"] == 12
    assert commitment == seal_prediction_packet(packet)


def test_grant_worker_rejects_non_digest_cohort_identity():
    states = torch.randn(8, 3)
    with pytest.raises(ValueError, match="64-character"):
        compute_grant_divergence_prediction_packet(
            protocol_id=PROTOCOL,
            cohort_id="development",
            natural_states=states,
            intervened_states=states,
            seed=1,
            sinkhorn_loss=lambda x, y: torch.tensor(0.0),
        )
