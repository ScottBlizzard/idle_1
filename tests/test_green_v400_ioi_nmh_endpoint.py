from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_ioi_nmh_endpoint import compute_ioi_nmh_endpoint
from green_v400_ioi_response_adapter import IOIInterventionSite


PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
ROW_ID = "56" * 32


def prediction_commitment():
    return seal_prediction_packet(
        {
            "schema_version": "green-v400-sfc-prediction-packet-v1",
            "protocol_id": PROTOCOL,
            "row_id": ROW_ID,
            "route": "prediction",
            "contains_endpoint_outcome": False,
            "committed_before_endpoint": True,
        }
    )


class FakeNMHModel:
    def __init__(self, width=3, heads=12):
        self.width = width
        self.heads = heads

    def run_with_hooks(self, tokens, fwd_hooks):
        seq = tokens.shape[1]
        context = tokens[0, 0].to(torch.float64)
        activation = torch.stack([tokens.to(torch.float64)] * self.width, dim=-1)
        activation[:, 1, :] = activation[:, 1, :] + context
        for name, hook in fwd_hooks:
            if name.endswith("hook_resid_post"):
                activation = hook(activation, None)
        for name, hook in fwd_hooks:
            if name.endswith("hook_pattern"):
                pattern = torch.zeros((1, self.heads, seq, seq), dtype=torch.float64)
                pattern[:, :, -1, 1] = activation[0, 1].mean() / 10.0
                hook(pattern, None)
        return torch.zeros((1, seq, 16), dtype=torch.float64)


def site(layer=8):
    return IOIInterventionSite(
        layer=layer, position=1, io_token_id=5, s_token_id=7
    )


def test_nmh_endpoint_recovers_clean_structural_readout_independently():
    model = FakeNMHModel()
    clean = torch.tensor([[3, 4, 5]])
    corrupt = torch.tensor([[1, 4, 5]])
    # clean attention 0.7, corrupt 0.5, patched 0.7; denominator is computed internally.
    packet, commitment = compute_ioi_nmh_endpoint(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        prediction_commitment=prediction_commitment(),
        model=model,
        clean_tokens=clean,
        corrupt_tokens=corrupt,
        site=site(),
    )
    assert packet["contains_prediction"] is False
    assert packet["endpoint_nmh_temporally_eligible_private"] is True
    assert packet["nmh_recovery_private"] == pytest.approx(1.0)
    assert commitment["prediction_committed_before_endpoint"] is True


def test_nmh_head_not_strictly_downstream_is_rejected():
    with pytest.raises(ValueError, match="strictly downstream"):
        compute_ioi_nmh_endpoint(
            protocol_id=PROTOCOL,
            row_id=ROW_ID,
            prediction_commitment=prediction_commitment(),
            model=FakeNMHModel(),
            clean_tokens=torch.tensor([[3, 4, 5]]),
            corrupt_tokens=torch.tensor([[1, 4, 5]]),
            site=site(layer=9),
        )


def test_degenerate_internal_clean_minus_corrupt_denominator_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        compute_ioi_nmh_endpoint(
            protocol_id=PROTOCOL,
            row_id=ROW_ID,
            prediction_commitment=prediction_commitment(),
            model=FakeNMHModel(),
            clean_tokens=torch.tensor([[3, 4, 5]]),
            corrupt_tokens=torch.tensor([[3, 4, 5]]),
            site=site(),
        )
