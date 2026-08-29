from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_greater_than_endpoint import compute_greater_than_mlp_endpoint
from green_v400_greater_than_response_adapter import (
    GreaterThanInterventionSite,
    build_target_and_patched_responses,
    capture_resid_post_center,
    greater_than_contrast,
)
from green_v400_response_baselines import (
    compare_batched_response_fields,
    compare_response_fields,
)


PROTOCOL = "GREEN_V400_SILENT_FAILURE_GT_REPLICATION_PREPARE_V1"
ROW_ID = "ab" * 32


class FakeGreaterThanModel:
    def __init__(self, width=3, vocab=120, clean_suffix=49):
        self.width = width
        self.vocab = vocab
        contrast = greater_than_contrast(
            clean_suffix, dtype=torch.float64, device=torch.device("cpu")
        )
        self.W_U = torch.zeros((width, vocab), dtype=torch.float64)
        self.W_U[0, :100] = contrast / contrast.square().sum()

    def run_with_hooks(self, tokens, fwd_hooks):
        batch, seq = tokens.shape
        activation = torch.stack([tokens.to(torch.float64)] * self.width, dim=-1)
        activation[:, 1, :] += tokens[:, :1].to(torch.float64)
        for name, hook in fwd_hooks:
            if name.endswith("hook_resid_post"):
                activation = hook(activation, None)
        mlp = activation[:, 1:2, :].expand(-1, seq, -1).clone()
        for name, hook in fwd_hooks:
            if name == "blocks.10.hook_mlp_out":
                mlp = hook(mlp, None)
        logits = torch.zeros((batch, seq, self.vocab), dtype=torch.float64)
        logits[:, -1, :] = activation[:, 1, :] @ self.W_U
        return logits


def site(layer=8):
    return GreaterThanInterventionSite(
        layer=layer,
        position=1,
        clean_suffix=49,
        suffix_token_ids=tuple(range(100)),
    )


def prediction_commitment():
    return seal_prediction_packet({
        "schema_version": "green-v400-sfc-prediction-packet-v1",
        "protocol_id": PROTOCOL,
        "row_id": ROW_ID,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
    })


def test_response_adapter_matches_scalar_and_batched_shared_baselines():
    model = FakeGreaterThanModel()
    clean = torch.tensor([[3, 4, 5]])
    corrupt = torch.tensor([[1, 4, 5]])
    target, patched = build_target_and_patched_responses(model, clean, corrupt, site())
    center = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    directions = torch.tensor([[0.1, 0.2, -0.3], [-0.2, 0.1, 0.4]], dtype=torch.float64)
    for method in ("exact", "first_order", "integrated_gradients", "hvp"):
        scalar = compare_response_fields(
            method, target, patched, center, directions, integrated_gradients_steps=5
        )
        batched = compare_batched_response_fields(
            method, target, patched, center, directions, integrated_gradients_steps=5
        )
        torch.testing.assert_close(batched.target_effects, scalar.target_effects)
        torch.testing.assert_close(batched.patched_effects, scalar.patched_effects)
        assert scalar.rmse == pytest.approx(0.0, abs=1e-12)


def test_layer10_mlp_endpoint_recovers_clean_projection_independently():
    model = FakeGreaterThanModel()
    clean = torch.tensor([[3, 4, 5]])
    corrupt = torch.tensor([[1, 4, 5]])
    packet, commitment = compute_greater_than_mlp_endpoint(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        prediction_commitment=prediction_commitment(),
        model=model,
        clean_tokens=clean,
        corrupt_tokens=corrupt,
        site=site(),
    )
    assert packet["contains_prediction"] is False
    assert packet["endpoint_temporally_eligible_private"] is True
    assert packet["greater_than_mlp_recovery_private"] == pytest.approx(1.0)
    assert commitment["prediction_committed_before_endpoint"] is True


def test_endpoint_and_adapter_fail_closed_on_invalid_contracts():
    with pytest.raises(ValueError, match="strictly downstream"):
        compute_greater_than_mlp_endpoint(
            protocol_id=PROTOCOL,
            row_id=ROW_ID,
            prediction_commitment=prediction_commitment(),
            model=FakeGreaterThanModel(),
            clean_tokens=torch.tensor([[3, 4, 5]]),
            corrupt_tokens=torch.tensor([[3, 4, 5]]),
            site=site(layer=10),
        )
    with pytest.raises(ValueError, match="degenerate"):
        compute_greater_than_mlp_endpoint(
            protocol_id=PROTOCOL,
            row_id=ROW_ID,
            prediction_commitment=prediction_commitment(),
            model=FakeGreaterThanModel(),
            clean_tokens=torch.tensor([[3, 4, 5]]),
            corrupt_tokens=torch.tensor([[3, 4, 5]]),
            site=site(),
        )
    with pytest.raises(ValueError, match="shape"):
        build_target_and_patched_responses(
            FakeGreaterThanModel(),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
            site(),
        )


def test_clean_center_capture_uses_declared_start_suffix_position():
    center = capture_resid_post_center(
        FakeGreaterThanModel(), torch.tensor([[3, 4, 5]]), site()
    )
    torch.testing.assert_close(
        center, torch.tensor([7.0, 7.0, 7.0], dtype=torch.float64)
    )
