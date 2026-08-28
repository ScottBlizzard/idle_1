from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_ioi_response_adapter import (
    IOIInterventionSite,
    build_target_and_patched_responses,
    capture_resid_post_center,
)
from green_v400_response_baselines import compare_response_fields
from green_v400_response_baselines import compare_batched_response_fields


class FakeHookedModel:
    def __init__(self, width=3, vocab=16):
        self.width = width
        self.vocab = vocab
        self.hook_calls = 0

    def run_with_hooks(self, tokens, fwd_hooks):
        activation = torch.stack(
            [tokens.to(torch.float64)] * self.width, dim=-1
        )
        for _, hook in fwd_hooks:
            self.hook_calls += 1
            activation = hook(activation, None)
        logits = torch.zeros(
            (tokens.shape[0], tokens.shape[1], self.vocab), dtype=activation.dtype
        )
        injected = activation[:, 1]
        context = tokens[:, 0].to(torch.float64)
        logits[:, -1, 5] = injected[:, 0] + 2.0 * injected[:, 1] + context
        logits[:, -1, 7] = -injected[:, 2]
        return logits


def site():
    return IOIInterventionSite(
        layer=2, position=1, io_token_id=5, s_token_id=7
    )


def test_adapter_preserves_gradients_for_all_shared_baselines():
    model = FakeHookedModel()
    clean = torch.tensor([[3, 4, 5]])
    corrupt = torch.tensor([[1, 4, 5]])
    target, patched = build_target_and_patched_responses(model, clean, corrupt, site())
    center = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    directions = torch.tensor([[0.1, 0.2, -0.3], [-0.2, 0.1, 0.4]], dtype=torch.float64)
    for method in ("exact", "first_order", "integrated_gradients", "hvp"):
        result = compare_response_fields(
            method,
            target,
            patched,
            center,
            directions,
            integrated_gradients_steps=5,
        )
        # Context shifts the center value but not this fake model's response field.
        assert result.rmse == pytest.approx(0.0, abs=1e-12)


def test_batched_adapter_matches_scalar_baselines_and_reduces_hook_calls():
    model = FakeHookedModel()
    clean = torch.tensor([[3, 4, 5]])
    corrupt = torch.tensor([[1, 4, 5]])
    target, patched = build_target_and_patched_responses(model, clean, corrupt, site())
    center = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    directions = torch.tensor([[0.1, 0.2, -0.3], [-0.2, 0.1, 0.4]], dtype=torch.float64)
    for method in ("exact", "first_order", "integrated_gradients", "hvp"):
        scalar = compare_response_fields(
            method, target, patched, center, directions, integrated_gradients_steps=7
        )
        batched = compare_batched_response_fields(
            method, target, patched, center, directions, integrated_gradients_steps=7
        )
        torch.testing.assert_close(batched.target_effects, scalar.target_effects)
        torch.testing.assert_close(batched.patched_effects, scalar.patched_effects)


def test_clean_center_capture_uses_the_declared_site_once():
    model = FakeHookedModel()
    clean = torch.tensor([[3, 4, 5]])
    center = capture_resid_post_center(model, clean, site())
    torch.testing.assert_close(center, torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64))
    assert model.hook_calls == 1


def test_invalid_hook_family_and_token_shape_fail_closed():
    with pytest.raises(ValueError, match="resid_post"):
        IOIInterventionSite(
            layer=0,
            position=0,
            io_token_id=1,
            s_token_id=2,
            hook_family="resid_mid",
        )
    model = FakeHookedModel()
    with pytest.raises(ValueError, match="shape"):
        build_target_and_patched_responses(
            model,
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
            site(),
        )


def test_injection_width_mismatch_is_rejected():
    model = FakeHookedModel(width=3)
    clean = torch.tensor([[3, 4, 5]])
    response, _ = build_target_and_patched_responses(model, clean, clean, site())
    with pytest.raises(ValueError, match="width"):
        response(torch.tensor([0.1, 0.2], dtype=torch.float64))
