from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_auto_lirpa_tail import GPT2ResidualTail, auto_lirpa_linf_bounds
from analysis.green_v400_auto_lirpa_historical_smoke import (
    classify_verifier_failure,
)


class AddBlock(nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        assert use_cache is False
        assert output_attentions is False
        assert attention_mask is not None
        return (hidden_states + self.amount,)


class TensorAddBlock(AddBlock):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        return hidden_states + self.amount


def build_tail():
    fixed = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
    )
    head = nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10)
    return GPT2ResidualTail(
        blocks=[AddBlock(0.25), AddBlock(-0.1)],
        final_layer_norm=nn.Identity(),
        lm_head=head,
        fixed_hidden=fixed,
        position=1,
        io_token_id=3,
        s_token_id=1,
    )


def test_tail_replaces_only_selected_position_and_returns_logit_contrast():
    tail = build_tail()
    injections = torch.tensor([[10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]])
    actual = tail(injections)
    # The selected position does not affect this identity-style tail's final
    # token, so both batch members have the same manually computed output.
    final = torch.tensor([7.15, 8.15, 9.15])
    expected_scalar = tail.lm_head(final)[3] - tail.lm_head(final)[1]
    torch.testing.assert_close(actual, expected_scalar.expand(2, 1))


def test_tail_accepts_current_transformers_tensor_block_return():
    tail = build_tail()
    tail.blocks[0] = TensorAddBlock(0.25)
    actual = tail(torch.tensor([[10.0, 20.0, 30.0]]))
    assert actual.shape == (1, 1)


@pytest.mark.parametrize(
    "fixed,position",
    [(torch.ones(3, 4), 0), (torch.ones(1, 3, 4), 3)],
)
def test_tail_validation_fails_closed(fixed, position):
    with pytest.raises(ValueError):
        GPT2ResidualTail(
            blocks=[AddBlock(0.0)],
            final_layer_norm=nn.Identity(),
            lm_head=nn.Linear(4, 5),
            fixed_hidden=fixed,
            position=position,
            io_token_id=1,
            s_token_id=2,
        )


def test_lirpa_argument_validation_precedes_optional_import():
    tail = build_tail()
    with pytest.raises(ValueError, match="shape"):
        auto_lirpa_linf_bounds(tail, torch.ones(3), epsilon=0.1)
    with pytest.raises(ValueError, match="positive"):
        auto_lirpa_linf_bounds(tail, torch.ones(1, 3), epsilon=0.0)


def test_only_exact_layer_norm_failure_signature_is_classified():
    assert classify_verifier_failure(
        AssertionError("Only positive values are supported in BoundReciprocal")
    ) == "STANDARD_LAYER_NORM_INTERVAL_DEPENDENCY"
    assert classify_verifier_failure(RuntimeError("unsupported operator")) is None
    assert classify_verifier_failure(AssertionError("different assertion")) is None
    assert classify_verifier_failure(
        RuntimeError(
            "The size of tensor a (196608) must match the size of tensor b "
            "(12288) at non-singleton dimension 0"
        )
    ) == "CROWN_INTERMEDIATE_BOUND_SHAPE_MISMATCH"
