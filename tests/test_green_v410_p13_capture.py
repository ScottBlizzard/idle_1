from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_p13_capture import capture_controlled_hook_t0, compute_response_only_panel


class Response:
    def __init__(self, linear: torch.Tensor, quadratic: float):
        self.linear = linear
        self.quadratic = quadratic

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.linear + self.quadratic * (value * value).sum(dim=-1)


def test_response_only_panel_uses_fixed_secant_and_independent_ad():
    center = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64)
    directions = torch.tensor([
        [1.0 + index / 16, -0.5, 0.25] for index in range(8)
    ], dtype=torch.float64)
    linear = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    branches = {
        "PAT_J": Response(linear, 0.5),
        "PAT_B": Response(linear * 0.5, -0.25),
        "TAR_J": Response(linear * -1.0, 0.75),
        "TAR_B": Response(linear * 2.0, 0.125),
    }
    result = compute_response_only_panel(
        branches=branches, center=center, directions=directions
    )
    assert result["central_secant_radius"] == [1, 4]
    for name, branch in branches.items():
        gradient = branch.linear + 2 * branch.quadratic * center
        expected = directions @ gradient
        for secant, derivative, value in zip(
            result["central_secants"][name],
            result["ad_derivatives"][name],
            expected,
        ):
            assert Fraction(*secant) == Fraction(*derivative)
            assert Fraction(*secant) == Fraction(*float(value).as_integer_ratio())


class KeywordHookModel:
    def __init__(self):
        self.calls = 0

    def run_with_hooks(self, tokens, *, fwd_hooks):
        self.calls += 1
        activation = torch.full(
            (1, tokens.shape[1], 3), float(self.calls), dtype=torch.float64
        )
        _, callback = fwd_hooks[0]
        return callback(activation, hook=object())


def test_controlled_capture_accepts_transformer_lens_keyword_hook_abi():
    model = KeywordHookModel()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    center, pat, tar = capture_controlled_hook_t0(
        model=model, clean_tokens=tokens, corrupt_tokens=tokens + 1,
        hook_name="blocks.0.hook_resid_post", site_position=2,
    )
    assert model.calls == 2
    assert center.tolist() == [1.0, 1.0, 1.0]
    assert torch.equal(pat[2], center)
    assert torch.equal(tar[2], center)
