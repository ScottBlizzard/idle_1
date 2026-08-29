from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_four_branch_baseline import (
    empirical_four_branch_interaction_response_batched,
)
from green_v400_matched_bypass_adapter import build_matched_bypass_four_branches


class FakeGateModel:
    """Tiny differentiable site -> selected gate -> residual bypass model."""

    def run_with_hooks(self, tokens, fwd_hooks):
        batch, seq = tokens.shape
        residual = torch.zeros((batch, seq, 2), dtype=torch.float64)
        residual[:, :, 0] = tokens.to(torch.float64)
        residual[:, :, 1] = 0.5 * tokens.to(torch.float64)
        for name, hook in fwd_hooks:
            if name == "blocks.2.hook_resid_post":
                residual = hook(residual, None)
        post = torch.zeros((batch, seq, 3), dtype=torch.float64)
        post[:, :, 0] = residual[:, :, 0] ** 2
        post[:, :, 1] = torch.sin(residual[:, :, 1])
        post[:, :, 2] = residual[:, :, 0] * residual[:, :, 1]
        for name, hook in fwd_hooks:
            if name == "blocks.10.mlp.hook_post":
                post = hook(post, None)
        logits = torch.zeros((batch, seq, 2), dtype=torch.float64)
        # Keep the direct residual bypass and selected-gate contribution.
        logits[:, -1, 0] = residual[:, -1, 0] + post[:, -1, 0]
        logits[:, -1, 1] = residual[:, -1, 1] + post[:, -1, 1]
        return logits


def test_matched_bypass_adapter_freezes_only_selected_gate_and_keeps_bypass():
    model = FakeGateModel()
    clean = torch.tensor([[3, 4, 5]])
    corrupt = torch.tensor([[1, 4, 5]])
    center = torch.tensor([0.2, -0.1], dtype=torch.float64)
    directions = torch.tensor([[0.3, 0.2], [-0.1, 0.4]], dtype=torch.float64)
    branches = build_matched_bypass_four_branches(
        model=model,
        clean_tokens=clean,
        corrupt_tokens=corrupt,
        intervention_hook="blocks.2.hook_resid_post",
        intervention_position=2,
        center=center,
        score_logits=lambda logits: logits[:, -1, 0] - logits[:, -1, 1],
        selected_gates=(0, 1),
    )
    result = empirical_four_branch_interaction_response_batched(
        branches, center, directions
    )
    # PAT and TAR share downstream mechanics in this fake model, so their
    # matched interaction cancels.  A missing residual bypass would not test
    # the intended branch construction.
    torch.testing.assert_close(result.psi_effects, torch.zeros(2, dtype=torch.float64))
    for response in branches.values():
        assert response.supports_batch is True
