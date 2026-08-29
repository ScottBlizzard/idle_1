from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_four_branch_baseline import (
    empirical_four_branch_interaction_response,
    empirical_four_branch_interaction_response_batched,
)


def test_four_branch_scalar_uses_binding_order_and_subtracts_center():
    center = torch.tensor([0.2], dtype=torch.float64)
    directions = torch.tensor([[0.3], [-0.1]], dtype=torch.float64)
    branches = {
        "PAT_J": lambda x: 3.0 * x[0] ** 2,
        "PAT_B": lambda x: 1.0 * x[0] ** 2,
        "TAR_J": lambda x: 0.5 * x[0] ** 2,
        "TAR_B": lambda x: 0.25 * x[0] ** 2,
    }
    result = empirical_four_branch_interaction_response(branches, center, directions)
    coefficient = 3.0 - 1.0 - 0.5 + 0.25
    expected = coefficient * ((center[0] + directions[:, 0]) ** 2 - center[0] ** 2)
    torch.testing.assert_close(result.psi_effects, expected)
    assert result.branch_order == ("PAT_J", "PAT_B", "TAR_J", "TAR_B")
    assert result.branch_weights == (1, -1, -1, 1)
    assert result.point_sampling_only is True
    assert result.certificate_claimed is False


def test_batched_four_branch_matches_scalar():
    center = torch.tensor([0.2, -0.3], dtype=torch.float64)
    directions = torch.tensor([[0.1, 0.4], [-0.2, 0.5]], dtype=torch.float64)
    scales = {"PAT_J": 2.0, "PAT_B": 0.7, "TAR_J": 1.1, "TAR_B": 0.4}
    scalar = {
        name: (lambda x, scale=scale: scale * torch.sin(x).sum())
        for name, scale in scales.items()
    }
    batched = {
        name: (lambda x, scale=scale: scale * torch.sin(x).sum(dim=1))
        for name, scale in scales.items()
    }
    expected = empirical_four_branch_interaction_response(scalar, center, directions)
    actual = empirical_four_branch_interaction_response_batched(
        batched, center, directions
    )
    torch.testing.assert_close(actual.psi_effects, expected.psi_effects)
    assert actual.rms_effect == pytest.approx(expected.rms_effect)


def test_four_branch_rejects_missing_or_nonfinite_branch():
    center = torch.zeros(1, dtype=torch.float64)
    directions = torch.ones((1, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="binding branch order"):
        empirical_four_branch_interaction_response(
            {"PAT_J": lambda x: x[0]}, center, directions
        )
    branches = {
        "PAT_J": lambda x: x[0],
        "PAT_B": lambda x: x[0],
        "TAR_J": lambda x: x[0],
        "TAR_B": lambda x: torch.tensor(float("nan"), dtype=x.dtype),
    }
    with pytest.raises(ValueError, match="non-finite"):
        empirical_four_branch_interaction_response(branches, center, directions)
