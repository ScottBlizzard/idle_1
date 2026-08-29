import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_grant_divergence import (
    grant_divergence_panel,
    normalized_sinkhorn_emd,
)


class RecordingLoss:
    def __init__(self):
        self.calls = []

    def __call__(self, first, second):
        self.calls.append((first.clone(), second.clone()))
        return (first - second).square().mean()


def test_sinkhorn_path_matches_official_standardize_then_sqrt_width_formula():
    first = torch.tensor([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    second = torch.tensor([[2.0, 7.0], [4.0, 7.0], [8.0, 7.0]])
    loss = RecordingLoss()
    result = normalized_sinkhorn_emd(first, second, sinkhorn_loss=loss)
    x, y = loss.calls[0]
    official_x = (first - first.mean(0)) / (first.std(0) + 1e-5)
    official_y = (second - second.mean(0)) / (second.std(0) + 1e-5)
    torch.testing.assert_close(x, official_x)
    torch.testing.assert_close(y, official_y)
    assert torch.allclose(x[:, 0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.equal(x[:, 1], torch.zeros(3))
    assert torch.equal(y[:, 1], torch.zeros(3))
    assert result == pytest.approx(float((x - y).square().mean()) / math.sqrt(2))


def test_actual_geomloss_matches_inline_primary_source_formula_when_available():
    geomloss = pytest.importorskip("geomloss")
    generator = torch.Generator().manual_seed(991)
    first = torch.randn(10, 6, generator=generator)
    second = torch.randn(10, 6, generator=generator) + 0.15
    official_first = (first - first.mean(0)) / (first.std(0) + 1e-5)
    official_second = (second - second.mean(0)) / (second.std(0) + 1e-5)
    official_loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    expected = float(official_loss(official_first, official_second)) / math.sqrt(6)
    actual = normalized_sinkhorn_emd(first, second)
    assert actual == pytest.approx(expected, rel=1e-7, abs=1e-9)


def test_grant_panel_is_deterministic_and_reports_natural_control():
    generator = torch.Generator().manual_seed(17)
    natural = torch.randn(20, 5, generator=generator)
    intervened = natural + 0.25
    fake = lambda x, y: (x - y).square().mean()
    first = grant_divergence_panel(
        natural, intervened, seed=123, sample_size=20, sinkhorn_loss=fake
    )
    second = grant_divergence_panel(
        natural, intervened, seed=123, sample_size=20, sinkhorn_loss=fake
    )
    assert first == second
    assert first.sample_size == 20
    assert first.split_size == 10
    assert first.mse == pytest.approx(0.25**2)
    for value in first.to_dict().values():
        if isinstance(value, float):
            assert math.isfinite(value)
    assert hasattr(first, "base_emd")
    assert hasattr(first, "base_cost_cos")
    assert hasattr(first, "base_nn_mse")


def test_matching_and_nearest_neighbor_metrics_have_expected_zero_identity_case():
    natural = torch.tensor(
        [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 5.0],
         [7.0, 2.0], [8.0, 9.0], [10.0, 3.0], [11.0, 12.0]]
    )
    fake = lambda x, y: (x - y).square().mean()
    panel = grant_divergence_panel(
        natural, natural.clone(), seed=9, sample_size=8, sinkhorn_loss=fake
    )
    assert panel.mse == 0.0
    # Cohort halves differ, so distribution distances need not vanish; observed
    # and natural-control paths must nevertheless be computed by the same rule.
    assert np.isfinite(panel.cost_cos)
    assert np.isfinite(panel.base_cost_cos)
    assert panel.emd == pytest.approx(panel.base_emd)
    assert panel.cost_mse == pytest.approx(panel.base_cost_mse)


def test_extension_requires_the_entire_even_sealed_cohort():
    states = torch.randn(8, 3)
    with pytest.raises(ValueError, match="entire sealed cohort"):
        grant_divergence_panel(
            states,
            states,
            seed=1,
            sample_size=6,
            sinkhorn_loss=lambda x, y: torch.tensor(0.0),
        )
    with pytest.raises(ValueError, match="even sealed cohort"):
        grant_divergence_panel(
            states[:7],
            states[:7],
            seed=1,
            sample_size=7,
            sinkhorn_loss=lambda x, y: torch.tensor(0.0),
        )


@pytest.mark.parametrize(
    "natural,intervened",
    [
        (torch.ones(3, 2), torch.ones(3, 2)),
        (torch.ones(4, 2), torch.ones(4, 3)),
        (torch.ones(4), torch.ones(4)),
        (torch.tensor([[float("nan")], [0.0], [1.0], [2.0]]), torch.ones(4, 1)),
    ],
)
def test_invalid_panels_fail_closed(natural, intervened):
    with pytest.raises(ValueError):
        grant_divergence_panel(
            natural,
            intervened,
            seed=1,
            sinkhorn_loss=lambda x, y: torch.tensor(0.0),
        )
