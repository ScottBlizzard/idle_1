from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_endpoint_firewall import (
    PREDICTION_FORBIDDEN_KEYS,
    seal_prediction_packet,
)
from green_v400_prediction_worker import (
    BASELINE_METHODS,
    compute_raw_snr_analytic_power,
    compute_response_baseline_packet,
)


PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
ROW_ID = "12" * 32


class BatchedQuadratic:
    supports_batch = True

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return self.scale * x.square().sum(dim=1)


def test_worker_serializes_and_commits_all_shared_baselines_without_endpoint_fields():
    center = torch.tensor([0.2, -0.1], dtype=torch.float64)
    directions = torch.tensor([[0.1, 0.2], [-0.3, 0.4]], dtype=torch.float64)
    target = lambda x: x[0] ** 2 + 0.5 * x[1]
    patched = lambda x: 1.2 * x[0] ** 2 + 0.5 * x[1]
    packet, commitment = compute_response_baseline_packet(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        target_response=target,
        patched_response=patched,
        center=center,
        green_directions=directions,
        ordinary_restoration=0.91,
        integrated_gradients_steps=17,
    )
    assert set(packet["response_baselines"]) == set(BASELINE_METHODS)
    assert packet["contains_endpoint_outcome"] is False
    assert not (set(packet) & PREDICTION_FORBIDDEN_KEYS)
    assert commitment == seal_prediction_packet(packet)
    analytic = packet["raw_snr_analytic_features"]
    assert analytic["direction_count"] == 2
    assert analytic["inferential_test_claimed"] is False
    assert analytic["assumption"].startswith("independent_direction_gaussian")
    assert packet["response_batching"] is False


def test_exact_ig_and_hvp_agree_for_quadratic_response_pair():
    center = torch.tensor([0.3], dtype=torch.float64)
    directions = torch.tensor([[0.2], [-0.1]], dtype=torch.float64)
    packet, _ = compute_response_baseline_packet(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        target_response=lambda x: x[0] ** 2,
        patched_response=lambda x: 2.0 * x[0] ** 2,
        center=center,
        green_directions=directions,
        ordinary_restoration=0.9,
        integrated_gradients_steps=9,
    )
    exact = packet["response_baselines"]["exact"]["discrepancies"]
    assert packet["response_baselines"]["integrated_gradients"]["discrepancies"] == pytest.approx(exact)
    assert packet["response_baselines"]["hvp"]["discrepancies"] == pytest.approx(exact)


def test_worker_uses_vectorized_path_only_for_explicit_batch_responses():
    center = torch.tensor([0.3, -0.2], dtype=torch.float64)
    directions = torch.tensor([[0.2, 0.1], [-0.1, 0.4]], dtype=torch.float64)
    packet, _ = compute_response_baseline_packet(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        target_response=BatchedQuadratic(1.0),
        patched_response=BatchedQuadratic(1.2),
        center=center,
        green_directions=directions,
        ordinary_restoration=0.9,
        integrated_gradients_steps=9,
    )
    assert packet["response_batching"] is True


def test_nonfinite_restoration_and_short_ig_grid_fail_closed():
    center = torch.tensor([0.0], dtype=torch.float64)
    directions = torch.tensor([[0.1]], dtype=torch.float64)
    kwargs = dict(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        target_response=lambda x: x[0],
        patched_response=lambda x: x[0],
        center=center,
        green_directions=directions,
    )
    with pytest.raises(ValueError, match="finite"):
        compute_response_baseline_packet(**kwargs, ordinary_restoration=float("nan"))
    with pytest.raises(ValueError, match="at least two"):
        compute_response_baseline_packet(
            **kwargs, ordinary_restoration=0.9, integrated_gradients_steps=1
        )


def test_raw_snr_analytic_power_is_scale_invariant_and_has_alpha_at_null():
    target = torch.ones(4, dtype=torch.float64)
    null = compute_raw_snr_analytic_power(target, torch.zeros_like(target))
    signal = compute_raw_snr_analytic_power(target, 0.5 * torch.ones_like(target))
    rescaled = compute_raw_snr_analytic_power(
        7.0 * target, 3.5 * torch.ones_like(target)
    )
    assert null["raw_snr"] == pytest.approx(0.0)
    assert null["gaussian_location_surrogate_power"] == pytest.approx(0.05)
    assert signal["raw_snr"] == pytest.approx(0.5)
    assert signal["raw_snr"] == pytest.approx(rescaled["raw_snr"])
    assert signal["gaussian_location_surrogate_power"] > null[
        "gaussian_location_surrogate_power"
    ]


def test_raw_snr_analytic_power_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same nonempty"):
        compute_raw_snr_analytic_power(
            torch.ones(2, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="finite"):
        compute_raw_snr_analytic_power(
            torch.tensor([1.0]), torch.tensor([float("nan")])
        )
    with pytest.raises(ValueError, match="alpha"):
        compute_raw_snr_analytic_power(
            torch.ones(2), torch.ones(2), alpha=1.0
        )
