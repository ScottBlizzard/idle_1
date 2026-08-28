from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_endpoint_worker import (
    compute_heldout_transport_endpoint,
    split_conformal_upper_tail_p,
)


PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
ROW_ID = "34" * 32


def prediction_commitment():
    packet = {
        "protocol_id": PROTOCOL,
        "row_id": ROW_ID,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "predictions": {"ordinary_restoration": 0.9},
    }
    return seal_prediction_packet(packet)


def test_split_conformal_upper_tail_is_conservative_with_ties():
    assert split_conformal_upper_tail_p([0.1, 0.2, 0.2, 0.4], 0.2) == pytest.approx(0.8)
    assert split_conformal_upper_tail_p([0.1, 0.2, 0.3, 0.4], 0.5) == pytest.approx(0.2)


def test_endpoint_detects_large_heldout_transport_failure_without_prediction_access():
    center = torch.tensor([0.0], dtype=torch.float64)
    directions = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    packet, commitment = compute_heldout_transport_endpoint(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        prediction_commitment=prediction_commitment(),
        target_response=lambda x: x[0],
        patched_response=lambda x: 3.0 * x[0],
        center=center,
        endpoint_directions=directions,
        endpoint_calibration_scores=[0.0] * 39,
        failure_alpha=0.05,
    )
    assert packet["contains_prediction"] is False
    assert packet["heldout_transport_normalized_error_private"] == pytest.approx(2.0)
    assert packet["endpoint_conformal_upper_tail_p_private"] == pytest.approx(0.025)
    assert packet["endpoint_failure_label_private"] is True
    assert commitment["prediction_packet_sha256"] == prediction_commitment()[
        "prediction_packet_sha256"
    ]


def test_identical_response_fields_are_not_failures():
    center = torch.tensor([0.2, -0.1], dtype=torch.float64)
    directions = torch.tensor([[0.1, 0.4], [-0.2, 0.3]], dtype=torch.float64)
    response = lambda x: x[0] ** 2 + x[1]
    packet, _ = compute_heldout_transport_endpoint(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        prediction_commitment=prediction_commitment(),
        target_response=response,
        patched_response=response,
        center=center,
        endpoint_directions=directions,
        endpoint_calibration_scores=[0.0] * 39,
    )
    assert packet["heldout_transport_error_private"] == pytest.approx(0.0)
    assert packet["endpoint_conformal_upper_tail_p_private"] == pytest.approx(1.0)
    assert packet["endpoint_failure_label_private"] is False


def test_endpoint_requires_calibration_and_valid_alpha():
    kwargs = dict(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        prediction_commitment=prediction_commitment(),
        target_response=lambda x: x[0],
        patched_response=lambda x: x[0],
        center=torch.tensor([0.0], dtype=torch.float64),
        endpoint_directions=torch.tensor([[1.0]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="nonempty"):
        compute_heldout_transport_endpoint(
            **kwargs, endpoint_calibration_scores=[]
        )
    with pytest.raises(ValueError, match="strictly"):
        compute_heldout_transport_endpoint(
            **kwargs, endpoint_calibration_scores=[0.0], failure_alpha=0.0
        )
    with pytest.raises(ValueError, match="cannot attain"):
        compute_heldout_transport_endpoint(
            **kwargs, endpoint_calibration_scores=[0.0] * 18, failure_alpha=0.05
        )
