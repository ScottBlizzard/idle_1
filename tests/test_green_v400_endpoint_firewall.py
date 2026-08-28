import copy

import pytest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_endpoint_firewall import (
    audit_commitment_pair,
    seal_endpoint_packet,
    seal_prediction_packet,
)


ROW_ID = "ab" * 32
PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"


def prediction_packet():
    return {
        "protocol_id": PROTOCOL,
        "row_id": ROW_ID,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "predictions": {
            "ordinary_restoration": 0.91,
            "green_certificate_status": "UNRESOLVED"
        },
    }


def endpoint_packet():
    return {
        "protocol_id": PROTOCOL,
        "row_id": ROW_ID,
        "route": "endpoint",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "endpoint_responses_private": [0.1, 0.2],
        "heldout_transport_error_private": 0.3,
        "nmh_recovery_private": 0.4,
    }


def test_valid_packets_commit_and_reaudit_exactly():
    prediction = prediction_packet()
    prediction_commitment = seal_prediction_packet(prediction)
    endpoint = endpoint_packet()
    endpoint_commitment = seal_endpoint_packet(endpoint, prediction_commitment)
    audit_commitment_pair(
        prediction, prediction_commitment, endpoint, endpoint_commitment
    )


@pytest.mark.parametrize(
    "forbidden_key",
    ["endpoint_directions", "nmh_recovery", "heldout_transport_failure"],
)
def test_prediction_packet_rejects_endpoint_fields_at_any_depth(forbidden_key):
    packet = prediction_packet()
    packet["nested"] = {"deeper": {forbidden_key: [1, 2, 3]}}
    with pytest.raises(ValueError, match="endpoint fields"):
        seal_prediction_packet(packet)


@pytest.mark.parametrize(
    "forbidden_key",
    ["green_certificate_status", "p13_status", "baseline_predictions", "green_panel"],
)
def test_endpoint_packet_rejects_prediction_fields_at_any_depth(forbidden_key):
    commitment = seal_prediction_packet(prediction_packet())
    packet = endpoint_packet()
    packet["nested"] = {"deeper": {forbidden_key: "leak"}}
    with pytest.raises(ValueError, match="prediction fields"):
        seal_endpoint_packet(packet, commitment)


def test_endpoint_cannot_bind_to_different_row():
    commitment = seal_prediction_packet(prediction_packet())
    packet = endpoint_packet()
    packet["row_id"] = "cd" * 32
    with pytest.raises(ValueError, match="row mismatch"):
        seal_endpoint_packet(packet, commitment)


def test_adaptive_endpoint_allocation_is_rejected():
    commitment = seal_prediction_packet(prediction_packet())
    packet = endpoint_packet()
    packet["adaptive_query_allocation"] = True
    with pytest.raises(ValueError, match="adaptive query"):
        seal_endpoint_packet(packet, commitment)


def test_post_commit_prediction_mutation_is_detected():
    prediction = prediction_packet()
    prediction_commitment = seal_prediction_packet(prediction)
    endpoint = endpoint_packet()
    endpoint_commitment = seal_endpoint_packet(endpoint, prediction_commitment)
    prediction["predictions"]["ordinary_restoration"] = 0.5
    with pytest.raises(ValueError, match="prediction packet"):
        audit_commitment_pair(
            prediction, prediction_commitment, endpoint, endpoint_commitment
        )


def test_post_commit_endpoint_mutation_is_detected():
    prediction = prediction_packet()
    prediction_commitment = seal_prediction_packet(prediction)
    endpoint = endpoint_packet()
    endpoint_commitment = seal_endpoint_packet(endpoint, prediction_commitment)
    endpoint["heldout_transport_error_private"] = 0.7
    with pytest.raises(ValueError, match="endpoint packet"):
        audit_commitment_pair(
            prediction, prediction_commitment, endpoint, endpoint_commitment
        )
