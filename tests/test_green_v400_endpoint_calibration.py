from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_endpoint_calibration import (
    compute_target_replay_packet,
    merge_target_replay_score,
)


PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
ROW_ID = "71" * 32
DIRECTION_COMMITMENT = "82" * 32


def replay(replay_id, worker_id, scale=1.0):
    return compute_target_replay_packet(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        target_response=lambda x: scale * x[0],
        center=torch.tensor([0.0], dtype=torch.float64),
        endpoint_directions=torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
        endpoint_direction_commitment=DIRECTION_COMMITMENT,
        replay_id=replay_id,
        worker_instance_id=worker_id,
    )


def test_two_distinct_replay_workers_produce_a_committed_calibration_score():
    packet_a, commitment_a = replay("A", "a1" * 32)
    packet_b, commitment_b = replay("B", "b2" * 32)
    score, commitment = merge_target_replay_score(
        packet_a, commitment_a, packet_b, commitment_b
    )
    assert score["endpoint_calibration_score_private"] == pytest.approx(0.0)
    assert score["worker_instances_distinct_private"] is True
    assert commitment["prediction_access_forbidden"] is True


def test_replay_score_detects_target_target_execution_disagreement():
    packet_a, commitment_a = replay("A", "a1" * 32, scale=1.0)
    packet_b, commitment_b = replay("B", "b2" * 32, scale=1.5)
    score, _ = merge_target_replay_score(
        packet_a, commitment_a, packet_b, commitment_b
    )
    assert score["endpoint_calibration_score_private"] == pytest.approx(0.5)


def test_same_worker_or_mutated_replay_fails_closed():
    packet_a, commitment_a = replay("A", "a1" * 32)
    packet_b, commitment_b = replay("B", "a1" * 32)
    with pytest.raises(ValueError, match="distinct worker"):
        merge_target_replay_score(packet_a, commitment_a, packet_b, commitment_b)
    packet_b, commitment_b = replay("B", "b2" * 32)
    packet_b["endpoint_replay_effects_private"][0] = 9.0
    with pytest.raises(ValueError, match="packet or commitment changed"):
        merge_target_replay_score(packet_a, commitment_a, packet_b, commitment_b)


def test_replay_identity_and_direction_commitment_are_validated():
    with pytest.raises(ValueError, match="replay_id"):
        replay("C", "a1" * 32)
    with pytest.raises(ValueError, match="64-character"):
        compute_target_replay_packet(
            protocol_id=PROTOCOL,
            row_id=ROW_ID,
            target_response=lambda x: x[0],
            center=torch.tensor([0.0]),
            endpoint_directions=torch.tensor([[1.0]]),
            endpoint_direction_commitment="short",
            replay_id="A",
            worker_instance_id="a1" * 32,
        )
