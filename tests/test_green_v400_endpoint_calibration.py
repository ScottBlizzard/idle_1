from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_endpoint_calibration import (
    compute_target_replay_packet,
    merge_target_replay_stability,
)
from green_v400_direction_binding import binding_sha256, build_direction_binding
from green_v400_response_precision import precision_receipt_sha256


PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
ROW_ID = "71" * 32
DIRECTION_COMMITMENT = "82" * 32
MODEL_MANIFEST = "83" * 32


def precision_receipt():
    receipt = {
        "schema_version": "green-v400-response-evaluation-precision-receipt-v1",
        "model_manifest_sha256": MODEL_MANIFEST,
        "checkpoint_storage_dtype": "float32",
        "response_evaluation_dtype": "float64",
        "model_manifest_tensor_hash_scheme": "sha256-contiguous-numpy-native-bytes-v1",
        "floating_tensor_count": 1,
        "all_manifest_tensor_hashes_matched_before_conversion": True,
        "all_float64_values_roundtrip_to_manifest_float32_exactly": True,
        "scientific_outcome_evaluated": False,
    }
    receipt["receipt_sha256"] = precision_receipt_sha256(receipt)
    return receipt


def replay(replay_id, worker_id, scale=1.0):
    directions = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    binding = build_direction_binding(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        panel_kind="endpoint",
        tensor=directions,
        direction_norm=1.0,
        generator_spec="unit-test-v1",
    )
    return compute_target_replay_packet(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        target_response=lambda x: scale * x[0],
        center=torch.tensor([0.0], dtype=torch.float64),
        endpoint_directions=directions,
        endpoint_direction_binding=binding,
        expected_endpoint_direction_binding_sha256=binding_sha256(binding),
        replay_id=replay_id,
        worker_instance_id=worker_id,
        model_manifest_sha256=MODEL_MANIFEST,
        response_precision_receipt=precision_receipt(),
    )


def test_two_distinct_replay_workers_produce_a_committed_stability_gate():
    packet_a, commitment_a = replay("A", "a1" * 32)
    packet_b, commitment_b = replay("B", "b2" * 32)
    gate, commitment = merge_target_replay_stability(
        packet_a, commitment_a, packet_b, commitment_b
    )
    assert gate["replay_max_absolute_difference_private"] == pytest.approx(0.0)
    assert gate["numerical_replay_stable_private"] is True
    assert gate["scientific_null_distribution_claimed"] is False
    assert gate["defines_transport_failure_label"] is False
    assert gate["worker_instances_distinct_private"] is True
    assert commitment["prediction_access_forbidden"] is True


def test_replay_gate_detects_target_target_execution_disagreement():
    packet_a, commitment_a = replay("A", "a1" * 32, scale=1.0)
    packet_b, commitment_b = replay("B", "b2" * 32, scale=1.5)
    gate, _ = merge_target_replay_stability(
        packet_a, commitment_a, packet_b, commitment_b
    )
    assert gate["replay_max_absolute_difference_private"] == pytest.approx(0.5)
    assert gate["numerical_replay_stable_private"] is False


def test_same_worker_or_mutated_replay_fails_closed():
    packet_a, commitment_a = replay("A", "a1" * 32)
    packet_b, commitment_b = replay("B", "a1" * 32)
    with pytest.raises(ValueError, match="distinct worker"):
        merge_target_replay_stability(packet_a, commitment_a, packet_b, commitment_b)
    packet_b, commitment_b = replay("B", "b2" * 32)
    packet_b["endpoint_replay_effects_private"][0] = 9.0
    with pytest.raises(ValueError, match="packet or commitment changed"):
        merge_target_replay_stability(packet_a, commitment_a, packet_b, commitment_b)


def test_replay_tolerances_are_frozen_and_never_define_scientific_failure():
    packet_a, commitment_a = replay("A", "a1" * 32)
    packet_b, commitment_b = replay("B", "b2" * 32)
    with pytest.raises(ValueError, match="at least one"):
        merge_target_replay_stability(
            packet_a,
            commitment_a,
            packet_b,
            commitment_b,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        )


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
            endpoint_direction_binding={},
            expected_endpoint_direction_binding_sha256="short",
            replay_id="A",
            worker_instance_id="a1" * 32,
            model_manifest_sha256=MODEL_MANIFEST,
            response_precision_receipt=precision_receipt(),
        )
