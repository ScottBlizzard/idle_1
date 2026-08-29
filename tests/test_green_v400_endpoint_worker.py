from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_direction_binding import binding_sha256, build_direction_binding
from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_endpoint_worker import compute_heldout_transport_endpoint
from green_v400_execution_receipts import float64_tensor_sha256, receipt_sha256
from green_v400_response_precision import precision_receipt_sha256


PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
ROW_ID = "34" * 32


def prediction_commitment():
    packet = {
        "schema_version": "green-v400-sfc-prediction-packet-v1",
        "protocol_id": PROTOCOL,
        "row_id": ROW_ID,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "ordinary_restoration": 0.9,
    }
    return seal_prediction_packet(packet)


def direction_binding(tensor):
    binding = build_direction_binding(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        panel_kind="endpoint",
        tensor=tensor,
        direction_norm=1.0,
        generator_spec="unit-test-v1",
    )
    return binding, binding_sha256(binding)


def endpoint_authorization(commitment, binding_hash):
    import hashlib

    source_hash = hashlib.sha256(
        (ROOT / "src/green_v400_endpoint_worker.py").read_bytes()
    ).hexdigest()
    receipt = {
        "schema_version": "green-v400-endpoint-authorization-receipt-v1",
        "protocol_id": PROTOCOL,
        "plan_sha256": "01" * 32,
        "endpoint_job_id": "02" * 32,
        "phase": "development",
        "site_row_id": ROW_ID,
        "prompt_row_id": "03" * 32,
        "layer": 0,
        "hook": "resid_post",
        "prediction_packet_sha256": commitment["prediction_packet_sha256"],
        "prediction_batch_completion_receipt_sha256": "10" * 32,
        "numerical_replay_layer_receipt_sha256": "04" * 32,
        "endpoint_direction_binding_sha256": binding_hash,
        "direction_registry_sha256": "05" * 32,
        "model_manifest_sha256": "06" * 32,
        "full_model_hash": "07" * 32,
        "decision_spec_sha256": "08" * 32,
        "endpoint_worker_source_sha256": source_hash,
        "response_adapter_source_path": "src/adapter.py",
        "response_adapter_source_sha256": "09" * 32,
        "clean_token_ids_sha256": "0a" * 32,
        "corrupt_token_ids_sha256": "0b" * 32,
        "phase_ledger_head_sha256": "0c" * 32,
        "grant_phase_receipts_sha256": "0f" * 32,
        "grant_phase_batch_completion_receipts_sha256": "11" * 32,
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    return receipt


def runtime_receipt(authorization, center):
    precision = response_precision_receipt(authorization)
    receipt = {
        "schema_version": "green-v400-runtime-input-receipt-v1",
        "protocol_id": PROTOCOL,
        "plan_sha256": authorization["plan_sha256"],
        "endpoint_authorization_receipt_sha256": authorization["receipt_sha256"],
        "model_session_receipt_sha256": "0d" * 32,
        "clean_token_ids_sha256": authorization["clean_token_ids_sha256"],
        "corrupt_token_ids_sha256": authorization["corrupt_token_ids_sha256"],
        "center_tensor_sha256": float64_tensor_sha256(
            center.to(torch.float64), "clean-resid-post-center-float64-v1"
        ),
        "response_precision_receipt_sha256": precision["receipt_sha256"],
        "response_adapter_source_sha256": authorization[
            "response_adapter_source_sha256"
        ],
        "formal_runner_source_sha256": "0e" * 32,
        "site_row_id": ROW_ID,
        "prompt_row_id": authorization["prompt_row_id"],
        "layer": 0,
        "hook": "resid_post",
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    return receipt


def response_precision_receipt(authorization):
    receipt = {
        "schema_version": "green-v400-response-evaluation-precision-receipt-v1",
        "model_manifest_sha256": authorization["model_manifest_sha256"],
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


def test_endpoint_detects_large_heldout_transport_failure_without_prediction_access():
    center = torch.tensor([0.0], dtype=torch.float64)
    directions = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    binding, binding_hash = direction_binding(directions)
    prediction = prediction_commitment()
    authorization = endpoint_authorization(prediction, binding_hash)
    packet, commitment = compute_heldout_transport_endpoint(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        prediction_commitment=prediction,
        endpoint_authorization_receipt=authorization,
        runtime_input_receipt=runtime_receipt(authorization, center),
        response_precision_receipt=response_precision_receipt(authorization),
        target_response=lambda x: x[0],
        patched_response=lambda x: 3.0 * x[0],
        center=center,
        endpoint_directions=directions,
        endpoint_direction_binding=binding,
        expected_endpoint_direction_binding_sha256=binding_hash,
    )
    assert packet["contains_prediction"] is False
    assert packet["heldout_transport_symmetric_normalized_error_private"] == pytest.approx(2.0 / 3.0)
    assert packet["endpoint_failure_label_private"] is True
    assert packet["endpoint_failure_label_role_private"] == "secondary_effect_size_label"
    assert commitment["prediction_packet_sha256"] == prediction_commitment()[
        "prediction_packet_sha256"
    ]


def test_identical_response_fields_are_not_failures():
    center = torch.tensor([0.2, -0.1], dtype=torch.float64)
    raw = torch.tensor([[0.1, 0.4], [-0.2, 0.3]], dtype=torch.float32)
    directions = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    binding, binding_hash = direction_binding(directions)
    prediction = prediction_commitment()
    authorization = endpoint_authorization(prediction, binding_hash)
    response = lambda x: x[0] ** 2 + x[1]
    packet, _ = compute_heldout_transport_endpoint(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        prediction_commitment=prediction,
        endpoint_authorization_receipt=authorization,
        runtime_input_receipt=runtime_receipt(authorization, center),
        response_precision_receipt=response_precision_receipt(authorization),
        target_response=response,
        patched_response=response,
        center=center,
        endpoint_directions=directions,
        endpoint_direction_binding=binding,
        expected_endpoint_direction_binding_sha256=binding_hash,
    )
    assert packet["heldout_transport_error_private"] == pytest.approx(0.0)
    assert packet["heldout_transport_symmetric_normalized_error_private"] == pytest.approx(0.0)
    assert packet["endpoint_failure_label_private"] is False


def test_endpoint_requires_a_committed_stable_replay_gate_and_valid_margin():
    kwargs = dict(
        protocol_id=PROTOCOL,
        row_id=ROW_ID,
        prediction_commitment=prediction_commitment(),
        target_response=lambda x: x[0],
        patched_response=lambda x: x[0],
        center=torch.tensor([0.0], dtype=torch.float64),
        endpoint_directions=torch.tensor([[1.0]], dtype=torch.float32),
    )
    binding, binding_hash = direction_binding(kwargs["endpoint_directions"])
    kwargs["endpoint_direction_binding"] = binding
    kwargs["expected_endpoint_direction_binding_sha256"] = binding_hash
    kwargs["endpoint_authorization_receipt"] = endpoint_authorization(
        kwargs["prediction_commitment"], binding_hash
    )
    kwargs["runtime_input_receipt"] = runtime_receipt(
        kwargs["endpoint_authorization_receipt"], kwargs["center"]
    )
    kwargs["response_precision_receipt"] = response_precision_receipt(
        kwargs["endpoint_authorization_receipt"]
    )
    mutated = dict(kwargs["endpoint_authorization_receipt"])
    mutated["decision_spec_sha256"] = "ff" * 32
    with pytest.raises(ValueError, match="self hash"):
        invalid_kwargs = dict(kwargs)
        invalid_kwargs["endpoint_authorization_receipt"] = mutated
        compute_heldout_transport_endpoint(
            **invalid_kwargs,
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        compute_heldout_transport_endpoint(
            **kwargs,
            transport_failure_threshold=0.99,
        )
