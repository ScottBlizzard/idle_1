import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.green_v400_phase_ledger import (
    append_phase_event,
    initialize_phase_ledger,
    validate_phase_ledger,
)
from green_v400_direction_binding import binding_sha256, build_direction_binding
from green_v400_endpoint_calibration import (
    compute_target_replay_packet,
    merge_target_replay_stability,
)
from green_v400_endpoint_firewall import (
    seal_endpoint_numerical_replay_packet,
    seal_prediction_packet,
)
from green_v400_execution_receipts import (
    build_endpoint_authorization_receipt,
    build_grant_cohort_receipt,
    build_model_session_receipt,
    build_numerical_replay_layer_receipt,
    receipt_sha256,
)
from green_v400_response_precision import precision_receipt_sha256


PROTOCOL = "P"
SITE = "11" * 32
PROMPT = "22" * 32
REPLAY_JOB = "33" * 32
PREDICTION_JOB = "44" * 32
ENDPOINT_JOB = "55" * 32
GRANT_JOB = "56" * 32
SOURCE = "66" * 32
RUNNER_SOURCE = "67" * 32
ENTRYPOINT_SOURCE = "68" * 32


def precision_receipt(model_manifest_sha256):
    receipt = {
        "schema_version": "green-v400-response-evaluation-precision-receipt-v1",
        "model_manifest_sha256": model_manifest_sha256,
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


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def plan(execution_enabled=True, phase="development", with_grant=False):
    payload = {
        "schema_version": "green-v400-sealed-execution-plan-v1",
        "protocol_id": PROTOCOL,
        "execution_enabled": execution_enabled,
        "model_manifest_sha256": "77" * 32,
        "full_model_hash": "88" * 32,
        "direction_registry_sha256": "99" * 32,
        "decision_spec_sha256": "aa" * 32,
        "grant_capture_spec_sha256": "ad" * 32,
        "model_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "source_file_sha256": {
            "src/green_v400_endpoint_calibration.py": SOURCE,
            "src/green_v400_formal_replay_runner.py": RUNNER_SOURCE,
            "analysis/green_v400_formal_worker.py": ENTRYPOINT_SOURCE,
            "src/green_v400_endpoint_worker.py": "bb" * 32,
            "src/green_v400_ioi_response_adapter.py": "bc" * 32,
            "src/green_v400_formal_grant_runner.py": "bd" * 32,
            "src/green_v400_grant_divergence.py": "be" * 32,
            "src/green_v400_grant_prediction_worker.py": "bf" * 32,
        },
        "queues": {
            "endpoint_numerical_replay": [{
                "job_id": REPLAY_JOB,
                "site_row_id": SITE,
                "prompt_row_id": PROMPT,
                "layer": 0,
                "hook": "resid_post",
                "endpoint_direction_binding_sha256": None,
            }],
            f"{phase}_prediction": [{
                "job_id": PREDICTION_JOB,
                "site_row_id": SITE,
                "prompt_row_id": PROMPT,
                "layer": 0,
                "hook": "resid_post",
            }],
            f"{phase}_endpoint": [{
                "job_id": ENDPOINT_JOB,
                "site_row_id": SITE,
                "prompt_row_id": PROMPT,
                "role": phase,
                "layer": 0,
                "hook": "resid_post",
                "endpoint_direction_binding_sha256": None,
            }],
        },
    }
    if with_grant:
        payload["queues"][f"{phase}_grant_cohort_prediction"] = [
            {
                "job_id": GRANT_JOB,
                "role": phase,
                "layer": 0,
                "cohort_site_row_ids_sha256": "de" * 32,
                "cohort_size": 4,
                "analysis_seed": 123,
                "grant_capture_spec_sha256": "ad" * 32,
            }
        ]
    directions = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    binding = build_direction_binding(
        protocol_id=PROTOCOL,
        row_id=SITE,
        panel_kind="endpoint",
        tensor=directions,
        direction_norm=1.0,
        generator_spec="unit-test-v1",
    )
    binding_hash = binding_sha256(binding)
    payload["queues"]["endpoint_numerical_replay"][0]["endpoint_direction_binding_sha256"] = binding_hash
    payload["queues"][f"{phase}_endpoint"][0]["endpoint_direction_binding_sha256"] = binding_hash
    payload["plan_sha256"] = digest(payload)
    return payload, directions, binding


def artifact(payload, directions, binding):
    kwargs = dict(
        protocol_id=PROTOCOL,
        row_id=SITE,
        target_response=lambda x: x[0],
        center=torch.tensor([0.0], dtype=torch.float64),
        endpoint_directions=directions,
        endpoint_direction_binding=binding,
        expected_endpoint_direction_binding_sha256=binding_sha256(binding),
        model_manifest_sha256=payload["model_manifest_sha256"],
        response_precision_receipt=precision_receipt(
            payload["model_manifest_sha256"]
        ),
    )
    a, ca = compute_target_replay_packet(
        **kwargs, replay_id="A", worker_instance_id="a1" * 32
    )
    b, cb = compute_target_replay_packet(
        **kwargs, replay_id="B", worker_instance_id="b2" * 32
    )
    gate, gate_commitment = merge_target_replay_stability(a, ca, b, cb)

    def worker(instance, pid, nonce):
        return {
            "worker_instance_id": instance,
            "pid": pid,
            "process_start_nonce": nonce,
            "python_executable_sha256": "cc" * 32,
            "source_file_sha256": ENTRYPOINT_SOURCE,
            "artifact_path": f"/mnt/sdb/ccj/iclr_1_runs/test/{instance}.json",
        }

    return {
        "job_id": REPLAY_JOB,
        "replay_a": a,
        "commitment_a": ca,
        "replay_b": b,
        "commitment_b": cb,
        "gate": gate,
        "gate_commitment": gate_commitment,
        "worker_a": worker("a1" * 32, 101, "dd" * 32),
        "worker_b": worker("b2" * 32, 102, "ee" * 32),
    }


def append(ledger, **event):
    return append_phase_event(ledger, {
        "plan_sha256": ledger["plan_sha256"],
        "previous_ledger_head_sha256": ledger["ledger_head_sha256"],
        **event,
    })


def test_replay_receipt_recomputes_every_pair_and_binds_process_plan_model_code_direction():
    payload, directions, binding = plan()
    receipt = build_numerical_replay_layer_receipt(
        plan=payload,
        layer=0,
        replay_artifacts=[artifact(payload, directions, binding)],
    )
    assert receipt["all_replays_stable"] is True
    assert receipt["plan_sha256"] == payload["plan_sha256"]
    assert receipt["direction_registry_sha256"] == payload["direction_registry_sha256"]
    assert receipt["formal_replay_runner_source_sha256"] == RUNNER_SOURCE
    assert receipt["formal_worker_entrypoint_source_sha256"] == ENTRYPOINT_SOURCE
    assert receipt["replay_core_source_sha256"] == SOURCE
    assert receipt["jobs"][0]["response_precision_receipt_sha256"]


def test_replay_receipt_rejects_wrong_manifest_dtype_or_runner_source():
    payload, directions, binding = plan()
    item = artifact(payload, directions, binding)
    item["replay_a"]["model_manifest_sha256_private"] = "00" * 32
    item["commitment_a"] = seal_endpoint_numerical_replay_packet(item["replay_a"])
    with pytest.raises(ValueError, match="model manifest"):
        build_numerical_replay_layer_receipt(plan=payload, layer=0, replay_artifacts=[item])

    item = artifact(payload, directions, binding)
    item["replay_a"]["response_evaluation_dtype_private"] = "float32"
    item["commitment_a"] = seal_endpoint_numerical_replay_packet(item["replay_a"])
    with pytest.raises(ValueError, match="float64"):
        build_numerical_replay_layer_receipt(plan=payload, layer=0, replay_artifacts=[item])

    item = artifact(payload, directions, binding)
    item["worker_a"]["source_file_sha256"] = SOURCE
    with pytest.raises(ValueError, match="source hash"):
        build_numerical_replay_layer_receipt(plan=payload, layer=0, replay_artifacts=[item])


def test_fake_gate_or_same_process_start_fails_closed():
    payload, directions, binding = plan()
    item = artifact(payload, directions, binding)
    item["gate"]["numerical_replay_stable_private"] = False
    with pytest.raises(ValueError, match="differs from recomputation"):
        build_numerical_replay_layer_receipt(plan=payload, layer=0, replay_artifacts=[item])
    item = artifact(payload, directions, binding)
    item["worker_b"]["pid"] = item["worker_a"]["pid"]
    item["worker_b"]["process_start_nonce"] = item["worker_a"]["process_start_nonce"]
    with pytest.raises(ValueError, match="distinct process starts"):
        build_numerical_replay_layer_receipt(plan=payload, layer=0, replay_artifacts=[item])


def test_phase_ledger_blocks_prepare_only_and_confirmation_before_development():
    payload, _, _ = plan(execution_enabled=False)
    ledger = initialize_phase_ledger(payload)
    with pytest.raises(ValueError, match="prepare-only"):
        append(
            ledger,
            kind="prediction_committed",
            phase="development",
            job_id=PREDICTION_JOB,
            commitment_sha256="01" * 32,
        )
    payload, _, _ = plan(phase="confirmation")
    ledger = initialize_phase_ledger(payload)
    with pytest.raises(ValueError, match="locked"):
        append(
            ledger,
            kind="prediction_committed",
            phase="confirmation",
            job_id=PREDICTION_JOB,
            commitment_sha256="01" * 32,
        )


def test_endpoint_authorization_requires_ledger_prediction_and_replay_receipts():
    payload, directions, binding = plan()
    replay = build_numerical_replay_layer_receipt(
        plan=payload,
        layer=0,
        replay_artifacts=[artifact(payload, directions, binding)],
    )
    prediction = {
        "schema_version": "green-v400-sfc-prediction-packet-v1",
        "protocol_id": PROTOCOL,
        "row_id": SITE,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "formal_execution_binding": {
            "plan_sha256": payload["plan_sha256"],
            "prediction_job_id": PREDICTION_JOB,
        },
    }
    commitment = seal_prediction_packet(prediction)
    ledger = initialize_phase_ledger(payload)
    ledger = append(
        ledger,
        kind="numerical_replay_layer_complete",
        layer=0,
        receipt_sha256=replay["receipt_sha256"],
    )
    with pytest.raises(ValueError, match="prediction job is absent"):
        build_endpoint_authorization_receipt(
            plan=payload,
            endpoint_job_id=ENDPOINT_JOB,
            prediction_packet=prediction,
            prediction_commitment=commitment,
            replay_layer_receipt=replay,
            phase_ledger=ledger,
            universe_row={
                "row_id": PROMPT,
                "clean_token_ids": [1, 2],
                "corrupt_token_ids": [1, 3],
            },
            response_adapter_source_path="src/green_v400_ioi_response_adapter.py",
        )
    ledger = append(
        ledger,
        kind="prediction_committed",
        phase="development",
        job_id=PREDICTION_JOB,
        commitment_sha256=commitment["prediction_packet_sha256"],
    )
    receipt = build_endpoint_authorization_receipt(
        plan=payload,
        endpoint_job_id=ENDPOINT_JOB,
        prediction_packet=prediction,
        prediction_commitment=commitment,
        replay_layer_receipt=replay,
        phase_ledger=ledger,
        universe_row={
            "row_id": PROMPT,
            "clean_token_ids": [1, 2],
            "corrupt_token_ids": [1, 3],
        },
        response_adapter_source_path="src/green_v400_ioi_response_adapter.py",
    )
    assert receipt["endpoint_job_id"] == ENDPOINT_JOB
    assert receipt["endpoint_direction_binding_sha256"] == binding_sha256(binding)


def test_ledger_replay_validation_rejects_derived_state_forgery():
    payload, _, _ = plan()
    ledger = initialize_phase_ledger(payload)
    ledger = append(
        ledger,
        kind="prediction_committed",
        phase="development",
        job_id=PREDICTION_JOB,
        commitment_sha256="01" * 32,
    )
    validate_phase_ledger(payload, ledger)
    forged = json.loads(json.dumps(ledger))
    forged["completed_prediction_job_ids"] = []
    with pytest.raises(ValueError, match="derived state"):
        validate_phase_ledger(payload, forged)


def test_endpoint_requires_every_typed_grant_receipt_in_plan_and_ledger():
    payload, directions, binding = plan(with_grant=True)
    replay = build_numerical_replay_layer_receipt(
        plan=payload,
        layer=0,
        replay_artifacts=[artifact(payload, directions, binding)],
    )
    prediction = {
        "schema_version": "green-v400-sfc-prediction-packet-v1",
        "protocol_id": PROTOCOL,
        "row_id": SITE,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "formal_execution_binding": {
            "plan_sha256": payload["plan_sha256"],
            "prediction_job_id": PREDICTION_JOB,
        },
    }
    commitment = seal_prediction_packet(prediction)
    ledger = initialize_phase_ledger(payload)
    ledger = append(
        ledger,
        kind="numerical_replay_layer_complete",
        layer=0,
        receipt_sha256=replay["receipt_sha256"],
    )
    ledger = append(
        ledger,
        kind="prediction_committed",
        phase="development",
        job_id=PREDICTION_JOB,
        commitment_sha256=commitment["prediction_packet_sha256"],
    )
    kwargs = dict(
        plan=payload,
        endpoint_job_id=ENDPOINT_JOB,
        prediction_packet=prediction,
        prediction_commitment=commitment,
        replay_layer_receipt=replay,
        phase_ledger=ledger,
        universe_row={
            "row_id": PROMPT,
            "clean_token_ids": [1, 2],
            "corrupt_token_ids": [1, 3],
        },
        response_adapter_source_path="src/green_v400_ioi_response_adapter.py",
    )
    with pytest.raises(ValueError, match="exactly cover"):
        build_endpoint_authorization_receipt(**kwargs)
    grant_receipt = {
        "schema_version": "green-v400-grant-cohort-receipt-v1",
        "protocol_id": PROTOCOL,
        "plan_sha256": payload["plan_sha256"],
        "job_id": GRANT_JOB,
        "phase": "development",
        "layer": 0,
        "cohort_site_row_ids_sha256": "de" * 32,
        "cohort_size": 4,
        "analysis_seed": 123,
        "grant_capture_spec_sha256": "ad" * 32,
        "prediction_packet_sha256": "ef" * 32,
        "model_session_receipt_sha256": "fe" * 32,
        "raw_activation_serialized": False,
    }
    grant_receipt["receipt_sha256"] = receipt_sha256(grant_receipt)
    ledger = append(
        ledger,
        kind="grant_cohort_committed",
        phase="development",
        job_id=GRANT_JOB,
        receipt_sha256=grant_receipt["receipt_sha256"],
    )
    kwargs["phase_ledger"] = ledger
    kwargs["grant_cohort_receipts"] = [grant_receipt]
    receipt = build_endpoint_authorization_receipt(**kwargs)
    assert receipt["grant_phase_receipts_sha256"]


def test_grant_receipt_recomputes_commitment_and_binds_exact_formal_route():
    payload, _, _ = plan(with_grant=True)
    session = build_model_session_receipt(
        plan=payload,
        observed_full_model_hash=payload["full_model_hash"],
        loader_source_sha256=ENTRYPOINT_SOURCE,
        process_start_nonce="c1" * 32,
        pid=909,
    )
    job = payload["queues"]["development_grant_cohort_prediction"][0]
    binding = {
        "plan_sha256": payload["plan_sha256"],
        "grant_job_id": GRANT_JOB,
        "model_session_receipt_sha256": session["receipt_sha256"],
        "grant_capture_spec_sha256": payload["grant_capture_spec_sha256"],
        "cohort_site_row_ids_sha256": job["cohort_site_row_ids_sha256"],
        "cohort_size": job["cohort_size"],
        "analysis_seed": job["analysis_seed"],
        "measurement_hook": "blocks.10.hook_resid_post",
        "measurement_position_rule": "final_prompt_position",
        "measurement_position_strictly_after_candidate": True,
        "vectors_per_site_row": 1,
        "raw_activation_serialized": False,
        "formal_grant_runner_source_sha256": "bd" * 32,
        "grant_core_source_sha256": "be" * 32,
        "grant_packet_source_sha256": "bf" * 32,
    }
    packet = {
        "schema_version": "green-v400-grant-divergence-prediction-v2",
        "protocol_id": PROTOCOL,
        "row_id": job["cohort_site_row_ids_sha256"],
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "scope": "development_phase_by_layer_cohort_only",
        "phase": "development",
        "diagnostic_label": "grant_style_downstream_contextual_divergence_extension",
        "grant_style_divergence": {},
        "source_repository_commit": "f2548d2ea9b4f4b87a87ba5d53db43838d15c521",
        "formal_execution_binding": binding,
    }
    artifact_payload = {
        "schema_version": "green-v400-formal-grant-artifact-v1",
        "job_id": GRANT_JOB,
        "grant_prediction": packet,
        "commitment": seal_prediction_packet(packet),
        "model_session": session,
    }
    receipt = build_grant_cohort_receipt(plan=payload, artifact=artifact_payload)
    assert receipt["job_id"] == GRANT_JOB
    assert receipt["raw_activation_serialized"] is False

    changed = json.loads(json.dumps(artifact_payload))
    changed["grant_prediction"]["formal_execution_binding"][
        "measurement_position_rule"
    ] = "candidate_position"
    changed["commitment"] = seal_prediction_packet(changed["grant_prediction"])
    with pytest.raises(ValueError, match="formal execution binding"):
        build_grant_cohort_receipt(plan=payload, artifact=changed)
