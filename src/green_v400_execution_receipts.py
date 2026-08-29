"""Typed execution receipts for the formal GREEN endpoint route.

Hashes provide immutable provenance, while the phase ledger enforces ordering.
These receipts do not turn a prepare-only plan into an authorized plan.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from green_v400_endpoint_calibration import merge_target_replay_stability
from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_response_precision import verify_precision_receipt


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def receipt_sha256(receipt: dict[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def token_ids_sha256(values: list[int]) -> str:
    return hashlib.sha256(_canonical(values)).hexdigest()


def float32_tensor_sha256(tensor: torch.Tensor, schema: str) -> str:
    if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError("runtime tensor must be finite float32")
    header = {"schema": schema, "shape": list(tensor.shape), "dtype": "float32-le"}
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    digest = hashlib.sha256(_canonical(header) + b"\0" + array.tobytes(order="C"))
    return digest.hexdigest()


def float64_tensor_sha256(tensor: torch.Tensor, schema: str) -> str:
    if tensor.dtype != torch.float64 or not torch.isfinite(tensor).all():
        raise ValueError("runtime tensor must be finite float64")
    header = {"schema": schema, "shape": list(tensor.shape), "dtype": "float64-le"}
    array = tensor.detach().cpu().contiguous().numpy().astype("<f8", copy=False)
    return hashlib.sha256(_canonical(header) + b"\0" + array.tobytes(order="C")).hexdigest()


def build_model_session_receipt(
    *,
    plan: dict[str, Any],
    observed_full_model_hash: str,
    loader_source_sha256: str,
    process_start_nonce: str,
    pid: int,
) -> dict[str, Any]:
    plan_hash = _plan_hash(plan)
    if observed_full_model_hash != plan.get("full_model_hash"):
        raise ValueError("loaded model weights do not match the plan")
    for value, label in (
        (loader_source_sha256, "loader source hash"),
        (process_start_nonce, "model process start nonce"),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label} must be a digest")
    if loader_source_sha256 != plan.get("source_file_sha256", {}).get(
        "analysis/green_v400_formal_worker.py"
    ):
        raise ValueError("model loader source differs from the plan")
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("model session PID is invalid")
    receipt = {
        "schema_version": "green-v400-model-session-receipt-v1",
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan_hash,
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "observed_full_model_hash": observed_full_model_hash,
        "model_revision": plan["model_revision"],
        "loader_source_sha256": loader_source_sha256,
        "process_start_nonce": process_start_nonce,
        "pid": pid,
        "weight_hash_recomputed_before_session": True,
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    return receipt


def validate_model_session_receipt(
    receipt: dict[str, Any], authorization: dict[str, Any]
) -> None:
    verify_receipt(receipt, "green-v400-model-session-receipt-v1")
    if receipt.get("plan_sha256") != authorization.get("plan_sha256"):
        raise ValueError("model session plan mismatch")
    if receipt.get("model_manifest_sha256") != authorization.get(
        "model_manifest_sha256"
    ):
        raise ValueError("model session manifest mismatch")
    if receipt.get("observed_full_model_hash") != authorization.get("full_model_hash"):
        raise ValueError("model session weight hash mismatch")
    if receipt.get("weight_hash_recomputed_before_session") is not True:
        raise ValueError("model weights were not rehashed before the session")


def validate_model_session_for_plan(
    receipt: dict[str, Any], plan: dict[str, Any]
) -> None:
    """Validate a model process before a prediction route has an authorization."""

    verify_receipt(receipt, "green-v400-model-session-receipt-v1")
    plan_hash = _plan_hash(plan)
    if receipt.get("plan_sha256") != plan_hash:
        raise ValueError("model session plan mismatch")
    if receipt.get("model_manifest_sha256") != plan.get("model_manifest_sha256"):
        raise ValueError("model session manifest mismatch")
    if receipt.get("observed_full_model_hash") != plan.get("full_model_hash"):
        raise ValueError("model session weight hash mismatch")
    if receipt.get("model_revision") != plan.get("model_revision"):
        raise ValueError("model session revision mismatch")
    if receipt.get("weight_hash_recomputed_before_session") is not True:
        raise ValueError("model weights were not rehashed before the session")


def build_runtime_input_receipt(
    *,
    authorization: dict[str, Any],
    model_session: dict[str, Any],
    center: torch.Tensor,
    clean_token_ids: list[int],
    corrupt_token_ids: list[int],
    adapter_source_sha256: str,
    formal_runner_source_sha256: str,
    response_precision_receipt: dict[str, Any],
) -> dict[str, Any]:
    validate_model_session_receipt(model_session, authorization)
    if token_ids_sha256(clean_token_ids) != authorization.get("clean_token_ids_sha256"):
        raise ValueError("clean tokens differ from endpoint authorization")
    if token_ids_sha256(corrupt_token_ids) != authorization.get(
        "corrupt_token_ids_sha256"
    ):
        raise ValueError("corrupt tokens differ from endpoint authorization")
    if adapter_source_sha256 != authorization.get("response_adapter_source_sha256"):
        raise ValueError("response adapter source differs from endpoint authorization")
    if not isinstance(formal_runner_source_sha256, str) or len(formal_runner_source_sha256) != 64:
        raise ValueError("formal runner source hash must be a digest")
    verify_precision_receipt(
        response_precision_receipt, authorization["model_manifest_sha256"]
    )
    receipt = {
        "schema_version": "green-v400-runtime-input-receipt-v1",
        "protocol_id": authorization["protocol_id"],
        "plan_sha256": authorization["plan_sha256"],
        "endpoint_authorization_receipt_sha256": authorization["receipt_sha256"],
        "model_session_receipt_sha256": model_session["receipt_sha256"],
        "clean_token_ids_sha256": token_ids_sha256(clean_token_ids),
        "corrupt_token_ids_sha256": token_ids_sha256(corrupt_token_ids),
        "center_tensor_sha256": float64_tensor_sha256(
            center, "clean-resid-post-center-float64-v1"
        ),
        "response_precision_receipt_sha256": response_precision_receipt[
            "receipt_sha256"
        ],
        "model_manifest_tensor_hash_scheme": response_precision_receipt[
            "model_manifest_tensor_hash_scheme"
        ],
        "response_adapter_source_sha256": adapter_source_sha256,
        "formal_runner_source_sha256": formal_runner_source_sha256,
        "site_row_id": authorization["site_row_id"],
        "prompt_row_id": authorization["prompt_row_id"],
        "layer": authorization["layer"],
        "hook": authorization["hook"],
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    return receipt


def validate_runtime_input_receipt(
    *,
    receipt: dict[str, Any],
    authorization: dict[str, Any],
    center: torch.Tensor,
    response_precision_receipt: dict[str, Any],
) -> None:
    verify_receipt(receipt, "green-v400-runtime-input-receipt-v1")
    if receipt.get("endpoint_authorization_receipt_sha256") != authorization.get(
        "receipt_sha256"
    ):
        raise ValueError("runtime input receipt authorization mismatch")
    if receipt.get("center_tensor_sha256") != float64_tensor_sha256(
        center, "clean-resid-post-center-float64-v1"
    ):
        raise ValueError("runtime activation center hash mismatch")
    verify_precision_receipt(
        response_precision_receipt, authorization["model_manifest_sha256"]
    )
    if receipt.get("response_precision_receipt_sha256") != response_precision_receipt.get(
        "receipt_sha256"
    ):
        raise ValueError("runtime response precision receipt hash mismatch")


def verify_receipt(receipt: dict[str, Any], schema: str) -> None:
    if receipt.get("schema_version") != schema:
        raise ValueError("typed receipt schema mismatch")
    if receipt.get("receipt_sha256") != receipt_sha256(receipt):
        raise ValueError("typed receipt self hash mismatch")


def validate_endpoint_authorization_receipt(
    *,
    receipt: dict[str, Any],
    protocol_id: str,
    row_id: str,
    prediction_commitment: dict[str, Any],
    endpoint_direction_binding_sha256: str,
    endpoint_worker_source_sha256: str,
) -> None:
    verify_receipt(receipt, "green-v400-endpoint-authorization-receipt-v1")
    if receipt.get("protocol_id") != protocol_id or receipt.get("site_row_id") != row_id:
        raise ValueError("endpoint authorization identity mismatch")
    if receipt.get("prediction_packet_sha256") != prediction_commitment.get(
        "prediction_packet_sha256"
    ):
        raise ValueError("endpoint authorization prediction hash mismatch")
    if receipt.get("endpoint_direction_binding_sha256") != endpoint_direction_binding_sha256:
        raise ValueError("endpoint authorization direction binding mismatch")
    if receipt.get("endpoint_worker_source_sha256") != endpoint_worker_source_sha256:
        raise ValueError("endpoint authorization worker source mismatch")
    for field in (
        "plan_sha256",
        "endpoint_job_id",
        "prediction_packet_sha256",
        "numerical_replay_layer_receipt_sha256",
        "endpoint_direction_binding_sha256",
        "direction_registry_sha256",
        "model_manifest_sha256",
        "full_model_hash",
        "decision_spec_sha256",
        "endpoint_worker_source_sha256",
        "phase_ledger_head_sha256",
        "clean_token_ids_sha256",
        "corrupt_token_ids_sha256",
        "response_adapter_source_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"endpoint authorization {field} must be a digest")


def _plan_hash(plan: dict[str, Any]) -> str:
    digest = plan.get("plan_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("execution plan requires plan_sha256")
    payload = dict(plan)
    payload.pop("plan_sha256", None)
    # The compiler defines plan_sha256 over the pre-hash payload.
    if hashlib.sha256(_canonical(payload)).hexdigest() != digest:
        raise ValueError("execution plan self hash mismatch")
    return digest


def _worker_receipt(
    receipt: dict[str, Any], packet: dict[str, Any], expected_source_sha256: str
) -> None:
    required = {
        "worker_instance_id",
        "pid",
        "process_start_nonce",
        "python_executable_sha256",
        "source_file_sha256",
        "artifact_path",
    }
    if set(receipt) != required:
        raise ValueError("worker process receipt must use the strict schema")
    if receipt["worker_instance_id"] != packet.get("worker_instance_id_private"):
        raise ValueError("worker receipt identity does not match replay packet")
    if not isinstance(receipt["pid"], int) or receipt["pid"] <= 0:
        raise ValueError("worker receipt PID is invalid")
    for field in (
        "worker_instance_id",
        "process_start_nonce",
        "python_executable_sha256",
        "source_file_sha256",
    ):
        value = receipt[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"worker receipt {field} must be a digest")
    if receipt["source_file_sha256"] != expected_source_sha256:
        raise ValueError("replay worker source hash differs from the plan")
    if not str(receipt["artifact_path"]).startswith("/mnt/sdb/ccj/iclr_1_runs/"):
        raise ValueError("worker artifact must be stored under /mnt/sdb")


def build_numerical_replay_layer_receipt(
    *,
    plan: dict[str, Any],
    layer: int,
    replay_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate every planned replay pair at one layer and issue a typed receipt."""

    plan_hash = _plan_hash(plan)
    planned = {
        job["job_id"]: job
        for job in plan.get("queues", {}).get("endpoint_numerical_replay", [])
        if job.get("layer") == layer
    }
    if not planned or {row.get("job_id") for row in replay_artifacts} != set(planned):
        raise ValueError("replay artifacts must exactly cover planned layer jobs")
    source_bindings = plan.get("source_file_sha256", {})
    runner_source_hash = source_bindings.get("src/green_v400_formal_replay_runner.py")
    replay_core_source_hash = source_bindings.get(
        "src/green_v400_endpoint_calibration.py"
    )
    for value, label in (
        (runner_source_hash, "formal replay runner source hash"),
        (replay_core_source_hash, "replay core source hash"),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label} is not bound by the plan")
    accepted = []
    for artifact in replay_artifacts:
        job = planned[artifact["job_id"]]
        a, ca = artifact.get("replay_a"), artifact.get("commitment_a")
        b, cb = artifact.get("replay_b"), artifact.get("commitment_b")
        if not all(isinstance(value, dict) for value in (a, ca, b, cb)):
            raise ValueError("replay artifact lacks A/B packets and commitments")
        if a.get("row_id") != job["site_row_id"] or b.get("row_id") != job["site_row_id"]:
            raise ValueError("replay packet row does not match planned job")
        expected_binding = job["endpoint_direction_binding_sha256"]
        if any(
            packet.get("endpoint_direction_binding_sha256_private") != expected_binding
            for packet in (a, b)
        ):
            raise ValueError("replay direction binding differs from planned job")
        if any(
            packet.get("model_manifest_sha256_private")
            != plan["model_manifest_sha256"]
            for packet in (a, b)
        ):
            raise ValueError("replay model manifest differs from the plan")
        if any(
            packet.get("response_evaluation_dtype_private") != "float64"
            for packet in (a, b)
        ):
            raise ValueError("replay response evaluation must use float64")
        precision_receipt_hashes = {
            packet.get("response_precision_receipt_sha256_private")
            for packet in (a, b)
        }
        if (
            len(precision_receipt_hashes) != 1
            or not isinstance(next(iter(precision_receipt_hashes)), str)
            or len(next(iter(precision_receipt_hashes))) != 64
        ):
            raise ValueError("replay precision receipt binding is invalid")
        precision_receipt_hash = next(iter(precision_receipt_hashes))
        _worker_receipt(artifact.get("worker_a", {}), a, runner_source_hash)
        _worker_receipt(artifact.get("worker_b", {}), b, runner_source_hash)
        if artifact["worker_a"]["worker_instance_id"] == artifact["worker_b"]["worker_instance_id"]:
            raise ValueError("replay worker instances must differ")
        if (
            artifact["worker_a"]["pid"],
            artifact["worker_a"]["process_start_nonce"],
        ) == (
            artifact["worker_b"]["pid"],
            artifact["worker_b"]["process_start_nonce"],
        ):
            raise ValueError("replay workers must be distinct process starts")
        gate, gate_commitment = merge_target_replay_stability(a, ca, b, cb)
        if gate.get("numerical_replay_stable_private") is not True:
            raise ValueError("numerical replay layer cannot pass with unstable pair")
        if gate != artifact.get("gate") or gate_commitment != artifact.get("gate_commitment"):
            raise ValueError("serialized numerical replay gate differs from recomputation")
        accepted.append({
            "job_id": job["job_id"],
            "site_row_id": job["site_row_id"],
            "gate_packet_sha256": gate_commitment[
                "endpoint_numerical_replay_packet_sha256"
            ],
            "response_precision_receipt_sha256": precision_receipt_hash,
            "worker_a_process_start_nonce": artifact["worker_a"]["process_start_nonce"],
            "worker_b_process_start_nonce": artifact["worker_b"]["process_start_nonce"],
        })
    accepted.sort(key=lambda row: row["job_id"])
    receipt = {
        "schema_version": "green-v400-numerical-replay-layer-receipt-v1",
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan_hash,
        "layer": layer,
        "job_count": len(accepted),
        "jobs": accepted,
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "full_model_hash": plan["full_model_hash"],
        "direction_registry_sha256": plan["direction_registry_sha256"],
        "formal_replay_runner_source_sha256": runner_source_hash,
        "replay_core_source_sha256": replay_core_source_hash,
        "scientific_null_distribution_claimed": False,
        "defines_transport_failure_label": False,
        "all_replays_stable": True,
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    return receipt


def build_endpoint_authorization_receipt(
    *,
    plan: dict[str, Any],
    endpoint_job_id: str,
    prediction_packet: dict[str, Any],
    prediction_commitment: dict[str, Any],
    replay_layer_receipt: dict[str, Any],
    phase_ledger: dict[str, Any],
    universe_row: dict[str, Any],
    response_adapter_source_path: str,
) -> dict[str, Any]:
    """Issue one endpoint authorization only after all typed prerequisites exist."""

    plan_hash = _plan_hash(plan)
    if plan.get("execution_enabled") is not True:
        raise ValueError("prepare-only plan cannot authorize endpoint execution")
    verify_receipt(
        replay_layer_receipt, "green-v400-numerical-replay-layer-receipt-v1"
    )
    jobs = {
        job["job_id"]: job
        for name in ("development_endpoint", "confirmation_endpoint")
        for job in plan.get("queues", {}).get(name, [])
    }
    job = jobs.get(endpoint_job_id)
    if job is None:
        raise ValueError("endpoint job is not present in execution plan")
    if replay_layer_receipt.get("plan_sha256") != plan_hash or replay_layer_receipt.get(
        "layer"
    ) != job["layer"]:
        raise ValueError("numerical replay receipt does not authorize endpoint layer")
    expected_prediction = seal_prediction_packet(prediction_packet)
    if expected_prediction != prediction_commitment:
        raise ValueError("prediction packet or commitment changed")
    if prediction_packet.get("row_id") != job["site_row_id"]:
        raise ValueError("prediction row does not match endpoint job")
    if universe_row.get("row_id") != job["prompt_row_id"]:
        raise ValueError("universe row does not match endpoint prompt")
    for field in ("clean_token_ids", "corrupt_token_ids"):
        values = universe_row.get(field)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError(f"universe row {field} is invalid")
    adapter_hash = plan.get("source_file_sha256", {}).get(
        response_adapter_source_path
    )
    if not isinstance(adapter_hash, str) or len(adapter_hash) != 64:
        raise ValueError("response adapter source is not bound by the plan")
    binding = prediction_packet.get("formal_execution_binding", {})
    if (
        binding.get("plan_sha256") != plan_hash
        or binding.get("prediction_job_id")
        != next(
            candidate["job_id"]
            for candidate in plan["queues"][f"{job['role']}_prediction"]
            if candidate["site_row_id"] == job["site_row_id"]
        )
    ):
        raise ValueError("prediction packet is not bound to the planned job")
    if phase_ledger.get("plan_sha256") != plan_hash:
        raise ValueError("phase ledger plan mismatch")
    completed = set(phase_ledger.get("completed_prediction_job_ids", []))
    if binding["prediction_job_id"] not in completed:
        raise ValueError("prediction job is absent from append-only phase ledger")
    replay_receipts = set(phase_ledger.get("numerical_replay_receipt_sha256", []))
    if replay_layer_receipt["receipt_sha256"] not in replay_receipts:
        raise ValueError("replay receipt is absent from append-only phase ledger")
    if job["role"] == "confirmation" and phase_ledger.get(
        "development_analysis_receipt_sha256"
    ) is None:
        raise ValueError("confirmation endpoint is locked before development analysis")
    receipt = {
        "schema_version": "green-v400-endpoint-authorization-receipt-v1",
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan_hash,
        "endpoint_job_id": endpoint_job_id,
        "phase": job["role"],
        "site_row_id": job["site_row_id"],
        "prompt_row_id": job["prompt_row_id"],
        "layer": job["layer"],
        "hook": job["hook"],
        "prediction_packet_sha256": prediction_commitment["prediction_packet_sha256"],
        "numerical_replay_layer_receipt_sha256": replay_layer_receipt["receipt_sha256"],
        "endpoint_direction_binding_sha256": job[
            "endpoint_direction_binding_sha256"
        ],
        "direction_registry_sha256": plan["direction_registry_sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "full_model_hash": plan["full_model_hash"],
        "decision_spec_sha256": plan["decision_spec_sha256"],
        "endpoint_worker_source_sha256": plan["source_file_sha256"][
            "src/green_v400_endpoint_worker.py"
        ],
        "response_adapter_source_path": response_adapter_source_path,
        "response_adapter_source_sha256": adapter_hash,
        "clean_token_ids_sha256": hashlib.sha256(
            _canonical(universe_row["clean_token_ids"])
        ).hexdigest(),
        "corrupt_token_ids_sha256": hashlib.sha256(
            _canonical(universe_row["corrupt_token_ids"])
        ).hexdigest(),
        "phase_ledger_head_sha256": phase_ledger["ledger_head_sha256"],
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    return receipt
