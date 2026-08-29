"""Persistent-model batch entry point for sealed GREEN prediction jobs.

One process exposes exactly one physical GPU, loads and verifies the frozen
model once, and executes one deterministic shard of either prediction or Grant
jobs.  Modes are intentionally separate so a Grant process never opens a
direction payload.  Every job still produces its own no-clobber formal artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis.green_v400_formal_worker import (
    FORMAL_OUTPUT_ROOT,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_frozen_model,
    load_json,
    model_session,
    planned_job,
    validate_runtime_envelope,
    verify_plan,
)
from analysis.green_v400_development_activation import validate_activated_plan
from green_v400_direction_binding import verify_direction_binding
from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_execution_receipts import (
    build_grant_cohort_receipt,
    receipt_sha256,
    validate_model_session_for_plan,
)
from green_v400_formal_grant_runner import run_formal_grant_prediction
from green_v400_formal_prediction_runner import run_formal_prediction
from green_v400_response_precision import (
    prepare_float64_response_evaluation,
    tensor_sha256,
)


BATCH_SOURCE_PATH = "analysis/green_v400_formal_batch_worker.py"


def select_shard_jobs(
    plan: dict[str, Any],
    *,
    mode: str,
    phase: str,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if mode not in {"prediction", "grant"}:
        raise ValueError("batch mode must be prediction or grant")
    if phase not in {"development", "confirmation"}:
        raise ValueError("batch phase must be development or confirmation")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("batch shard coordinates are invalid")
    suffix = "prediction" if mode == "prediction" else "grant_cohort_prediction"
    queue = plan.get("queues", {}).get(f"{phase}_{suffix}", [])
    job_ids = [job.get("job_id") for job in queue]
    if not queue or any(not isinstance(value, str) for value in job_ids):
        raise ValueError("planned batch queue is empty or malformed")
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("planned batch queue has duplicate job identifiers")
    return [
        job for ordinal, job in enumerate(queue) if ordinal % shard_count == shard_index
    ]


class DirectionPanel:
    """One-time verified mmap of the public GREEN direction panel."""

    def __init__(
        self,
        *,
        plan: dict[str, Any],
        registry: dict[str, Any],
        payload_path: Path,
    ) -> None:
        registry_without_hash = dict(registry)
        registry_hash = registry_without_hash.pop("registry_sha256", None)
        if canonical_sha256(registry_without_hash) != registry_hash:
            raise ValueError("direction registry self hash mismatch")
        if registry_hash != plan.get("direction_registry_sha256"):
            raise ValueError("direction registry differs from the execution plan")
        record = registry.get("panels", {}).get("green", {})
        if record.get("prediction_process_access") is not True:
            raise ValueError("GREEN direction panel is not prediction-authorized")
        if payload_path.name != record.get("payload_filename"):
            raise ValueError("direction payload filename differs from registry")
        if file_sha256(payload_path) != record.get("payload_file_sha256"):
            raise ValueError("direction payload file hash mismatch")
        payload = np.load(payload_path, mmap_mode="r", allow_pickle=False)
        if list(payload.shape) != record.get("shape") or payload.dtype.str != "<f4":
            raise ValueError("direction payload array metadata mismatch")
        rows = record.get("row_bindings", [])
        self._rows = {row.get("row_id"): row for row in rows}
        if len(self._rows) != len(rows):
            raise ValueError("direction registry row identifiers are not unique")
        self._payload = payload
        self._protocol_id = plan["protocol_id"]

    def row(self, row_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self._rows.get(row_id)
        if record is None:
            raise ValueError("direction row is absent from the verified panel")
        tensor = torch.from_numpy(
            np.array(self._payload[int(record["row_index"])], copy=True)
        )
        verify_direction_binding(
            tensor=tensor,
            binding=record["binding"],
            expected_binding_sha256=record["binding_sha256"],
            protocol_id=self._protocol_id,
            row_id=row_id,
            panel_kind="green",
        )
        return tensor, record["binding"]


def validate_existing_artifact(
    *, plan: dict[str, Any], mode: str, job: dict[str, Any], artifact: dict[str, Any]
) -> None:
    if mode == "grant":
        build_grant_cohort_receipt(plan=plan, artifact=artifact)
        return
    if set(artifact) != {
        "schema_version",
        "job_id",
        "prediction",
        "commitment",
        "model_session",
    } or artifact.get("schema_version") != "green-v400-formal-prediction-artifact-v1":
        raise ValueError("existing prediction artifact schema is invalid")
    if artifact.get("job_id") != job.get("job_id"):
        raise ValueError("existing prediction artifact job differs from shard")
    packet = artifact["prediction"]
    if seal_prediction_packet(packet) != artifact["commitment"]:
        raise ValueError("existing prediction packet or commitment changed")
    validate_model_session_for_plan(artifact["model_session"], plan)
    binding = packet.get("formal_execution_binding", {})
    expected_binding_fields = {
        "plan_sha256",
        "prediction_job_id",
        "model_session_receipt_sha256",
        "green_direction_binding_sha256",
        "formal_prediction_runner_source_sha256",
        "response_precision_receipt_sha256",
        "response_evaluation_dtype",
    }
    execution = plan.get("prediction_execution", {})
    if (
        packet.get("schema_version") != "green-v400-sfc-prediction-packet-v2"
        or packet.get("protocol_id") != plan.get("protocol_id")
        or packet.get("row_id") != job.get("site_row_id")
        or set(binding) != expected_binding_fields
        or binding.get("plan_sha256") != plan.get("plan_sha256")
        or binding.get("prediction_job_id") != job.get("job_id")
        or binding.get("green_direction_binding_sha256")
        != job.get("green_direction_binding_sha256")
        or binding.get("model_session_receipt_sha256")
        != artifact["model_session"].get("receipt_sha256")
        or binding.get("formal_prediction_runner_source_sha256")
        != plan.get("source_file_sha256", {}).get(
            "src/green_v400_formal_prediction_runner.py"
        )
        or binding.get("response_evaluation_dtype") != "float64"
        or not isinstance(binding.get("response_precision_receipt_sha256"), str)
        or len(binding["response_precision_receipt_sha256"]) != 64
        or packet.get("integrated_gradients_steps")
        != execution.get("integrated_gradients_steps")
        or packet.get("ms_hvp_segments") != execution.get("ms_hvp_segments")
        or packet.get("response_batch_chunk_size")
        != execution.get("response_batch_chunk_size")
    ):
        raise ValueError("existing prediction formal binding differs from plan")


def validate_clean_model_exit(
    *, model: Any, model_manifest: dict[str, Any], plan: dict[str, Any]
) -> str:
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("batch model retains parameter gradients at exit")
    for hook in getattr(model, "hook_dict", {}).values():
        if any(
            len(getattr(hook, field, [])) != 0
            for field in ("fwd_hooks", "bwd_hooks")
        ):
            raise ValueError("batch model retains active hooks at exit")
    expected = model_manifest.get("weight_tensor_hashes", {})
    state = model.state_dict()
    observed = {
        name: tensor_sha256(value.float() if value.is_floating_point() else value)
        for name, value in state.items()
    }
    if observed != expected:
        raise ValueError("batch model weights changed during execution")
    digest = canonical_sha256(observed)
    if digest != plan.get("full_model_hash"):
        raise ValueError("batch exit model hash differs from the plan")
    return digest


def validate_batch_completion_receipt(
    *,
    receipt: dict[str, Any],
    plan: dict[str, Any],
    mode: str,
    phase: str,
    shard_index: int,
    shard_count: int,
    jobs: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    if receipt.get("schema_version") != "green-v400-batch-completion-receipt-v1":
        raise ValueError("batch completion receipt schema is invalid")
    if receipt.get("receipt_sha256") != receipt_sha256(receipt):
        raise ValueError("batch completion receipt self hash mismatch")
    job_ids = [job["job_id"] for job in jobs]
    if (
        receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("mode") != mode
        or receipt.get("phase") != phase
        or receipt.get("shard_index") != shard_index
        or receipt.get("shard_count") != shard_count
        or receipt.get("ordered_job_ids_sha256") != canonical_sha256(job_ids)
        or receipt.get("completed_job_count") != len(job_ids)
        or receipt.get("all_artifacts_valid") is not True
        or receipt.get("model_clean_exit") is not True
    ):
        raise ValueError("batch completion receipt differs from the sealed shard")
    artifacts = receipt.get("artifact_file_sha256", {})
    sessions = receipt.get("artifact_model_session_receipt_sha256", {})
    if set(artifacts) != set(job_ids) or set(sessions) != set(job_ids):
        raise ValueError("batch completion artifact set differs from the sealed shard")
    if receipt.get("distinct_artifact_model_session_receipt_sha256") != sorted(
        set(sessions.values())
    ):
        raise ValueError("batch completion session set differs from artifacts")
    final_session = receipt.get("final_validation_model_session_receipt_sha256")
    if not isinstance(final_session, str) or len(final_session) != 64:
        raise ValueError("batch final validation session is invalid")
    for job in jobs:
        path = output_directory / f"{job['job_id']}.json"
        if not path.is_file() or file_sha256(path) != artifacts[job["job_id"]]:
            raise ValueError("batch completion artifact file changed")
        artifact = load_json(path)
        if artifact.get("model_session", {}).get("receipt_sha256") != sessions[
            job["job_id"]
        ]:
            raise ValueError("batch completion artifact session mapping changed")
        validate_existing_artifact(
            plan=plan, mode=mode, job=job, artifact=artifact
        )


def run_batch(
    *,
    plan: dict[str, Any],
    universe: dict[str, Any],
    model_manifest: dict[str, Any],
    mode: str,
    phase: str,
    shard_index: int,
    shard_count: int,
    output_directory: Path,
    device: str,
    direction_registry: dict[str, Any] | None = None,
    direction_payload: Path | None = None,
    grant_capture_spec: dict[str, Any] | None = None,
    integrated_gradients_steps: int = 65,
    ms_hvp_segments: int = 8,
    response_batch_chunk_size: int = 16,
    resume: bool = False,
    parent_plan: dict[str, Any] | None = None,
    development_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verify_plan(plan)
    if plan.get("development_authorized") is True:
        if parent_plan is None or development_authorization is None:
            raise ValueError("development batch requires parent plan and authorization")
        validate_activated_plan(
            parent_plan=parent_plan,
            authorization=development_authorization,
            activated_plan=plan,
        )
    validate_runtime_envelope(plan, device)
    if plan.get("source_file_sha256", {}).get(BATCH_SOURCE_PATH) != file_sha256(
        Path(__file__)
    ):
        raise ValueError("formal batch worker source differs from the execution plan")
    jobs = select_shard_jobs(
        plan,
        mode=mode,
        phase=phase,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    execution = plan.get("prediction_execution", {})
    if (
        shard_count != execution.get("shard_count")
        or execution.get("physical_gpu_by_shard", {}).get(str(shard_index))
        != int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
    ):
        raise ValueError("batch shard or physical GPU differs from the plan")
    if mode == "prediction" and any(
        execution.get(key) != value
        for key, value in {
            "integrated_gradients_steps": integrated_gradients_steps,
            "ms_hvp_segments": ms_hvp_segments,
            "response_batch_chunk_size": response_batch_chunk_size,
        }.items()
    ):
        raise ValueError("prediction numerical parameters differ from the plan")
    for job in jobs:
        planned_job(plan, mode, job["job_id"])
    resolved_root = FORMAL_OUTPUT_ROOT.resolve(strict=True)
    resolved_output = output_directory.resolve(strict=True)
    if not resolved_output.is_relative_to(resolved_root):
        raise ValueError("batch output directory must remain under /mnt/sdb")
    completion_path = resolved_output / (
        f"_completion_{mode}_{phase}_{shard_index:02d}_of_{shard_count:02d}.json"
    )
    if completion_path.exists():
        if not resume:
            raise FileExistsError("batch completion already exists and resume is disabled")
        receipt = load_json(completion_path)
        validate_batch_completion_receipt(
            receipt=receipt,
            plan=plan,
            mode=mode,
            phase=phase,
            shard_index=shard_index,
            shard_count=shard_count,
            jobs=jobs,
            output_directory=resolved_output,
        )
        return {
            "planned": len(jobs),
            "completed": len(jobs),
            "newly_completed": 0,
            "all_artifacts_valid": True,
            "batch_completion_receipt_sha256": receipt["receipt_sha256"],
        }
    pending = []
    completed = []
    for job in jobs:
        path = resolved_output / f"{job['job_id']}.json"
        if not path.exists():
            pending.append((job, path))
            continue
        if not resume:
            raise FileExistsError("batch output already exists and resume is disabled")
        validate_existing_artifact(
            plan=plan,
            mode=mode,
            job=job,
            artifact=load_json(path),
        )
        completed.append(job["job_id"])
    if mode == "grant":
        if direction_registry is not None or direction_payload is not None:
            raise ValueError("Grant batch forbids every direction input")
        if grant_capture_spec is None:
            raise ValueError("Grant batch requires the frozen capture specification")
        panel = None
    else:
        if grant_capture_spec is not None:
            raise ValueError("prediction batch forbids a Grant capture input")
        if direction_registry is None or direction_payload is None:
            raise ValueError("prediction batch requires the public GREEN panel")
        panel = DirectionPanel(
            plan=plan,
            registry=direction_registry,
            payload_path=direction_payload,
        )
    model = load_frozen_model(model_manifest, device)
    session = model_session(plan, model)
    precision_receipt = None
    if mode == "prediction":
        precision_receipt = prepare_float64_response_evaluation(
            model=model,
            model_manifest=model_manifest,
            expected_model_manifest_sha256=plan["model_manifest_sha256"],
        )
    for job, path in pending:
        if mode == "grant":
            packet, commitment = run_formal_grant_prediction(
                plan=plan,
                universe=universe,
                capture_spec=grant_capture_spec,
                grant_job_id=job["job_id"],
                model_session_receipt=session,
                model=model,
            )
            artifact = {
                "schema_version": "green-v400-formal-grant-artifact-v1",
                "job_id": job["job_id"],
                "grant_prediction": packet,
                "commitment": commitment,
                "model_session": session,
            }
        else:
            directions, binding = panel.row(job["site_row_id"])
            packet, commitment = run_formal_prediction(
                plan=plan,
                universe=universe,
                prediction_job_id=job["job_id"],
                model_session_receipt=session,
                model_manifest=model_manifest,
                model=model,
                green_directions=directions,
                green_direction_binding=binding,
                integrated_gradients_steps=integrated_gradients_steps,
                ms_hvp_segments=ms_hvp_segments,
                response_batch_chunk_size=response_batch_chunk_size,
                response_precision_receipt=precision_receipt,
            )
            artifact = {
                "schema_version": "green-v400-formal-prediction-artifact-v1",
                "job_id": job["job_id"],
                "prediction": packet,
                "commitment": commitment,
                "model_session": session,
            }
        atomic_write_json(path, artifact)
        completed.append(job["job_id"])
    exit_model_hash = validate_clean_model_exit(
        model=model, model_manifest=model_manifest, plan=plan
    )
    artifact_hashes = {
        job["job_id"]: file_sha256(resolved_output / f"{job['job_id']}.json")
        for job in jobs
    }
    artifact_sessions = {
        job["job_id"]: load_json(
            resolved_output / f"{job['job_id']}.json"
        )["model_session"]["receipt_sha256"]
        for job in jobs
    }
    receipt = {
        "schema_version": "green-v400-batch-completion-receipt-v1",
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan["plan_sha256"],
        "mode": mode,
        "phase": phase,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "ordered_job_ids_sha256": canonical_sha256(
            [job["job_id"] for job in jobs]
        ),
        "completed_job_count": len(completed),
        "artifact_file_sha256": artifact_hashes,
        "artifact_model_session_receipt_sha256": artifact_sessions,
        "distinct_artifact_model_session_receipt_sha256": sorted(
            set(artifact_sessions.values())
        ),
        "final_validation_model_session_receipt_sha256": session["receipt_sha256"],
        "exit_full_model_hash": exit_model_hash,
        "batch_worker_source_sha256": file_sha256(Path(__file__)),
        "model_clean_exit": True,
        "all_artifacts_valid": len(completed) == len(jobs),
        "raw_activation_serialized": False,
        "authorizes_endpoint": False,
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    atomic_write_json(completion_path, receipt)
    return {
        "planned": len(jobs),
        "completed": len(completed),
        "newly_completed": len(pending),
        "all_artifacts_valid": len(completed) == len(jobs),
        "batch_completion_receipt_sha256": receipt["receipt_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prediction", "grant"), required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--development-authorization", type=Path)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--direction-registry", type=Path)
    parser.add_argument("--direction-payload", type=Path)
    parser.add_argument("--grant-capture-spec", type=Path)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--integrated-gradients-steps", type=int, default=65)
    parser.add_argument("--ms-hvp-segments", type=int, default=8)
    parser.add_argument("--response-batch-chunk-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_batch(
        plan=load_json(args.plan),
        universe=load_json(args.universe),
        model_manifest=load_json(args.model_manifest),
        mode=args.mode,
        phase=args.phase,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        output_directory=args.output_directory,
        device=args.device,
        direction_registry=(
            load_json(args.direction_registry) if args.direction_registry else None
        ),
        direction_payload=args.direction_payload,
        grant_capture_spec=(
            load_json(args.grant_capture_spec) if args.grant_capture_spec else None
        ),
        integrated_gradients_steps=args.integrated_gradients_steps,
        ms_hvp_segments=args.ms_hvp_segments,
        response_batch_chunk_size=args.response_batch_chunk_size,
        resume=args.resume,
        parent_plan=(load_json(args.parent_plan) if args.parent_plan else None),
        development_authorization=(
            load_json(args.development_authorization)
            if args.development_authorization
            else None
        ),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
