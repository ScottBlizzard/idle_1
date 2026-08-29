"""Plan-bound process entry point for GREEN prediction, replay, and endpoint jobs.

The checked-in plans remain prepare-only, so this command fails before model or
payload loading until a later binding decision supplies a separately audited
execution-enabled plan.  Prediction mode never opens the endpoint payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

from green_v400_direction_binding import verify_direction_binding
from green_v400_execution_receipts import build_model_session_receipt
from green_v400_response_precision import tensor_sha256


WORKER_SOURCE_PATH = "analysis/green_v400_formal_worker.py"
FORMAL_OUTPUT_ROOT = Path("/mnt/sdb/ccj/iclr_1_runs")
ALLOWED_PHYSICAL_GPUS = {"4", "5", "6", "7"}
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    resolved_root = FORMAL_OUTPUT_ROOT.resolve(strict=True)
    resolved_parent = path.parent.resolve(strict=True)
    resolved_path = resolved_parent / path.name
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("formal worker output must remain under /mnt/sdb/ccj/iclr_1_runs")
    if resolved_path.exists():
        raise FileExistsError("formal worker refuses to overwrite an existing artifact")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=resolved_parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A same-filesystem hard link is an atomic no-clobber publication:
        # concurrent creators receive FileExistsError and neither artifact is
        # overwritten.  The temporary inode is removed only after publication.
        os.link(temporary, resolved_path)
        if os.name != "nt":
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(resolved_parent, directory_flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_runtime_envelope(plan: dict[str, Any], device: str) -> None:
    if plan.get("execution_enabled") is not True:
        raise ValueError("prepare-only plan cannot start a formal worker")
    if plan.get("real_outcomes_authorized") is not True:
        raise ValueError("formal worker requires a binding real-outcome authorization")
    if device != "cuda:0":
        raise ValueError("formal worker device must be cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError("formal worker must expose exactly one physical GPU in 4 through 7")
    if plan.get("gpu_policy", {}).get("physical_gpu_indices") != [4, 5, 6, 7]:
        raise ValueError("formal worker plan GPU policy mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("formal worker requires the frozen CUBLAS workspace setting")
    if any(os.environ.get(name) != "1" for name in THREAD_ENVIRONMENT):
        raise ValueError("formal worker requires every frozen thread count to equal one")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    source_hash = file_sha256(Path(__file__))
    if plan.get("source_file_sha256", {}).get(WORKER_SOURCE_PATH) != source_hash:
        raise ValueError("formal worker source differs from the execution plan")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify_plan(plan: dict[str, Any]) -> None:
    claimed = plan.get("plan_sha256")
    payload = dict(plan)
    payload.pop("plan_sha256", None)
    if canonical_sha256(payload) != claimed:
        raise ValueError("formal worker plan self hash mismatch")


def load_direction_row(
    *,
    plan: dict[str, Any],
    registry: dict[str, Any],
    payload_path: Path,
    panel: str,
    row_id: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if panel not in {"green", "endpoint"}:
        raise ValueError("direction panel must be green or endpoint")
    registry_without_hash = dict(registry)
    registry_hash = registry_without_hash.pop("registry_sha256", None)
    if canonical_sha256(registry_without_hash) != registry_hash:
        raise ValueError("direction registry self hash mismatch")
    if registry_hash != plan.get("direction_registry_sha256"):
        raise ValueError("direction registry differs from the execution plan")
    record = registry.get("panels", {}).get(panel, {})
    if payload_path.name != record.get("payload_filename"):
        raise ValueError("direction payload filename differs from registry")
    if file_sha256(payload_path) != record.get("payload_file_sha256"):
        raise ValueError("direction payload file hash mismatch")
    matches = [
        row for row in record.get("row_bindings", []) if row.get("row_id") == row_id
    ]
    if len(matches) != 1:
        raise ValueError("direction row does not resolve exactly once")
    row = matches[0]
    payload = np.load(payload_path, mmap_mode="r", allow_pickle=False)
    if list(payload.shape) != record.get("shape") or payload.dtype.str != "<f4":
        raise ValueError("direction payload array metadata mismatch")
    tensor = torch.from_numpy(np.array(payload[int(row["row_index"])], copy=True))
    verify_direction_binding(
        tensor=tensor,
        binding=row["binding"],
        expected_binding_sha256=row["binding_sha256"],
        protocol_id=plan["protocol_id"],
        row_id=row_id,
        panel_kind=panel,
    )
    return tensor, row["binding"]


def load_frozen_model(model_manifest: dict[str, Any], device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    model_id = model_manifest["model_name"]
    revision = model_manifest["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().to(device)
    hf_model.config.use_cache = False
    model = HookedTransformer.from_pretrained_no_processing(
        "gpt2",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float32,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        refactor_factored_attn_matrices=False,
        default_prepend_bos=False,
    ).eval()
    del hf_model
    return model


def model_session(plan: dict[str, Any], model: Any) -> dict[str, Any]:
    observed_hashes = {
        name: tensor_sha256(value) for name, value in model.state_dict().items()
    }
    nonce = hashlib.sha256(os.urandom(32)).hexdigest()
    return build_model_session_receipt(
        plan=plan,
        observed_full_model_hash=canonical_sha256(observed_hashes),
        loader_source_sha256=file_sha256(Path(__file__)),
        process_start_nonce=nonce,
        pid=os.getpid(),
    )


def planned_job(plan: dict[str, Any], mode: str, job_id: str) -> dict[str, Any]:
    queue_names = {
        "prediction": ("development_prediction", "confirmation_prediction"),
        "grant": (
            "development_grant_cohort_prediction",
            "confirmation_grant_cohort_prediction",
        ),
        "replay": ("endpoint_numerical_replay",),
        "endpoint": ("development_endpoint", "confirmation_endpoint"),
    }[mode]
    matches = [
        job
        for name in queue_names
        for job in plan.get("queues", {}).get(name, [])
        if job.get("job_id") == job_id
    ]
    if len(matches) != 1:
        raise ValueError("formal worker job does not resolve exactly once")
    job = matches[0]
    role = job.get("role")
    if mode == "replay":
        if plan.get("development_authorized") is not True:
            raise ValueError("numerical replay requires development authorization")
    elif role == "development":
        if plan.get("development_authorized") is not True:
            raise ValueError("development job is not authorized")
    elif role == "confirmation":
        if plan.get("confirmation_authorized") is not True:
            raise ValueError("confirmation job is not authorized")
    else:
        raise ValueError("formal worker job has an invalid scientific role")
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("prediction", "grant", "replay", "endpoint"), required=True
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--development-authorization", type=Path)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--direction-registry", type=Path)
    parser.add_argument("--direction-payload", type=Path)
    parser.add_argument("--grant-capture-spec", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--integrated-gradients-steps", type=int, default=65)
    parser.add_argument("--ms-hvp-segments", type=int, default=8)
    parser.add_argument("--response-batch-chunk-size", type=int, default=16)
    parser.add_argument("--replay-id", choices=("A", "B"))
    parser.add_argument("--prediction-commitment", type=Path)
    parser.add_argument("--endpoint-authorization-receipt", type=Path)
    args = parser.parse_args()

    plan = load_json(args.plan)
    verify_plan(plan)
    if plan.get("development_authorized") is True:
        if args.parent_plan is None or args.development_authorization is None:
            parser.error(
                "development execution requires --parent-plan and "
                "--development-authorization"
            )
        from analysis.green_v400_development_activation import (
            file_sha256 as activation_file_sha256,
            validate_activated_plan,
        )

        validate_activated_plan(
            parent_plan=load_json(args.parent_plan),
            authorization=load_json(args.development_authorization),
            activated_plan=plan,
            parent_plan_file_sha256=activation_file_sha256(args.parent_plan),
        )
    validate_runtime_envelope(plan, args.device)
    universe = load_json(args.universe)
    model_manifest = load_json(args.model_manifest)
    job = planned_job(plan, args.mode, args.job_id)
    if args.mode == "prediction":
        frozen = plan.get("prediction_execution", {})
        observed = {
            "integrated_gradients_steps": args.integrated_gradients_steps,
            "ms_hvp_segments": args.ms_hvp_segments,
            "response_batch_chunk_size": args.response_batch_chunk_size,
        }
        if any(frozen.get(key) != value for key, value in observed.items()):
            raise ValueError("prediction numerical parameters differ from the plan")
    directions = None
    binding = None
    if args.mode == "grant":
        from green_v400_formal_grant_runner import run_formal_grant_prediction

        if args.grant_capture_spec is None:
            parser.error("grant mode requires --grant-capture-spec")
        forbidden_route_inputs = (
            args.direction_registry,
            args.direction_payload,
            args.prediction_commitment,
            args.endpoint_authorization_receipt,
            args.replay_id,
        )
        if any(value is not None for value in forbidden_route_inputs):
            parser.error("grant mode forbids direction, replay, and adjudication inputs")
    else:
        if args.direction_registry is None or args.direction_payload is None:
            parser.error("non-Grant modes require direction registry and payload")
        registry = load_json(args.direction_registry)
        panel = "green" if args.mode == "prediction" else "endpoint"
        directions, binding = load_direction_row(
            plan=plan,
            registry=registry,
            payload_path=args.direction_payload,
            panel=panel,
            row_id=job["site_row_id"],
        )
    model = load_frozen_model(model_manifest, args.device)
    session = model_session(plan, model)

    if args.mode == "grant":
        packet, commitment = run_formal_grant_prediction(
            plan=plan,
            universe=universe,
            capture_spec=load_json(args.grant_capture_spec),
            grant_job_id=args.job_id,
            model_session_receipt=session,
            model=model,
        )
        artifact = {
            "schema_version": "green-v400-formal-grant-artifact-v1",
            "job_id": args.job_id,
            "grant_prediction": packet,
            "commitment": commitment,
            "model_session": session,
        }
    elif args.mode == "prediction":
        from green_v400_formal_prediction_runner import run_formal_prediction

        packet, commitment = run_formal_prediction(
            plan=plan,
            universe=universe,
            prediction_job_id=args.job_id,
            model_session_receipt=session,
            model_manifest=model_manifest,
            model=model,
            green_directions=directions,
            green_direction_binding=binding,
            integrated_gradients_steps=args.integrated_gradients_steps,
            ms_hvp_segments=args.ms_hvp_segments,
            response_batch_chunk_size=args.response_batch_chunk_size,
        )
        artifact = {
            "schema_version": "green-v400-formal-prediction-artifact-v1",
            "job_id": args.job_id,
            "prediction": packet,
            "commitment": commitment,
            "model_session": session,
        }
    elif args.mode == "replay":
        from green_v400_formal_replay_runner import run_formal_target_replay

        if args.replay_id is None:
            parser.error("--replay-id is required in replay mode")
        worker_instance_id = canonical_sha256(
            [session["process_start_nonce"], args.job_id, args.replay_id]
        )
        packet, commitment = run_formal_target_replay(
            plan=plan,
            universe=universe,
            replay_job_id=args.job_id,
            model_session_receipt=session,
            model_manifest=model_manifest,
            model=model,
            endpoint_directions=directions,
            endpoint_direction_binding=binding,
            replay_id=args.replay_id,
            worker_instance_id=worker_instance_id,
        )
        artifact = {
            "schema_version": "green-v400-formal-replay-worker-artifact-v1",
            "job_id": args.job_id,
            "replay": packet,
            "commitment": commitment,
            "model_session": session,
            "worker": {
                "worker_instance_id": worker_instance_id,
                "pid": os.getpid(),
                "process_start_nonce": session["process_start_nonce"],
                "python_executable_sha256": file_sha256(Path(sys.executable)),
                "source_file_sha256": file_sha256(Path(__file__)),
                "artifact_path": str(args.output.resolve()),
            },
        }
    else:
        from green_v400_formal_endpoint_runner import (
            run_formal_heldout_transport_endpoint,
        )

        if args.prediction_commitment is None or args.endpoint_authorization_receipt is None:
            parser.error(
                "endpoint mode requires --prediction-commitment and "
                "--endpoint-authorization-receipt"
            )
        endpoint, commitment = run_formal_heldout_transport_endpoint(
            plan=plan,
            universe=universe,
            endpoint_authorization_receipt=load_json(
                args.endpoint_authorization_receipt
            ),
            model_session_receipt=session,
            model_manifest=model_manifest,
            prediction_commitment=load_json(args.prediction_commitment),
            model=model,
            endpoint_directions=directions,
            endpoint_direction_binding=binding,
        )
        artifact = {
            "schema_version": "green-v400-formal-endpoint-artifact-v1",
            "job_id": args.job_id,
            "endpoint": endpoint,
            "commitment": commitment,
            "model_session": session,
        }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"job_id": args.job_id, "artifact_sha256": canonical_sha256(artifact)}))


if __name__ == "__main__":
    main()
