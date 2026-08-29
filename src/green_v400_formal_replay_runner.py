"""Only plan-bound route for endpoint numerical-replay workers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from green_v400_direction_binding import verify_direction_binding
from green_v400_endpoint_calibration import compute_target_replay_packet
from green_v400_execution_receipts import validate_model_session_for_plan
from green_v400_response_precision import prepare_float64_response_evaluation


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ValueError("formal replay model has no parameters") from exc


def run_formal_target_replay(
    *,
    plan: dict[str, Any],
    universe: dict[str, Any],
    replay_job_id: str,
    model_session_receipt: dict[str, Any],
    model_manifest: dict[str, Any],
    model: Any,
    endpoint_directions: torch.Tensor,
    endpoint_direction_binding: dict[str, Any],
    replay_id: str,
    worker_instance_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan.get("execution_enabled") is not True:
        raise ValueError("prepare-only plan cannot run endpoint numerical replay")
    if plan.get("response_evaluation_precision", {}).get(
        "response_evaluation_dtype"
    ) != "float64":
        raise ValueError("plan does not bind float64 response evaluation")
    if plan["response_evaluation_precision"].get(
        "model_manifest_tensor_hash_scheme"
    ) != "sha256-contiguous-numpy-native-bytes-v1":
        raise ValueError("plan does not bind the frozen model-manifest hash scheme")
    validate_model_session_for_plan(model_session_receipt, plan)
    jobs = [
        job
        for job in plan.get("queues", {}).get("endpoint_numerical_replay", [])
        if job.get("job_id") == replay_job_id
    ]
    if len(jobs) != 1:
        raise ValueError("numerical replay job does not resolve exactly once")
    job = jobs[0]
    verify_direction_binding(
        tensor=endpoint_directions,
        binding=endpoint_direction_binding,
        expected_binding_sha256=job["endpoint_direction_binding_sha256"],
        protocol_id=plan["protocol_id"],
        row_id=job["site_row_id"],
        panel_kind="endpoint",
    )
    if _sha256_value(universe) != plan.get("universe_sha256"):
        raise ValueError("full universe artifact differs from the execution plan")
    rows = [
        row
        for row in universe.get("rows", [])
        if row.get("row_id") == job["prompt_row_id"]
    ]
    if len(rows) != 1:
        raise ValueError("planned replay prompt does not resolve exactly once")
    row = rows[0]

    source_bindings = plan.get("source_file_sha256", {})
    is_gt = "GT_REPLICATION" in plan["protocol_id"]
    if is_gt:
        from green_v400_greater_than_response_adapter import (
            GreaterThanInterventionSite,
            build_target_and_patched_responses,
            capture_resid_post_center,
        )

        adapter_path = Path(__file__).with_name("green_v400_greater_than_response_adapter.py")
        site = GreaterThanInterventionSite(
            layer=job["layer"],
            position=int(row["site_position"]),
            clean_suffix=int(row["y"]),
            suffix_token_ids=tuple(int(value) for value in universe["suffix_token_ids"]),
        )
    else:
        from green_v400_ioi_response_adapter import (
            IOIInterventionSite,
            build_target_and_patched_responses,
            capture_resid_post_center,
        )

        adapter_path = Path(__file__).with_name("green_v400_ioi_response_adapter.py")
        site = IOIInterventionSite(
            layer=job["layer"],
            position=int(row["signature"][1]),
            io_token_id=int(row["io_token_id"]),
            s_token_id=int(row["s_token_id"]),
        )
    if site.hook_name != f"blocks.{job['layer']}.hook_{job['hook']}":
        raise ValueError("internally constructed replay adapter differs from planned site")
    required_sources = {
        "src/green_v400_formal_replay_runner.py": Path(__file__),
        "src/green_v400_endpoint_calibration.py": Path(__file__).with_name(
            "green_v400_endpoint_calibration.py"
        ),
        "src/green_v400_response_precision.py": Path(__file__).with_name(
            "green_v400_response_precision.py"
        ),
        f"src/{adapter_path.name}": adapter_path,
    }
    for relative, path in required_sources.items():
        if source_bindings.get(relative) != _source_sha256(path):
            raise ValueError(f"formal replay source differs from plan: {relative}")

    precision_receipt = prepare_float64_response_evaluation(
        model=model,
        model_manifest=model_manifest,
        expected_model_manifest_sha256=plan["model_manifest_sha256"],
    )
    device = _model_device(model)
    clean_tokens = torch.tensor([row["clean_token_ids"]], dtype=torch.long, device=device)
    corrupt_tokens = torch.tensor([row["corrupt_token_ids"]], dtype=torch.long, device=device)
    center = capture_resid_post_center(model, clean_tokens, site).to(torch.float64)
    target_response, _ = build_target_and_patched_responses(
        model, clean_tokens, corrupt_tokens, site
    )
    return compute_target_replay_packet(
        protocol_id=plan["protocol_id"],
        row_id=job["site_row_id"],
        target_response=target_response,
        center=center,
        endpoint_directions=endpoint_directions.to(device=device),
        endpoint_direction_binding=endpoint_direction_binding,
        expected_endpoint_direction_binding_sha256=job[
            "endpoint_direction_binding_sha256"
        ],
        replay_id=replay_id,
        worker_instance_id=worker_instance_id,
        model_manifest_sha256=plan["model_manifest_sha256"],
        response_precision_receipt=precision_receipt,
    )
