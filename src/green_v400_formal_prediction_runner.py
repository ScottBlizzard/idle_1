"""Only plan-bound route for untouched GREEN prediction packets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from green_bridge_spec import SELECTED_GATES
from green_v400_direction_binding import verify_direction_binding
from green_v400_execution_receipts import validate_model_session_for_plan
from green_v400_prediction_worker import (
    compute_ordinary_restoration,
    compute_response_baseline_packet,
)
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
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise ValueError("formal model must expose parameters for device binding")
    try:
        return next(parameters()).device
    except StopIteration as exc:
        raise ValueError("formal model has no parameters") from exc


def _planned_prediction_job(plan: dict[str, Any], prediction_job_id: str) -> dict[str, Any]:
    matches = [
        job
        for phase in ("development", "confirmation")
        for job in plan.get("queues", {}).get(f"{phase}_prediction", [])
        if job.get("job_id") == prediction_job_id
    ]
    if len(matches) != 1:
        raise ValueError("prediction job does not resolve exactly once in the plan")
    return matches[0]


def _universe_row(
    plan: dict[str, Any], universe: dict[str, Any], prompt_row_id: str
) -> dict[str, Any]:
    if _sha256_value(universe) != plan.get("universe_sha256"):
        raise ValueError("full universe artifact differs from the execution plan")
    matches = [row for row in universe.get("rows", []) if row.get("row_id") == prompt_row_id]
    if len(matches) != 1:
        raise ValueError("planned prompt does not resolve to exactly one universe row")
    return matches[0]


def run_formal_prediction(
    *,
    plan: dict[str, Any],
    universe: dict[str, Any],
    prediction_job_id: str,
    model_session_receipt: dict[str, Any],
    model_manifest: dict[str, Any],
    model: Any,
    green_directions: torch.Tensor,
    green_direction_binding: dict[str, Any],
    integrated_gradients_steps: int,
    ms_hvp_segments: int,
    response_batch_chunk_size: int = 32,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve every scientific input internally and emit one sealed prediction."""

    if plan.get("execution_enabled") is not True:
        raise ValueError("prepare-only plan cannot run an untouched prediction")
    if plan.get("response_evaluation_precision", {}).get(
        "response_evaluation_dtype"
    ) != "float64":
        raise ValueError("plan does not bind float64 response evaluation")
    if plan["response_evaluation_precision"].get(
        "model_manifest_tensor_hash_scheme"
    ) != "sha256-contiguous-numpy-native-bytes-v1":
        raise ValueError("plan does not bind the frozen model-manifest hash scheme")
    validate_model_session_for_plan(model_session_receipt, plan)
    precision_receipt = prepare_float64_response_evaluation(
        model=model,
        model_manifest=model_manifest,
        expected_model_manifest_sha256=plan["model_manifest_sha256"],
    )
    job = _planned_prediction_job(plan, prediction_job_id)
    if green_directions.dtype != torch.float32:
        raise ValueError("formal GREEN directions must be float32")
    verify_direction_binding(
        tensor=green_directions,
        binding=green_direction_binding,
        expected_binding_sha256=job["green_direction_binding_sha256"],
        protocol_id=plan["protocol_id"],
        row_id=job["site_row_id"],
        panel_kind="green",
    )
    row = _universe_row(plan, universe, job["prompt_row_id"])
    device = _model_device(model)
    clean_tokens = torch.tensor([row["clean_token_ids"]], dtype=torch.long, device=device)
    corrupt_tokens = torch.tensor([row["corrupt_token_ids"]], dtype=torch.long, device=device)

    is_gt = "GT_REPLICATION" in plan["protocol_id"]
    if is_gt:
        from green_v400_greater_than_response_adapter import (
            GreaterThanInterventionSite,
            build_matched_bypass_four_branch_responses,
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
            build_matched_bypass_four_branch_responses,
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
        raise ValueError("internally constructed adapter hook differs from planned site")

    source_bindings = plan.get("source_file_sha256", {})
    required_sources = {
        "src/green_v400_formal_prediction_runner.py": Path(__file__),
        "src/green_v400_prediction_worker.py": Path(__file__).with_name(
            "green_v400_prediction_worker.py"
        ),
        "src/green_v400_matched_bypass_adapter.py": Path(__file__).with_name(
            "green_v400_matched_bypass_adapter.py"
        ),
        "src/green_v400_four_branch_baseline.py": Path(__file__).with_name(
            "green_v400_four_branch_baseline.py"
        ),
        "src/green_v400_response_precision.py": Path(__file__).with_name(
            "green_v400_response_precision.py"
        ),
        f"src/{adapter_path.name}": adapter_path,
    }
    for relative, path in required_sources.items():
        if source_bindings.get(relative) != _source_sha256(path):
            raise ValueError(f"formal prediction source differs from plan: {relative}")

    center = capture_resid_post_center(model, clean_tokens, site).to(torch.float64)
    corrupt_center = capture_resid_post_center(model, corrupt_tokens, site).to(torch.float64)
    target_response, patched_response = build_target_and_patched_responses(
        model, clean_tokens, corrupt_tokens, site
    )
    clean_score = target_response(center)
    corrupt_score = patched_response(corrupt_center)
    patched_score = patched_response(center)
    restoration = compute_ordinary_restoration(
        clean_score, corrupt_score, patched_score
    )
    four_branches = build_matched_bypass_four_branch_responses(
        model,
        clean_tokens,
        corrupt_tokens,
        site,
        center,
        selected_gates=tuple(SELECTED_GATES),
    )
    formal_binding = {
        "plan_sha256": plan["plan_sha256"],
        "prediction_job_id": job["job_id"],
        "model_session_receipt_sha256": model_session_receipt["receipt_sha256"],
        "green_direction_binding_sha256": job["green_direction_binding_sha256"],
        "formal_prediction_runner_source_sha256": source_bindings[
            "src/green_v400_formal_prediction_runner.py"
        ],
        "response_precision_receipt_sha256": precision_receipt["receipt_sha256"],
        "response_evaluation_dtype": "float64",
    }
    return compute_response_baseline_packet(
        protocol_id=plan["protocol_id"],
        row_id=job["site_row_id"],
        target_response=target_response,
        patched_response=patched_response,
        four_branch_responses=four_branches,
        center=center,
        green_directions=green_directions.to(device=device, dtype=torch.float64),
        ordinary_restoration=restoration,
        integrated_gradients_steps=integrated_gradients_steps,
        ms_hvp_segments=ms_hvp_segments,
        response_batch_chunk_size=response_batch_chunk_size,
        formal_execution_binding=formal_binding,
    )
