"""Only authorized construction route for formal held-out transport endpoints.

Unlike the low-level numerical core, this runner does not accept a center or
response callables.  It resolves the immutable universe row and planned site,
constructs task adapters internally, captures the clean activation center, and
binds every runtime input before evaluating the endpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from green_v400_endpoint_worker import compute_heldout_transport_endpoint
from green_v400_execution_receipts import (
    build_runtime_input_receipt,
    validate_endpoint_authorization_receipt,
    validate_model_session_receipt,
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


def _planned_endpoint_job(
    plan: dict[str, Any], authorization: dict[str, Any]
) -> dict[str, Any]:
    matches = [
        job
        for phase in ("development", "confirmation")
        for job in plan.get("queues", {}).get(f"{phase}_endpoint", [])
        if job.get("job_id") == authorization.get("endpoint_job_id")
    ]
    if len(matches) != 1:
        raise ValueError("authorization does not identify exactly one planned endpoint job")
    job = matches[0]
    for field, receipt_field in (
        ("site_row_id", "site_row_id"),
        ("prompt_row_id", "prompt_row_id"),
        ("layer", "layer"),
        ("hook", "hook"),
        ("endpoint_direction_binding_sha256", "endpoint_direction_binding_sha256"),
    ):
        if job.get(field) != authorization.get(receipt_field):
            raise ValueError(f"endpoint authorization changed planned {field}")
    return job


def _universe_row(
    plan: dict[str, Any], universe: dict[str, Any], prompt_row_id: str
) -> dict[str, Any]:
    if _sha256_value(universe) != plan.get("universe_sha256"):
        raise ValueError("full universe artifact differs from the execution plan")
    matches = [row for row in universe.get("rows", []) if row.get("row_id") == prompt_row_id]
    if len(matches) != 1:
        raise ValueError("planned prompt does not resolve to exactly one universe row")
    return matches[0]


def run_formal_heldout_transport_endpoint(
    *,
    plan: dict[str, Any],
    universe: dict[str, Any],
    endpoint_authorization_receipt: dict[str, Any],
    model_session_receipt: dict[str, Any],
    model_manifest: dict[str, Any],
    prediction_commitment: dict[str, Any],
    model: Any,
    endpoint_directions: torch.Tensor,
    endpoint_direction_binding: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan.get("execution_enabled") is not True:
        raise ValueError("prepare-only plan cannot run a formal endpoint")
    if plan.get("response_evaluation_precision", {}).get(
        "response_evaluation_dtype"
    ) != "float64":
        raise ValueError("plan does not bind float64 response evaluation")
    if plan["response_evaluation_precision"].get(
        "model_manifest_tensor_hash_scheme"
    ) != "sha256-contiguous-numpy-native-bytes-v1":
        raise ValueError("plan does not bind the frozen model-manifest hash scheme")
    job = _planned_endpoint_job(plan, endpoint_authorization_receipt)
    endpoint_worker_hash = _source_sha256(Path(__file__).with_name("green_v400_endpoint_worker.py"))
    validate_endpoint_authorization_receipt(
        receipt=endpoint_authorization_receipt,
        protocol_id=plan["protocol_id"],
        row_id=job["site_row_id"],
        prediction_commitment=prediction_commitment,
        endpoint_direction_binding_sha256=job["endpoint_direction_binding_sha256"],
        endpoint_worker_source_sha256=endpoint_worker_hash,
    )
    validate_model_session_receipt(model_session_receipt, endpoint_authorization_receipt)
    precision_receipt = prepare_float64_response_evaluation(
        model=model,
        model_manifest=model_manifest,
        expected_model_manifest_sha256=plan["model_manifest_sha256"],
    )
    row = _universe_row(plan, universe, job["prompt_row_id"])
    device = _model_device(model)
    clean_tokens = torch.tensor([row["clean_token_ids"]], dtype=torch.long, device=device)
    corrupt_tokens = torch.tensor([row["corrupt_token_ids"]], dtype=torch.long, device=device)

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
        raise ValueError("internally constructed adapter hook differs from planned site")
    adapter_hash = _source_sha256(adapter_path)
    if adapter_hash != endpoint_authorization_receipt[
        "response_adapter_source_sha256"
    ]:
        raise ValueError("runtime response adapter differs from authorization")

    precision_source = Path(__file__).with_name("green_v400_response_precision.py")
    if _source_sha256(precision_source) != plan.get("source_file_sha256", {}).get(
        "src/green_v400_response_precision.py"
    ):
        raise ValueError("response precision source differs from plan")
    center = capture_resid_post_center(model, clean_tokens, site).to(torch.float64)
    target_response, patched_response = build_target_and_patched_responses(
        model, clean_tokens, corrupt_tokens, site
    )
    formal_runner_hash = _source_sha256(Path(__file__))
    if formal_runner_hash != plan.get("source_file_sha256", {}).get(
        "src/green_v400_formal_endpoint_runner.py"
    ):
        raise ValueError("formal endpoint runner source differs from plan")
    runtime_receipt = build_runtime_input_receipt(
        authorization=endpoint_authorization_receipt,
        model_session=model_session_receipt,
        center=center,
        clean_token_ids=[int(value) for value in row["clean_token_ids"]],
        corrupt_token_ids=[int(value) for value in row["corrupt_token_ids"]],
        adapter_source_sha256=adapter_hash,
        formal_runner_source_sha256=formal_runner_hash,
        response_precision_receipt=precision_receipt,
    )
    return compute_heldout_transport_endpoint(
        protocol_id=plan["protocol_id"],
        row_id=job["site_row_id"],
        prediction_commitment=prediction_commitment,
        endpoint_authorization_receipt=endpoint_authorization_receipt,
        runtime_input_receipt=runtime_receipt,
        response_precision_receipt=precision_receipt,
        target_response=target_response,
        patched_response=patched_response,
        center=center,
        endpoint_directions=endpoint_directions.to(device=device, dtype=torch.float32),
        endpoint_direction_binding=endpoint_direction_binding,
        expected_endpoint_direction_binding_sha256=job[
            "endpoint_direction_binding_sha256"
        ],
    )
