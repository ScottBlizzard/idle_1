"""Plan-bound Grant-style downstream contextual-divergence route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from green_v400_execution_receipts import validate_model_session_for_plan
from green_v400_grant_prediction_worker import (
    compute_grant_divergence_prediction_packet,
)


CAPTURE_SPEC_PATH = "configs/green_v400_grant_capture_spec.json"
RUNNER_SOURCE_PATH = "src/green_v400_formal_grant_runner.py"
GRANT_CORE_SOURCE_PATH = "src/green_v400_grant_divergence.py"
GRANT_PACKET_SOURCE_PATH = "src/green_v400_grant_prediction_worker.py"


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
        raise ValueError("formal model must expose at least one parameter") from exc


def _planned_grant_job(
    plan: dict[str, Any], grant_job_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = [
        job
        for phase in ("development", "confirmation")
        for job in plan.get("queues", {}).get(
            f"{phase}_grant_cohort_prediction", []
        )
        if job.get("job_id") == grant_job_id
    ]
    if len(matches) != 1:
        raise ValueError("Grant job does not resolve exactly once in the plan")
    job = matches[0]
    phase = job.get("role")
    cohort = sorted(
        (
            row
            for row in plan.get("queues", {}).get(f"{phase}_prediction", [])
            if row.get("layer") == job.get("layer")
            and row.get("hook") == job.get("hook")
        ),
        key=lambda row: row.get("site_row_id", ""),
    )
    site_ids = [row.get("site_row_id") for row in cohort]
    if (
        not cohort
        or len(site_ids) != len(set(site_ids))
        or len(cohort) != job.get("cohort_size")
        or _sha256_value(site_ids) != job.get("cohort_site_row_ids_sha256")
    ):
        raise ValueError("Grant cohort does not exactly match the planned row queue")
    return job, cohort


def _universe_rows(
    plan: dict[str, Any], universe: dict[str, Any], cohort: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if _sha256_value(universe) != plan.get("universe_sha256"):
        raise ValueError("full universe artifact differs from the execution plan")
    rows = universe.get("rows", [])
    by_id = {row.get("row_id"): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("universe prompt identifiers are not unique")
    result = []
    for site in cohort:
        row = by_id.get(site.get("prompt_row_id"))
        if row is None:
            raise ValueError("planned Grant prompt is absent from the universe")
        result.append((site, row))
    return result


def _candidate_position(plan: dict[str, Any], row: dict[str, Any]) -> int:
    if "GT_REPLICATION" in plan.get("protocol_id", ""):
        value = row.get("site_position")
    else:
        signature = row.get("signature")
        value = signature[1] if isinstance(signature, list) and len(signature) > 1 else None
    if not isinstance(value, int) or value < 0:
        raise ValueError("Grant task-defined token position is invalid")
    return value


def _capture_triplet(
    *,
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    candidate_hook: str,
    measurement_hook: str,
    candidate_position: int,
    measurement_position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        clean_tokens.ndim != 2
        or clean_tokens.shape[0] != 1
        or clean_tokens.shape != corrupt_tokens.shape
        or candidate_position >= clean_tokens.shape[1]
        or measurement_position >= clean_tokens.shape[1]
        or measurement_position <= candidate_position
    ):
        raise ValueError("Grant token pair or position is invalid")
    center: list[torch.Tensor] = []
    natural: list[torch.Tensor] = []

    def capture_center(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        center.append(activation[0, candidate_position, :].detach().clone())
        return activation

    def capture_natural(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        natural.append(
            activation[0, measurement_position, :].detach().cpu().float().clone()
        )
        return activation

    with torch.no_grad():
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[
                (candidate_hook, capture_center),
                (measurement_hook, capture_natural),
            ],
        )
    if len(center) != 1 or len(natural) != 1:
        raise RuntimeError("Grant clean-run hooks must each fire exactly once")
    corrupt_control: list[torch.Tensor] = []

    def capture_corrupt(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        corrupt_control.append(
            activation[0, measurement_position, :].detach().cpu().float().clone()
        )
        return activation

    with torch.no_grad():
        model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[(measurement_hook, capture_corrupt)],
        )
    if len(corrupt_control) != 1:
        raise RuntimeError("Grant unpatched-corrupt measurement hook must fire exactly once")
    intervened: list[torch.Tensor] = []

    def patch_center(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        if activation.ndim != 3 or activation.shape[0] != 1:
            raise ValueError("Grant patch activation has invalid shape")
        result = activation.clone()
        result[0, candidate_position, :] = center[0]
        return result

    def capture_intervened(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        intervened.append(
            activation[0, measurement_position, :].detach().cpu().float().clone()
        )
        return activation

    with torch.no_grad():
        model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[
                (candidate_hook, patch_center),
                (measurement_hook, capture_intervened),
            ],
        )
    if len(intervened) != 1:
        raise RuntimeError("Grant intervened-run measurement hook must fire exactly once")
    if natural[0].shape != intervened[0].shape or natural[0].ndim != 1:
        raise ValueError("Grant captured state vectors have incompatible shapes")
    if not torch.isfinite(natural[0]).all() or not torch.isfinite(intervened[0]).all():
        raise ValueError("Grant captured state vectors must be finite")
    if corrupt_control[0].shape != natural[0].shape:
        raise ValueError("Grant corrupt-control state vector has incompatible shape")
    if not torch.isfinite(corrupt_control[0]).all():
        raise ValueError("Grant corrupt-control state vector must be finite")
    return natural[0], intervened[0], corrupt_control[0]


def run_formal_grant_prediction(
    *,
    plan: dict[str, Any],
    universe: dict[str, Any],
    capture_spec: dict[str, Any],
    grant_job_id: str,
    model_session_receipt: dict[str, Any],
    model: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect only plan-derived states and emit one cohort commitment."""

    if plan.get("execution_enabled") is not True:
        raise ValueError("prepare-only plan cannot run a Grant prediction")
    validate_model_session_for_plan(model_session_receipt, plan)
    capture_hash = _sha256_value(capture_spec)
    if (
        capture_hash != plan.get("grant_capture_spec_sha256")
        or plan.get("grant_capture_spec_path") != CAPTURE_SPEC_PATH
        or capture_spec.get("contains_scientific_outcome") is not False
        or capture_spec.get("measurement_hook") != "blocks.10.hook_resid_post"
        or capture_spec.get("measurement_position") != "final_prompt_position"
        or capture_spec.get(
            "measurement_position_must_be_strictly_after_candidate_position"
        )
        is not True
        or capture_spec.get("vectors_per_site_row") != 1
        or capture_spec.get("firewall", {}).get("green_direction_payload_access")
        is not False
        or capture_spec.get("firewall", {}).get(
            "heldout_direction_payload_access"
        )
        is not False
        or capture_spec.get("firewall", {}).get("heldout_outcome_access")
        is not False
    ):
        raise ValueError("Grant capture specification differs from the sealed plan")
    source_bindings = plan.get("source_file_sha256", {})
    source_paths = {
        RUNNER_SOURCE_PATH: Path(__file__),
        GRANT_CORE_SOURCE_PATH: Path(__file__).with_name("green_v400_grant_divergence.py"),
        GRANT_PACKET_SOURCE_PATH: Path(__file__).with_name(
            "green_v400_grant_prediction_worker.py"
        ),
    }
    for relative, path in source_paths.items():
        if source_bindings.get(relative) != _source_sha256(path):
            raise ValueError(f"formal Grant source differs from plan: {relative}")
    job, cohort = _planned_grant_job(plan, grant_job_id)
    if job.get("grant_capture_spec_sha256") != capture_hash:
        raise ValueError("Grant job is not bound to the capture specification")
    measurement_hook = capture_spec["measurement_hook"]
    measurement_layer = int(measurement_hook.split(".")[1])
    if measurement_layer <= int(job["layer"]):
        raise ValueError("Grant measurement hook is not strictly downstream")
    device = _model_device(model)
    natural_states = []
    intervened_states = []
    unpatched_corrupt_states = []
    for site, row in _universe_rows(plan, universe, cohort):
        candidate_position = _candidate_position(plan, row)
        measurement_position = len(row["clean_token_ids"]) - 1
        candidate_hook = f"blocks.{site['layer']}.hook_{site['hook']}"
        natural, intervened, corrupt_control = _capture_triplet(
            model=model,
            clean_tokens=torch.tensor(
                [row["clean_token_ids"]], dtype=torch.long, device=device
            ),
            corrupt_tokens=torch.tensor(
                [row["corrupt_token_ids"]], dtype=torch.long, device=device
            ),
            candidate_hook=candidate_hook,
            measurement_hook=measurement_hook,
            candidate_position=candidate_position,
            measurement_position=measurement_position,
        )
        natural_states.append(natural)
        intervened_states.append(intervened)
        unpatched_corrupt_states.append(corrupt_control)
    natural_panel = torch.stack(natural_states)
    intervened_panel = torch.stack(intervened_states)
    unpatched_corrupt_panel = torch.stack(unpatched_corrupt_states)
    formal_binding = {
        "plan_sha256": plan["plan_sha256"],
        "grant_job_id": job["job_id"],
        "model_session_receipt_sha256": model_session_receipt["receipt_sha256"],
        "grant_capture_spec_sha256": capture_hash,
        "cohort_site_row_ids_sha256": job["cohort_site_row_ids_sha256"],
        "cohort_size": job["cohort_size"],
        "analysis_seed": job["analysis_seed"],
        "measurement_hook": measurement_hook,
        "measurement_position_rule": capture_spec["measurement_position"],
        "measurement_position_strictly_after_candidate": True,
        "vectors_per_site_row": 1,
        "raw_activation_serialized": False,
        "formal_grant_runner_source_sha256": source_bindings[RUNNER_SOURCE_PATH],
        "grant_core_source_sha256": source_bindings[GRANT_CORE_SOURCE_PATH],
        "grant_packet_source_sha256": source_bindings[GRANT_PACKET_SOURCE_PATH],
    }
    return compute_grant_divergence_prediction_packet(
        protocol_id=plan["protocol_id"],
        cohort_id=job["cohort_site_row_ids_sha256"],
        phase=job["role"],
        diagnostic_label=capture_spec["diagnostic_name"],
        natural_states=natural_panel,
        intervened_states=intervened_panel,
        unpatched_corrupt_states=unpatched_corrupt_panel,
        seed=int(job["analysis_seed"]),
        sample_size=len(cohort),
        formal_execution_binding=formal_binding,
    )
