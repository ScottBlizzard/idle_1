"""Hash-chained, plan-derived phase ledger for GREEN formal execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value


def _planned_ids(plan: dict[str, Any], suffix: str) -> dict[str, list[str]]:
    return {
        phase: sorted(
            job["job_id"]
            for job in plan.get("queues", {}).get(f"{phase}_{suffix}", [])
        )
        for phase in ("development", "confirmation")
    }


def initialize_phase_ledger(plan: dict[str, Any]) -> dict[str, Any]:
    replay_layers = sorted(
        {
            int(job["layer"])
            for job in plan.get("queues", {}).get("endpoint_numerical_replay", [])
        }
    )
    ledger = {
        "schema_version": "green-v400-phase-ledger-v3",
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan["plan_sha256"],
        "execution_enabled": plan.get("execution_enabled") is True,
        "development_authorized": plan.get("development_authorized") is True,
        "confirmation_authorized": plan.get("confirmation_authorized") is True,
        "parent_plan_sha256": plan.get("parent_plan_sha256"),
        "development_authorization_sha256": plan.get(
            "development_authorization_sha256"
        ),
        "planned_prediction_job_ids": _planned_ids(plan, "prediction"),
        "planned_grant_job_ids": _planned_ids(plan, "grant_cohort_prediction"),
        "planned_replay_layers": replay_layers,
        "events": [],
        "completed_prediction_job_ids": [],
        "completed_prediction_commitment_sha256": {},
        "completed_prediction_batch_receipt_sha256": {},
        "completed_grant_job_ids": [],
        "completed_grant_receipt_sha256": {},
        "completed_grant_batch_receipt_sha256": {},
        "numerical_replay_receipt_sha256": [],
        "numerical_replay_receipt_by_layer": {},
        "development_analysis_receipt_sha256": None,
    }
    ledger["ledger_head_sha256"] = _hash({"genesis": ledger["plan_sha256"]})
    return ledger


def _require_exact_event_fields(event: dict[str, Any], extras: set[str]) -> None:
    expected = {
        "plan_sha256",
        "previous_ledger_head_sha256",
        "kind",
        *extras,
    }
    if set(event) != expected:
        raise ValueError("phase event does not use the strict schema")


def append_phase_event(
    ledger: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    if ledger.get("schema_version") != "green-v400-phase-ledger-v3":
        raise ValueError("phase ledger schema is invalid")
    if ledger.get("execution_enabled") is not True:
        raise ValueError("prepare-only ledger cannot record execution events")
    if event.get("plan_sha256") != ledger.get("plan_sha256"):
        raise ValueError("phase event plan mismatch")
    if event.get("previous_ledger_head_sha256") != ledger.get(
        "ledger_head_sha256"
    ):
        raise ValueError("phase event does not extend current ledger head")
    kind = event.get("kind")
    updated = json.loads(json.dumps(ledger))
    if kind == "numerical_replay_layer_complete":
        _require_exact_event_fields(event, {"layer", "receipt_sha256"})
        layer = event.get("layer")
        digest = _digest(event.get("receipt_sha256"), "replay receipt")
        if layer not in updated["planned_replay_layers"]:
            raise ValueError("numerical replay layer is absent from the plan")
        key = str(layer)
        if key in updated["numerical_replay_receipt_by_layer"]:
            raise ValueError("numerical replay layer already recorded")
        updated["numerical_replay_receipt_by_layer"][key] = digest
        updated["numerical_replay_receipt_sha256"].append(digest)
    elif kind == "prediction_committed":
        _require_exact_event_fields(
            event,
            {
                "phase",
                "job_id",
                "commitment_sha256",
                "batch_completion_receipt_sha256",
            },
        )
        phase = event.get("phase")
        job_id = event.get("job_id")
        if phase == "development" and updated["development_authorized"] is not True:
            raise ValueError("development prediction is not authorized")
        if phase == "confirmation" and updated["confirmation_authorized"] is not True:
            raise ValueError("confirmation prediction is not authorized")
        digest = _digest(event.get("commitment_sha256"), "prediction commitment")
        batch_digest = _digest(
            event.get("batch_completion_receipt_sha256"),
            "prediction batch completion receipt",
        )
        if job_id not in updated["planned_prediction_job_ids"].get(phase, []):
            raise ValueError("prediction job is absent from the planned phase queue")
        if phase == "confirmation" and updated[
            "development_analysis_receipt_sha256"
        ] is None:
            raise ValueError("confirmation prediction is locked before development analysis")
        if job_id in updated["completed_prediction_commitment_sha256"]:
            raise ValueError("prediction job already recorded")
        updated["completed_prediction_job_ids"].append(job_id)
        updated["completed_prediction_commitment_sha256"][job_id] = digest
        updated["completed_prediction_batch_receipt_sha256"][job_id] = batch_digest
    elif kind == "grant_cohort_committed":
        _require_exact_event_fields(
            event,
            {
                "phase",
                "job_id",
                "receipt_sha256",
                "batch_completion_receipt_sha256",
            },
        )
        phase = event.get("phase")
        job_id = event.get("job_id")
        if phase == "development" and updated["development_authorized"] is not True:
            raise ValueError("development Grant cohort is not authorized")
        if phase == "confirmation" and updated["confirmation_authorized"] is not True:
            raise ValueError("confirmation Grant cohort is not authorized")
        digest = _digest(event.get("receipt_sha256"), "Grant receipt")
        batch_digest = _digest(
            event.get("batch_completion_receipt_sha256"),
            "Grant batch completion receipt",
        )
        if job_id not in updated["planned_grant_job_ids"].get(phase, []):
            raise ValueError("Grant cohort job is absent from the planned phase queue")
        if phase == "confirmation" and updated[
            "development_analysis_receipt_sha256"
        ] is None:
            raise ValueError("confirmation Grant cohort is locked before development analysis")
        if job_id in updated["completed_grant_receipt_sha256"]:
            raise ValueError("Grant cohort job already recorded")
        updated["completed_grant_job_ids"].append(job_id)
        updated["completed_grant_receipt_sha256"][job_id] = digest
        updated["completed_grant_batch_receipt_sha256"][job_id] = batch_digest
    elif kind == "development_analysis_complete":
        _require_exact_event_fields(event, {"receipt_sha256"})
        digest = _digest(
            event.get("receipt_sha256"), "development analysis receipt"
        )
        if updated["development_authorized"] is not True:
            raise ValueError("development analysis is not authorized")
        if updated["development_analysis_receipt_sha256"] is not None:
            raise ValueError("development analysis is one-shot")
        if set(updated["completed_prediction_job_ids"]) != set(
            updated["planned_prediction_job_ids"]["development"]
        ):
            raise ValueError("development analysis requires every prediction commitment")
        if set(updated["completed_grant_job_ids"]) != set(
            updated["planned_grant_job_ids"]["development"]
        ):
            raise ValueError("development analysis requires every Grant cohort receipt")
        updated["development_analysis_receipt_sha256"] = digest
    else:
        raise ValueError("unsupported phase event kind")
    serialized_event = dict(event)
    serialized_event["event_sha256"] = _hash(event)
    updated["events"].append(serialized_event)
    updated["completed_prediction_job_ids"].sort()
    updated["completed_grant_job_ids"].sort()
    updated["numerical_replay_receipt_sha256"].sort()
    updated["ledger_head_sha256"] = _hash(
        {
            "previous": ledger["ledger_head_sha256"],
            "event": serialized_event["event_sha256"],
        }
    )
    return updated


def validate_phase_ledger(plan: dict[str, Any], ledger: dict[str, Any]) -> None:
    expected = initialize_phase_ledger(plan)
    for serialized in ledger.get("events", []):
        if not isinstance(serialized, dict):
            raise ValueError("phase ledger event must be an object")
        event = dict(serialized)
        claimed = event.pop("event_sha256", None)
        if _hash(event) != claimed:
            raise ValueError("phase ledger event hash mismatch")
        expected = append_phase_event(expected, event)
    if expected != ledger:
        raise ValueError("phase ledger derived state or hash chain mismatch")
