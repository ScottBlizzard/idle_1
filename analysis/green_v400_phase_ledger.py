"""Hash-chained phase ledger for GREEN formal execution ordering."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def initialize_phase_ledger(plan: dict[str, Any]) -> dict[str, Any]:
    ledger = {
        "schema_version": "green-v400-phase-ledger-v1",
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan["plan_sha256"],
        "execution_enabled": plan.get("execution_enabled") is True,
        "events": [],
        "completed_prediction_job_ids": [],
        "completed_grant_job_ids": [],
        "numerical_replay_receipt_sha256": [],
        "development_analysis_receipt_sha256": None,
    }
    ledger["ledger_head_sha256"] = _hash({"genesis": ledger["plan_sha256"]})
    return ledger


def append_phase_event(
    ledger: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    if ledger.get("execution_enabled") is not True:
        raise ValueError("prepare-only ledger cannot record execution events")
    if event.get("plan_sha256") != ledger.get("plan_sha256"):
        raise ValueError("phase event plan mismatch")
    if event.get("previous_ledger_head_sha256") != ledger.get("ledger_head_sha256"):
        raise ValueError("phase event does not extend current ledger head")
    kind = event.get("kind")
    updated = json.loads(json.dumps(ledger))
    if kind == "numerical_replay_layer_complete":
        digest = event.get("receipt_sha256")
        if digest in updated["numerical_replay_receipt_sha256"]:
            raise ValueError("numerical replay receipt already recorded")
        updated["numerical_replay_receipt_sha256"].append(digest)
    elif kind == "prediction_committed":
        job_id = event.get("job_id")
        if event.get("phase") == "confirmation" and updated[
            "development_analysis_receipt_sha256"
        ] is None:
            raise ValueError("confirmation prediction is locked before development analysis")
        if job_id in updated["completed_prediction_job_ids"]:
            raise ValueError("prediction job already recorded")
        updated["completed_prediction_job_ids"].append(job_id)
    elif kind == "grant_cohort_committed":
        job_id = event.get("job_id")
        if job_id in updated["completed_grant_job_ids"]:
            raise ValueError("Grant cohort job already recorded")
        updated["completed_grant_job_ids"].append(job_id)
    elif kind == "development_analysis_complete":
        if updated["development_analysis_receipt_sha256"] is not None:
            raise ValueError("development analysis is one-shot")
        updated["development_analysis_receipt_sha256"] = event.get("receipt_sha256")
    else:
        raise ValueError("unsupported phase event kind")
    serialized_event = dict(event)
    serialized_event["event_sha256"] = _hash(event)
    updated["events"].append(serialized_event)
    updated["completed_prediction_job_ids"].sort()
    updated["completed_grant_job_ids"].sort()
    updated["numerical_replay_receipt_sha256"].sort()
    updated["ledger_head_sha256"] = _hash({
        "previous": ledger["ledger_head_sha256"],
        "event": serialized_event["event_sha256"],
    })
    return updated
