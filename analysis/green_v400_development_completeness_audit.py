"""Fail-closed audit of GREEN v4 development analysis readiness.

This is a diagnostic assembler, not the frozen scientific analyzer.  It checks
that completed prediction and endpoint artifacts exactly cover the activated
development plan, verifies their packet commitments, and reports whether the
prespecified analyzer inputs actually exist.  Missing GREEN certificate or
Greater-Than clean-validity records are never inferred from endpoint outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from green_v400_endpoint_firewall import audit_commitment_pair
from green_v400_execution_receipts import validate_model_session_for_plan


PRIMARY_METHODS = (
    "finite_activation_patching",
    "first_order_attribution",
    "ms_hvp",
    "empirical_four_branch_interaction",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_plan(plan: dict[str, Any]) -> None:
    claimed = plan.get("plan_sha256")
    payload = dict(plan)
    payload.pop("plan_sha256", None)
    if not isinstance(claimed, str) or _sha256(payload) != claimed:
        raise ValueError("activated plan self hash mismatch")
    if (
        plan.get("development_authorized") is not True
        or plan.get("execution_enabled") is not True
        or plan.get("confirmation_authorized") is not False
        or plan.get("plan_gate") != "DEVELOPMENT_ONLY_AUTHORIZED"
    ):
        raise ValueError("plan is not a development-only activated plan")


def _artifact_index(root: Path, identity: Callable[[dict[str, Any]], Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = identity(payload)
        if key is None:
            continue
        if not isinstance(key, str) or not key:
            raise ValueError(f"invalid artifact identity in {path}")
        if key in result:
            raise ValueError(f"duplicate artifact identity: {key}")
        result[key] = payload
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty value list")

    def at(probability: float) -> float:
        location = (len(ordered) - 1) * probability
        lower = int(location)
        fraction = location - lower
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "min": at(0.0),
        "p25": at(0.25),
        "median": at(0.5),
        "p75": at(0.75),
        "max": at(1.0),
        "mean": fmean(ordered),
    }


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    wins = sum(
        float(left > right) + 0.5 * float(left == right)
        for left in positive
        for right in negative
    )
    return wins / (len(positive) * len(negative))


def _baseline_scores(packet: dict[str, Any]) -> dict[str, float]:
    baselines = packet.get("response_baselines", {})
    result = {}
    for method in PRIMARY_METHODS:
        record = baselines.get(method, {})
        field = "normalized_risk_score" if method == "empirical_four_branch_interaction" else "normalized_rmse"
        result[method] = _finite_nonnegative(record.get(field), f"{method}.{field}")
    return result


def audit_development_completeness(
    *,
    plan: dict[str, Any],
    prediction_artifacts: dict[str, dict[str, Any]],
    endpoint_artifacts: dict[str, dict[str, Any]],
    task: str,
) -> dict[str, Any]:
    _validate_plan(plan)
    if task not in {"ioi", "greater_than"}:
        raise ValueError("task must be ioi or greater_than")
    prediction_jobs = plan.get("queues", {}).get("development_prediction", [])
    endpoint_jobs = plan.get("queues", {}).get("development_endpoint", [])
    expected_prediction = {job["job_id"] for job in prediction_jobs}
    expected_endpoint = {job["job_id"] for job in endpoint_jobs}
    if set(prediction_artifacts) != expected_prediction:
        raise ValueError("prediction artifacts do not exactly cover the development plan")
    if set(endpoint_artifacts) != expected_endpoint:
        raise ValueError("endpoint artifacts do not exactly cover the development plan")
    prediction_by_site = {job["site_row_id"]: job for job in prediction_jobs}
    endpoint_by_site = {job["site_row_id"]: job for job in endpoint_jobs}
    if set(prediction_by_site) != set(endpoint_by_site):
        raise ValueError("prediction and endpoint site universes differ")

    rows = []
    endpoint_status_counts: dict[str, int] = {}
    green_status_count = 0
    clean_validity_count = 0
    validated_model_sessions: set[str] = set()
    for site_row_id in sorted(prediction_by_site):
        prediction_job = prediction_by_site[site_row_id]
        endpoint_job = endpoint_by_site[site_row_id]
        prediction_artifact = prediction_artifacts[prediction_job["job_id"]]
        endpoint_artifact = endpoint_artifacts[endpoint_job["job_id"]]
        if prediction_artifact.get("job_id") != prediction_job["job_id"]:
            raise ValueError("prediction artifact job identity mismatch")
        if endpoint_artifact.get("job_id") != endpoint_job["job_id"]:
            raise ValueError("endpoint artifact job identity mismatch")
        for model_session in (
            prediction_artifact.get("model_session", {}),
            endpoint_artifact.get("model_session", {}),
        ):
            receipt_digest = model_session.get("receipt_sha256")
            if not isinstance(receipt_digest, str):
                raise ValueError("model session receipt digest is missing")
            if receipt_digest not in validated_model_sessions:
                validate_model_session_for_plan(model_session, plan)
                validated_model_sessions.add(receipt_digest)
        audit_commitment_pair(
            prediction_artifact.get("prediction", {}),
            prediction_artifact.get("commitment", {}),
            endpoint_artifact.get("endpoint", {}),
            endpoint_artifact.get("commitment", {}),
        )
        prediction = prediction_artifact["prediction"]
        endpoint = endpoint_artifact["endpoint"]
        if prediction.get("row_id") != site_row_id or endpoint.get("row_id") != site_row_id:
            raise ValueError("packet site identity mismatch")
        status = endpoint.get("endpoint_status_private")
        if not isinstance(status, str) or not status:
            raise ValueError("endpoint status is missing")
        endpoint_status_counts[status] = endpoint_status_counts.get(status, 0) + 1
        green_status = prediction.get("green_certificate_status")
        clean_validity = prediction.get("clean_task_valid")
        green_status_count += int(isinstance(green_status, str) and bool(green_status))
        clean_validity_count += int(isinstance(clean_validity, bool))
        rows.append(
            {
                "row_id": site_row_id,
                "prompt_row_id": prediction_job["prompt_row_id"],
                "ordinary_restoration": _finite(
                    prediction.get("ordinary_restoration"), "ordinary_restoration"
                ),
                "endpoint_status": status,
                "heldout_transport_symmetric_normalized_error": _finite_nonnegative(
                    endpoint.get("heldout_transport_symmetric_normalized_error_private"),
                    "heldout transport symmetric normalized error",
                ),
                "heldout_transport_error": _finite_nonnegative(
                    endpoint.get("heldout_transport_error_private"),
                    "heldout transport absolute error",
                ),
                "baseline_risk_scores": _baseline_scores(prediction),
                "green_status": green_status,
                "clean_task_valid": clean_validity,
            }
        )

    valid = [row for row in rows if row["endpoint_status"] == "VALID"]
    high = [row for row in valid if row["ordinary_restoration"] >= 0.8]
    labels = [row["heldout_transport_symmetric_normalized_error"] > 0.2 for row in high]
    blockers = []
    if green_status_count != len(rows):
        blockers.append(
            {
                "code": "MISSING_GREEN_CERTIFICATE_STATUS",
                "present": green_status_count,
                "required": len(rows),
            }
        )
    if task == "greater_than" and clean_validity_count != len(rows):
        blockers.append(
            {
                "code": "MISSING_GREATER_THAN_CLEAN_TASK_VALIDITY",
                "present": clean_validity_count,
                "required": len(rows),
            }
        )
    if len(valid) != len(rows):
        blockers.append(
            {
                "code": "NONVALID_ENDPOINT_ROWS",
                "present": len(rows) - len(valid),
                "required": 0,
            }
        )
    return {
        "schema_version": "green-v400-development-completeness-audit-v1",
        "contains_scientific_outcome": True,
        "diagnostic_only": True,
        "confirmation_accessed": False,
        "confirmation_authorized": False,
        "task": task,
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan["plan_sha256"],
        "row_count": len(rows),
        "endpoint_status_counts": endpoint_status_counts,
        "high_restoration_valid_count": len(high),
        "high_restoration_valid_fraction": len(high) / len(rows),
        "high_restoration_transport_failure_count": sum(labels),
        "high_restoration_transport_failure_fraction": sum(labels) / len(labels) if labels else None,
        "all_row_summaries": {
            "ordinary_restoration": _quantiles([row["ordinary_restoration"] for row in rows]),
            "heldout_transport_symmetric_normalized_error": _quantiles(
                [row["heldout_transport_symmetric_normalized_error"] for row in valid]
            ),
            "heldout_transport_error": _quantiles(
                [row["heldout_transport_error"] for row in valid]
            ),
        },
        "high_restoration_baseline_auc_for_transport_error_gt_0_20": {
            method: _auc(labels, [row["baseline_risk_scores"][method] for row in high])
            for method in PRIMARY_METHODS
        },
        "frozen_primary_analyzer_ready": not blockers,
        "primary_analysis_blockers": blockers,
        "rows_sha256": _sha256(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--task", choices=("ioi", "greater_than"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    predictions = _artifact_index(args.prediction_root, lambda value: value.get("job_id"))
    endpoints = _artifact_index(args.endpoint_root, lambda value: value.get("job_id"))
    report = audit_development_completeness(
        plan=plan,
        prediction_artifacts=predictions,
        endpoint_artifacts=endpoints,
        task=args.task,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)


if __name__ == "__main__":
    main()
