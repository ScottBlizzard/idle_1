"""Strict aggregation and terminal receipts for GREEN v4.1 P13."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from green_v410_p13 import aggregate_task_site_ratios
from green_v410_p13_queue import validate_p13_queue
from green_v410_protocol import PROTOCOL_ID, sha256_canonical, strict_fields
from green_v410_schemas import validate_artifact


_RECORD_FIELDS = {
    "schema_version", "protocol_id", "attempt_index", "task",
    "queue_manifest_sha256", "job_id", "site_row_id", "direction_ordinal",
    "direction_certificate_row_id", "direction_certificate_sha256",
    "branch_order", "branch_centers", "branch_official_intervals",
    "independent_ad_derivatives", "ad_overlaps_official_intervals",
    "relational_official_interval", "contraction_ratio", "resource_record",
    "contains_endpoint_material", "record_sha256",
}


def _validate_record(record: Mapping[str, Any], queue: Mapping[str, Any]) -> None:
    strict_fields(record, _RECORD_FIELDS, "P13 direction record")
    payload = dict(record)
    claimed = payload.pop("record_sha256")
    if claimed != sha256_canonical(payload):
        raise ValueError("P13 direction record self hash mismatch")
    jobs = {job["job_id"]: job for job in queue["jobs"]}
    job = jobs.get(record["job_id"])
    ordinal = record["direction_ordinal"]
    if (
        job is None or type(ordinal) is not int or ordinal not in range(8)
        or record["schema_version"] != "green-v410-p13-direction-contraction-v1"
        or record["protocol_id"] != PROTOCOL_ID
        or record["attempt_index"] != 1
        or record["task"] != queue["task"]
        or record["queue_manifest_sha256"] != queue["queue_manifest_sha256"]
        or record["site_row_id"] != job["site_row_id"]
        or record["branch_order"] != ["PAT_J", "PAT_B", "TAR_J", "TAR_B"]
        or set(record["ad_overlaps_official_intervals"])
            != {"PAT_J", "PAT_B", "TAR_J", "TAR_B"}
        or not all(record["ad_overlaps_official_intervals"].values())
        or record["contains_endpoint_material"] is not False
    ):
        raise ValueError("P13 direction record binding mismatch")


def finalize_task(
    *, queue: Mapping[str, Any], direction_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one complete task and return its site manifest and receipt."""
    validate_p13_queue(queue)
    if len(direction_records) != 864:
        raise ValueError(f"P13 requires 864 direction records, got {len(direction_records)}")
    for record in direction_records:
        _validate_record(record, queue)
    expected_keys = {
        (job["job_id"], ordinal)
        for job in queue["jobs"] for ordinal in range(8)
    }
    observed_keys = {
        (record["job_id"], record["direction_ordinal"])
        for record in direction_records
    }
    if observed_keys != expected_keys or len(observed_keys) != 864:
        raise ValueError("P13 direction record set is not exactly the frozen queue")
    aggregate = aggregate_task_site_ratios(direction_records)
    site_manifest_payload = {
        "schema_version": "green-v410-p13-site-ratio-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": queue["task"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "site_count": 108,
        "direction_count": 864,
        "site_values": aggregate["site_values"],
        "site_values_sha256": aggregate["site_values_sha256"],
        "contains_endpoint_material": False,
    }
    site_manifest = site_manifest_payload | {
        "manifest_sha256": sha256_canonical(site_manifest_payload)
    }
    task_state = aggregate["terminal_state"]
    receipt_payload = {
        "schema_version": "green-v410-sfc-jwtec-p13-task-receipt-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": queue["task"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "resource_manifest_sha256": queue["resource_manifest_sha256"],
        "expected_prompt_clusters": 12,
        "expected_sites": 108,
        "expected_directions": 864,
        "observed_prompt_clusters": len({job["prompt_row_id"] for job in queue["jobs"]}),
        "observed_sites": len({record["site_row_id"] for record in direction_records}),
        "observed_directions": len(direction_records),
        "site_ratio_manifest_sha256": site_manifest["manifest_sha256"],
        "median_ratio": aggregate["median_ratio"],
        "p90_ratio": aggregate["p90_ratio"],
        "median_threshold": [1, 5],
        "p90_threshold": [1, 2],
        "all_records_valid": True,
        "terminal_state": task_state,
        "first_failure_code": None if task_state == "PASS" else "P13_THRESHOLD_FAILED",
    }
    receipt = receipt_payload | {"receipt_sha256": sha256_canonical(receipt_payload)}
    validate_artifact(receipt)
    return site_manifest, receipt


def finalize_two_tasks(task_receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the terminal two-task qualification receipt."""
    if len(task_receipts) != 2:
        raise ValueError("P13 global receipt requires exactly two task receipts")
    by_task = {receipt.get("task"): receipt for receipt in task_receipts}
    if set(by_task) != {"ioi", "greater_than"}:
        raise ValueError("P13 global receipt task set mismatch")
    for receipt in by_task.values():
        validate_artifact(receipt)
    states = {task: by_task[task]["terminal_state"] for task in ("ioi", "greater_than")}
    passed = all(state == "PASS" for state in states.values())
    payload = {
        "schema_version": "green-v410-sfc-jwtec-p13-two-task-receipt-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task_states": states,
        "task_receipt_sha256s": {
            task: by_task[task]["receipt_sha256"] for task in ("ioi", "greater_than")
        },
        "terminal_state": "PASS" if passed else "STOP_P13_QUALIFICATION_FAILED",
        "first_failure_code": None if passed else "P13_TASK_THRESHOLD_FAILED",
    }
    receipt = payload | {"receipt_sha256": sha256_canonical(payload)}
    validate_artifact(receipt)
    return receipt
