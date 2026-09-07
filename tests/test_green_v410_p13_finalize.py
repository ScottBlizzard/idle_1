from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

from green_v410_p13_finalize import finalize_task, finalize_two_tasks
from green_v410_protocol import PROTOCOL_ID, sha256_canonical


def queue(task: str) -> dict:
    jobs = []
    for site in range(108):
        jobs.append({
            "job_id": f"job-{task}-{site}",
            "site_row_id": f"site-{task}-{site}",
            "prompt_row_id": f"prompt-{site % 12}",
            "direction_panel": [{"direction_ordinal": ordinal} for ordinal in range(8)],
        })
    payload = {
        "schema_version": "green-v410-p13-certificate-queue-v1",
        "protocol_id": PROTOCOL_ID, "attempt_index": 1,
        "queue_id": {"ioi": "GREEN_V410_P13_IOI_CERTIFICATE_QUEUE_A1", "greater_than": "GREEN_V410_P13_GT_CERTIFICATE_QUEUE_A1"}[task],
        "phase": "qualification", "task": task, "job_count": 108,
        "direction_record_count": 864,
        "sort_key": ["task", "phase", "prompt_row_id", "layer", "site_row_id"],
        "qualification_manifest_sha256": "a" * 64, "direction_registry_sha256": "b" * 64,
        "direction_payload_file_sha256": "c" * 64, "causal_cone_manifest_sha256": "d" * 64,
        "resource_manifest_sha256": "e" * 64, "confirmation_seal_sha256": "f" * 64,
        "source_bundle_sha256": "1" * 64, "compiled_backend_sha256": "8" * 64,
        "schema_bundle_sha256": "2" * 64,
        "model_manifest_sha256": "3" * 64, "tokenizer_manifest_sha256": "4" * 64,
        "selected_gate_spec_sha256": "5" * 64, "task_metric_spec_sha256": "6" * 64,
        "decision_rule_sha256": "7" * 64, "contains_endpoint_material": False,
        "jobs": jobs,
    }
    return payload | {"queue_manifest_sha256": sha256_canonical(payload)}


def records(q: dict, ratio: Fraction = Fraction(1, 10)) -> list[dict]:
    rows = []
    for job in q["jobs"]:
        for ordinal in range(8):
            payload = {
                "schema_version": "green-v410-p13-direction-contraction-v1",
                "protocol_id": PROTOCOL_ID, "attempt_index": 1, "task": q["task"],
                "queue_manifest_sha256": q["queue_manifest_sha256"],
                "job_id": job["job_id"], "site_row_id": job["site_row_id"],
                "direction_ordinal": ordinal, "direction_certificate_row_id": "row",
                "direction_certificate_sha256": "a" * 64,
                "branch_order": ["PAT_J", "PAT_B", "TAR_J", "TAR_B"],
                "branch_centers": {}, "branch_official_intervals": {},
                "independent_ad_derivatives": {},
                "ad_overlaps_official_intervals": {name: True for name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B")},
                "relational_official_interval": {},
                "contraction_ratio": [ratio.numerator, ratio.denominator],
                "resource_record": {}, "contains_endpoint_material": False,
            }
            rows.append(payload | {"record_sha256": sha256_canonical(payload)})
    return rows


def test_complete_two_task_pass():
    task_receipts = []
    for task in ("ioi", "greater_than"):
        q = queue(task)
        site_manifest, receipt = finalize_task(queue=q, direction_records=records(q))
        assert site_manifest["site_count"] == 108
        assert receipt["terminal_state"] == "PASS"
        task_receipts.append(receipt)
    assert finalize_two_tasks(task_receipts)["terminal_state"] == "PASS"


def test_tampered_record_is_rejected():
    q = queue("ioi")
    rows = records(q)
    rows[0] = deepcopy(rows[0])
    rows[0]["contraction_ratio"] = [1, 1]
    try:
        finalize_task(queue=q, direction_records=rows)
    except ValueError as error:
        assert "self hash" in str(error)
    else:
        raise AssertionError("tampered P13 record was accepted")
