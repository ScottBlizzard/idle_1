from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_p13_queue import build_p13_queue, validate_p13_queue
from green_v410_schemas import with_artifact_self_hash


PREPARE = ROOT / "analysis" / "GREEN_V410_P13_QUALIFICATION_PREPARE_20260831"


def _resource():
    return with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-resource-manifest-v2",
        "protocol_id": "GREEN-V410-SFC-JWTEC-20260830",
        "attempt_index": 1,
        "selected_leaf_budget": 4,
        "selection_rule": "largest passing candidate",
        "candidate_receipt_sha256s": [str(index) * 64 for index in range(1, 5)],
        "max_depth": 24,
        "graph_nodes_metric": "root_only_peak_live_dependent_scalar_outputs_v1",
        "max_graph_nodes": 2_000_000,
        "executor_source_sha256": "5" * 64,
        "resource_config_sha256": "6" * 64,
        "memory_max_bytes": 68_719_476_736,
        "direction_wall_max_seconds": 85_800,
        "guardband": "5/4",
        "max_process_launches": 2,
        "official_precision_bits": 384,
        "audit_precision_bits": 512,
        "radii": ["1", "1/2", "1/4"],
    })


def _seal():
    summary = {
        "schema_version": "green-v410-confirmation-seal-scanner-summary-v1",
        "protocol_id": "GREEN-V410-SFC-JWTEC-20260830",
        "attempt_index": 1,
        "roots": [],
        "scanned_file_count": 0,
        "scanned_byte_count": 0,
        "findings": [],
        "seal_policy_sha256": "7" * 64,
    }
    reports = [
        {"scanner_id": name, "summary": summary, "summary_sha256": "8" * 64}
        for name in ("PATHLIB_Rglob_V1", "OS_SCANDIR_V1")
    ]
    return with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-confirmation-seal-v1",
        "protocol_id": "GREEN-V410-SFC-JWTEC-20260830",
        "attempt_index": 1,
        "scanner_reports": reports,
        "scanner_report_sha256s": ["9" * 64, "a" * 64],
        "summaries_identical": True,
        "forbidden_findings": [],
        "terminal_state": "PASS",
        "scanned_roots": [],
        "seal_policy_sha256": "7" * 64,
    })


def test_builds_closed_endpoint_free_p13_queues():
    model = json.loads((
        ROOT / "analysis" / "GREEN_V400_FORMAL_PREPARE_ARTIFACTS_20260826"
        / "model_manifest.json"
    ).read_text(encoding="utf-8"))
    for task, expected_id in (
        ("ioi", "GREEN_V410_P13_IOI_CERTIFICATE_QUEUE_A1"),
        ("greater_than", "GREEN_V410_P13_GT_CERTIFICATE_QUEUE_A1"),
    ):
        queue = build_p13_queue(
            task=task,
            prepare_root=PREPARE,
            resource_manifest=_resource(),
            confirmation_seal=_seal(),
            model_manifest=model,
            source_bundle_sha256="b" * 64,
            compiled_backend_sha256="c" * 64,
        )
        validate_p13_queue(queue)
        assert queue["queue_id"] == expected_id
        assert queue["job_count"] == 108
        assert queue["direction_record_count"] == 864
        assert queue["contains_endpoint_material"] is False
        assert len({row["job_id"] for row in queue["jobs"]}) == 108
        assert all(
            [item["direction_ordinal"] for item in row["direction_panel"]]
            == list(range(8))
            for row in queue["jobs"]
        )
