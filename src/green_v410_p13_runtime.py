"""Outcome-blind runtime closure and progress checks for P13 execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from green_v410_artifacts import file_sha256
from green_v410_p13_queue import validate_p13_queue
from green_v410_protocol import sha256_canonical


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_unstarted_certificate_unit(log_root: Path, unit_id: str) -> None:
    """A log claims an earlier launch; absent output is not a resume checkpoint.

    Call only after skipping a complete, identity-valid record. Even an empty
    log can belong to a running or interrupted worker and must block a replay.
    """
    if any(log_root.glob(f"{unit_id}.attempt_*.log")):
        raise RuntimeError(
            f"P13_CERTIFICATE_RELAUNCH_FORBIDDEN unit={unit_id}: "
            "prior launch has no valid completed record or charged checkpoint"
        )


def verify_runtime_closure(
    *, source_root: Path, queue: Mapping[str, Any],
    source_manifest: Mapping[str, Any], compiled_backend: Path,
) -> None:
    validate_p13_queue(queue)
    payload = dict(source_manifest)
    claimed = payload.pop("bundle_sha256", None)
    if claimed != sha256_canonical(payload) or claimed != queue["source_bundle_sha256"]:
        raise ValueError("P13 source bundle identity mismatch")
    records = source_manifest.get("files")
    if not isinstance(records, list) or len(records) != source_manifest.get("file_count"):
        raise ValueError("P13 source bundle file count mismatch")
    for record in records:
        path = source_root / record["path"]
        if (
            not path.is_file() or path.stat().st_size != record["byte_length"]
            or file_sha256(path) != record["file_sha256"]
        ):
            raise ValueError(f"P13 frozen source changed: {record['path']}")
    if (
        not compiled_backend.is_file()
        or file_sha256(compiled_backend) != queue["compiled_backend_sha256"]
    ):
        raise ValueError("P13 compiled backend identity mismatch")


def valid_capture(path: Path, *, queue_sha256: str, job_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = load_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return (
        value.get("manifest_sha256") == sha256_canonical(payload)
        and value.get("queue_manifest_sha256") == queue_sha256
        and value.get("job_id") == job_id
        and value.get("contains_endpoint_material") is False
    )


def valid_graph(
    path: Path, *, site_row_id: str, direction_ordinal: int,
    direction_payload_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = load_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    payload = {key: item for key, item in value.items() if key != "graph_manifest_sha256"}
    site_path = path.parent.parent / "site_identity.json"
    if not site_path.is_file():
        return False
    try:
        site = load_object(site_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        value.get("graph_manifest_sha256") == sha256_canonical(payload)
        and value.get("direction_ordinal") == direction_ordinal
        and value.get("direction_payload_sha256") == direction_payload_sha256
        and site.get("parent_site_row_id") == site_row_id
        and value.get("site_identity_sha256") == site.get("site_identity_sha256")
        and value.get("contains_endpoint_material") is False
    )


def valid_certificate_record(
    path: Path, *, queue_sha256: str, job_id: str,
    site_row_id: str, direction_ordinal: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = load_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    payload = dict(value)
    claimed = payload.pop("record_sha256", None)
    return (
        claimed == sha256_canonical(payload)
        and value.get("queue_manifest_sha256") == queue_sha256
        and value.get("job_id") == job_id
        and value.get("site_row_id") == site_row_id
        and value.get("direction_ordinal") == direction_ordinal
        and value.get("contains_endpoint_material") is False
    )


def progress_counts(*, queue: Mapping[str, Any], run_root: Path) -> dict[str, int]:
    task = queue["task"]
    capture_root = run_root / "capture" / task
    graph_root = run_root / "graphs" / task
    certificate_root = run_root / "certificates" / task
    captures = graphs = certificates = 0
    for job in queue["jobs"]:
        captures += valid_capture(
            capture_root / job["job_id"] / "capture_manifest.json",
            queue_sha256=queue["queue_manifest_sha256"], job_id=job["job_id"],
        )
        for direction in job["direction_panel"]:
            ordinal = direction["direction_ordinal"]
            graphs += valid_graph(
                graph_root / job["job_id"] / f"direction_{ordinal:02d}" / "graph_manifest.json",
                site_row_id=job["site_row_id"], direction_ordinal=ordinal,
                direction_payload_sha256=direction["direction_payload_sha256"],
            )
            certificates += valid_certificate_record(
                certificate_root / job["job_id"] / f"direction_{ordinal:02d}" / "p13_record.json",
                queue_sha256=queue["queue_manifest_sha256"], job_id=job["job_id"],
                site_row_id=job["site_row_id"], direction_ordinal=ordinal,
            )
    return {"captures": captures, "graphs": graphs, "certificates": certificates}
