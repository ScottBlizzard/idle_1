"""Immutable publication, adoption, and identity helpers for GREEN v4.1."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from green_v410_protocol import (
    DECISION_RULE_SHA256,
    PROTOCOL_ID,
    RADIUS_PANEL,
    canonical_json_bytes,
    is_lower_sha256,
    sha256_bytes,
    sha256_canonical,
)
from green_v410_schemas import with_artifact_self_hash


RADIUS_PANEL_ID = "GREEN_V410_RADIUS_PANEL_T1_T1_2_T1_4_V1"
BRANCH_SEMANTICS_ID = "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_no_clobber_bytes(
    path: Path,
    payload: bytes,
    *,
    job_id: str,
    immutable_permissions: int = 0o444,
) -> dict[str, str]:
    """Atomically publish bytes without ever replacing a final artifact."""

    if not job_id or any(character in job_id for character in "/\\\0"):
        raise ValueError("job_id is invalid for an atomic temporary name")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    if path.exists():
        if path.is_file() and file_sha256(path) == digest and path.read_bytes() == payload:
            return {"publication": "EXISTING_IDENTICAL", "file_sha256": digest}
        raise FileExistsError(f"INVALID_CONFLICTING_ARTIFACT: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.{job_id}.{os.getpid()}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    publication = "CREATED"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_file() and file_sha256(path) == digest and path.read_bytes() == payload:
                publication = "EXISTING_IDENTICAL"
            else:
                raise FileExistsError(f"INVALID_CONFLICTING_ARTIFACT: {path}")
    finally:
        try:
            os.chmod(temporary, stat.S_IWRITE | stat.S_IREAD)
            temporary.unlink()
        except FileNotFoundError:
            pass
    if publication == "CREATED":
        os.chmod(path, immutable_permissions)
        _fsync_directory(path.parent)
    return {"publication": publication, "file_sha256": digest}


def atomic_no_clobber_json(path: Path, value: Mapping[str, Any], *, job_id: str) -> dict[str, str]:
    return atomic_no_clobber_bytes(
        path,
        canonical_json_bytes(value) + b"\n",
        job_id=job_id,
    )


def build_adoption_receipt(
    *,
    artifact_role: str,
    parent_path: Path,
    successor_role: str,
    successor_path: Path,
) -> dict[str, Any]:
    """Adopt only byte-identical immutable parent material."""

    if not parent_path.is_file() or not successor_path.is_file():
        raise FileNotFoundError("adoption input is missing")
    parent_hash = file_sha256(parent_path)
    successor_hash = file_sha256(successor_path)
    if parent_hash != successor_hash or parent_path.read_bytes() != successor_path.read_bytes():
        raise ValueError("adoption is not byte-identical")
    proof = {
        "schema_version": "green-v410-byte-identical-adoption-proof-v1",
        "parent_sha256": parent_hash,
        "successor_sha256": successor_hash,
        "byte_length": parent_path.stat().st_size,
    }
    return with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-adoption-receipt-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "artifact_role": artifact_role,
        "parent_path": parent_path.as_posix(),
        "parent_sha256": parent_hash,
        "successor_role": successor_role,
        "successor_path": successor_path.as_posix(),
        "successor_sha256": successor_hash,
        "semantic_equivalence": "BYTE_IDENTICAL",
        "semantic_proof_sha256": sha256_canonical(proof),
    })


def direction_certificate_row_id(
    *,
    task: str,
    phase: str,
    parent_site_row_id: str,
    site_identity_sha256: str,
    direction_ordinal: int,
    direction_payload_sha256: str,
    graph_manifest_sha256: str,
    resource_manifest_sha256: str,
    decision_rule_sha256: str = DECISION_RULE_SHA256,
) -> str:
    if task not in {"ioi", "greater_than"} or phase not in {"qualification", "confirmation"}:
        raise ValueError("direction certificate task or phase is invalid")
    if type(direction_ordinal) is not int or direction_ordinal not in range(8):
        raise ValueError("direction certificate ordinal is invalid")
    hashes = (
        site_identity_sha256, direction_payload_sha256, graph_manifest_sha256,
        resource_manifest_sha256, decision_rule_sha256,
    )
    if any(not is_lower_sha256(value) for value in hashes):
        raise ValueError("direction certificate identity contains malformed hash")
    return sha256_canonical({
        "schema_version": "green-v410-sfc-jwtec-direction-certificate-identity-v1",
        "protocol_id": PROTOCOL_ID,
        "task": task,
        "phase": phase,
        "parent_site_row_id": parent_site_row_id,
        "site_identity_sha256": site_identity_sha256,
        "direction_ordinal": direction_ordinal,
        "direction_payload_sha256": direction_payload_sha256,
        "radius_panel_id": RADIUS_PANEL_ID,
        "branch_semantics_id": BRANCH_SEMANTICS_ID,
        "graph_manifest_sha256": graph_manifest_sha256,
        "resource_manifest_sha256": resource_manifest_sha256,
        "decision_rule_sha256": decision_rule_sha256,
    })


def site_certificate_id(
    *,
    task: str,
    phase: str,
    parent_site_row_id: str,
    site_identity_sha256: str,
    direction_certificate_row_ids: Sequence[str],
    p13_two_task_pass_receipt_sha256: str,
    classifier_source_sha256: str,
    decision_rule_sha256: str = DECISION_RULE_SHA256,
) -> str:
    if task not in {"ioi", "greater_than"} or phase not in {"qualification", "confirmation"}:
        raise ValueError("site certificate task or phase is invalid")
    direction_ids = sorted(direction_certificate_row_ids)
    if len(direction_ids) != 8 or len(set(direction_ids)) != 8:
        raise ValueError("site certificate identity requires eight unique directions")
    hashes = (
        site_identity_sha256, *direction_ids, p13_two_task_pass_receipt_sha256,
        classifier_source_sha256, decision_rule_sha256,
    )
    if any(not is_lower_sha256(value) for value in hashes):
        raise ValueError("site certificate identity contains malformed hash")
    return sha256_canonical({
        "schema_version": "green-v410-sfc-jwtec-site-certificate-identity-v1",
        "protocol_id": PROTOCOL_ID,
        "task": task,
        "phase": phase,
        "parent_site_row_id": parent_site_row_id,
        "site_identity_sha256": site_identity_sha256,
        "sorted_direction_certificate_row_ids": direction_ids,
        "p13_two_task_pass_receipt_sha256": p13_two_task_pass_receipt_sha256,
        "decision_rule_sha256": decision_rule_sha256,
        "classifier_source_sha256": classifier_source_sha256,
    })
