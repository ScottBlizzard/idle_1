from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_artifacts import (
    atomic_no_clobber_bytes,
    atomic_no_clobber_json,
    build_adoption_receipt,
    direction_certificate_row_id,
    site_certificate_id,
)
from green_v410_protocol import canonical_json_bytes
from green_v410_schemas import validate_artifact


H = "a" * 64


def test_atomic_no_clobber_create_and_identical_resume(tmp_path):
    path = tmp_path / "result.json"
    value = {"z": 1, "a": [2, 3]}
    created = atomic_no_clobber_json(path, value, job_id="job-1")
    assert created["publication"] == "CREATED"
    assert path.read_bytes() == canonical_json_bytes(value) + b"\n"
    resumed = atomic_no_clobber_json(path, value, job_id="job-1")
    assert resumed["publication"] == "EXISTING_IDENTICAL"


def test_atomic_no_clobber_rejects_conflict(tmp_path):
    path = tmp_path / "result.bin"
    atomic_no_clobber_bytes(path, b"first", job_id="job-1")
    with pytest.raises(FileExistsError, match="INVALID_CONFLICTING_ARTIFACT"):
        atomic_no_clobber_bytes(path, b"second", job_id="job-1")
    assert path.read_bytes() == b"first"


def test_atomic_publication_leaves_no_partial_files(tmp_path):
    path = tmp_path / "result.bin"
    atomic_no_clobber_bytes(path, b"payload", job_id="job-1")
    assert [candidate for candidate in tmp_path.iterdir() if ".partial." in candidate.name] == []


def test_byte_identical_adoption_receipt(tmp_path):
    parent = tmp_path / "parent.json"
    successor = tmp_path / "successor.json"
    parent.write_bytes(b"same")
    successor.write_bytes(b"same")
    receipt = build_adoption_receipt(
        artifact_role="parent-confirmation-direction-manifest",
        parent_path=parent,
        successor_role="v410-confirmation-direction-manifest",
        successor_path=successor,
    )
    validate_artifact(receipt)
    assert receipt["semantic_equivalence"] == "BYTE_IDENTICAL"


def test_adoption_rejects_semantic_guessing(tmp_path):
    parent = tmp_path / "parent.json"
    successor = tmp_path / "successor.json"
    parent.write_text('{"a":1}', encoding="utf-8")
    successor.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not byte-identical"):
        build_adoption_receipt(
            artifact_role="parent", parent_path=parent,
            successor_role="successor", successor_path=successor,
        )


def test_direction_identity_is_deterministic_and_ordinal_sensitive():
    common = dict(
        task="ioi", phase="confirmation", parent_site_row_id="site",
        site_identity_sha256=H, direction_payload_sha256=H,
        graph_manifest_sha256=H, resource_manifest_sha256=H,
    )
    first = direction_certificate_row_id(direction_ordinal=0, **common)
    assert first == direction_certificate_row_id(direction_ordinal=0, **common)
    assert first != direction_certificate_row_id(direction_ordinal=1, **common)


def test_site_identity_sorts_direction_ids_but_requires_all_eight():
    ids = [f"{index + 1:064x}" for index in range(8)]
    common = dict(
        task="greater_than", phase="qualification", parent_site_row_id="site",
        site_identity_sha256=H, p13_two_task_pass_receipt_sha256=H,
        classifier_source_sha256=H,
    )
    forward = site_certificate_id(direction_certificate_row_ids=ids, **common)
    reverse = site_certificate_id(direction_certificate_row_ids=list(reversed(ids)), **common)
    assert forward == reverse
    with pytest.raises(ValueError, match="eight unique"):
        site_certificate_id(direction_certificate_row_ids=ids[:-1], **common)

