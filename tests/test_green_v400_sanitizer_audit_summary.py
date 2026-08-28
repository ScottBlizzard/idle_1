from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_v400_sanitizer_audit_summary import (
    EXIT_SCHEMA, INPUT_SCHEMA, main, summarize_manifest,
)
from green_bridge_v400_schemas import canonical_json, sha256_canonical


def _write_report(path: Path, status: str) -> None:
    payload = {
        "schema_version": "green-v400-synthetic-sanitizer-fixture-v1",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "status": status,
        "unretained_internal_fixture": {"value": [1, 2, 3]},
    }
    payload["report_semantic_hash"] = sha256_canonical(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_exit(path: Path, exit_code: int = 0, signal_number=None) -> None:
    path.write_text(json.dumps({
        "schema_version": EXIT_SCHEMA,
        "exit_code": exit_code,
        "termination_signal": signal_number,
    }), encoding="utf-8")


def _entry(tmp_path: Path, name: str, sanitizer: str, status: str) -> dict:
    report = tmp_path / f"{name}.json"
    log = tmp_path / f"{name}.log"
    exit_record = tmp_path / f"{name}.exit.json"
    _write_report(report, status)
    log.write_text("audit completed without sanitizer diagnostics\n", encoding="utf-8")
    _write_exit(exit_record)
    return {
        "name": name,
        "sanitizer": sanitizer,
        "expected_status": status,
        "audit_json": report.name,
        "sanitizer_log": log.name,
        "process_exit_json": exit_record.name,
    }


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": INPUT_SCHEMA,
        "entries": [
            _entry(tmp_path, "tsan_close", "TSAN", "PASS_CLOSE"),
            _entry(tmp_path, "asan_context", "ASAN_UBSAN", "PASS_CONTEXT"),
        ],
    }), encoding="utf-8")
    return path


def test_sanitizer_summary_is_canonical_outcome_free_and_hash_closed(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "summary.json"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_SANITIZER_AUDIT_SUMMARY"
    assert summary["required_sanitizer_families_present"] is True
    assert summary["report_contains_scientific_outcome"] is False
    assert summary["supervisor_applied_scientific_threshold"] is False
    assert [entry["name"] for entry in summary["entries"]] == [
        "asan_context", "tsan_close",
    ]
    assert "unretained_internal_fixture" not in output.read_text(encoding="utf-8")
    recorded_hash = summary.pop("report_semantic_hash")
    assert recorded_hash == sha256_canonical(summary)
    summary["report_semantic_hash"] = recorded_hash
    assert output.read_text(encoding="utf-8") == canonical_json(summary) + "\n"


@pytest.mark.parametrize("marker", [
    "AddressSanitizer", "UBSan", "runtime error", "ThreadSanitizer",
    "WARNING:data race", "FATAL: sanitizer runtime failed",
])
def test_sanitizer_summary_fails_closed_on_forbidden_log_markers(tmp_path, marker):
    manifest = _manifest(tmp_path)
    (tmp_path / "tsan_close.log").write_text(marker + "\n", encoding="utf-8")
    output = tmp_path / "summary.json"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 2
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL_SANITIZER_AUDIT_SUMMARY"
    entry = next(item for item in summary["entries"] if item["name"] == "tsan_close")
    assert entry["checks"]["sanitizer_log_clean"] is False
    assert entry["forbidden_log_markers"]


@pytest.mark.parametrize("mutation", ["status", "exit", "signal", "outcome"])
def test_sanitizer_summary_requires_pass_exit_and_outcome_free_report(tmp_path, mutation):
    manifest = _manifest(tmp_path)
    if mutation == "status":
        _write_report(tmp_path / "tsan_close.json", "FAIL_CLOSE")
    elif mutation == "exit":
        _write_exit(tmp_path / "tsan_close.exit.json", exit_code=66)
    elif mutation == "signal":
        _write_exit(tmp_path / "tsan_close.exit.json", signal_number=6)
    else:
        path = tmp_path / "tsan_close.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["contains_scientific_outcome"] = True
        payload["report_semantic_hash"] = sha256_canonical({
            key: value for key, value in payload.items()
            if key != "report_semantic_hash"
        })
        path.write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize_manifest(manifest)
    assert summary["status"] == "FAIL_SANITIZER_AUDIT_SUMMARY"


def test_sanitizer_summary_rejects_tampered_report_and_duplicate_names(tmp_path):
    manifest = _manifest(tmp_path)
    report = tmp_path / "tsan_close.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["status"] = "PASS_DIFFERENT"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="report_semantic_hash mismatch"):
        summarize_manifest(manifest)

    duplicate_root = tmp_path / "duplicates"
    duplicate_root.mkdir()
    first = _entry(duplicate_root, "same", "TSAN", "PASS_ONE")
    second = dict(first)
    second["sanitizer"] = "ASAN_UBSAN"
    duplicate_manifest = duplicate_root / "manifest.json"
    duplicate_manifest.write_text(json.dumps({
        "schema_version": INPUT_SCHEMA, "entries": [first, second],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sanitizer entry name"):
        summarize_manifest(duplicate_manifest)
