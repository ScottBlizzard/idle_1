"""Canonical outcome-free summary of explicit native sanitizer audit artifacts.

The input manifest is deliberately explicit and contains no globbing::

    {
      "schema_version": "green-v400-sanitizer-audit-input-v1",
      "entries": [{
        "name": "tsan_close_first",
        "sanitizer": "TSAN",
        "expected_status": "PASS_CLOSE_FIRST_PRELOCK_WAITER",
        "audit_json": "/absolute/or/manifest-relative/report.json",
        "sanitizer_log": "/absolute/or/manifest-relative/run.log",
        "process_exit_json": "/absolute/or/manifest-relative/exit.json"
      }]
    }

Each process-exit record must use schema
``green-v400-sanitizer-process-exit-v1`` and contain integer ``exit_code`` plus
integer-or-null ``termination_signal``.  This tool never copies arbitrary audit
payloads or log text into its output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_schemas import canonical_json, sha256_canonical


INPUT_SCHEMA = "green-v400-sanitizer-audit-input-v1"
EXIT_SCHEMA = "green-v400-sanitizer-process-exit-v1"
SUMMARY_SCHEMA = "green-v400-sanitizer-audit-summary-v1"
SANITIZER_FAMILIES = ("ASAN_UBSAN", "TSAN")
ENTRY_FIELDS = {
    "name", "sanitizer", "expected_status", "audit_json",
    "sanitizer_log", "process_exit_json",
}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_LOG_BYTES = 256 * 1024 * 1024
FORBIDDEN_LOG_PATTERNS = (
    ("AddressSanitizer", re.compile(r"AddressSanitizer", re.IGNORECASE)),
    ("UBSan", re.compile(r"\bUBSan\b", re.IGNORECASE)),
    ("runtime error", re.compile(r"\bruntime\s+error\b", re.IGNORECASE)),
    ("ThreadSanitizer", re.compile(r"ThreadSanitizer", re.IGNORECASE)),
    ("WARNING:data race", re.compile(r"WARNING:\s*data\s+race", re.IGNORECASE)),
    ("FATAL", re.compile(r"(?<![A-Za-z])FATAL(?=\s|:|$)", re.IGNORECASE)),
    ("LeakSanitizer", re.compile(r"LeakSanitizer", re.IGNORECASE)),
)


def _strict_fields(payload: dict, expected: set[str], label: str) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"{label} field mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _resolve_input(manifest_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must identify a regular file")
    return resolved


def _read_bytes(path: Path, maximum: int, label: str) -> bytes:
    size = path.stat().st_size
    if size > maximum:
        raise ValueError(f"{label} exceeds the {maximum}-byte audit limit")
    return path.read_bytes()


def _load_json(path: Path, label: str) -> tuple[dict, bytes]:
    raw = _read_bytes(path, MAX_JSON_BYTES, label)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verify_report_semantic_hash(report: dict, label: str) -> str:
    recorded = report.get("report_semantic_hash")
    if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
        raise ValueError(f"{label} lacks a canonical report_semantic_hash")
    semantic_payload = {
        key: value for key, value in report.items()
        if key != "report_semantic_hash"
    }
    if sha256_canonical(semantic_payload) != recorded:
        raise ValueError(f"{label} report_semantic_hash mismatch")
    return recorded


def _outcome_free_flags(report: dict) -> tuple[bool, bool]:
    outcome_keys = [
        key for key in (
            "contains_scientific_outcome", "report_contains_scientific_outcome",
        ) if key in report
    ]
    threshold_keys = [
        key for key in (
            "scientific_threshold_applied", "supervisor_applied_scientific_threshold",
        ) if key in report
    ]
    outcome_free = bool(outcome_keys) and all(report[key] is False for key in outcome_keys)
    threshold_free = bool(threshold_keys) and all(
        report[key] is False for key in threshold_keys
    )
    return outcome_free, threshold_free


def _scan_log(raw: bytes) -> list[str]:
    text = raw.decode("utf-8", errors="replace")
    return [name for name, pattern in FORBIDDEN_LOG_PATTERNS if pattern.search(text)]


def _validate_exit_record(record: dict, label: str) -> tuple[int, int | None]:
    _strict_fields(
        record, {"schema_version", "exit_code", "termination_signal"}, label,
    )
    if record["schema_version"] != EXIT_SCHEMA:
        raise ValueError(f"{label} schema_version mismatch")
    exit_code = record["exit_code"]
    signal_number = record["termination_signal"]
    if type(exit_code) is not int:
        raise ValueError(f"{label} exit_code must be an integer")
    if signal_number is not None and type(signal_number) is not int:
        raise ValueError(f"{label} termination_signal must be integer or null")
    return exit_code, signal_number


def summarize_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest, manifest_raw = _load_json(manifest_path, "sanitizer manifest")
    _strict_fields(manifest, {"schema_version", "entries"}, "sanitizer manifest")
    if manifest["schema_version"] != INPUT_SCHEMA:
        raise ValueError("sanitizer manifest schema_version mismatch")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("sanitizer manifest entries must be a nonempty list")

    names: set[str] = set()
    families: set[str] = set()
    summarized = []
    for ordinal, entry in enumerate(entries):
        label = f"sanitizer manifest entry {ordinal}"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be an object")
        _strict_fields(entry, ENTRY_FIELDS, label)
        name = entry["name"]
        sanitizer = entry["sanitizer"]
        expected_status = entry["expected_status"]
        if (not isinstance(name, str) or len(name) > 128
                or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)):
            raise ValueError(f"{label} has an invalid name")
        if name in names:
            raise ValueError(f"duplicate sanitizer entry name: {name}")
        names.add(name)
        if sanitizer not in SANITIZER_FAMILIES:
            raise ValueError(f"{label} has unsupported sanitizer family")
        families.add(sanitizer)
        if (not isinstance(expected_status, str)
                or not re.fullmatch(r"PASS_[A-Z0-9_]{1,123}", expected_status)):
            raise ValueError(f"{label} expected_status must begin with PASS_")

        audit_path = _resolve_input(manifest_path, entry["audit_json"], f"{label} audit_json")
        log_path = _resolve_input(
            manifest_path, entry["sanitizer_log"], f"{label} sanitizer_log",
        )
        exit_path = _resolve_input(
            manifest_path, entry["process_exit_json"], f"{label} process_exit_json",
        )
        report, report_raw = _load_json(audit_path, f"{label} audit_json")
        exit_record, exit_raw = _load_json(exit_path, f"{label} process_exit_json")
        log_raw = _read_bytes(log_path, MAX_LOG_BYTES, f"{label} sanitizer_log")
        report_hash = _verify_report_semantic_hash(report, f"{label} audit_json")
        exit_code, termination_signal = _validate_exit_record(
            exit_record, f"{label} process_exit_json",
        )
        outcome_free, threshold_free = _outcome_free_flags(report)
        forbidden_markers = _scan_log(log_raw)
        actual_status = report.get("status")
        if (not isinstance(actual_status, str)
                or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", actual_status)):
            raise ValueError(f"{label} audit status must be a bounded status string")
        checks = {
            "expected_pass_status": actual_status == expected_status,
            "process_exit_zero": exit_code == 0,
            "process_not_signal_terminated": termination_signal is None,
            "sanitizer_log_clean": not forbidden_markers,
            "audit_report_outcome_free": outcome_free,
            "audit_report_threshold_free": threshold_free,
        }
        summarized.append({
            "name": name,
            "sanitizer": sanitizer,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "exit_code": exit_code,
            "termination_signal": termination_signal,
            "checks": checks,
            "forbidden_log_markers": forbidden_markers,
            "audit_report_semantic_hash": report_hash,
            "audit_json_sha256": _sha256(report_raw),
            "sanitizer_log_sha256": _sha256(log_raw),
            "process_exit_json_sha256": _sha256(exit_raw),
        })

    summarized.sort(key=lambda item: item["name"])
    required_families_present = families == set(SANITIZER_FAMILIES)
    passed = required_families_present and all(
        all(item["checks"].values()) for item in summarized
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "report_contains_scientific_outcome": False,
        "supervisor_applied_scientific_threshold": False,
        "status": "PASS_SANITIZER_AUDIT_SUMMARY" if passed else "FAIL_SANITIZER_AUDIT_SUMMARY",
        "manifest_file_sha256": _sha256(manifest_raw),
        "manifest_semantic_hash": sha256_canonical(manifest),
        "required_sanitizer_families": list(SANITIZER_FAMILIES),
        "required_sanitizer_families_present": required_families_present,
        "entries": summarized,
        "instrumented_scope": {
            "translation_units": [
                "native/green_v400_mpfr_backend.cpp",
                "native/green_v400_native_plan_loader.cpp",
            ],
            "audit_test_hooks_enabled": True,
            "asan_ubsan": (
                "instrumented repository C++ memory-safety and undefined-behavior paths"
            ),
            "tsan": "instrumented repository C++ synchronization and executed race paths",
        },
        "limitations": [
            "The Python interpreter, libstdc++, MPFR, and GMP are not "
            "established as instrumented by this summary.",
            "Sanitizer-family labels are explicit manifest assertions; this "
            "summary does not independently inspect compiler commands or "
            "binary instrumentation.",
            "A clean sanitizer log covers only paths and thread interleavings "
            "exercised by the listed audits.",
            "Active close races for info/export and plan-close versus "
            "context-open require explicit listed audits before they are claimed.",
            "Audit-hook sanitizer binaries differ from the production backend binary and its hash.",
            "This engineering summary authorizes no scientific outcome or formal certificate claim.",
        ],
        "claim_scope": (
            "outcome-free engineering aggregation of explicit sanitizer audit reports, "
            "process exit records, and sanitizer logs"
        ),
    }
    summary["report_semantic_hash"] = sha256_canonical(summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_manifest(Path(args.manifest))
    with output.open("xb") as stream:
        stream.write((canonical_json(summary) + "\n").encode("utf-8"))
        stream.flush()
    print(canonical_json({
        "status": summary["status"],
        "entry_count": len(summary["entries"]),
        "report": str(output),
        "report_semantic_hash": summary["report_semantic_hash"],
    }))
    return 0 if summary["status"] == "PASS_SANITIZER_AUDIT_SUMMARY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
