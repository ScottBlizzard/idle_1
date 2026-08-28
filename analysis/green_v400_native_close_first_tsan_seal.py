"""Seal one completed native-only TSan close-first audit.

The native harness remains the only process that loads or calls the audit DSO.
This post-process runs after that process has exited and only validates/hashes its
strict outcome-free JSON, sanitizer log, and process-exit record.
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


RAW_SCHEMA = "green-v400-native-close-first-tsan-v1"
EXIT_SCHEMA = "green-v400-sanitizer-process-exit-v1"
SEAL_SCHEMA = "green-v400-native-close-first-tsan-seal-v1"
EXPECTED_STATUS = "PASS_CLOSE_FIRST_PRELOCK_WAITER"
MAX_RAW_BYTES = 64 * 1024
MAX_LOG_BYTES = 256 * 1024 * 1024
RAW_FIELDS = {
    "audit_only_backend_build",
    "close_completed_before_hook_release",
    "contains_scientific_outcome",
    "context_close_status",
    "context_info_status",
    "context_open_status",
    "dispatch_done_before_release",
    "dispatch_status",
    "hook_enable_status",
    "hook_reached_status",
    "hook_release_status",
    "metric_active_dispatches",
    "metric_dispatch_entries",
    "metric_info_status",
    "metric_peak_dispatches",
    "metric_reset_status",
    "pass",
    "plan_close_status",
    "plan_info_status",
    "plan_open_status",
    "post_close_info_status",
    "schema_version",
    "scientific_threshold_applied",
    "status",
}
FORBIDDEN_LOG_PATTERNS = (
    re.compile(r"AddressSanitizer", re.IGNORECASE),
    re.compile(r"\bUBSan\b", re.IGNORECASE),
    re.compile(r"\bruntime\s+error\b", re.IGNORECASE),
    re.compile(r"ThreadSanitizer", re.IGNORECASE),
    re.compile(r"WARNING:\s*data\s+race", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])FATAL(?=\s|:|$)", re.IGNORECASE),
    re.compile(r"LeakSanitizer", re.IGNORECASE),
)


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    if path.stat().st_size > maximum:
        raise ValueError(f"{label} exceeds {maximum} bytes")
    return path.read_bytes()


def _strict_canonical_object(raw: bytes, label: str) -> dict:
    if not raw or raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
        raise ValueError(f"{label} must be exactly one LF-terminated JSON line")
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object")
    if text != canonical_json(payload) + "\n":
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def _strict_fields(payload: dict, expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{label} field mismatch; missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )


def _require_exact_native_pass(report: dict) -> None:
    _strict_fields(report, RAW_FIELDS, "raw native audit")
    expected = {
        "audit_only_backend_build": True,
        "close_completed_before_hook_release": True,
        "contains_scientific_outcome": False,
        "context_close_status": 0,
        "context_info_status": 0,
        "context_open_status": 0,
        "dispatch_done_before_release": False,
        "dispatch_status": 2,
        "hook_enable_status": 0,
        "hook_reached_status": 1,
        "hook_release_status": 0,
        "metric_active_dispatches": 0,
        "metric_dispatch_entries": 0,
        "metric_info_status": 0,
        "metric_peak_dispatches": 0,
        "metric_reset_status": 0,
        "pass": True,
        "plan_close_status": 0,
        "plan_info_status": 0,
        "plan_open_status": 0,
        "post_close_info_status": 2,
        "schema_version": RAW_SCHEMA,
        "scientific_threshold_applied": False,
        "status": EXPECTED_STATUS,
    }
    for key, value in expected.items():
        if type(report[key]) is not type(value) or report[key] != value:
            raise ValueError(f"raw native audit has invalid {key}")


def seal_native_audit(
    raw_audit_path: Path, sanitizer_log_path: Path, exit_json_path: Path,
) -> dict:
    raw_audit = _read_bounded(raw_audit_path, MAX_RAW_BYTES, "raw native audit")
    log_raw = _read_bounded(sanitizer_log_path, MAX_LOG_BYTES, "sanitizer log")
    exit_raw = _read_bounded(exit_json_path, MAX_RAW_BYTES, "process exit record")
    report = _strict_canonical_object(raw_audit, "raw native audit")
    exit_record = _strict_canonical_object(exit_raw, "process exit record")
    _require_exact_native_pass(report)
    _strict_fields(
        exit_record,
        {"exit_code", "schema_version", "termination_signal"},
        "process exit record",
    )
    if (
        exit_record["schema_version"] != EXIT_SCHEMA
        or type(exit_record["exit_code"]) is not int
        or exit_record["exit_code"] != 0
        or exit_record["termination_signal"] is not None
    ):
        raise ValueError("native TSan process did not exit cleanly")
    log_text = log_raw.decode("utf-8", errors="replace")
    if any(pattern.search(log_text) for pattern in FORBIDDEN_LOG_PATTERNS):
        raise ValueError("sanitizer log contains a forbidden diagnostic")

    sealed = dict(report)
    sealed.update({
        "evidence_seal_schema_version": SEAL_SCHEMA,
        "process_exit_json_sha256": hashlib.sha256(exit_raw).hexdigest(),
        "process_exit_zero": True,
        "raw_native_audit_sha256": hashlib.sha256(raw_audit).hexdigest(),
        "sanitizer_log_clean": True,
        "sanitizer_log_sha256": hashlib.sha256(log_raw).hexdigest(),
    })
    sealed["report_semantic_hash"] = sha256_canonical(sealed)
    return sealed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-audit", required=True)
    parser.add_argument("--sanitizer-log", required=True)
    parser.add_argument("--process-exit-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal_native_audit(
        Path(args.raw_audit), Path(args.sanitizer_log),
        Path(args.process_exit_json),
    )
    with output.open("xb") as stream:
        stream.write((canonical_json(sealed) + "\n").encode("utf-8"))
        stream.flush()
    print(canonical_json({
        "output": str(output),
        "report_semantic_hash": sealed["report_semantic_hash"],
        "status": sealed["status"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
