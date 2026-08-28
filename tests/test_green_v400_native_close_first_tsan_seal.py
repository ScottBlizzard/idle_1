from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import canonical_json, sha256_canonical
from green_v400_native_close_first_tsan_seal import main, seal_native_audit


def _raw_payload() -> dict:
    return {
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
        "schema_version": "green-v400-native-close-first-tsan-v1",
        "scientific_threshold_applied": False,
        "status": "PASS_CLOSE_FIRST_PRELOCK_WAITER",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw.json"
    log = tmp_path / "tsan.log"
    exit_record = tmp_path / "exit.json"
    raw.write_bytes((canonical_json(_raw_payload()) + "\n").encode("utf-8"))
    log.write_bytes(b"")
    exit_record.write_bytes((canonical_json({
        "exit_code": 0,
        "schema_version": "green-v400-sanitizer-process-exit-v1",
        "termination_signal": None,
    }) + "\n").encode("utf-8"))
    return raw, log, exit_record


def test_sealer_emits_canonical_hash_closed_report_without_overwriting_raw(tmp_path):
    raw, log, exit_record = _fixture(tmp_path)
    original_raw = raw.read_bytes()
    output = tmp_path / "sealed.json"
    assert main([
        "--raw-audit", str(raw), "--sanitizer-log", str(log),
        "--process-exit-json", str(exit_record), "--output", str(output),
    ]) == 0
    sealed = json.loads(output.read_text(encoding="utf-8"))
    recorded_hash = sealed.pop("report_semantic_hash")
    assert recorded_hash == sha256_canonical(sealed)
    sealed["report_semantic_hash"] = recorded_hash
    assert output.read_text(encoding="utf-8") == canonical_json(sealed) + "\n"
    assert sealed["contains_scientific_outcome"] is False
    assert sealed["scientific_threshold_applied"] is False
    assert raw.read_bytes() == original_raw


@pytest.mark.parametrize("mutation", ["exit", "signal", "log", "status", "outcome"])
def test_sealer_fails_closed_on_nonpassing_evidence(tmp_path, mutation):
    raw, log, exit_record = _fixture(tmp_path)
    if mutation == "exit":
        payload = json.loads(exit_record.read_text())
        payload["exit_code"] = 124
        exit_record.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    elif mutation == "signal":
        payload = json.loads(exit_record.read_text())
        payload["termination_signal"] = 9
        exit_record.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    elif mutation == "log":
        log.write_text("WARNING: data race\n")
    else:
        payload = _raw_payload()
        if mutation == "status":
            payload["status"] = "FAIL_CLOSE_FIRST_CONTRACT"
            payload["pass"] = False
        else:
            payload["contains_scientific_outcome"] = True
        raw.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    with pytest.raises(ValueError):
        seal_native_audit(raw, log, exit_record)


def test_sealer_rejects_noncanonical_or_multiline_raw_json(tmp_path):
    raw, log, exit_record = _fixture(tmp_path)
    raw.write_text(json.dumps(_raw_payload(), indent=2) + "\n")
    with pytest.raises(ValueError, match="exactly one|canonical"):
        seal_native_audit(raw, log, exit_record)
