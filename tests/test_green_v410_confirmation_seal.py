from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_confirmation_seal import build_seal_receipt, dual_scan, load_policy
from green_v410_protocol import PROTOCOL_ID
from green_v410_schemas import validate_artifact


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_dual_scanners_agree_on_clean_prepare_tree(tmp_path):
    write_json(tmp_path / "configs" / "prepare.json", {
        "protocol_id": "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1",
        "confirmation_authorized": False,
        "phase": "prepare",
    })
    write_json(tmp_path / "queues" / "confirmation_prepare_manifest.json", {
        "protocol_id": "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1",
        "phase": "confirmation",
        "jobs": [{"job_id": "prepared-but-unopened"}],
    })
    policy = load_policy()
    first, second = dual_scan({"repo": tmp_path}, policy)
    assert first["summary"] == second["summary"]
    assert first["summary"]["findings"] == []
    receipt, _ = build_seal_receipt({"repo": tmp_path}, policy)
    assert receipt["terminal_state"] == "PASS"
    validate_artifact(receipt)


def test_model_derived_confirmation_json_fails_both_scanners(tmp_path):
    write_json(tmp_path / "outputs" / "confirmation_prediction_result.json", {
        "protocol_id": PROTOCOL_ID,
        "phase": "confirmation",
        "ordinary_restoration": 0.9,
    })
    receipt, reports = build_seal_receipt({"server": tmp_path}, load_policy())
    assert receipt["terminal_state"] == "STOP_CONFIRMATION_SEAL_FAILED"
    assert receipt["summaries_identical"] is True
    assert reports[0]["summary"] == reports[1]["summary"]
    assert reports[0]["summary"]["findings"][0]["reason"] == "CONFIRMATION_MODEL_DERIVED_JSON"


def test_true_confirmation_execution_flag_fails(tmp_path):
    write_json(tmp_path / "state.json", {
        "protocol_id": PROTOCOL_ID,
        "confirmation_started": True,
    })
    receipt, _ = build_seal_receipt({"server": tmp_path}, load_policy())
    assert receipt["terminal_state"] == "STOP_CONFIRMATION_SEAL_FAILED"
    assert receipt["forbidden_findings"][0]["reason"] == "CONFIRMATION_EXECUTION_FLAG_TRUE"


def test_suspicious_binary_confirmation_output_fails(tmp_path):
    path = tmp_path / "cache" / "confirmation_endpoint_payload.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not actually numpy; path alone is forbidden")
    receipt, _ = build_seal_receipt({"cache": tmp_path}, load_policy())
    assert receipt["terminal_state"] == "STOP_CONFIRMATION_SEAL_FAILED"
    assert receipt["forbidden_findings"][0]["reason"] == "CONFIRMATION_BINARY_OUTPUT_PATH"


def test_historical_unrelated_confirmation_result_is_out_of_scope(tmp_path):
    write_json(tmp_path / "historical" / "confirmation_result.json", {
        "protocol_id": "UNRELATED_OLD_PROTOCOL",
        "phase": "confirmation",
        "ordinary_restoration": 0.9,
    })
    receipt, _ = build_seal_receipt({"repo": tmp_path}, load_policy())
    # The suspicious output path itself keeps this conservative even when the
    # embedded protocol is old.  Such paths must be explicitly hash-allowlisted.
    assert receipt["terminal_state"] == "STOP_CONFIRMATION_SEAL_FAILED"

