from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis import green_v400_endpoint_authorization_prepare as prepare


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def plan_payload():
    return {
        "protocol_id": "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1",
        "plan_sha256": "aa" * 32,
        "development_authorized": True,
        "confirmation_authorized": False,
        "queues": {
            "development_endpoint": [
                {
                    "job_id": "e1",
                    "site_row_id": "s1",
                    "prompt_row_id": "u1",
                    "layer": 0,
                }
            ],
            "development_grant_cohort_prediction": [{"job_id": "g1"}],
        },
    }


def test_prepare_maps_exact_prediction_replay_grant_and_universe(
    monkeypatch, tmp_path: Path
):
    plan = plan_payload()
    ledger = {"planned_replay_layers": [0]}
    prediction = {
        "prediction": {"packet": True},
        "commitment": {"prediction_packet_sha256": "11" * 32},
    }
    monkeypatch.setattr(prepare, "verify_plan", lambda value: None)
    monkeypatch.setattr(prepare, "validate_phase_ledger", lambda p, l: None)
    monkeypatch.setattr(
        prepare,
        "_prediction_artifacts",
        lambda **kwargs: {"s1": prediction},
    )
    grant_directory = tmp_path / "grant"
    replay_directory = tmp_path / "replay"
    write_json(grant_directory / "g1.json", {"job_id": "g1"})
    write_json(replay_directory / "layer_00.json", {"layer": 0})
    observed = {}

    def fake_authorization(**kwargs):
        observed.update(kwargs)
        return {
            "protocol_id": plan["protocol_id"],
            "plan_sha256": plan["plan_sha256"],
            "phase_ledger_head_sha256": "22" * 32,
            "receipt_sha256": "33" * 32,
        }

    monkeypatch.setattr(
        prepare, "build_endpoint_authorization_receipt", fake_authorization
    )
    authorizations, commitments = prepare.prepare_endpoint_authorizations(
        plan=plan,
        phase="development",
        ledger=ledger,
        universe={"rows": [{"row_id": "u1"}]},
        prediction_root=tmp_path / "prediction",
        grant_receipt_directory=grant_directory,
        replay_receipt_directory=replay_directory,
    )
    assert set(authorizations) == {"e1"}
    assert commitments["e1"] == prediction["commitment"]
    assert observed["prediction_packet"] == prediction["prediction"]
    assert observed["replay_layer_receipt"] == {"layer": 0}
    assert observed["grant_cohort_receipts"] == [{"job_id": "g1"}]
    assert observed["response_adapter_source_path"] == (
        "src/green_v400_ioi_response_adapter.py"
    )


def test_confirmation_remains_locked(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(prepare, "verify_plan", lambda value: None)
    monkeypatch.setattr(prepare, "validate_phase_ledger", lambda p, l: None)
    with pytest.raises(ValueError, match="confirmation endpoint execution is not authorized"):
        prepare.prepare_endpoint_authorizations(
            plan=plan_payload(),
            phase="confirmation",
            ledger={},
            universe={},
            prediction_root=tmp_path,
            grant_receipt_directory=tmp_path,
            replay_receipt_directory=tmp_path,
        )


def test_duplicate_universe_rows_fail_before_authorization(monkeypatch, tmp_path: Path):
    plan = plan_payload()
    monkeypatch.setattr(prepare, "verify_plan", lambda value: None)
    monkeypatch.setattr(prepare, "validate_phase_ledger", lambda p, l: None)
    monkeypatch.setattr(
        prepare,
        "_prediction_artifacts",
        lambda **kwargs: {"s1": {"prediction": {}, "commitment": {}}},
    )
    write_json(tmp_path / "grant" / "g1.json", {"job_id": "g1"})
    write_json(tmp_path / "replay" / "layer_00.json", {"layer": 0})
    with pytest.raises(ValueError, match="duplicate row identifiers"):
        prepare.prepare_endpoint_authorizations(
            plan=plan,
            phase="development",
            ledger={"planned_replay_layers": [0]},
            universe={"rows": [{"row_id": "u1"}, {"row_id": "u1"}]},
            prediction_root=tmp_path,
            grant_receipt_directory=tmp_path / "grant",
            replay_receipt_directory=tmp_path / "replay",
        )
