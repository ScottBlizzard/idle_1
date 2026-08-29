from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis import green_v400_batch_ledger_ingest as ingest
from analysis.green_v400_phase_ledger import validate_phase_ledger


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def plan_payload():
    plan = {
        "protocol_id": "protocol",
        "execution_enabled": True,
        "development_authorized": True,
        "confirmation_authorized": False,
        "parent_plan_sha256": "aa" * 32,
        "development_authorization_sha256": "bb" * 32,
        "prediction_execution": {"shard_count": 4},
        "queues": {
            "development_prediction": [
                {"job_id": "p2"},
                {"job_id": "p1"},
            ],
            "confirmation_prediction": [],
            "development_grant_cohort_prediction": [{"job_id": "g1"}],
            "confirmation_grant_cohort_prediction": [],
            "endpoint_numerical_replay": [],
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def test_completed_batches_derive_deterministic_ledger(monkeypatch, tmp_path: Path):
    plan = plan_payload()
    predictions = {
        "p1": {"commitment": {"prediction_packet_sha256": "11" * 32}},
        "p2": {"commitment": {"prediction_packet_sha256": "22" * 32}},
    }
    grants = {"g1": {"grant_prediction": {"opaque": True}}}

    def fake_batches(*, mode, **kwargs):
        if mode == "prediction":
            return copy.deepcopy(predictions), {"p1": "31" * 32, "p2": "32" * 32}
        return copy.deepcopy(grants), {"g1": "33" * 32}

    monkeypatch.setattr(ingest, "_validated_mode_artifacts", fake_batches)
    monkeypatch.setattr(
        ingest,
        "build_grant_cohort_receipt",
        lambda **kwargs: {
            "schema_version": "green-v400-grant-cohort-receipt-v1",
            "job_id": "g1",
            "receipt_sha256": "44" * 32,
        },
    )
    ledger, receipts = ingest.derive_completed_phase_ledger(
        plan=plan,
        phase="development",
        prediction_root=tmp_path / "prediction",
        grant_root=tmp_path / "grant",
    )
    validate_phase_ledger(plan, ledger)
    assert ledger["completed_prediction_job_ids"] == ["p1", "p2"]
    assert ledger["completed_grant_job_ids"] == ["g1"]
    assert [event["job_id"] for event in ledger["events"]] == ["p1", "p2", "g1"]
    assert receipts["g1"]["receipt_sha256"] == "44" * 32


def test_confirmation_cannot_be_ingested_from_development_only_plan(tmp_path: Path):
    with pytest.raises(ValueError, match="confirmation is not authorized"):
        ingest.derive_completed_phase_ledger(
            plan=plan_payload(),
            phase="confirmation",
            prediction_root=tmp_path,
            grant_root=tmp_path,
        )


def test_requires_frozen_four_shard_topology(tmp_path: Path):
    plan = plan_payload()
    plan["prediction_execution"]["shard_count"] = 2
    plan.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(plan)
    with pytest.raises(ValueError, match="four-shard topology"):
        ingest.derive_completed_phase_ledger(
            plan=plan,
            phase="development",
            prediction_root=tmp_path,
            grant_root=tmp_path,
        )
