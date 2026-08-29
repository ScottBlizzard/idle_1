from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis import green_v400_replay_receipt_assembler as assembler
from analysis.green_v400_phase_ledger import (
    initialize_phase_ledger,
    validate_phase_ledger,
)


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
        "queues": {
            "development_prediction": [],
            "confirmation_prediction": [],
            "development_grant_cohort_prediction": [],
            "confirmation_grant_cohort_prediction": [],
            "endpoint_numerical_replay": [
                {"job_id": "r2", "layer": 1},
                {"job_id": "r1", "layer": 0},
                {"job_id": "r3", "layer": 1},
            ],
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def test_assembly_appends_one_receipt_per_layer(monkeypatch, tmp_path: Path):
    plan = plan_payload()
    ledger = initialize_phase_ledger(plan)
    monkeypatch.setattr(
        assembler,
        "_load_replay_pair",
        lambda **kwargs: {"job_id": kwargs["job_id"], "validated": True},
    )

    def fake_layer_receipt(*, layer, replay_artifacts, **kwargs):
        return {
            "schema_version": "green-v400-numerical-replay-layer-receipt-v1",
            "layer": layer,
            "jobs": sorted(row["job_id"] for row in replay_artifacts),
            "receipt_sha256": f"{layer + 1:064x}",
        }

    monkeypatch.setattr(
        assembler, "build_numerical_replay_layer_receipt", fake_layer_receipt
    )
    updated, receipts, combined = assembler.assemble_replay_receipts(
        plan=plan, ledger=ledger, replay_root=tmp_path
    )
    validate_phase_ledger(plan, updated)
    assert list(receipts) == [0, 1]
    assert receipts[1]["jobs"] == ["r2", "r3"]
    assert set(combined) == {"r1", "r2", "r3"}
    assert updated["numerical_replay_receipt_by_layer"] == {
        "0": f"{1:064x}",
        "1": f"{2:064x}",
    }


def test_replay_requires_development_authorization(tmp_path: Path):
    plan = plan_payload()
    plan["development_authorized"] = False
    plan.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(plan)
    ledger = initialize_phase_ledger(plan)
    with pytest.raises(ValueError, match="requires development authorization"):
        assembler.assemble_replay_receipts(
            plan=plan, ledger=ledger, replay_root=tmp_path
        )


def test_replay_cannot_be_appended_twice(monkeypatch, tmp_path: Path):
    plan = plan_payload()
    ledger = initialize_phase_ledger(plan)
    ledger["numerical_replay_receipt_sha256"] = ["11" * 32]
    with pytest.raises(ValueError, match="derived state or hash chain mismatch"):
        assembler.assemble_replay_receipts(
            plan=plan, ledger=ledger, replay_root=tmp_path
        )
