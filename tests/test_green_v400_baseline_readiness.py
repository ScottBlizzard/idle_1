import copy
import json
from pathlib import Path

import pytest

from analysis.green_v400_baseline_readiness import (
    assert_baselines_ready,
    audit_baseline_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "green_v400_baseline_readiness.json"
GT_CONFIG = ROOT / "configs" / "green_v400_greater_than_baseline_readiness.json"


def payload():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_checked_in_registry_passes_after_grant_capture_binding_is_closed():
    audit = audit_baseline_readiness(payload(), ROOT)
    assert audit["verdict"] == "PASS_BASELINES_READY"
    assert audit["ready_for_untouched_execution"] is True
    assert audit["not_ready_required"] == []
    assert len(
        audit["evidence_file_sha256"]["grant_divergence"][
            "analysis/CODEX_GREEN_V400_GRANT_CAPTURE_SEMANTICS_AUDIT_20260829.md"
        ]
    ) == 64
    assert_baselines_ready(payload(), ROOT)


def test_checked_in_greater_than_registry_passes_same_frozen_grant_binding():
    registry = json.loads(GT_CONFIG.read_text(encoding="utf-8"))
    audit = audit_baseline_readiness(registry, ROOT)
    assert audit["verdict"] == "PASS_BASELINES_READY"
    assert audit["ready_for_untouched_execution"] is True
    assert audit["not_ready_required"] == []
    assert_baselines_ready(registry, ROOT)


def test_grant_binding_fails_closed_if_ready_evidence_is_removed():
    changed = payload()
    changed["baselines"]["grant_divergence"]["evidence"].remove(
        "configs/green_v400_grant_capture_spec.json"
    )
    changed["baselines"]["grant_divergence"]["evidence"].append(
        "configs/missing_grant_capture_spec.json"
    )
    audit = audit_baseline_readiness(changed, ROOT)
    assert audit["verdict"] == "BLOCK_BASELINES_NOT_READY"
    assert audit["missing_evidence"] == {
        "grant_divergence": ["configs/missing_grant_capture_spec.json"]
    }


def test_gate_passes_only_when_every_required_baseline_is_ready():
    ready = payload()
    for entry in ready["baselines"].values():
        if entry["required"]:
            entry["status"] = "READY"
    audit = audit_baseline_readiness(ready, ROOT)
    assert audit["verdict"] == "PASS_BASELINES_READY"
    assert audit["ready_for_untouched_execution"] is True


def test_missing_evidence_file_blocks_even_if_status_claims_ready():
    ready = payload()
    for entry in ready["baselines"].values():
        if entry["required"]:
            entry["status"] = "READY"
    ready["baselines"]["ordinary_restoration"]["evidence"].append("missing.py")
    audit = audit_baseline_readiness(ready, ROOT)
    assert audit["verdict"] == "BLOCK_BASELINES_NOT_READY"
    assert audit["missing_evidence"] == {"ordinary_restoration": ["missing.py"]}


def test_registry_cannot_authorize_real_outcomes():
    changed = payload()
    changed["real_outcomes_authorized"] = True
    audit = audit_baseline_readiness(changed, ROOT)
    assert "cannot authorize" in audit["errors"][0]


def test_documented_verifier_failure_cannot_be_relabelled_as_prediction():
    changed = payload()
    changed["baselines"]["generic_verifier"]["prediction_available"] = True
    audit = audit_baseline_readiness(changed, ROOT)
    assert "cannot expose a prediction" in " ".join(audit["errors"])


def test_atp_scope_cannot_claim_full_atp_star_execution():
    changed = payload()
    changed["baselines"]["AtP_star_or_closest_exact_attribution"][
        "AtP_star_claimed_as_executed"
    ] = True
    audit = audit_baseline_readiness(changed, ROOT)
    assert "cannot claim AtP*" in " ".join(audit["errors"])
