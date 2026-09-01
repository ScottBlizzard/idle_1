from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import green_v410_schemas as schemas
from green_v410_protocol import PROTOCOL_ID, is_lower_sha256


H = "a" * 64


def site_identity() -> dict:
    return schemas.with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-site-identity-v1",
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": "PARENT",
        "phase": "confirmation",
        "task": "ioi",
        "parent_prompt_row_id": "prompt",
        "parent_site_row_id": "site",
        "layer": 4,
        "hook_family": "resid_post",
        "hook_name": "blocks.4.hook_resid_post",
        "token_position": 3,
        "clean_tokens_sha256": H,
        "corrupt_tokens_sha256": H,
        "clean_center_tensor_sha256": H,
        "pat_controlled_hook_tensor_sha256": H,
        "tar_controlled_hook_tensor_sha256": H,
        "green_direction_binding_sha256": H,
        "selected_gate_spec_sha256": H,
        "task_metric_spec_sha256": H,
        "model_manifest_sha256": H,
        "tokenizer_manifest_sha256": H,
        "graph_semantics_id": "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1",
    })


def test_schema_bundle_is_canonical_and_complete():
    expected = {
        "green-v410-sfc-jwtec-adoption-receipt-v1",
        "green-v410-sfc-jwtec-confirmation-seal-v1",
        "green-v410-sfc-jwtec-resource-calibration-v1",
        "green-v410-resource-minimum-budget-failfast-stop-v1",
        "green-v410-sfc-jwtec-resource-manifest-v1",
        "green-v410-sfc-jwtec-site-identity-v1",
        "green-v410-sfc-jwtec-graph-manifest-v1",
        "green-v410-sfc-jwtec-direction-certificate-v1",
        "green-v410-sfc-jwtec-site-certificate-v1",
        "green-v410-sfc-jwtec-p13-task-receipt-v1",
        "green-v410-sfc-jwtec-p13-two-task-receipt-v1",
        "green-v410-sfc-jwtec-clean-validity-v1",
        "green-v410-sfc-jwtec-status-manifest-v1",
        "green-v410-sfc-jwtec-endpoint-transition-v1",
        "green-v410-sfc-jwtec-final-analysis-receipt-v1",
    }
    assert set(schemas.SCHEMA_REGISTRY) == expected
    assert is_lower_sha256(schemas.SCHEMA_BUNDLE_SHA256)


def test_site_identity_self_hash_and_closed_fields():
    payload = site_identity()
    schemas.validate_artifact(payload)
    changed = deepcopy(payload)
    changed["unknown"] = 1
    with pytest.raises(ValueError, match="field mismatch"):
        schemas.validate_artifact(changed)


def test_site_identity_rejects_wrong_hook_even_when_rehashed():
    payload = site_identity()
    payload["hook_name"] = "blocks.5.hook_resid_post"
    with pytest.raises(ValueError, match="hook name"):
        schemas.with_artifact_self_hash({
            key: value for key, value in payload.items() if key != "site_identity_sha256"
        })


def test_self_hash_detects_mutation():
    payload = site_identity()
    payload["token_position"] += 1
    with pytest.raises(ValueError, match="self hash mismatch"):
        schemas.validate_artifact(payload)


def test_nonfinite_values_are_rejected_before_publication():
    payload = site_identity()
    payload["token_position"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        schemas.with_artifact_self_hash({
            key: value for key, value in payload.items() if key != "site_identity_sha256"
        })


def test_confirmation_seal_pass_requires_identical_clean_reports():
    payload = schemas.with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-confirmation-seal-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "scanner_reports": [H, H],
        "scanner_report_sha256s": [H, H],
        "summaries_identical": True,
        "forbidden_findings": [],
        "terminal_state": "PASS",
        "scanned_roots": ["repo", "server"],
        "seal_policy_sha256": H,
    })
    schemas.validate_artifact(payload)
    payload["forbidden_findings"] = [{"path": "bad"}]
    with pytest.raises(ValueError, match="terminal state"):
        schemas.with_artifact_self_hash({
            key: value for key, value in payload.items() if key != "receipt_sha256"
        })


def test_clean_validity_uses_strict_inequality():
    base = {
        "schema_version": "green-v410-sfc-jwtec-clean-validity-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": "greater_than",
        "parent_prompt_row_id": "prompt",
        "correct_probability_mass": 0.5,
        "incorrect_probability_mass": 0.5,
        "validity_margin": 0.0,
        "clean_task_valid": False,
        "model_manifest_sha256": H,
        "tokenizer_manifest_sha256": H,
        "task_metric_spec_sha256": H,
    }
    schemas.validate_artifact(schemas.with_artifact_self_hash(base))
    base["clean_task_valid"] = True
    with pytest.raises(ValueError, match="not strict"):
        schemas.with_artifact_self_hash(base)
