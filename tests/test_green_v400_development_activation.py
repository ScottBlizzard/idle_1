import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.green_v400_development_activation import (
    ACTIVATION_SOURCE_PATH,
    ALLOWED_CHANGES,
    canonical_sha256,
    derive_development_plan,
    file_sha256,
    validate_activated_plan,
)


def parent():
    payload = {
        "schema_version": "green-v400-sealed-execution-plan-v1",
        "protocol_id": "P",
        "execution_enabled": False,
        "real_outcomes_authorized": False,
        "plan_gate": "PLAN_COMPILED_AWAITING_SCIENTIFIC_AUTHORIZATION",
        "baseline_readiness": {
            "ready_for_untouched_execution": True,
            "not_ready_required": [],
        },
        "source_file_sha256": {
            ACTIVATION_SOURCE_PATH: file_sha256(ROOT / ACTIVATION_SOURCE_PATH)
        },
        "queues": {"development_prediction": [{"job_id": "1"}]},
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def authorization(payload):
    directives = ["持续推进", "服务器一直开着"]
    value = {
        "schema_version": "green-v400-development-authorization-v1",
        "authorization_id": "TEST",
        "contains_scientific_outcome": False,
        "authority": {
            "kind": "project_owner_continuous_execution_directive",
            "verbatim_directives": directives,
            "verbatim_directives_sha256": canonical_sha256(directives),
            "interpretation": "authorize frozen development only",
            "authentication_claim": "operator_attestation_not_external_signature",
        },
        "targets": [
            {
                "protocol_id": payload["protocol_id"],
                "parent_plan_sha256": payload["plan_sha256"],
                "parent_plan_file_sha256": "aa" * 32,
            }
        ],
        "scope": {
            "development_authorized": True,
            "confirmation_authorized": False,
            "automatic_confirmation_unlock": False,
            "post_outcome_rule_changes_allowed": False,
        },
        "allowed_parent_to_child_changes": list(ALLOWED_CHANGES),
    }
    value["authorization_sha256"] = canonical_sha256(value)
    return value


def test_exact_mechanical_activation_preserves_all_scientific_fields():
    base = parent()
    authority = authorization(base)
    child = derive_development_plan(parent_plan=base, authorization=authority)
    assert child["development_authorized"] is True
    assert child["confirmation_authorized"] is False
    assert child["queues"] == base["queues"]
    validate_activated_plan(
        parent_plan=base, authorization=authority, activated_plan=child
    )


def test_rehashed_child_with_queue_change_is_still_rejected():
    base = parent()
    authority = authorization(base)
    child = derive_development_plan(parent_plan=base, authorization=authority)
    child["queues"]["development_prediction"].append({"job_id": "2"})
    child["plan_sha256"] = canonical_sha256(
        {key: value for key, value in child.items() if key != "plan_sha256"}
    )
    with pytest.raises(ValueError, match="exact mechanical"):
        validate_activated_plan(
            parent_plan=base, authorization=authority, activated_plan=child
        )


def test_confirmation_or_attestation_change_fails_closed():
    base = parent()
    authority = authorization(base)
    changed = copy.deepcopy(authority)
    changed["scope"]["confirmation_authorized"] = True
    changed["authorization_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "authorization_sha256"}
    )
    with pytest.raises(ValueError, match="development-only"):
        derive_development_plan(parent_plan=base, authorization=changed)
