from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_classifier import ExactInterval, classify_site, outward_panel_ratio_squared
from green_v410_protocol import DECISION_RULE_SHA256, PROTOCOL_ID, sha256_canonical


H = "a" * 64


def site(task: str = "ioi") -> dict:
    return {
        "schema_version": "green-v410-sfc-jwtec-site-identity-v1",
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1",
        "phase": "confirmation",
        "task": task,
        "parent_prompt_row_id": "prompt-1",
        "parent_site_row_id": "site-1",
        "layer": 0,
        "hook_family": "resid_post",
        "hook_name": "blocks.0.hook_resid_post",
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
    }


def interval(lower: int, upper: int, denominator: int = 1) -> dict:
    return {
        "lower": [lower, denominator],
        "upper": [upper, denominator],
    }


def records(parent: dict, *, psi=(1, 1), pat=(10, 10), tar=(10, 10)) -> list[dict]:
    identity_hash = sha256_canonical(parent)
    output = []
    for ordinal in range(8):
        output.append({
            "schema_version": "green-v410-sfc-jwtec-direction-certificate-v1",
            "protocol_id": PROTOCOL_ID,
            "certificate_row_id": f"{ordinal + 1:064x}",
            "parent_site_row_id": parent["parent_site_row_id"],
            "site_identity_sha256": identity_hash,
            "direction_ordinal": ordinal,
            "direction_payload_sha256": H,
            "graph_manifest_sha256": H,
            "resource_manifest_sha256": H,
            "decision_rule_sha256": DECISION_RULE_SHA256,
            "status": "INTERVAL_COMPUTED",
            "reason_code": "INTERVAL_COMPUTED",
            "invalid_scope": "NONE",
            "radii": ["1", "1/2", "1/4"],
            "intervals": {
                "psi": interval(*psi),
                "pat_joint": interval(*pat),
                "tar_joint": interval(*tar),
            },
            "precision_nested": True,
        })
    return output


def p13_pass() -> dict:
    return {
        "schema_version": "green-v410-sfc-jwtec-p13-two-task-receipt-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "terminal_state": "PASS",
        "task_states": {"ioi": "PASS", "greater_than": "PASS"},
    }


def flags() -> dict:
    return {
        "frozen_source_schema_graph_hashes_match": True,
        "phase_manifest_conflict": False,
        "any_hard_site_graph_numeric_invalidity": False,
        "first_canonical_reason": None,
    }


def resources() -> dict:
    return {
        "any_required_child_resource_inconclusive": False,
        "first_canonical_reason": None,
    }


def classify(parent: dict, rows: object, **changes):
    kwargs = {
        "site_identity": parent,
        "direction_certificate_records": rows,
        "p13_specification": p13_pass(),
        "validity_flags": flags(),
        "clean_task_validity": None if parent["task"] == "ioi" else True,
        "resource_flags": resources(),
    }
    kwargs.update(changes)
    return classify_site(**kwargs)


def test_positive_below_threshold():
    parent = site()
    result = classify(parent, records(parent, psi=(1, 1)))
    assert result.status == "CERTIFIED_POSITIVE"
    assert result.ratio_squared_upper == Fraction(1, 100)


def test_exact_boundary_is_positive():
    parent = site()
    result = classify(parent, records(parent, psi=(2, 2)))
    assert result.status == "CERTIFIED_POSITIVE"
    assert result.ratio_squared_upper == Fraction(1, 25)


def test_straddling_boundary_is_unresolved():
    parent = site()
    result = classify(parent, records(parent, psi=(1, 3)))
    assert result.status == "UNRESOLVED"
    assert result.ratio_squared_lower < Fraction(1, 25) < result.ratio_squared_upper


def test_strictly_above_boundary_is_negative():
    parent = site()
    result = classify(parent, records(parent, psi=(3, 3)))
    assert result.status == "CERTIFIED_NEGATIVE"


def test_independent_direction_sign_reversal_is_invariant():
    parent = site()
    rows = records(parent, psi=(1, 1))
    for ordinal in (0, 2, 7):
        rows[ordinal]["intervals"]["psi"] = interval(-1, -1)
        rows[ordinal]["intervals"]["pat_joint"] = interval(-10, -10)
    assert classify(parent, rows).status == "CERTIFIED_POSITIVE"


def test_resource_state_does_not_become_invalid():
    parent = site()
    rows = records(parent)
    rows[4]["status"] = "RESOURCE_INCONCLUSIVE"
    rows[4]["reason_code"] = "WALL_LIMIT"
    rows[4]["intervals"] = {}
    rows[4]["precision_nested"] = False
    result = classify(parent, rows)
    assert result.status == "RESOURCE_INCONCLUSIVE"


def test_hard_invalidity_precedes_resource_state():
    parent = site()
    rows = records(parent)
    rows[4]["status"] = "RESOURCE_INCONCLUSIVE"
    rows[4]["intervals"] = {}
    bad_flags = flags()
    bad_flags["any_hard_site_graph_numeric_invalidity"] = True
    bad_flags["first_canonical_reason"] = "GRAPH_PARITY_FAILED"
    result = classify(parent, rows, validity_flags=bad_flags)
    assert result.status == "INVALID"
    assert result.reason_code == "GRAPH_PARITY_FAILED"


def test_incomplete_panel_is_method_invalid():
    parent = site()
    result = classify(parent, records(parent)[:-1])
    assert result == result.__class__("INVALID", "DIRECTION_PANEL_INCOMPLETE", "METHOD")


def test_gt_clean_false_is_task_precondition_invalid_before_records():
    parent = site("greater_than")
    result = classify(parent, [], clean_task_validity=False)
    assert result.status == "INVALID"
    assert result.invalid_scope == "TASK_PRECONDITION"


def test_missing_p13_is_run_contract_invalid():
    parent = site()
    result = classify(parent, records(parent), p13_specification={})
    assert result.status == "INVALID"
    assert result.reason_code == "P13_GLOBAL_NOT_PASSED"
    assert result.invalid_scope == "RUN_CONTRACT"


@pytest.mark.parametrize("bad", [None, {}, "rows", 3])
def test_malformed_record_container_is_total(bad):
    parent = site()
    result = classify(parent, bad)
    assert result.status == "INVALID"


def test_noncanonical_fraction_is_invalid_not_rounded():
    parent = site()
    rows = records(parent)
    rows[0]["intervals"]["psi"]["lower"] = [2, 2]
    result = classify(parent, rows)
    assert result.status == "INVALID"
    assert result.reason_code == "SCALAR_INTERVAL_MISSING_OR_MALFORMED"


def test_outward_ratio_uses_denominator_epsilon_exactly():
    zero = [ExactInterval(Fraction(0), Fraction(0)) for _ in range(8)]
    psi = [ExactInterval(Fraction(1, 10**13), Fraction(1, 10**13)) for _ in range(8)]
    result = outward_panel_ratio_squared(psi, zero, zero)
    assert result.lower == result.upper == Fraction(1, 100)

