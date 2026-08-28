import copy
import json
from pathlib import Path

from analysis.green_v400_silent_failure_protocol import (
    load_and_validate_prepare_config,
    validate_prepare_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "green_v400_silent_failure_challenge_prepare.json"


def valid_payload():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_checked_in_prepare_config_is_valid_and_cannot_authorize_outcomes():
    payload = load_and_validate_prepare_config(CONFIG)
    assert payload["status"] == "FORMAL_PREPARE_ONLY"
    assert payload["real_outcomes_authorized"] is False


def test_prediction_and_endpoint_routes_cannot_overlap():
    payload = valid_payload()
    payload["route_firewall"]["endpoint_routes"].append("GREEN_certificate")
    assert any("must be disjoint" in e for e in validate_prepare_config(payload))


def test_endpoint_must_forbid_certificate_status():
    payload = valid_payload()
    payload["primary_endpoint"]["forbidden_inputs"].remove("GREEN_certificate_status")
    assert any("GREEN_certificate_status" in e for e in validate_prepare_config(payload))


def test_direction_seed_domains_must_differ():
    payload = valid_payload()
    payload["direction_panels"]["heldout_endpoint_panel_seed_domain"] = payload[
        "direction_panels"
    ]["green_panel_seed_domain"]
    assert any("direction domains must differ" in e for e in validate_prepare_config(payload))


def test_old_low_lcb_gate_is_rejected():
    payload = valid_payload()
    payload["transition_gate_correction"]["low_regime"] = (
        "simultaneous_95pct_LCB_le_0.20"
    )
    assert any("must use the simultaneous UCB" in e for e in validate_prepare_config(payload))


def test_real_outcome_authorization_is_rejected():
    payload = valid_payload()
    payload["real_outcomes_authorized"] = True
    assert any("real_outcomes_authorized must be false" in e for e in validate_prepare_config(payload))
