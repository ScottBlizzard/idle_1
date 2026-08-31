from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import green_v410_protocol as protocol


def test_decision_document_is_frozen_by_exact_bytes():
    assert protocol.decision_document_sha256() == protocol.DECISION_DOCUMENT_SHA256


def test_protocol_config_is_closed_and_self_hashed():
    payload = protocol.load_protocol_config()
    assert payload["required_hashes"]["decision_rule_sha256"] == protocol.DECISION_RULE_SHA256
    assert (
        payload["required_hashes"]["protocol_config_sha256"]
        == protocol.protocol_config_self_hash(payload)
    )


def test_protocol_config_self_hash_is_order_invariant():
    payload = protocol.load_protocol_config()
    reordered = dict(reversed(list(payload.items())))
    assert protocol.protocol_config_self_hash(payload) == protocol.protocol_config_self_hash(reordered)


def test_protocol_config_rejects_unknown_fields():
    payload = protocol.load_protocol_config()
    payload["posthoc_override"] = True
    with pytest.raises(ValueError, match="field mismatch"):
        protocol.validate_protocol_config(payload)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("radii",), ["1", "1/2"], "radius panel"),
        (("branch_weights",), [1, -1, 1, -1], "branch weights"),
        (("precision_bits",), {"official": 512, "audit": 384}, "precision panel"),
        (("scientific_retry_allowed",), True, "attempt contract"),
    ],
)
def test_protocol_config_rejects_scientific_drift(path, value, message):
    payload = protocol.load_protocol_config()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        protocol.validate_protocol_config(payload)


def test_canonical_json_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        protocol.canonical_json_bytes({"bad": float("nan")})


def test_decision_rule_forbids_certified_null():
    assert "CERTIFIED_NULL" not in protocol.FINAL_STATUSES
    assert protocol.DECISION_RULE_SPEC["certified_null"] == "FORBIDDEN"


def test_checked_in_config_is_strict_json():
    text = protocol.CONFIG_PATH.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert protocol.canonical_json_bytes(payload)

