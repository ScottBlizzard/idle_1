"""Frozen protocol constants and canonical hashing for GREEN v4.1.

This module contains no experiment execution.  It is the closed, prepare-side
representation of GPT Pro decision GREEN-V410-SFC-JWTEC-20260830.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "green_v410_sfc_jwtec_protocol.json"
DECISION_PATH = (
    ROOT
    / "analysis"
    / "GPTPRO_GREEN_V4_SUCCESSOR_CERTIFICATE_PROTOCOL_DECISION_20260830.md"
)

SCHEMA_VERSION = "green-v410-sfc-jwtec-protocol-spec-v1"
PROTOCOL_ID = "GREEN-V410-SFC-JWTEC-20260830"
PARENT_COMMIT = "91741e32bfaad05667fb2d4c8d1de26373e90a15"
OUTCOME_BLIND_CUTOFF_COMMIT = "6aab5837bf3950b86b166ef938097e9739810d9b"
DECISION_DOCUMENT_SHA256 = (
    "5fb0d7dd936b4f44e6d96e41e1d4e3e9f63f75cf032013a5a687c42b21d43da0"
)

TASKS = ("ioi", "greater_than")
SITE_LAYERS = tuple(range(9))
SITE_HOOK_FAMILY = "resid_post"
BRANCH_ORDER = ("PAT_J", "PAT_B", "TAR_J", "TAR_B")
BRANCH_WEIGHTS = (1, -1, -1, 1)
DIRECTION_ORDINALS = tuple(range(8))
RADIUS_PANEL = ("1", "1/2", "1/4")
OFFICIAL_PRECISION_BITS = 384
AUDIT_PRECISION_BITS = 512
RATIO_THRESHOLD_NUMERATOR = 1
RATIO_THRESHOLD_DENOMINATOR = 5
RATIO_THRESHOLD_SQUARED_NUMERATOR = 1
RATIO_THRESHOLD_SQUARED_DENOMINATOR = 25
DENOMINATOR_EPSILON = "1e-12"

FINAL_STATUSES = (
    "CERTIFIED_POSITIVE",
    "CERTIFIED_NEGATIVE",
    "UNRESOLVED",
    "INVALID",
    "RESOURCE_INCONCLUSIVE",
)
INVALID_SCOPES = ("NONE", "TASK_PRECONDITION", "METHOD", "RUN_CONTRACT")
DIRECTION_TERMINAL_STATES = ("INTERVAL_COMPUTED", "INVALID", "RESOURCE_INCONCLUSIVE")

DECISION_RULE_SPEC = {
    "schema_version": "green-v410-sfc-jwtec-decision-rule-v1",
    "protocol_id": PROTOCOL_ID,
    "tasks": list(TASKS),
    "direction_ordinals": list(DIRECTION_ORDINALS),
    "radii": list(RADIUS_PANEL),
    "required_scalars": ["psi", "pat_joint", "tar_joint"],
    "direction_aggregation": "outward_interval_RMS_across_exactly_eight_directions",
    "denominator": "max(RMS(pat_joint),RMS(tar_joint),1e-12)",
    "comparison": {
        "positive": "ratio_interval.upper <= 1/5",
        "negative": "ratio_interval.lower > 1/5",
        "unresolved": "otherwise",
    },
    "precedence": [
        "RUN_CONTRACT_INVALID",
        "TASK_PRECONDITION_INVALID",
        "METHOD_INVALID",
        "RESOURCE_INCONCLUSIVE",
        "MATHEMATICAL_CLASSIFICATION",
    ],
    "certified_null": "FORBIDDEN",
    "greater_than_clean_validity": "strict_correct_mass_gt_incorrect_mass",
    "p13_role": "required_two_task_global_pass_receipt_only",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the protocol's canonical UTF-8 JSON encoding."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_canonical(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


DECISION_RULE_SHA256 = sha256_canonical(DECISION_RULE_SPEC)


def is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_fields(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise ValueError(
            f"{name} field mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )


def _assert_no_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_nonfinite(child, f"{path}[{index}]")


def load_protocol_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant {value}")
        ))
    _assert_no_nonfinite(payload)
    validate_protocol_config(payload)
    return payload


def validate_protocol_config(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "protocol_id", "parent_commit",
        "outcome_blind_cutoff_commit", "attempt_index",
        "scientific_retry_allowed", "tasks", "site_layers",
        "site_hook_family", "gate", "directions", "radii",
        "precision_bits", "branch_order", "branch_weights",
        "site_estimand", "status_rule", "p13", "primary_analysis",
        "resource_selector", "development_use", "confirmation",
        "required_hashes",
    }
    strict_fields(payload, expected, "protocol config")
    if payload["schema_version"] != SCHEMA_VERSION or payload["protocol_id"] != PROTOCOL_ID:
        raise ValueError("protocol config identity mismatch")
    if payload["parent_commit"] != PARENT_COMMIT:
        raise ValueError("protocol parent commit mismatch")
    if payload["outcome_blind_cutoff_commit"] != OUTCOME_BLIND_CUTOFF_COMMIT:
        raise ValueError("outcome-blind cutoff mismatch")
    if payload["attempt_index"] != 1 or payload["scientific_retry_allowed"] is not False:
        raise ValueError("protocol attempt contract mismatch")
    if tuple(payload["tasks"]) != TASKS or tuple(payload["site_layers"]) != SITE_LAYERS:
        raise ValueError("task or site layer panel mismatch")
    if payload["site_hook_family"] != SITE_HOOK_FAMILY:
        raise ValueError("site hook family mismatch")
    if tuple(payload["branch_order"]) != BRANCH_ORDER:
        raise ValueError("branch order mismatch")
    if tuple(payload["branch_weights"]) != BRANCH_WEIGHTS:
        raise ValueError("branch weights mismatch")
    if tuple(payload["radii"]) != RADIUS_PANEL:
        raise ValueError("radius panel mismatch")
    if payload["precision_bits"] != {"official": 384, "audit": 512}:
        raise ValueError("precision panel mismatch")
    if payload["directions"] != {
        "count": 8,
        "width": 768,
        "payload_dtype": "float32-little-endian",
        "nominal_norm": 0.001,
        "generator": "numpy-PCG64DXSM-rowwise-normal-v1",
    }:
        raise ValueError("direction contract mismatch")
    if payload["required_hashes"]["decision_document_sha256"] != DECISION_DOCUMENT_SHA256:
        raise ValueError("decision document hash mismatch")
    decision_hash = payload["required_hashes"]["decision_rule_sha256"]
    if decision_hash not in {"<computed>", DECISION_RULE_SHA256}:
        raise ValueError("decision rule hash mismatch")
    if payload["status_rule"].get("CERTIFIED_NULL") != "FORBIDDEN":
        raise ValueError("CERTIFIED_NULL must remain forbidden")


def protocol_config_self_hash(payload: Mapping[str, Any]) -> str:
    """Hash a protocol config without creating a self-referential fixed point.

    The normative document places ``protocol_config_sha256`` inside the
    protocol config.  Its canonical meaning is the hash after replacing only
    that field by the literal ``<computed>`` from the normative template.
    """

    validate_protocol_config(payload)
    normalized = copy.deepcopy(dict(payload))
    normalized["required_hashes"]["protocol_config_sha256"] = "<computed>"
    return sha256_canonical(normalized)


def decision_document_sha256(path: Path = DECISION_PATH) -> str:
    return sha256_bytes(path.read_bytes())

