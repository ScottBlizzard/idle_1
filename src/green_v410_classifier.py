"""Total, deterministic GREEN v4.1 site-status classifier."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

from green_v410_protocol import (
    DECISION_RULE_SHA256,
    DIRECTION_ORDINALS,
    DIRECTION_TERMINAL_STATES,
    INVALID_SCOPES,
    PROTOCOL_ID,
    RADIUS_PANEL,
    RATIO_THRESHOLD_SQUARED_DENOMINATOR,
    RATIO_THRESHOLD_SQUARED_NUMERATOR,
    is_lower_sha256,
    sha256_canonical,
    strict_fields,
)


SITE_IDENTITY_SCHEMA = "green-v410-sfc-jwtec-site-identity-v1"
DIRECTION_CERTIFICATE_SCHEMA = "green-v410-sfc-jwtec-direction-certificate-v1"
P13_TWO_TASK_SCHEMA = "green-v410-sfc-jwtec-p13-two-task-receipt-v1"


@dataclass(frozen=True)
class ExactInterval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], name: str) -> "ExactInterval":
        strict_fields(payload, {"lower", "upper"}, name)
        return cls(
            _fraction_pair(payload["lower"], f"{name}.lower"),
            _fraction_pair(payload["upper"], f"{name}.upper"),
        )


@dataclass(frozen=True)
class Classification:
    status: str
    reason_code: str
    invalid_scope: str = "NONE"
    ratio_squared_lower: Fraction | None = None
    ratio_squared_upper: Fraction | None = None

    def __post_init__(self) -> None:
        if self.invalid_scope not in INVALID_SCOPES:
            raise ValueError("invalid classification scope")
        if self.status == "INVALID" and self.invalid_scope == "NONE":
            raise ValueError("invalid status requires a non-NONE scope")
        if self.status != "INVALID" and self.invalid_scope != "NONE":
            raise ValueError("only INVALID may carry a non-NONE scope")


def _fraction_pair(value: object, name: str) -> Fraction:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"{name} must be an exact integer rational pair")
    numerator, denominator = value
    if denominator <= 0:
        raise ValueError(f"{name} denominator must be positive")
    fraction = Fraction(numerator, denominator)
    if fraction.numerator != numerator or fraction.denominator != denominator:
        raise ValueError(f"{name} must be canonically reduced")
    return fraction


def _invalid(reason: str, scope: str) -> Classification:
    return Classification("INVALID", reason, scope)


def _resource(reason: str) -> Classification:
    return Classification("RESOURCE_INCONCLUSIVE", reason)


def _square_interval(interval: ExactInterval) -> ExactInterval:
    a, b = interval.lower, interval.upper
    if a <= 0 <= b:
        return ExactInterval(Fraction(0), max(a * a, b * b))
    return ExactInterval(min(a * a, b * b), max(a * a, b * b))


def _mean_square_bounds(intervals: Iterable[ExactInterval]) -> ExactInterval:
    squares = [_square_interval(interval) for interval in intervals]
    if len(squares) != 8:
        raise ValueError("exactly eight direction intervals are required")
    return ExactInterval(
        sum((interval.lower for interval in squares), Fraction(0)) / 8,
        sum((interval.upper for interval in squares), Fraction(0)) / 8,
    )


def outward_panel_ratio_squared(
    psi: Sequence[ExactInterval],
    pat_joint: Sequence[ExactInterval],
    tar_joint: Sequence[ExactInterval],
) -> ExactInterval:
    """Return exact bounds on R^2, avoiding an inexact square root.

    R and the 1/5 threshold are nonnegative, so comparing R to 1/5 is
    exactly equivalent to comparing R^2 to 1/25.  This is a mechanical
    implementation of the normative RMS rule, not a change to it.
    """

    j2 = _mean_square_bounds(psi)
    p2 = _mean_square_bounds(pat_joint)
    q2 = _mean_square_bounds(tar_joint)
    epsilon_squared = Fraction(1, 10**24)
    d2_lower = max(p2.lower, q2.lower, epsilon_squared)
    d2_upper = max(p2.upper, q2.upper, epsilon_squared)
    if d2_lower <= 0:
        raise ValueError("site denominator lower bound must be positive")
    return ExactInterval(j2.lower / d2_upper, j2.upper / d2_lower)


def _validate_site_identity(site: Mapping[str, Any]) -> str | None:
    required = {
        "schema_version", "protocol_id", "parent_protocol_id", "phase", "task",
        "parent_prompt_row_id", "parent_site_row_id", "layer", "hook_family",
        "hook_name", "token_position", "clean_tokens_sha256",
        "corrupt_tokens_sha256", "clean_center_tensor_sha256",
        "pat_controlled_hook_tensor_sha256", "tar_controlled_hook_tensor_sha256",
        "green_direction_binding_sha256", "selected_gate_spec_sha256",
        "task_metric_spec_sha256", "model_manifest_sha256",
        "tokenizer_manifest_sha256", "graph_semantics_id",
    }
    try:
        strict_fields(site, required, "site identity")
    except ValueError:
        return "RUN_IDENTITY_MISMATCH"
    if site["schema_version"] != SITE_IDENTITY_SCHEMA or site["protocol_id"] != PROTOCOL_ID:
        return "RUN_IDENTITY_MISMATCH"
    if site["phase"] not in {"qualification", "confirmation"}:
        return "RUN_IDENTITY_MISMATCH"
    if site["task"] not in {"ioi", "greater_than"}:
        return "UNKNOWN_TASK"
    if type(site["layer"]) is not int or site["layer"] not in DIRECTION_ORDINALS + (8,):
        return "RUN_IDENTITY_MISMATCH"
    expected_hook = f"blocks.{site['layer']}.hook_resid_post"
    if site["hook_family"] != "resid_post" or site["hook_name"] != expected_hook:
        return "RUN_IDENTITY_MISMATCH"
    if site["graph_semantics_id"] != "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1":
        return "RUN_IDENTITY_MISMATCH"
    hash_fields = {
        "clean_tokens_sha256", "corrupt_tokens_sha256", "clean_center_tensor_sha256",
        "pat_controlled_hook_tensor_sha256", "tar_controlled_hook_tensor_sha256",
        "green_direction_binding_sha256", "selected_gate_spec_sha256",
        "task_metric_spec_sha256", "model_manifest_sha256", "tokenizer_manifest_sha256",
    }
    if any(not is_lower_sha256(site[field]) for field in hash_fields):
        return "RUN_IDENTITY_MISMATCH"
    return None


def _p13_is_exact_pass(receipt: Mapping[str, Any]) -> bool:
    return (
        isinstance(receipt, Mapping)
        and receipt.get("schema_version") == P13_TWO_TASK_SCHEMA
        and receipt.get("protocol_id") == PROTOCOL_ID
        and receipt.get("attempt_index") == 1
        and receipt.get("terminal_state") == "PASS"
        and receipt.get("task_states") == {"ioi": "PASS", "greater_than": "PASS"}
    )


def _direction_payload(
    record: Mapping[str, Any], site: Mapping[str, Any]
) -> tuple[str | None, dict[str, ExactInterval] | None]:
    required = {
        "schema_version", "protocol_id", "certificate_row_id", "parent_site_row_id",
        "site_identity_sha256", "direction_ordinal", "direction_payload_sha256",
        "graph_manifest_sha256", "resource_manifest_sha256", "decision_rule_sha256",
        "status", "reason_code", "invalid_scope", "radii", "intervals",
        "precision_nested",
    }
    try:
        strict_fields(record, required, "direction certificate")
    except ValueError:
        return "DIRECTION_RECORD_MALFORMED", None
    if (
        record["schema_version"] != DIRECTION_CERTIFICATE_SCHEMA
        or record["protocol_id"] != PROTOCOL_ID
        or record["parent_site_row_id"] != site["parent_site_row_id"]
        or record["site_identity_sha256"] != sha256_canonical(site)
        or record["decision_rule_sha256"] != DECISION_RULE_SHA256
    ):
        return "DIRECTION_IDENTITY_MISMATCH", None
    hash_fields = {
        "certificate_row_id", "site_identity_sha256", "direction_payload_sha256",
        "graph_manifest_sha256", "resource_manifest_sha256", "decision_rule_sha256",
    }
    if any(not is_lower_sha256(record[field]) for field in hash_fields):
        return "DIRECTION_IDENTITY_MISMATCH", None
    if type(record["direction_ordinal"]) is not int:
        return "DIRECTION_RECORD_MALFORMED", None
    if record["status"] not in DIRECTION_TERMINAL_STATES:
        return "UNKNOWN_DIRECTION_TERMINAL_STATE", None
    if record["invalid_scope"] not in INVALID_SCOPES:
        return "DIRECTION_RECORD_MALFORMED", None
    if not isinstance(record["reason_code"], str) or not record["reason_code"]:
        return "DIRECTION_RECORD_MALFORMED", None
    if record["status"] != "INTERVAL_COMPUTED":
        return None, None
    if record["precision_nested"] is not True:
        return "DIRECTION_CERTIFICATE_INVALID", None
    if tuple(record["radii"]) != RADIUS_PANEL:
        return "RADIUS_PANEL_INCOMPLETE", None
    if not isinstance(record["intervals"], Mapping):
        return "SCALAR_INTERVAL_MISSING_OR_MALFORMED", None
    try:
        strict_fields(record["intervals"], {"psi", "pat_joint", "tar_joint"}, "intervals")
        intervals = {
            name: ExactInterval.from_payload(record["intervals"][name], f"intervals.{name}")
            for name in ("psi", "pat_joint", "tar_joint")
        }
    except (KeyError, TypeError, ValueError):
        return "SCALAR_INTERVAL_MISSING_OR_MALFORMED", None
    return None, intervals


def classify_site(
    *,
    site_identity: Mapping[str, Any],
    direction_certificate_records: object,
    p13_specification: Mapping[str, Any],
    validity_flags: Mapping[str, Any],
    clean_task_validity: object,
    resource_flags: Mapping[str, Any],
) -> Classification:
    """Implement the total classifier and binding precedence in Section 7."""

    if not isinstance(site_identity, Mapping):
        return _invalid("RUN_IDENTITY_MISMATCH", "RUN_CONTRACT")
    if not isinstance(validity_flags, Mapping):
        return _invalid("FROZEN_HASH_MISMATCH", "RUN_CONTRACT")
    if not isinstance(resource_flags, Mapping):
        return _invalid("RESOURCE_FLAGS_MALFORMED", "RUN_CONTRACT")

    identity_error = _validate_site_identity(site_identity)
    if identity_error is not None:
        return _invalid(identity_error, "RUN_CONTRACT")
    if not _p13_is_exact_pass(p13_specification):
        return _invalid("P13_GLOBAL_NOT_PASSED", "RUN_CONTRACT")
    if validity_flags.get("frozen_source_schema_graph_hashes_match") is not True:
        return _invalid("FROZEN_HASH_MISMATCH", "RUN_CONTRACT")
    if validity_flags.get("phase_manifest_conflict") is True:
        return _invalid("PHASE_MANIFEST_CONFLICT", "RUN_CONTRACT")

    task = site_identity["task"]
    if task == "greater_than":
        if clean_task_validity is None or clean_task_validity in ("MISSING", "MALFORMED"):
            return _invalid("GT_CLEAN_VALIDITY_MISSING", "RUN_CONTRACT")
        if clean_task_validity is False:
            return _invalid("GT_CLEAN_TASK_INVALID", "TASK_PRECONDITION")
        if clean_task_validity is not True:
            return _invalid("GT_CLEAN_VALIDITY_BAD_DOMAIN", "RUN_CONTRACT")
    elif clean_task_validity is not None and clean_task_validity != "NOT_APPLICABLE":
        return _invalid("IOI_CLEAN_VALIDITY_MUST_BE_NA", "RUN_CONTRACT")

    if not isinstance(direction_certificate_records, list):
        return _invalid("DIRECTION_RECORD_CONTAINER_MALFORMED", "METHOD")
    ordinals: list[int] = []
    canonical_records: dict[int, Mapping[str, Any]] = {}
    for record in direction_certificate_records:
        if not isinstance(record, Mapping) or type(record.get("direction_ordinal")) is not int:
            return _invalid("DIRECTION_RECORD_MALFORMED", "METHOD")
        ordinal = record["direction_ordinal"]
        if ordinal in canonical_records:
            if dict(canonical_records[ordinal]) != dict(record):
                return _invalid("DUPLICATE_OR_CONFLICTING_DIRECTION_RECORD", "METHOD")
            return _invalid("DUPLICATE_OR_CONFLICTING_DIRECTION_RECORD", "METHOD")
        canonical_records[ordinal] = record
        ordinals.append(ordinal)
    if set(ordinals) != set(DIRECTION_ORDINALS):
        return _invalid("DIRECTION_PANEL_INCOMPLETE", "METHOD")

    parsed: dict[int, dict[str, ExactInterval]] = {}
    for ordinal in DIRECTION_ORDINALS:
        record = canonical_records[ordinal]
        error, intervals = _direction_payload(record, site_identity)
        if error in {"DIRECTION_IDENTITY_MISMATCH", "DIRECTION_RECORD_MALFORMED"}:
            return _invalid(error, "METHOD")
        if record.get("status") == "INVALID":
            return _invalid("DIRECTION_CERTIFICATE_INVALID", "METHOD")
        if error is not None:
            return _invalid(error, "METHOD")
        if intervals is not None:
            parsed[ordinal] = intervals

    if validity_flags.get("any_hard_site_graph_numeric_invalidity") is True:
        reason = validity_flags.get("first_canonical_reason")
        if not isinstance(reason, str) or not reason:
            reason = "SITE_GRAPH_NUMERIC_INVALID"
        return _invalid(reason, "METHOD")

    if resource_flags.get("any_required_child_resource_inconclusive") is True:
        reason = resource_flags.get("first_canonical_reason")
        return _resource(reason if isinstance(reason, str) and reason else "CHILD_RESOURCE_INCONCLUSIVE")
    if any(record.get("status") == "RESOURCE_INCONCLUSIVE" for record in canonical_records.values()):
        return _resource("DIRECTION_RESOURCE_INCONCLUSIVE")

    if len(parsed) != 8:
        return _invalid("UNKNOWN_DIRECTION_TERMINAL_STATE", "METHOD")
    try:
        ratio_squared = outward_panel_ratio_squared(
            [parsed[index]["psi"] for index in DIRECTION_ORDINALS],
            [parsed[index]["pat_joint"] for index in DIRECTION_ORDINALS],
            [parsed[index]["tar_joint"] for index in DIRECTION_ORDINALS],
        )
    except (ArithmeticError, ValueError):
        return _invalid("SITE_RATIO_INTERVAL_INVALID", "METHOD")

    threshold_squared = Fraction(
        RATIO_THRESHOLD_SQUARED_NUMERATOR,
        RATIO_THRESHOLD_SQUARED_DENOMINATOR,
    )
    if ratio_squared.upper <= threshold_squared:
        return Classification(
            "CERTIFIED_POSITIVE", "JWTEC_MARGIN_NONNEGATIVE",
            ratio_squared_lower=ratio_squared.lower,
            ratio_squared_upper=ratio_squared.upper,
        )
    if ratio_squared.lower > threshold_squared:
        return Classification(
            "CERTIFIED_NEGATIVE", "JWTEC_MARGIN_STRICTLY_NEGATIVE",
            ratio_squared_lower=ratio_squared.lower,
            ratio_squared_upper=ratio_squared.upper,
        )
    return Classification(
        "UNRESOLVED", "JWTEC_INTERVAL_STRADDLES_BOUNDARY",
        ratio_squared_lower=ratio_squared.lower,
        ratio_squared_upper=ratio_squared.upper,
    )
