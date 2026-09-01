"""Closed artifact schemas for GREEN-V410-SFC-JWTEC-20260830.

The schemas are intentionally represented as canonical data and hashed as one
bundle.  Runtime validators reject unknown fields, non-finite numbers, unknown
enum values, malformed hashes, and invalid self hashes.
"""

from __future__ import annotations

import copy
import math
from fractions import Fraction
from typing import Any, Mapping

from green_v410_protocol import (
    FINAL_STATUSES,
    INVALID_SCOPES,
    PROTOCOL_ID,
    is_lower_sha256,
    sha256_canonical,
    strict_fields,
)


def _schema(fields: tuple[str, ...], self_hash: str, **metadata: Any) -> dict[str, Any]:
    return {
        "closed": True,
        "fields": list(fields),
        "self_hash_field": self_hash,
        **metadata,
    }


SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {
    "green-v410-sfc-jwtec-adoption-receipt-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "artifact_role",
            "parent_path", "parent_sha256", "successor_role", "successor_path",
            "successor_sha256", "semantic_equivalence", "semantic_proof_sha256",
            "receipt_sha256",
        ),
        "receipt_sha256",
        enums={"semantic_equivalence": ["BYTE_IDENTICAL", "VALIDATED_ADAPTER"]},
    ),
    "green-v410-sfc-jwtec-confirmation-seal-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "scanner_reports",
            "scanner_report_sha256s", "summaries_identical", "forbidden_findings",
            "terminal_state", "scanned_roots", "seal_policy_sha256",
            "receipt_sha256",
        ),
        "receipt_sha256",
        enums={"terminal_state": ["PASS", "STOP_CONFIRMATION_SEAL_FAILED"]},
    ),
    "green-v410-sfc-jwtec-resource-calibration-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "candidate_leaf_budget",
            "profiles", "fixture_kinds", "precision_bits", "run_records",
            "all_theorem_checks_pass", "max_depth", "max_graph_nodes",
            "max_process_tree_rss_bytes", "projected_complete_direction_wall_seconds",
            "deterministic_replay", "candidate_pass", "first_failure_code",
            "receipt_sha256",
        ),
        "receipt_sha256",
    ),
    "green-v410-resource-minimum-budget-failfast-stop-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "decision",
            "candidate_leaf_budget", "failure_code",
            "trigger_run_artifact_sha256", "observed_max_depth",
            "max_depth_limit", "observed_graph_nodes", "max_graph_nodes_limit",
            "observed_process_tree_peak_rss_bytes", "guardband",
            "guarded_process_tree_peak_rss_numerator",
            "guarded_process_tree_peak_rss_denominator", "memory_max_bytes",
            "candidate_order_semantics", "larger_candidates_scheduled",
            "contains_scientific_outcome", "contains_endpoint_material",
            "receipt_sha256",
        ),
        "receipt_sha256",
        enums={
            "decision": ["STOP_RESOURCE_LOCK_INFEASIBLE"],
            "failure_code": [
                "MAX_DEPTH_EXCEEDED",
                "MAX_GRAPH_NODES_EXCEEDED",
                "MEMORY_GUARDBAND_EXCEEDED",
            ],
            "candidate_order_semantics": ["minimum_budget_prefix_admission"],
        },
    ),
    "green-v410-sfc-jwtec-resource-manifest-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "selected_leaf_budget",
            "selection_rule", "candidate_receipt_sha256s", "max_depth",
            "max_graph_nodes", "memory_max_bytes", "direction_wall_max_seconds",
            "guardband", "max_process_launches", "official_precision_bits",
            "audit_precision_bits", "radii", "manifest_sha256",
        ),
        "manifest_sha256",
    ),
    "green-v410-sfc-jwtec-site-identity-v1": _schema(
        (
            "schema_version", "protocol_id", "parent_protocol_id", "phase", "task",
            "parent_prompt_row_id", "parent_site_row_id", "layer", "hook_family",
            "hook_name", "token_position", "clean_tokens_sha256",
            "corrupt_tokens_sha256", "clean_center_tensor_sha256",
            "pat_controlled_hook_tensor_sha256", "tar_controlled_hook_tensor_sha256",
            "green_direction_binding_sha256", "selected_gate_spec_sha256",
            "task_metric_spec_sha256", "model_manifest_sha256",
            "tokenizer_manifest_sha256", "graph_semantics_id", "site_identity_sha256",
        ),
        "site_identity_sha256",
        enums={"phase": ["qualification", "confirmation"], "task": ["ioi", "greater_than"]},
    ),
    "green-v410-sfc-jwtec-graph-manifest-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "site_identity_sha256",
            "direction_ordinal", "direction_payload_sha256", "graph_semantics_id",
            "branch_order", "branch_weights", "output_names", "output_node_ids",
            "node_count", "dependency_closure_sha256", "provenance_map_sha256",
            "model_manifest_sha256", "task_metric_spec_sha256",
            "selected_gate_spec_sha256", "contains_endpoint_material",
            "graph_manifest_sha256",
        ),
        "graph_manifest_sha256",
    ),
    "green-v410-sfc-jwtec-direction-certificate-v1": _schema(
        (
            "schema_version", "protocol_id", "certificate_row_id",
            "parent_site_row_id", "site_identity_sha256", "direction_ordinal",
            "direction_payload_sha256", "graph_manifest_sha256",
            "resource_manifest_sha256", "decision_rule_sha256", "status",
            "reason_code", "invalid_scope", "radii", "intervals",
            "precision_nested", "partition_manifest_sha256", "artifact_sha256",
        ),
        "artifact_sha256",
        enums={
            "status": ["INTERVAL_COMPUTED", "INVALID", "RESOURCE_INCONCLUSIVE"],
            "invalid_scope": list(INVALID_SCOPES),
        },
    ),
    "green-v410-sfc-jwtec-site-certificate-v1": _schema(
        (
            "schema_version", "protocol_id", "site_certificate_id",
            "parent_site_row_id", "site_identity_sha256",
            "direction_certificate_row_ids", "p13_two_task_pass_receipt_sha256",
            "decision_rule_sha256", "classifier_source_sha256", "status",
            "reason_code", "invalid_scope", "ratio_squared_interval",
            "artifact_sha256",
        ),
        "artifact_sha256",
        enums={"status": list(FINAL_STATUSES), "invalid_scope": list(INVALID_SCOPES)},
    ),
    "green-v410-sfc-jwtec-p13-task-receipt-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "task",
            "queue_manifest_sha256", "resource_manifest_sha256",
            "expected_prompt_clusters", "expected_sites", "expected_directions",
            "observed_prompt_clusters", "observed_sites", "observed_directions",
            "site_ratio_manifest_sha256", "median_ratio", "p90_ratio",
            "median_threshold", "p90_threshold", "all_records_valid",
            "terminal_state", "first_failure_code", "receipt_sha256",
        ),
        "receipt_sha256",
        enums={"task": ["ioi", "greater_than"], "terminal_state": ["PASS", "P13_FAIL"]},
    ),
    "green-v410-sfc-jwtec-p13-two-task-receipt-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "task_states",
            "task_receipt_sha256s", "terminal_state", "first_failure_code",
            "receipt_sha256",
        ),
        "receipt_sha256",
        enums={"terminal_state": ["PASS", "STOP_P13_QUALIFICATION_FAILED"]},
    ),
    "green-v410-sfc-jwtec-clean-validity-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "task",
            "parent_prompt_row_id", "correct_probability_mass",
            "incorrect_probability_mass", "validity_margin", "clean_task_valid",
            "model_manifest_sha256", "tokenizer_manifest_sha256",
            "task_metric_spec_sha256", "artifact_sha256",
        ),
        "artifact_sha256",
        enums={"task": ["greater_than"]},
    ),
    "green-v410-sfc-jwtec-status-manifest-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "task_counts",
            "site_certificate_ids", "site_certificate_sha256s", "status_counts",
            "p13_two_task_pass_receipt_sha256", "confirmation_seal_sha256",
            "decision_rule_sha256", "source_bundle_sha256", "schema_bundle_sha256",
            "manifest_sha256",
        ),
        "manifest_sha256",
    ),
    "green-v410-sfc-jwtec-endpoint-transition-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "terminal_state",
            "status_manifest_sha256", "confirmation_seal_sha256",
            "p13_two_task_pass_receipt_sha256", "prediction_receipt_sha256s",
            "grant_receipt_sha256s", "clean_validity_manifest_sha256",
            "replay_receipt_sha256s", "source_bundle_sha256", "schema_bundle_sha256",
            "endpoint_payload_materialized_before_transition", "receipt_sha256",
        ),
        "receipt_sha256",
        enums={"terminal_state": ["GREEN_V410_CONFIRM_NONENDPOINT_FREEZE_A1"]},
    ),
    "green-v410-sfc-jwtec-final-analysis-receipt-v1": _schema(
        (
            "schema_version", "protocol_id", "attempt_index", "terminal_state",
            "task_results", "gate_results", "input_sha256s", "source_bundle_sha256",
            "schema_bundle_sha256", "first_failed_gate", "receipt_sha256",
        ),
        "receipt_sha256",
        enums={"terminal_state": ["COMPLETE", "STOP"]},
    ),
}

SCHEMA_BUNDLE_SHA256 = sha256_canonical(SCHEMA_REGISTRY)


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def artifact_self_hash(payload: Mapping[str, Any]) -> str:
    schema_name = payload.get("schema_version")
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError("unknown GREEN v4.1 artifact schema")
    field = SCHEMA_REGISTRY[schema_name]["self_hash_field"]
    unhashed = copy.deepcopy(dict(payload))
    unhashed.pop(field, None)
    return sha256_canonical(unhashed)


def with_artifact_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    schema_name = result.get("schema_version")
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError("unknown GREEN v4.1 artifact schema")
    field = SCHEMA_REGISTRY[schema_name]["self_hash_field"]
    result.pop(field, None)
    _assert_finite(result)
    result[field] = sha256_canonical(result)
    validate_artifact(result)
    return result


def validate_artifact(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("artifact must be an object")
    schema_name = payload.get("schema_version")
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError("unknown GREEN v4.1 artifact schema")
    specification = SCHEMA_REGISTRY[schema_name]
    strict_fields(payload, set(specification["fields"]), schema_name)
    _assert_finite(payload)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("artifact protocol ID mismatch")
    if "attempt_index" in payload and payload["attempt_index"] != 1:
        raise ValueError("artifact attempt index mismatch")
    for field, allowed in specification.get("enums", {}).items():
        if payload[field] not in allowed:
            raise ValueError(f"{schema_name} unknown enum value for {field}")
    for field, value in payload.items():
        if field.endswith("_sha256") and not is_lower_sha256(value):
            raise ValueError(f"{schema_name} malformed SHA-256 field {field}")
        if field.endswith("_sha256s"):
            values = value.values() if isinstance(value, Mapping) else value
            if not isinstance(values, (list, tuple)) and not hasattr(values, "__iter__"):
                raise ValueError(f"{schema_name} malformed SHA-256 collection {field}")
            if any(not is_lower_sha256(item) for item in values):
                raise ValueError(f"{schema_name} malformed SHA-256 collection {field}")
    self_hash_field = specification["self_hash_field"]
    if payload[self_hash_field] != artifact_self_hash(payload):
        raise ValueError(f"{schema_name} self hash mismatch")
    _validate_semantics(payload)


def _validate_semantics(payload: Mapping[str, Any]) -> None:
    schema = payload["schema_version"]
    if schema == "green-v410-sfc-jwtec-site-identity-v1":
        if type(payload["layer"]) is not int or payload["layer"] not in range(9):
            raise ValueError("site layer mismatch")
        if payload["hook_family"] != "resid_post":
            raise ValueError("site hook family mismatch")
        if payload["hook_name"] != f"blocks.{payload['layer']}.hook_resid_post":
            raise ValueError("site hook name mismatch")
        if payload["graph_semantics_id"] != "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1":
            raise ValueError("site graph semantics mismatch")
    elif schema == "green-v410-sfc-jwtec-graph-manifest-v1":
        if payload["branch_order"] != ["PAT_J", "PAT_B", "TAR_J", "TAR_B"]:
            raise ValueError("graph branch order mismatch")
        if payload["branch_weights"] != [1, -1, -1, 1]:
            raise ValueError("graph branch weights mismatch")
        if payload["output_names"] != ["PAT_J", "PAT_B", "TAR_J", "TAR_B", "PSI"]:
            raise ValueError("graph outputs mismatch")
        if payload["contains_endpoint_material"] is not False:
            raise ValueError("endpoint material is forbidden in certificate graph")
    elif schema == "green-v410-sfc-jwtec-direction-certificate-v1":
        if type(payload["direction_ordinal"]) is not int or payload["direction_ordinal"] not in range(8):
            raise ValueError("direction ordinal mismatch")
        if payload["status"] == "INTERVAL_COMPUTED":
            if payload["radii"] != ["1", "1/2", "1/4"] or payload["precision_nested"] is not True:
                raise ValueError("complete interval record contract mismatch")
    elif schema == "green-v410-sfc-jwtec-site-certificate-v1":
        if len(payload["direction_certificate_row_ids"]) != 8:
            raise ValueError("site certificate requires eight directions")
        if payload["status"] == "INVALID" and payload["invalid_scope"] == "NONE":
            raise ValueError("invalid site certificate requires scope")
        if payload["status"] != "INVALID" and payload["invalid_scope"] != "NONE":
            raise ValueError("non-invalid site certificate cannot carry invalid scope")
    elif schema == "green-v410-sfc-jwtec-p13-task-receipt-v1":
        expected = (12, 108, 864)
        observed = (
            payload["observed_prompt_clusters"], payload["observed_sites"],
            payload["observed_directions"],
        )
        if (
            payload["expected_prompt_clusters"], payload["expected_sites"],
            payload["expected_directions"],
        ) != expected:
            raise ValueError("P13 expected counts mismatch")
        if payload["terminal_state"] == "PASS" and (
            observed != expected or payload["all_records_valid"] is not True
        ):
            raise ValueError("P13 PASS has incomplete records")
    elif schema == "green-v410-sfc-jwtec-p13-two-task-receipt-v1":
        if set(payload["task_states"]) != {"ioi", "greater_than"}:
            raise ValueError("P13 two-task domain mismatch")
        if payload["terminal_state"] == "PASS" and payload["task_states"] != {
            "ioi": "PASS", "greater_than": "PASS"
        }:
            raise ValueError("P13 global PASS requires both tasks")
    elif schema == "green-v410-sfc-jwtec-clean-validity-v1":
        expected = payload["correct_probability_mass"] > payload["incorrect_probability_mass"]
        if payload["clean_task_valid"] is not expected:
            raise ValueError("Greater-Than clean validity is not strict")
    elif schema == "green-v410-sfc-jwtec-confirmation-seal-v1":
        passing = (
            payload["summaries_identical"] is True
            and payload["forbidden_findings"] == []
        )
        expected_state = "PASS" if passing else "STOP_CONFIRMATION_SEAL_FAILED"
        if payload["terminal_state"] != expected_state:
            raise ValueError("confirmation seal terminal state mismatch")
    elif schema == "green-v410-sfc-jwtec-resource-calibration-v1":
        if (
            payload["candidate_leaf_budget"] not in [4, 8, 16, 32]
            or payload["profiles"] != [
                "ioi:layer0", "ioi:layer4", "ioi:layer8",
                "greater_than:layer0", "greater_than:layer4", "greater_than:layer8",
            ]
            or payload["fixture_kinds"] != [
                "affine", "positive_curvature", "negative_curvature",
                "signed_cancellation", "deep_dyadic",
            ]
            or payload["precision_bits"] != [384, 512]
            or len(payload["run_records"]) != 60
            or type(payload["candidate_pass"]) is not bool
            or (payload["first_failure_code"] == "NONE") is not payload["candidate_pass"]
        ):
            raise ValueError("resource calibration receipt contract mismatch")
    elif schema == "green-v410-resource-minimum-budget-failfast-stop-v1":
        guarded_rss = Fraction(
            payload["guarded_process_tree_peak_rss_numerator"],
            payload["guarded_process_tree_peak_rss_denominator"],
        )
        expected_guarded_rss = (
            Fraction(payload["guardband"])
            * payload["observed_process_tree_peak_rss_bytes"]
        )
        failure_holds = {
            "MAX_DEPTH_EXCEEDED": (
                payload["observed_max_depth"] > payload["max_depth_limit"]
            ),
            "MAX_GRAPH_NODES_EXCEEDED": (
                payload["observed_graph_nodes"] > payload["max_graph_nodes_limit"]
            ),
            "MEMORY_GUARDBAND_EXCEEDED": (
                guarded_rss > payload["memory_max_bytes"]
            ),
        }
        if (
            payload["candidate_leaf_budget"] != 4
            or payload["max_depth_limit"] != 24
            or payload["max_graph_nodes_limit"] != 2_000_000
            or payload["guardband"] != "5/4"
            or payload["memory_max_bytes"] != 68_719_476_736
            or guarded_rss != expected_guarded_rss
            or not failure_holds[payload["failure_code"]]
            or payload["larger_candidates_scheduled"] is not False
            or payload["contains_scientific_outcome"] is not False
            or payload["contains_endpoint_material"] is not False
        ):
            raise ValueError("minimum-budget fail-fast STOP contract mismatch")
    elif schema == "green-v410-sfc-jwtec-resource-manifest-v1":
        if (
            payload["selected_leaf_budget"] not in [4, 8, 16, 32]
            or payload["selection_rule"] != "largest passing candidate"
            or len(payload["candidate_receipt_sha256s"]) != 4
            or payload["max_depth"] != 24
            or payload["max_graph_nodes"] != 2_000_000
            or payload["memory_max_bytes"] != 68_719_476_736
            or payload["direction_wall_max_seconds"] != 85_800
            or payload["guardband"] != "5/4"
            or payload["max_process_launches"] != 2
            or payload["official_precision_bits"] != 384
            or payload["audit_precision_bits"] != 512
            or payload["radii"] != ["1", "1/2", "1/4"]
        ):
            raise ValueError("resource manifest contract mismatch")
    elif schema == "green-v410-sfc-jwtec-endpoint-transition-v1":
        if payload["endpoint_payload_materialized_before_transition"] is not False:
            raise ValueError("endpoint payload was materialized before transition")
