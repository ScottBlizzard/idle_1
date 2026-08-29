"""Static validator for the prepare-only GREEN silent-failure challenge.

The validator deliberately has no model or outcome-loading dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORBIDDEN_ENDPOINT_INPUTS = {
    "GREEN_certificate_status",
    "P13_status",
    "certificate_width",
    "certificate_budget_history",
    "GREEN_direction_panel",
    "baseline_prediction",
    "synthetic_attenuation_or_dither_amplitude",
}


def load_and_validate_prepare_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_prepare_config(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def validate_prepare_config(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if payload.get("status") != "FORMAL_PREPARE_ONLY":
        errors.append("status must remain FORMAL_PREPARE_ONLY")
    for field in (
        "real_outcomes_authorized",
        "development_authorized",
        "confirmation_authorized",
    ):
        if payload.get(field) is not False:
            errors.append(f"{field} must be false")

    decision = payload.get("shared_decision_rule", {})
    digest = decision.get("canonical_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("shared decision rule requires a 64-character canonical hash")
    if decision.get("same_rule_across_tasks") is not True:
        errors.append("shared decision rule must be identical across tasks")

    firewall = payload.get("route_firewall", {})
    prediction_routes = set(firewall.get("prediction_routes", []))
    endpoint_routes = set(firewall.get("endpoint_routes", []))
    if not prediction_routes or not endpoint_routes:
        errors.append("prediction and endpoint routes must both be nonempty")
    if prediction_routes & endpoint_routes:
        errors.append("prediction and endpoint routes must be disjoint")
    for field in (
        "prediction_committed_before_endpoint",
        "separate_route_processes",
        "cross_route_shared_model_instance_forbidden",
        "prediction_route_persistent_model_allowed",
        "grant_route_separate_from_direction_bearing_prediction_route",
        "endpoint_payload_in_prediction_process_forbidden",
        "adaptive_query_allocation_forbidden",
    ):
        if firewall.get(field) is not True:
            errors.append(f"route_firewall.{field} must be true")

    panels = payload.get("direction_panels", {})
    if panels.get("green_panel_seed_domain") == panels.get(
        "heldout_endpoint_panel_seed_domain"
    ):
        errors.append("GREEN and endpoint direction domains must differ")
    if panels.get("panels_must_be_disjoint") is not True:
        errors.append("direction panels must be disjoint")
    if panels.get("endpoint_panel_hidden_from_prediction_workers") is not True:
        errors.append("endpoint panel must remain hidden from prediction workers")
    if panels.get("direction_width") != 768:
        errors.append("GPT-2 Small direction width must equal 768")
    if panels.get("direction_norm") != 0.001:
        errors.append("direction norm must equal the frozen value 0.001")
    if panels.get("payload_generator_spec") != "numpy-PCG64DXSM-rowwise-normal-v1":
        errors.append("direction payload generator spec is not frozen")
    if panels.get("actual_tensor_payload_hash_required") is not True:
        errors.append("actual direction tensor payload hashes are required")

    precision = payload.get("response_evaluation_precision", {})
    if precision.get("checkpoint_storage_dtype") != "float32":
        errors.append("checkpoint storage dtype must remain float32")
    if precision.get("response_evaluation_dtype") != "float64":
        errors.append("finite response evaluation dtype must equal float64")
    if precision.get("model_manifest_tensor_hash_scheme") != (
        "sha256-contiguous-numpy-native-bytes-v1"
    ):
        errors.append("model manifest tensor hash scheme is not frozen")
    if precision.get(
        "checkpoint_values_must_roundtrip_to_float32_bit_exactly"
    ) is not True:
        errors.append("float64 evaluation must preserve float32 checkpoint values exactly")
    if precision.get("float32_response_outcomes_forbidden") is not True:
        errors.append("float32 finite response outcomes must be forbidden")
    audit_hash = precision.get("historical_audit_file_sha256")
    if not isinstance(audit_hash, str) or len(audit_hash) != 64:
        errors.append("response precision historical audit hash is missing")
    if precision.get("historical_audit_verdict") != (
        "PASS_REQUIRE_FLOAT64_SAME_CHECKPOINT_RESPONSE_EVALUATION"
    ):
        errors.append("response precision historical audit verdict is not binding")

    endpoint = payload.get("primary_endpoint", {})
    forbidden = set(endpoint.get("forbidden_inputs", []))
    missing_forbidden = FORBIDDEN_ENDPOINT_INPUTS - forbidden
    if missing_forbidden:
        errors.append(
            "primary endpoint does not forbid: " + ", ".join(sorted(missing_forbidden))
        )
    if endpoint.get("transport_failure_threshold") != 0.2:
        errors.append("transport failure effect-size threshold must equal 0.20")
    if endpoint.get("per_row_inferential_p_value_claimed") is not False:
        errors.append("the endpoint must not claim a per-row inferential p-value")
    if "symmetric normalized response RMSE > 0.20" not in endpoint.get(
        "failure_label", ""
    ):
        errors.append("secondary failure label must use the frozen 0.20 effect-size margin")

    calibration = payload.get("endpoint_numerical_replay_protocol", {})
    if calibration.get("mode") != "same_row_same_direction_target_target_replay":
        errors.append("endpoint numerical gate must use same-row target-target replay")
    if calibration.get("replay_workers_per_pair") != 2:
        errors.append("each endpoint numerical replay pair must use two workers")
    for field in (
        "worker_instances_must_be_distinct",
        "prediction_access_forbidden",
        "adaptive_query_allocation_forbidden",
    ):
        if calibration.get(field) is not True:
            errors.append(f"endpoint_numerical_replay_protocol.{field} must be true")
    for field in ("scientific_null_distribution_claimed", "defines_transport_failure_label"):
        if calibration.get(field) is not False:
            errors.append(f"endpoint_numerical_replay_protocol.{field} must be false")
    for field in ("absolute_tolerance", "relative_tolerance"):
        value = calibration.get(field)
        if not isinstance(value, (int, float)) or value < 0:
            errors.append(f"endpoint numerical replay {field} must be nonnegative")
    if calibration.get("absolute_tolerance", 0) == 0 and calibration.get(
        "relative_tolerance", 0
    ) == 0:
        errors.append("at least one endpoint numerical replay tolerance must be positive")

    gate = payload.get("transition_gate_correction", {})
    if gate.get("low_regime") != "simultaneous_95pct_UCB_le_0.20":
        errors.append("low-regime gate must use the simultaneous UCB")
    if gate.get("high_regime") != "simultaneous_95pct_LCB_ge_0.80":
        errors.append("high-regime gate must use the simultaneous LCB")

    if not payload.get("replication_requirement", {}).get("required_for_oral_claim"):
        errors.append("cross-setting replication must be required for the Oral claim")

    return errors
