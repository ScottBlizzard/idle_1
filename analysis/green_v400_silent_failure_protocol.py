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

    firewall = payload.get("route_firewall", {})
    prediction_routes = set(firewall.get("prediction_routes", []))
    endpoint_routes = set(firewall.get("endpoint_routes", []))
    if not prediction_routes or not endpoint_routes:
        errors.append("prediction and endpoint routes must both be nonempty")
    if prediction_routes & endpoint_routes:
        errors.append("prediction and endpoint routes must be disjoint")
    for field in (
        "prediction_committed_before_endpoint",
        "separate_worker_processes",
        "shared_model_instance_forbidden",
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

    endpoint = payload.get("primary_endpoint", {})
    forbidden = set(endpoint.get("forbidden_inputs", []))
    missing_forbidden = FORBIDDEN_ENDPOINT_INPUTS - forbidden
    if missing_forbidden:
        errors.append(
            "primary endpoint does not forbid: " + ", ".join(sorted(missing_forbidden))
        )
    if "split-conformal" not in endpoint.get("failure_label", ""):
        errors.append("primary failure label must use endpoint-only split-conformal calibration")

    gate = payload.get("transition_gate_correction", {})
    if gate.get("low_regime") != "simultaneous_95pct_UCB_le_0.20":
        errors.append("low-regime gate must use the simultaneous UCB")
    if gate.get("high_regime") != "simultaneous_95pct_LCB_ge_0.80":
        errors.append("high-regime gate must use the simultaneous LCB")

    if not payload.get("replication_requirement", {}).get("required_for_oral_claim"):
        errors.append("cross-setting replication must be required for the Oral claim")

    return errors

