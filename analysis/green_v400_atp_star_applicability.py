"""Fail-closed scope statement for AtP* on coarse residual sites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VERDICT = "ATP_STAR_NOT_PRIMARY_FOR_COARSE_FULL_RESIDUAL_SITES"


def audit_atp_star_applicability(
    challenge: dict[str, Any], readiness: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    population = challenge.get("candidate_population", {})
    hook = population.get("hook")
    layers = population.get("layers")
    site_family = str(population.get("primary_site_family", ""))
    if hook != "resid_post" or "resid_post" not in site_family:
        errors.append("candidate sites are not full resid_post vectors")
    if not isinstance(layers, list) or not layers:
        errors.append("candidate layer list is missing")
        layers = []
    elif len(set(layers)) != len(layers) or not all(
        isinstance(layer, int) and layer >= 0 for layer in layers
    ):
        errors.append("candidate layers must be unique nonnegative integers")
    # The scope statement is valid only while the frozen sweep remains coarse.
    # It makes no claim that finite patching implements or beats AtP*.
    if len(layers) > 32:
        errors.append("coarse-site sweep exceeds the exact-patching applicability ceiling")

    baselines = readiness.get("baselines", {})
    for required in ("finite_activation_patching_response", "first_order_attribution"):
        if baselines.get(required, {}).get("status") != "READY":
            errors.append(f"{required} is not READY")
    scope = baselines.get("AtP_star_or_closest_exact_attribution", {})
    if scope.get("status") != "READY":
        errors.append("AtP* scope registry entry is not READY")
    if scope.get("comparison_method") != "finite_activation_patching_response":
        errors.append("coarse-site comparison method must be finite activation patching")
    if scope.get("supersedes_atp_star") is not False:
        errors.append("coarse-site comparison cannot claim to supersede AtP*")
    if scope.get("applicability") != "coarse_full_residual_sites_only":
        errors.append("AtP* scope applicability is not narrowly frozen")

    route = challenge.get("route_firewall", {}).get("prediction_routes", [])
    if "finite_activation_patching_response" not in route:
        errors.append("prediction route omits finite activation patching")
    if "first_order_attribution" not in route:
        errors.append("prediction route omits first-order attribution comparator")
    if "AtP_star_NA_for_coarse_full_residual_sites" not in route:
        errors.append("prediction route omits the narrow AtP* N/A scope statement")
    if challenge.get("real_outcomes_authorized") is not False:
        errors.append("applicability audit cannot authorize real outcomes")

    return {
        "schema_version": "green-v400-atp-star-applicability-audit-v1",
        "protocol_id": challenge.get("protocol_id"),
        "real_outcomes_authorized": False,
        "hook": hook,
        "coarse_site_count_per_prompt": len(layers),
        "comparison_method": "finite_activation_patching_response",
        "atp_star_claimed_as_executed": False,
        "supersedes_atp_star": False,
        "errors": errors,
        "verdict": VERDICT if not errors else "BLOCK_ATP_STAR_APPLICABILITY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    args = parser.parse_args()
    challenge = json.loads(args.challenge.read_text(encoding="utf-8"))
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    print(json.dumps(audit_atp_star_applicability(challenge, readiness), indent=2))


if __name__ == "__main__":
    main()
