"""Fail-closed applicability gate for AtP* versus exact activation patching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VERDICT = "PASS_EXACT_PATCHING_SUPERSEDES_ATP_STAR_FOR_COARSE_SITES"


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
    # Exact evaluation is only the cost-dominant substitute while the frozen
    # sweep remains small.  This ceiling prevents silent reuse for neuron/head
    # universes where AtP* was designed to provide the scalability benefit.
    if len(layers) > 32:
        errors.append("coarse-site sweep exceeds the exact-patching applicability ceiling")

    baselines = readiness.get("baselines", {})
    for required in ("exact_finite_response", "first_order_attribution"):
        if baselines.get(required, {}).get("status") != "READY":
            errors.append(f"{required} is not READY")
    replacement = baselines.get("AtP_star_or_closest_exact_attribution", {})
    if replacement.get("status") != "READY":
        errors.append("AtP*/exact replacement registry entry is not READY")
    if replacement.get("replacement_method") != "exact_finite_response":
        errors.append("AtP* replacement must be exact_finite_response")
    if replacement.get("applicability") != "coarse_full_residual_sites_only":
        errors.append("AtP* replacement applicability is not narrowly frozen")

    route = challenge.get("route_firewall", {}).get("prediction_routes", [])
    if "exact_activation_or_path_patching" not in route:
        errors.append("prediction route omits exact activation/path patching")
    if "attribution_patching_or_AtP_star" not in route:
        errors.append("prediction route omits attribution patching comparator")
    if challenge.get("real_outcomes_authorized") is not False:
        errors.append("applicability audit cannot authorize real outcomes")

    return {
        "schema_version": "green-v400-atp-star-applicability-audit-v1",
        "protocol_id": challenge.get("protocol_id"),
        "real_outcomes_authorized": False,
        "hook": hook,
        "coarse_site_count_per_prompt": len(layers),
        "replacement_method": "exact_finite_response",
        "atp_star_claimed_as_executed": False,
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
