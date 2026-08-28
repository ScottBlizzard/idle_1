"""Bind the untouched IOI universe to the silent-failure challenge manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.green_v400_silent_failure_prepare import (
    _atomic_write_json,
    build_prepare_manifest,
    sha256_value,
)
from analysis.green_v400_silent_failure_protocol import load_and_validate_prepare_config


PREDICTION_ROLES = {"development", "confirmation"}


def _expand_sites(
    prompt_row_ids: list[str], layers: list[int], hook: str
) -> list[dict[str, Any]]:
    sites = []
    for prompt_row_id in sorted(prompt_row_ids):
        for layer in layers:
            payload = {
                "prompt_row_id": prompt_row_id,
                "layer": layer,
                "hook": hook,
            }
            sites.append({"row_id": sha256_value(payload), **payload})
    return sites


def finalize_prepare_manifest(
    challenge_config: dict[str, Any], universe: dict[str, Any]
) -> dict[str, Any]:
    if universe.get("contains_scientific_outcome") is not False:
        raise ValueError("universe must contain no scientific outcome")
    if universe.get("model_weights_loaded") is not False:
        raise ValueError("universe must be tokenizer-only")
    if universe.get("execution_authorized") is not False:
        raise ValueError("universe cannot authorize execution")
    if universe.get("protocol_id") != challenge_config.get("protocol_id"):
        raise ValueError("protocol identifiers do not match")

    rows = universe.get("rows", [])
    if len(rows) != universe.get("row_count"):
        raise ValueError("row count does not match serialized universe")
    if sha256_value(rows) != universe.get("rows_sha256"):
        raise ValueError("universe row hash mismatch")

    by_role: dict[str, list[str]] = {}
    for row in rows:
        by_role.setdefault(row["role"], []).append(row["row_id"])

    prediction_prompts = sorted(
        row_id
        for role in PREDICTION_ROLES
        for row_id in by_role.get(role, [])
    )
    endpoint_calibration_prompts = sorted(by_role.get("endpoint_calibration", []))
    reserve_prompts = sorted(by_role.get("unused_reserve", []))
    if not prediction_prompts or not endpoint_calibration_prompts or not reserve_prompts:
        raise ValueError("prediction, endpoint-calibration, and reserve rows are required")

    prompt_role_sets = [
        set(prediction_prompts), set(endpoint_calibration_prompts), set(reserve_prompts)
    ]
    if any(
        prompt_role_sets[i] & prompt_role_sets[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise ValueError("role row identifiers must be disjoint")

    population = challenge_config.get("candidate_population", {})
    layers = population.get("layers")
    hook = population.get("hook")
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(layer, int) or layer < 0 for layer in layers)
        or len(layers) != len(set(layers))
    ):
        raise ValueError("candidate layers must be unique nonnegative integers")
    if not isinstance(hook, str) or not hook:
        raise ValueError("candidate hook must be nonempty")

    prediction_sites = _expand_sites(prediction_prompts, layers, hook)
    endpoint_calibration_sites = _expand_sites(
        endpoint_calibration_prompts, layers, hook
    )
    reserve_sites = _expand_sites(reserve_prompts, layers, hook)

    site_role_sets = [
        {row["row_id"] for row in prediction_sites},
        {row["row_id"] for row in endpoint_calibration_sites},
        {row["row_id"] for row in reserve_sites},
    ]
    if any(
        site_role_sets[i] & site_role_sets[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise ValueError("expanded site identifiers must be disjoint")

    manifest = build_prepare_manifest(
        challenge_config, [row["row_id"] for row in prediction_sites]
    )
    manifest.update(
        {
            "schema_version": "green-v400-sfc-finalized-prepare-manifest-v1",
            "universe_rows_sha256": universe["rows_sha256"],
            "universe_config_sha256": universe["config_sha256"],
            "universe_row_count": universe["row_count"],
            "site_definition": {"hook": hook, "layers": layers},
            "prediction_role_counts": {
                role: len(by_role.get(role, [])) for role in sorted(PREDICTION_ROLES)
            },
            "prediction_prompt_count": len(prediction_prompts),
            "prediction_site_count": len(prediction_sites),
            "prediction_sites_sha256": sha256_value(prediction_sites),
            "prediction_sites": prediction_sites,
            "endpoint_calibration": {
                "prompt_count": len(endpoint_calibration_prompts),
                "site_count": len(endpoint_calibration_sites),
                "sites_sha256": sha256_value(endpoint_calibration_sites),
                "sites": endpoint_calibration_sites,
                "prediction_access_forbidden": True,
            },
            "unused_reserve": {
                "prompt_count": len(reserve_prompts),
                "site_count": len(reserve_sites),
                "sites_sha256": sha256_value(reserve_sites),
                "sites": reserve_sites,
                "execution_forbidden": True,
            },
            "all_role_sets_disjoint": True,
        }
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-config", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_and_validate_prepare_config(args.challenge_config)
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    _atomic_write_json(args.output, finalize_prepare_manifest(config, universe))


if __name__ == "__main__":
    main()
