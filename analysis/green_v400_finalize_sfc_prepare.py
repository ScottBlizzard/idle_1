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

    prediction_rows = sorted(
        row_id
        for role in PREDICTION_ROLES
        for row_id in by_role.get(role, [])
    )
    endpoint_calibration = sorted(by_role.get("endpoint_calibration", []))
    reserve = sorted(by_role.get("unused_reserve", []))
    if not prediction_rows or not endpoint_calibration or not reserve:
        raise ValueError("prediction, endpoint-calibration, and reserve rows are required")

    role_sets = [set(prediction_rows), set(endpoint_calibration), set(reserve)]
    if any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("role row identifiers must be disjoint")

    manifest = build_prepare_manifest(challenge_config, prediction_rows)
    manifest.update(
        {
            "schema_version": "green-v400-sfc-finalized-prepare-manifest-v1",
            "universe_rows_sha256": universe["rows_sha256"],
            "universe_config_sha256": universe["config_sha256"],
            "universe_row_count": universe["row_count"],
            "prediction_role_counts": {
                role: len(by_role.get(role, [])) for role in sorted(PREDICTION_ROLES)
            },
            "endpoint_calibration": {
                "row_count": len(endpoint_calibration),
                "row_ids_sha256": sha256_value(endpoint_calibration),
                "row_ids": endpoint_calibration,
                "prediction_access_forbidden": True,
            },
            "unused_reserve": {
                "row_count": len(reserve),
                "row_ids_sha256": sha256_value(reserve),
                "row_ids": reserve,
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

