"""Build an outcome-free manifest for the GREEN silent-failure challenge.

Only record identifiers and deterministic direction commitments are accepted.
No model, cache, activation, logit, certificate, or endpoint value is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from analysis.green_v400_silent_failure_protocol import load_and_validate_prepare_config


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def direction_commitment(protocol_id: str, row_id: str, domain: str, ordinal: int) -> str:
    payload = f"{protocol_id}\0{row_id}\0{domain}\0{ordinal}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_row_ids(row_ids: Iterable[str]) -> tuple[str, ...]:
    rows = tuple(str(row).strip() for row in row_ids if str(row).strip())
    if not rows:
        raise ValueError("row universe must be nonempty")
    if len(rows) != len(set(rows)):
        raise ValueError("row identifiers must be unique")
    return tuple(sorted(rows))


def build_prepare_manifest(
    config: dict[str, Any],
    row_ids: Iterable[str],
    *,
    green_direction_count: int = 8,
    endpoint_direction_count: int = 8,
) -> dict[str, Any]:
    rows = _normalize_row_ids(row_ids)
    if green_direction_count <= 0 or endpoint_direction_count <= 0:
        raise ValueError("direction counts must be positive")

    protocol_id = config["protocol_id"]
    panels = config["direction_panels"]
    green_domain = panels["green_panel_seed_domain"]
    endpoint_domain = panels["heldout_endpoint_panel_seed_domain"]
    if green_domain == endpoint_domain:
        raise ValueError("GREEN and endpoint direction domains must differ")

    commitments: dict[str, dict[str, list[str]]] = {}
    all_green: set[str] = set()
    all_endpoint: set[str] = set()
    for row_id in rows:
        green = [
            direction_commitment(protocol_id, row_id, green_domain, ordinal)
            for ordinal in range(green_direction_count)
        ]
        endpoint = [
            direction_commitment(protocol_id, row_id, endpoint_domain, ordinal)
            for ordinal in range(endpoint_direction_count)
        ]
        commitments[row_id] = {"green": green, "endpoint": endpoint}
        all_green.update(green)
        all_endpoint.update(endpoint)

    if all_green & all_endpoint:
        raise AssertionError("direction-domain separation failed")

    endpoint = config["primary_endpoint"]
    return {
        "schema_version": "green-v400-silent-failure-manifest-v1",
        "protocol_id": protocol_id,
        "status": "FORMAL_PREPARE_ONLY",
        "contains_scientific_outcome": False,
        "real_outcomes_authorized": False,
        "config_sha256": sha256_value(config),
        "row_universe_sha256": sha256_value(rows),
        "row_count": len(rows),
        "row_ids": list(rows),
        "direction_counts": {
            "green": green_direction_count,
            "endpoint": endpoint_direction_count,
        },
        "direction_commitments": commitments,
        "direction_panels_disjoint": True,
        "prediction_committed_before_endpoint": True,
        "endpoint_contract": {
            "name": endpoint["name"],
            "forbidden_inputs": list(endpoint["forbidden_inputs"]),
            "endpoint_panel_hidden_from_prediction_workers": True,
        },
    }


def _read_row_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = canonical_bytes(payload) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--row-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--green-directions", type=int, default=8)
    parser.add_argument("--endpoint-directions", type=int, default=8)
    args = parser.parse_args()

    config = load_and_validate_prepare_config(args.config)
    manifest = build_prepare_manifest(
        config,
        _read_row_ids(args.row_ids),
        green_direction_count=args.green_directions,
        endpoint_direction_count=args.endpoint_directions,
    )
    _atomic_write_json(args.output, manifest)


if __name__ == "__main__":
    main()

