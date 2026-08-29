"""Generate and bind actual outcome-free GREEN/endpoint direction tensors."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis.green_v400_silent_failure_prepare import (
    _atomic_write_json,
    sha256_value,
)
from src.green_v400_direction_binding import binding_sha256, build_direction_binding


GENERATOR_SPEC = "numpy-PCG64DXSM-rowwise-normal-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_seed(path: Path) -> bytes:
    seed = path.read_bytes()
    if len(seed) != 32:
        raise ValueError(f"direction seed {path} must contain exactly 32 bytes")
    return seed


def _row_seed(master_seed: bytes, protocol_id: str, row_id: str, panel: str) -> int:
    message = f"{protocol_id}\0{row_id}\0{panel}\0{GENERATOR_SPEC}".encode("utf-8")
    return int.from_bytes(hmac.new(master_seed, message, hashlib.sha256).digest(), "little")


def _row_directions(
    *,
    master_seed: bytes,
    protocol_id: str,
    row_id: str,
    panel: str,
    count: int,
    width: int,
    direction_norm: float,
) -> np.ndarray:
    generator = np.random.Generator(
        np.random.PCG64DXSM(_row_seed(master_seed, protocol_id, row_id, panel))
    )
    values = generator.standard_normal((count, width), dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise RuntimeError("direction generator produced a degenerate row")
    values = (values / norms * direction_norm).astype("<f4")
    # Bind and use the exact serialized float32 payload, not the float64 precursor.
    return np.ascontiguousarray(values)


def build_direction_payload_registry(
    *,
    manifest: dict[str, Any],
    green_seed: bytes,
    endpoint_seed: bytes,
    output_directory: Path,
    direction_width: int,
    direction_norm: float,
) -> dict[str, Any]:
    if len(green_seed) != 32 or len(endpoint_seed) != 32:
        raise ValueError("direction seeds must each contain exactly 32 bytes")
    if green_seed == endpoint_seed:
        raise ValueError("GREEN and endpoint master seeds must differ")
    if manifest.get("contains_scientific_outcome") is not False:
        raise ValueError("direction payloads require an outcome-free manifest")
    prediction_sites = manifest.get("prediction_sites", [])
    prediction_row_ids = (
        [site["row_id"] for site in prediction_sites]
        if prediction_sites
        else list(manifest.get("row_ids", []))
    )
    replay_sites = manifest.get("endpoint_calibration", {}).get("sites", [])
    replay_row_ids = [site["row_id"] for site in replay_sites]
    if set(prediction_row_ids) & set(replay_row_ids):
        raise ValueError("prediction and numerical-replay direction rows must be disjoint")
    counts = manifest.get("direction_counts", {})
    if not prediction_row_ids or counts.get("green", 0) <= 0 or counts.get("endpoint", 0) <= 0:
        raise ValueError("manifest direction rows and counts must be nonempty")
    protocol_id = manifest["protocol_id"]
    output_directory.mkdir(parents=True, exist_ok=True)

    panels: dict[str, Any] = {}
    for panel, master_seed in (("green", green_seed), ("endpoint", endpoint_seed)):
        count = int(counts[panel])
        panel_row_ids = (
            prediction_row_ids
            if panel == "green"
            else prediction_row_ids + replay_row_ids
        )
        payload = np.lib.format.open_memmap(
            output_directory / f"{panel}_directions.npy",
            mode="w+",
            dtype="<f4",
            shape=(len(panel_row_ids), count, direction_width),
        )
        row_bindings = []
        for index, row_id in enumerate(panel_row_ids):
            row_tensor = _row_directions(
                master_seed=master_seed,
                protocol_id=protocol_id,
                row_id=row_id,
                panel=panel,
                count=count,
                width=direction_width,
                direction_norm=direction_norm,
            )
            payload[index] = row_tensor
            binding = build_direction_binding(
                protocol_id=protocol_id,
                row_id=row_id,
                panel_kind=panel,
                tensor=torch.from_numpy(row_tensor),
                direction_norm=direction_norm,
                generator_spec=GENERATOR_SPEC,
            )
            row_bindings.append(
                {
                    "row_index": index,
                    "row_id": row_id,
                    "binding_sha256": binding_sha256(binding),
                    "binding": binding,
                }
            )
        payload.flush()
        del payload
        payload_path = output_directory / f"{panel}_directions.npy"
        panels[panel] = {
            "seed_sha256": hashlib.sha256(master_seed).hexdigest(),
            "seed_value_serialized": False,
            "prediction_process_access": panel == "green",
            "payload_filename": payload_path.name,
            "payload_file_sha256": _file_sha256(payload_path),
            "shape": [len(panel_row_ids), count, direction_width],
            "dtype": "float32-little-endian",
            "row_ids_sha256": sha256_value(panel_row_ids),
            "row_bindings_sha256": sha256_value(row_bindings),
            "row_bindings": row_bindings,
        }

    registry = {
        "schema_version": "green-v400-direction-payload-registry-v1",
        "protocol_id": protocol_id,
        "contains_scientific_outcome": False,
        "manifest_sha256": sha256_value(manifest),
        "generator_spec": GENERATOR_SPEC,
        "numpy_version": np.__version__,
        "direction_width": direction_width,
        "direction_norm": direction_norm,
        "prediction_row_ids_sha256": sha256_value(prediction_row_ids),
        "endpoint_numerical_replay_row_ids_sha256": sha256_value(replay_row_ids),
        "panels": panels,
        "panels_have_distinct_seed_commitments": (
            panels["green"]["seed_sha256"] != panels["endpoint"]["seed_sha256"]
        ),
        "endpoint_payload_hidden_from_prediction_process": True,
    }
    registry["registry_sha256"] = sha256_value(registry)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--green-seed-file", type=Path, required=True)
    parser.add_argument("--endpoint-seed-file", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--direction-width", type=int, default=768)
    parser.add_argument("--direction-norm", type=float, default=0.001)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    registry = build_direction_payload_registry(
        manifest=manifest,
        green_seed=_read_seed(args.green_seed_file),
        endpoint_seed=_read_seed(args.endpoint_seed_file),
        output_directory=args.output_directory,
        direction_width=args.direction_width,
        direction_norm=args.direction_norm,
    )
    _atomic_write_json(args.registry, registry)
    print(json.dumps({
        "registry_sha256": registry["registry_sha256"],
        "green_file_sha256": registry["panels"]["green"]["payload_file_sha256"],
        "endpoint_file_sha256": registry["panels"]["endpoint"]["payload_file_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
