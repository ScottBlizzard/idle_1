import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.green_v400_direction_payload_prepare import (
    build_direction_payload_registry,
)
from analysis.green_v400_silent_failure_prepare import build_prepare_manifest
from src.green_v400_direction_binding import verify_direction_binding


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "green_v400_silent_failure_challenge_prepare.json"


def manifest():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return build_prepare_manifest(
        config,
        ["row-a", "row-b"],
        green_direction_count=2,
        endpoint_direction_count=3,
    )


def test_actual_direction_files_and_per_row_bindings_are_deterministic(tmp_path):
    first = build_direction_payload_registry(
        manifest=manifest(),
        green_seed=b"g" * 32,
        endpoint_seed=b"e" * 32,
        output_directory=tmp_path / "first",
        direction_width=4,
        direction_norm=0.001,
    )
    second = build_direction_payload_registry(
        manifest=manifest(),
        green_seed=b"g" * 32,
        endpoint_seed=b"e" * 32,
        output_directory=tmp_path / "second",
        direction_width=4,
        direction_norm=0.001,
    )
    assert first["registry_sha256"] == second["registry_sha256"]
    assert first["panels"]["endpoint"]["payload_file_sha256"] == second["panels"]["endpoint"]["payload_file_sha256"]
    assert first["panels"]["green"]["seed_sha256"] != first["panels"]["endpoint"]["seed_sha256"]
    payload = np.load(tmp_path / "first" / "endpoint_directions.npy", mmap_mode="r")
    row = first["panels"]["endpoint"]["row_bindings"][0]
    verify_direction_binding(
        tensor=torch.from_numpy(np.array(payload[0], copy=True)),
        binding=row["binding"],
        expected_binding_sha256=row["binding_sha256"],
        protocol_id=manifest()["protocol_id"],
        row_id=row["row_id"],
        panel_kind="endpoint",
    )


def test_seed_reuse_and_manifest_outcome_flag_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="must differ"):
        build_direction_payload_registry(
            manifest=manifest(),
            green_seed=b"x" * 32,
            endpoint_seed=b"x" * 32,
            output_directory=tmp_path,
            direction_width=4,
            direction_norm=0.001,
        )
    changed = manifest()
    changed["contains_scientific_outcome"] = True
    with pytest.raises(ValueError, match="outcome-free"):
        build_direction_payload_registry(
            manifest=changed,
            green_seed=b"g" * 32,
            endpoint_seed=b"e" * 32,
            output_directory=tmp_path,
            direction_width=4,
            direction_norm=0.001,
        )
