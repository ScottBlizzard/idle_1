import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.green_v400_formal_worker import (
    WORKER_SOURCE_PATH,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_direction_row,
    planned_job,
    validate_runtime_envelope,
    verify_plan,
)
from green_v400_direction_binding import binding_sha256, build_direction_binding


def test_formal_worker_rejects_prepare_plan_before_payload_or_model(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    plan = {
        "execution_enabled": False,
        "gpu_policy": {"physical_gpu_indices": [4, 5, 6, 7]},
        "source_file_sha256": {WORKER_SOURCE_PATH: file_sha256(Path(__file__).parents[1] / WORKER_SOURCE_PATH)},
    }
    with pytest.raises(ValueError, match="prepare-only"):
        validate_runtime_envelope(plan, "cuda:0")


def test_formal_worker_loads_only_bound_direction_row(tmp_path):
    protocol = "P"
    row_id = "12" * 32
    values = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype="<f4")
    payload_path = tmp_path / "green.npy"
    np.save(payload_path, values, allow_pickle=False)
    tensor = torch.from_numpy(values[0])
    binding = build_direction_binding(
        protocol_id=protocol,
        row_id=row_id,
        panel_kind="green",
        tensor=tensor,
        direction_norm=1.0,
        generator_spec="unit-test-v1",
    )
    registry = {
        "panels": {
            "green": {
                "payload_filename": "green.npy",
                "payload_file_sha256": file_sha256(payload_path),
                "shape": list(values.shape),
                "row_bindings": [{
                    "row_id": row_id,
                    "row_index": 0,
                    "binding_sha256": binding_sha256(binding),
                    "binding": binding,
                }],
            }
        }
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    plan = {
        "protocol_id": protocol,
        "direction_registry_sha256": registry["registry_sha256"],
    }
    loaded, loaded_binding = load_direction_row(
        plan=plan,
        registry=registry,
        payload_path=payload_path,
        panel="green",
        row_id=row_id,
    )
    assert torch.equal(loaded, tensor)
    assert loaded_binding == binding
    payload_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file hash"):
        load_direction_row(
            plan=plan,
            registry=registry,
            payload_path=payload_path,
            panel="green",
            row_id=row_id,
        )


def test_worker_plan_and_job_resolution_fail_closed():
    plan = {
        "development_authorized": True,
        "queues": {
            "development_prediction": [{"job_id": "a", "role": "development"}],
            "confirmation_prediction": [],
        }
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    verify_plan(plan)
    assert planned_job(plan, "prediction", "a")["job_id"] == "a"
    changed = json.loads(json.dumps(plan))
    changed["queues"]["development_prediction"][0]["job_id"] = "b"
    with pytest.raises(ValueError, match="self hash"):
        verify_plan(changed)


def test_worker_requires_binding_authority_and_frozen_runtime(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    plan = {
        "execution_enabled": True,
        "real_outcomes_authorized": False,
        "gpu_policy": {"physical_gpu_indices": [4, 5, 6, 7]},
        "source_file_sha256": {
            WORKER_SOURCE_PATH: file_sha256(Path(__file__).parents[1] / WORKER_SOURCE_PATH)
        },
    }
    with pytest.raises(ValueError, match="real-outcome authorization"):
        validate_runtime_envelope(plan, "cuda:0")
    plan["real_outcomes_authorized"] = True
    validate_runtime_envelope(plan, "cuda:0")


def test_worker_output_is_fail_closed_outside_server_root(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "analysis.green_v400_formal_worker.FORMAL_OUTPUT_ROOT", tmp_path / "allowed"
    )
    (tmp_path / "allowed").mkdir()
    outside = tmp_path / "allowed-suffix"
    outside.mkdir()
    with pytest.raises(ValueError, match="must remain under"):
        atomic_write_json(outside / "artifact.json", {"x": 1})


def test_worker_output_refuses_to_overwrite(monkeypatch, tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setattr("analysis.green_v400_formal_worker.FORMAL_OUTPUT_ROOT", root)
    output = root / "artifact.json"
    atomic_write_json(output, {"x": 1})
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        atomic_write_json(output, {"x": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"x": 1}
