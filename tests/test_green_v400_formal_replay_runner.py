import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_direction_binding import binding_sha256, build_direction_binding
from green_v400_execution_receipts import build_model_session_receipt
from green_v400_formal_replay_runner import run_formal_target_replay
from green_v400_response_precision import tensor_sha256
from tests.test_green_v400_formal_endpoint_runner import FakeFormalModel


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def file_hash(name):
    return hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()


def artifacts():
    protocol = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
    prompt = "12" * 32
    site = "23" * 32
    directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    binding = build_direction_binding(
        protocol_id=protocol,
        row_id=site,
        panel_kind="endpoint",
        tensor=directions,
        direction_norm=1.0,
        generator_spec="unit-test-v1",
    )
    universe = {
        "rows": [{
            "row_id": prompt,
            "clean_token_ids": [3, 4, 5],
            "corrupt_token_ids": [1, 4, 5],
            "signature": [0, 1],
            "io_token_id": 5,
            "s_token_id": 7,
        }]
    }
    job = {
        "job_id": "34" * 32,
        "site_row_id": site,
        "prompt_row_id": prompt,
        "layer": 0,
        "hook": "resid_post",
        "endpoint_direction_binding_sha256": binding_sha256(binding),
    }
    model = FakeFormalModel()
    manifest = {
        "weight_tensor_hashes": {
            name: tensor_sha256(value) for name, value in model.state_dict().items()
        }
    }
    plan = {
        "protocol_id": protocol,
        "execution_enabled": True,
        "universe_sha256": digest(universe),
        "model_manifest_sha256": digest(manifest),
        "full_model_hash": "45" * 32,
        "model_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "response_evaluation_precision": {
            "response_evaluation_dtype": "float64",
            "model_manifest_tensor_hash_scheme": "sha256-contiguous-numpy-native-bytes-v1",
        },
        "source_file_sha256": {
            "src/green_v400_formal_replay_runner.py": file_hash(
                "green_v400_formal_replay_runner.py"
            ),
            "src/green_v400_endpoint_calibration.py": file_hash(
                "green_v400_endpoint_calibration.py"
            ),
            "src/green_v400_response_precision.py": file_hash(
                "green_v400_response_precision.py"
            ),
            "src/green_v400_ioi_response_adapter.py": file_hash(
                "green_v400_ioi_response_adapter.py"
            ),
        },
        "queues": {"endpoint_numerical_replay": [job]},
    }
    plan["plan_sha256"] = digest(plan)
    session = build_model_session_receipt(
        plan=plan,
        observed_full_model_hash=plan["full_model_hash"],
        loader_source_sha256="56" * 32,
        process_start_nonce="67" * 32,
        pid=321,
    )
    return plan, universe, job, directions, binding, manifest, session


def test_formal_replay_binds_float64_same_checkpoint_execution():
    plan, universe, job, directions, binding, manifest, session = artifacts()
    packet, commitment = run_formal_target_replay(
        plan=plan,
        universe=universe,
        replay_job_id=job["job_id"],
        model_session_receipt=session,
        model_manifest=manifest,
        model=FakeFormalModel(),
        endpoint_directions=directions,
        endpoint_direction_binding=binding,
        replay_id="A",
        worker_instance_id="78" * 32,
    )
    assert packet["response_evaluation_dtype_private"] == "float64"
    assert packet["response_precision_receipt_sha256_private"]
    assert commitment["prediction_access_forbidden"] is True


def test_formal_replay_rejects_prepare_only_plan():
    plan, universe, job, directions, binding, manifest, session = artifacts()
    plan["execution_enabled"] = False
    plan.pop("plan_sha256")
    plan["plan_sha256"] = digest(plan)
    with pytest.raises(ValueError, match="prepare-only"):
        run_formal_target_replay(
            plan=plan,
            universe=universe,
            replay_job_id=job["job_id"],
            model_session_receipt=session,
            model_manifest=manifest,
            model=FakeFormalModel(),
            endpoint_directions=directions,
            endpoint_direction_binding=binding,
            replay_id="A",
            worker_instance_id="78" * 32,
        )
