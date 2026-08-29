import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_direction_binding import build_direction_binding, binding_sha256
from green_v400_execution_receipts import build_model_session_receipt
from green_v400_formal_prediction_runner import run_formal_prediction
from green_v400_response_precision import tensor_sha256


def canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_hash(name):
    return hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()


class FakeFormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def run_with_hooks(self, tokens, fwd_hooks):
        batch, seq = tokens.shape
        dtype = self.anchor.dtype
        residual = torch.stack(
            [tokens.to(dtype), 0.5 * tokens.to(dtype)], dim=-1
        ) + 0.0 * self.anchor
        for name, hook in fwd_hooks:
            if name == "blocks.2.hook_resid_post":
                residual = hook(residual, None)
        post = torch.zeros((batch, seq, 3072), dtype=dtype)
        signal = residual[:, 1, 0]
        for gate in (2326, 1138, 2287, 606, 2848, 2305, 46, 2659, 946, 1616):
            post[:, -1, gate] = signal.square() / 100.0
        for name, hook in fwd_hooks:
            if name == "blocks.10.mlp.hook_post":
                post = hook(post, None)
        logits = torch.zeros((batch, seq, 16), dtype=dtype)
        gate_signal = post[:, -1, :].sum(dim=1)
        logits[:, -1, 5] = tokens[:, 0].float() + residual[:, 1, 0] + gate_signal
        logits[:, -1, 7] = residual[:, 1, 1]
        return logits


def fixtures():
    protocol = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
    prompt_row = "11" * 32
    site_row = "22" * 32
    directions = torch.tensor([[0.0006, 0.0008]], dtype=torch.float32)
    binding = build_direction_binding(
        protocol_id=protocol,
        row_id=site_row,
        panel_kind="green",
        tensor=directions,
        direction_norm=0.001,
        generator_spec="test-generator-v1",
    )
    universe = {
        "rows": [
            {
                "row_id": prompt_row,
                "clean_token_ids": [3, 4, 5],
                "corrupt_token_ids": [1, 4, 5],
                "signature": [0, 1],
                "io_token_id": 5,
                "s_token_id": 7,
            }
        ]
    }
    job = {
        "job_id": "33" * 32,
        "kind": "prediction",
        "role": "development",
        "site_row_id": site_row,
        "prompt_row_id": prompt_row,
        "layer": 2,
        "hook": "resid_post",
        "green_direction_binding_sha256": binding_sha256(binding),
        "contains_scientific_outcome": False,
    }
    sources = {
        "analysis/green_v400_formal_worker.py": "66" * 32,
        "src/green_v400_formal_prediction_runner.py": file_hash(
            "green_v400_formal_prediction_runner.py"
        ),
        "src/green_v400_prediction_worker.py": file_hash("green_v400_prediction_worker.py"),
        "src/green_v400_matched_bypass_adapter.py": file_hash(
            "green_v400_matched_bypass_adapter.py"
        ),
        "src/green_v400_four_branch_baseline.py": file_hash(
            "green_v400_four_branch_baseline.py"
        ),
        "src/green_v400_ioi_response_adapter.py": file_hash(
            "green_v400_ioi_response_adapter.py"
        ),
        "src/green_v400_response_precision.py": file_hash(
            "green_v400_response_precision.py"
        ),
    }
    manifest = {
        "weight_tensor_hashes": {
            name: tensor_sha256(value)
            for name, value in FakeFormalModel().state_dict().items()
        }
    }
    plan = {
        "protocol_id": protocol,
        "execution_enabled": True,
        "universe_sha256": canonical_hash(universe),
        "model_manifest_sha256": canonical_hash(manifest),
        "full_model_hash": "55" * 32,
        "model_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "response_evaluation_precision": {
            "response_evaluation_dtype": "float64",
            "model_manifest_tensor_hash_scheme": "sha256-contiguous-numpy-native-bytes-v1",
        },
        "source_file_sha256": sources,
        "queues": {
            "development_prediction": [job],
            "confirmation_prediction": [],
        },
    }
    plan["plan_sha256"] = canonical_hash(plan)
    session = build_model_session_receipt(
        plan=plan,
        observed_full_model_hash=plan["full_model_hash"],
        loader_source_sha256="66" * 32,
        process_start_nonce="77" * 32,
        pid=1234,
    )
    return plan, universe, job, directions, binding, session, manifest


def test_formal_prediction_binds_plan_model_direction_and_four_branch_route():
    plan, universe, job, directions, binding, session, manifest = fixtures()
    packet, commitment = run_formal_prediction(
        plan=plan,
        universe=universe,
        prediction_job_id=job["job_id"],
        model_session_receipt=session,
        model_manifest=manifest,
        model=FakeFormalModel(),
        green_directions=directions,
        green_direction_binding=binding,
        integrated_gradients_steps=3,
        ms_hvp_segments=2,
    )
    assert packet["schema_version"] == "green-v400-sfc-prediction-packet-v2"
    assert packet["formal_execution_binding"]["plan_sha256"] == plan["plan_sha256"]
    assert packet["formal_execution_binding"]["response_evaluation_dtype"] == "float64"
    assert "empirical_four_branch_interaction" in packet["response_baselines"]
    assert packet["response_baselines"]["empirical_four_branch_interaction"][
        "diagnostics"
    ]["certificate_claimed"] is False
    assert commitment["prediction_packet_sha256"]


def test_formal_prediction_rejects_prepare_plan_and_changed_direction_payload():
    plan, universe, job, directions, binding, session, manifest = fixtures()
    changed = dict(plan)
    changed["execution_enabled"] = False
    changed.pop("plan_sha256")
    changed["plan_sha256"] = canonical_hash(changed)
    with pytest.raises(ValueError, match="prepare-only"):
        run_formal_prediction(
            plan=changed,
            universe=universe,
            prediction_job_id=job["job_id"],
            model_session_receipt=session,
            model_manifest=manifest,
            model=FakeFormalModel(),
            green_directions=directions,
            green_direction_binding=binding,
            integrated_gradients_steps=3,
            ms_hvp_segments=2,
        )
    mutated = directions.clone()
    mutated[0, 0] += 1e-5
    with pytest.raises(ValueError, match="payload hash"):
        run_formal_prediction(
            plan=plan,
            universe=universe,
            prediction_job_id=job["job_id"],
            model_session_receipt=session,
            model_manifest=manifest,
            model=FakeFormalModel(),
            green_directions=mutated,
            green_direction_binding=binding,
            integrated_gradients_steps=3,
            ms_hvp_segments=2,
        )
