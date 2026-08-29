import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_direction_binding import binding_sha256, build_direction_binding
from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_execution_receipts import (
    build_model_session_receipt,
    receipt_sha256,
    token_ids_sha256,
)
from green_v400_formal_endpoint_runner import run_formal_heldout_transport_endpoint
from green_v400_response_precision import tensor_sha256


PROTOCOL = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
PROMPT = "10" * 32
SITE = "20" * 32
PREDICTION_JOB = "30" * 32
ENDPOINT_JOB = "40" * 32


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(name):
    return hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()


class FakeFormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def run_with_hooks(self, tokens, fwd_hooks):
        batch, seq = tokens.shape
        dtype = self.anchor.dtype
        activation = torch.stack([tokens.to(dtype)] * 3, dim=-1)
        activation[:, 1, :] += tokens[:, :1].to(dtype)
        for name, hook in fwd_hooks:
            if name.endswith("hook_resid_post"):
                activation = hook(activation, None)
        logits = torch.zeros((batch, seq, 16), dtype=dtype)
        logits[:, -1, 5] = activation[:, 1, 0]
        return logits


def artifacts(execution_enabled=True):
    row = {
        "row_id": PROMPT,
        "role": "development",
        "clean_token_ids": [3, 4, 5],
        "corrupt_token_ids": [1, 4, 5],
        "signature": [3, 1, 0, 2],
        "io_token_id": 5,
        "s_token_id": 7,
    }
    universe = {"protocol_id": PROTOCOL, "rows": [row]}
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    binding = build_direction_binding(
        protocol_id=PROTOCOL,
        row_id=SITE,
        panel_kind="endpoint",
        tensor=directions,
        direction_norm=1.0,
        generator_spec="unit-test-v1",
    )
    binding_hash = binding_sha256(binding)
    manifest = {
        "weight_tensor_hashes": {
            name: tensor_sha256(value)
            for name, value in FakeFormalModel().state_dict().items()
        }
    }
    plan = {
        "protocol_id": PROTOCOL,
        "execution_enabled": execution_enabled,
        "universe_sha256": value_hash(universe),
        "model_manifest_sha256": value_hash(manifest),
        "full_model_hash": "60" * 32,
        "model_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "response_evaluation_precision": {
            "response_evaluation_dtype": "float64",
            "model_manifest_tensor_hash_scheme": "sha256-contiguous-numpy-native-bytes-v1",
        },
        "source_file_sha256": {
            "analysis/green_v400_formal_worker.py": "b0" * 32,
            "src/green_v400_endpoint_worker.py": file_hash("green_v400_endpoint_worker.py"),
            "src/green_v400_formal_endpoint_runner.py": file_hash("green_v400_formal_endpoint_runner.py"),
            "src/green_v400_ioi_response_adapter.py": file_hash("green_v400_ioi_response_adapter.py"),
            "src/green_v400_response_precision.py": file_hash("green_v400_response_precision.py"),
        },
        "queues": {
            "development_prediction": [{
                "job_id": PREDICTION_JOB,
                "site_row_id": SITE,
                "prompt_row_id": PROMPT,
                "layer": 0,
                "hook": "resid_post",
            }],
            "development_endpoint": [{
                "job_id": ENDPOINT_JOB,
                "site_row_id": SITE,
                "prompt_row_id": PROMPT,
                "role": "development",
                "layer": 0,
                "hook": "resid_post",
                "endpoint_direction_binding_sha256": binding_hash,
            }],
            "confirmation_endpoint": [],
        },
    }
    plan["plan_sha256"] = value_hash(plan)
    prediction = {
        "schema_version": "green-v400-sfc-prediction-packet-v1",
        "protocol_id": PROTOCOL,
        "row_id": SITE,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
    }
    prediction_commitment = seal_prediction_packet(prediction)
    authorization = {
        "schema_version": "green-v400-endpoint-authorization-receipt-v1",
        "protocol_id": PROTOCOL,
        "plan_sha256": plan["plan_sha256"],
        "endpoint_job_id": ENDPOINT_JOB,
        "phase": "development",
        "site_row_id": SITE,
        "prompt_row_id": PROMPT,
        "layer": 0,
        "hook": "resid_post",
        "prediction_packet_sha256": prediction_commitment["prediction_packet_sha256"],
        "numerical_replay_layer_receipt_sha256": "70" * 32,
        "endpoint_direction_binding_sha256": binding_hash,
        "direction_registry_sha256": "80" * 32,
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "full_model_hash": plan["full_model_hash"],
        "decision_spec_sha256": "90" * 32,
        "endpoint_worker_source_sha256": file_hash("green_v400_endpoint_worker.py"),
        "response_adapter_source_path": "src/green_v400_ioi_response_adapter.py",
        "response_adapter_source_sha256": file_hash("green_v400_ioi_response_adapter.py"),
        "clean_token_ids_sha256": token_ids_sha256(row["clean_token_ids"]),
        "corrupt_token_ids_sha256": token_ids_sha256(row["corrupt_token_ids"]),
        "phase_ledger_head_sha256": "a0" * 32,
    }
    authorization["receipt_sha256"] = receipt_sha256(authorization)
    session = build_model_session_receipt(
        plan=plan,
        observed_full_model_hash=plan["full_model_hash"],
        loader_source_sha256="b0" * 32,
        process_start_nonce="c0" * 32,
        pid=123,
    )
    return plan, universe, directions, binding, prediction_commitment, authorization, session, manifest


def test_formal_runner_constructs_center_and_adapters_internally():
    plan, universe, directions, binding, commitment, authorization, session, manifest = artifacts()
    packet, _ = run_formal_heldout_transport_endpoint(
        plan=plan,
        universe=universe,
        endpoint_authorization_receipt=authorization,
        model_session_receipt=session,
        model_manifest=manifest,
        prediction_commitment=commitment,
        model=FakeFormalModel(),
        endpoint_directions=directions,
        endpoint_direction_binding=binding,
    )
    assert packet["endpoint_status_private"] == "VALID"
    assert packet["runtime_input_receipt_sha256_private"]
    signature = inspect.signature(run_formal_heldout_transport_endpoint)
    assert "center" not in signature.parameters
    assert "target_response" not in signature.parameters
    assert "patched_response" not in signature.parameters


def test_formal_runner_rejects_full_universe_substitution_and_prepare_only_plan():
    plan, universe, directions, binding, commitment, authorization, session, manifest = artifacts()
    changed = json.loads(json.dumps(universe))
    changed["rows"][0]["clean_token_ids"][0] = 9
    with pytest.raises(ValueError, match="full universe"):
        run_formal_heldout_transport_endpoint(
            plan=plan,
            universe=changed,
            endpoint_authorization_receipt=authorization,
            model_session_receipt=session,
            model_manifest=manifest,
            prediction_commitment=commitment,
            model=FakeFormalModel(),
            endpoint_directions=directions,
            endpoint_direction_binding=binding,
        )
    plan, universe, directions, binding, commitment, authorization, session, manifest = artifacts(
        execution_enabled=False
    )
    with pytest.raises(ValueError, match="prepare-only"):
        run_formal_heldout_transport_endpoint(
            plan=plan,
            universe=universe,
            endpoint_authorization_receipt=authorization,
            model_session_receipt=session,
            model_manifest=manifest,
            prediction_commitment=commitment,
            model=FakeFormalModel(),
            endpoint_directions=directions,
            endpoint_direction_binding=binding,
        )
