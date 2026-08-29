import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.green_v400_formal_batch_worker import (
    select_shard_jobs,
    validate_clean_model_exit,
    validate_existing_artifact,
)
from analysis.green_v400_formal_worker import canonical_sha256
from green_v400_endpoint_firewall import seal_prediction_packet
from green_v400_execution_receipts import receipt_sha256


def plan():
    queues = {"development_prediction": []}
    for index in range(10):
        queues["development_prediction"].append(
            {
                "job_id": f"{index + 1:064x}",
                "role": "development",
                "site_row_id": f"{index + 101:064x}",
                "green_direction_binding_sha256": f"{index + 201:064x}",
            }
        )
    return {"queues": queues}


def test_shards_are_disjoint_complete_and_preserve_queue_order():
    payload = plan()
    shards = [
        select_shard_jobs(
            payload,
            mode="prediction",
            phase="development",
            shard_index=index,
            shard_count=4,
        )
        for index in range(4)
    ]
    observed = [job["job_id"] for shard in shards for job in shard]
    expected = [job["job_id"] for job in payload["queues"]["development_prediction"]]
    assert set(observed) == set(expected)
    assert len(observed) == len(set(observed))
    assert [job["job_id"] for job in shards[0]] == expected[0::4]


def test_invalid_shard_or_duplicate_queue_fails_closed():
    payload = plan()
    with pytest.raises(ValueError, match="coordinates"):
        select_shard_jobs(
            payload,
            mode="prediction",
            phase="development",
            shard_index=4,
            shard_count=4,
        )
    payload["queues"]["development_prediction"][1]["job_id"] = payload["queues"][
        "development_prediction"
    ][0]["job_id"]
    with pytest.raises(ValueError, match="duplicate"):
        select_shard_jobs(
            payload,
            mode="prediction",
            phase="development",
            shard_index=0,
            shard_count=1,
        )


def test_resume_validation_rejects_changed_prediction_commitment():
    payload = {
        "model_manifest_sha256": "22" * 32,
        "full_model_hash": "33" * 32,
        "model_revision": "rev",
        "protocol_id": "P",
        "prediction_execution": {
            "integrated_gradients_steps": 65,
            "ms_hvp_segments": 8,
            "response_batch_chunk_size": 16,
        },
        "source_file_sha256": {
            "src/green_v400_formal_prediction_runner.py": "aa" * 32
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    job = {
        "job_id": "44" * 32,
        "site_row_id": "55" * 32,
        "green_direction_binding_sha256": "66" * 32,
    }
    session = {
        "schema_version": "green-v400-model-session-receipt-v1",
        "protocol_id": "P",
        "plan_sha256": payload["plan_sha256"],
        "model_manifest_sha256": payload["model_manifest_sha256"],
        "observed_full_model_hash": payload["full_model_hash"],
        "model_revision": "rev",
        "loader_source_sha256": "77" * 32,
        "process_start_nonce": "88" * 32,
        "pid": 1,
        "weight_hash_recomputed_before_session": True,
    }
    session["receipt_sha256"] = receipt_sha256(session)
    packet = {
        "schema_version": "green-v400-sfc-prediction-packet-v2",
        "protocol_id": "P",
        "row_id": job["site_row_id"],
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "formal_execution_binding": {
            "plan_sha256": payload["plan_sha256"],
            "prediction_job_id": job["job_id"],
            "green_direction_binding_sha256": job[
                "green_direction_binding_sha256"
            ],
            "model_session_receipt_sha256": session["receipt_sha256"],
            "formal_prediction_runner_source_sha256": "aa" * 32,
            "response_precision_receipt_sha256": "bb" * 32,
            "response_evaluation_dtype": "float64",
        },
        "ordinary_restoration": 0.9,
        "response_baselines": {},
        "normalized_mismatch_description": {},
        "integrated_gradients_steps": 65,
        "ms_hvp_segments": 8,
        "response_batch_chunk_size": 16,
        "response_batching": True,
    }
    artifact = {
        "schema_version": "green-v400-formal-prediction-artifact-v1",
        "job_id": job["job_id"],
        "prediction": packet,
        "commitment": seal_prediction_packet(packet),
        "model_session": session,
    }
    validate_existing_artifact(
        plan=payload, mode="prediction", job=job, artifact=artifact
    )
    changed = json.loads(json.dumps(artifact))
    changed["prediction"]["row_id"] = "99" * 32
    with pytest.raises(ValueError, match="commitment changed"):
        validate_existing_artifact(
            plan=payload, mode="prediction", job=job, artifact=changed
        )


def test_batch_exit_detects_gradient_or_weight_mutation():
    import torch
    from green_v400_response_precision import tensor_sha256

    model = torch.nn.Linear(2, 1).float()
    expected = {
        name: tensor_sha256(value) for name, value in model.state_dict().items()
    }
    payload = {"full_model_hash": canonical_sha256(expected)}
    manifest = {"weight_tensor_hashes": expected}
    assert validate_clean_model_exit(
        model=model, model_manifest=manifest, plan=payload
    ) == payload["full_model_hash"]
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(ValueError, match="gradients"):
        validate_clean_model_exit(
            model=model, model_manifest=manifest, plan=payload
        )
