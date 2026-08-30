from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "src")]

from green_v400_development_completeness_audit import (
    audit_development_completeness,
)
from green_v400_endpoint_firewall import seal_endpoint_packet, seal_prediction_packet
from green_v400_execution_receipts import receipt_sha256


ROW = "a" * 64
PROMPT = "b" * 64
PREDICTION_JOB = "c" * 64
ENDPOINT_JOB = "d" * 64


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def fixture(task="ioi"):
    plan = {
        "schema_version": "green-v400-development-execution-plan-v1",
        "protocol_id": "GREEN_V400_SILENT_FAILURE_GT_REPLICATION_PREPARE_V1" if task == "greater_than" else "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1",
        "development_authorized": True,
        "execution_enabled": True,
        "confirmation_authorized": False,
        "plan_gate": "DEVELOPMENT_ONLY_AUTHORIZED",
        "model_manifest_sha256": "1" * 64,
        "full_model_hash": "2" * 64,
        "model_revision": "revision",
        "queues": {
            "development_prediction": [{"job_id": PREDICTION_JOB, "site_row_id": ROW, "prompt_row_id": PROMPT}],
            "development_endpoint": [{"job_id": ENDPOINT_JOB, "site_row_id": ROW, "prompt_row_id": PROMPT}],
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    session = {
        "schema_version": "green-v400-model-session-receipt-v1",
        "protocol_id": plan["protocol_id"],
        "plan_sha256": plan["plan_sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "observed_full_model_hash": plan["full_model_hash"],
        "model_revision": plan["model_revision"],
        "weight_hash_recomputed_before_session": True,
    }
    session["receipt_sha256"] = receipt_sha256(session)
    baselines = {
        method: ({"normalized_risk_score": 0.1} if method == "empirical_four_branch_interaction" else {"normalized_rmse": 0.1})
        for method in (
            "finite_activation_patching",
            "first_order_attribution",
            "ms_hvp",
            "empirical_four_branch_interaction",
        )
    }
    prediction = {
        "schema_version": "green-v400-sfc-prediction-packet-v2",
        "protocol_id": plan["protocol_id"],
        "row_id": ROW,
        "route": "prediction",
        "contains_endpoint_outcome": False,
        "committed_before_endpoint": True,
        "ordinary_restoration": 0.9,
        "response_baselines": baselines,
        "normalized_mismatch_description": {},
        "integrated_gradients_steps": 65,
        "ms_hvp_segments": 8,
        "response_batch_chunk_size": 16,
        "response_batching": True,
    }
    prediction_commitment = seal_prediction_packet(prediction)
    endpoint = {
        "schema_version": "green-v400-sfc-endpoint-packet-v2",
        "protocol_id": plan["protocol_id"],
        "row_id": ROW,
        "route": "endpoint",
        "contains_prediction": False,
        "adaptive_query_allocation": False,
        "endpoint_status_private": "VALID",
        "endpoint_direction_count_private": 2,
        "endpoint_target_effects_private": [0.1, 0.2],
        "endpoint_patched_effects_private": [0.1, 0.1],
        "endpoint_discrepancies_private": [0.0, -0.1],
        "heldout_transport_error_private": 0.01,
        "heldout_transport_target_rms_private": 0.1,
        "heldout_transport_patched_rms_private": 0.1,
        "heldout_transport_symmetric_scale_private": 0.1,
        "heldout_transport_symmetric_normalized_error_private": 0.1,
        "endpoint_transport_failure_threshold_private": 0.2,
        "endpoint_normalization_floor_private": 1e-12,
        "endpoint_failure_label_private": False,
        "endpoint_failure_label_role_private": "secondary_prespecified_effect_size_label_not_per_row_inference",
        "numerical_replay_layer_receipt_sha256_private": "3" * 64,
        "endpoint_direction_binding_sha256_private": "4" * 64,
        "endpoint_authorization_receipt_sha256_private": "5" * 64,
        "decision_spec_sha256_private": "6" * 64,
        "runtime_input_receipt_sha256_private": "7" * 64,
        "response_precision_receipt_sha256_private": "8" * 64,
        "scientific_null_distribution_claimed_private": False,
        "scientific_outcome_evaluated_private": True,
    }
    return plan, {
        PREDICTION_JOB: {
            "job_id": PREDICTION_JOB,
            "prediction": prediction,
            "commitment": prediction_commitment,
            "model_session": session,
        }
    }, {
        ENDPOINT_JOB: {
            "job_id": ENDPOINT_JOB,
            "endpoint": endpoint,
            "commitment": seal_endpoint_packet(endpoint, prediction_commitment),
            "model_session": session,
        }
    }


def test_missing_prespecified_inputs_fail_closed_without_inferring_outcomes():
    plan, predictions, endpoints = fixture("greater_than")
    report = audit_development_completeness(
        plan=plan, prediction_artifacts=predictions, endpoint_artifacts=endpoints, task="greater_than"
    )
    assert report["frozen_primary_analyzer_ready"] is False
    assert {item["code"] for item in report["primary_analysis_blockers"]} == {
        "MISSING_GREEN_CERTIFICATE_STATUS",
        "MISSING_GREATER_THAN_CLEAN_TASK_VALIDITY",
    }
    assert report["confirmation_accessed"] is False


def test_tampered_endpoint_commitment_is_rejected():
    plan, predictions, endpoints = fixture()
    broken = deepcopy(endpoints)
    broken[ENDPOINT_JOB]["endpoint"]["heldout_transport_error_private"] = 0.02
    with pytest.raises(ValueError, match="changed"):
        audit_development_completeness(
            plan=plan, prediction_artifacts=predictions, endpoint_artifacts=broken, task="ioi"
        )


def test_extra_artifact_is_rejected():
    plan, predictions, endpoints = fixture()
    predictions["e" * 64] = deepcopy(predictions[PREDICTION_JOB])
    with pytest.raises(ValueError, match="exactly cover"):
        audit_development_completeness(
            plan=plan, prediction_artifacts=predictions, endpoint_artifacts=endpoints, task="ioi"
        )
