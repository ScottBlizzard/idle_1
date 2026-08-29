import copy
import json
from pathlib import Path

import pytest

from analysis.green_v400_execution_plan_prepare import compile_execution_plan
from analysis.green_v400_finalize_sfc_prepare import finalize_prepare_manifest
from analysis.green_v400_ioi_universe_prepare import build_untouched_universe
from analysis.green_v400_silent_failure_prepare import sha256_value
from tests.test_green_v400_ioi_universe_prepare import FakeTokenizer, small_config


ROOT = Path(__file__).resolve().parents[1]


def fake_direction_registry(manifest):
    protocol = manifest["protocol_id"]
    prediction_ids = [site["row_id"] for site in manifest["prediction_sites"]]
    replay_ids = [site["row_id"] for site in manifest["endpoint_calibration"]["sites"]]

    def panel(name, row_ids):
        rows = []
        for index, row_id in enumerate(row_ids):
            binding = {
                "schema_version": "green-v400-direction-binding-v1",
                "protocol_id": protocol,
                "row_id": row_id,
                "panel_kind": name,
                "dtype": "float32-little-endian",
                "shape": [8, 768],
                "direction_norm": 0.001,
                "generator_spec": "numpy-PCG64DXSM-rowwise-normal-v1",
                "tensor_sha256": sha256_value([name, row_id, "tensor"]),
            }
            rows.append({
                "row_index": index,
                "row_id": row_id,
                "binding_sha256": sha256_value(binding),
                "binding": binding,
            })
        return {
            "payload_file_sha256": sha256_value([name, "payload"]),
            "row_bindings": rows,
        }

    registry = {
        "schema_version": "green-v400-direction-payload-registry-v1",
        "protocol_id": protocol,
        "contains_scientific_outcome": False,
        "manifest_sha256": sha256_value(manifest),
        "panels": {
            "green": panel("green", prediction_ids),
            "endpoint": panel("endpoint", prediction_ids + replay_ids),
        },
    }
    registry["registry_sha256"] = sha256_value(registry)
    return registry


def payloads():
    challenge = json.loads(
        (ROOT / "configs/green_v400_silent_failure_challenge_prepare.json").read_text(
            encoding="utf-8"
        )
    )
    universe = build_untouched_universe(FakeTokenizer(), small_config())
    manifest = finalize_prepare_manifest(challenge, universe)
    readiness = json.loads(
        (ROOT / "configs/green_v400_baseline_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    decision = json.loads(
        (ROOT / "configs/green_v400_shared_decision_spec.json").read_text(
            encoding="utf-8"
        )
    )
    model = json.loads(
        (ROOT / "analysis/GREEN_V400_FORMAL_PREPARE_ARTIFACTS_20260826/model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return challenge, universe, manifest, readiness, fake_direction_registry(manifest), decision, model


def test_plan_compiles_all_routes_but_authorizes_nothing():
    plan = compile_execution_plan(*payloads(), repository_root=ROOT)
    assert plan["execution_enabled"] is False
    assert plan["untouched_rows_evaluated"] == 0
    assert plan["plan_gate"] == "PLAN_COMPILED_AWAITING_SCIENTIFIC_AUTHORIZATION"
    assert plan["baseline_readiness"]["not_ready_required"] == []
    assert plan["queue_counts"] == {
        "development_prediction": 6 * 9,
        "development_grant_cohort_prediction": 9,
        "development_endpoint": 6 * 9,
        "confirmation_prediction": 6 * 9,
        "confirmation_grant_cohort_prediction": 9,
        "confirmation_endpoint": 6 * 9,
        "endpoint_numerical_replay": 3 * 9,
    }
    assert plan["gpu_policy"]["physical_gpu_indices"] == [4, 5, 6, 7]
    assert plan["worker_routes"] == {
        "prediction": "persistent_prediction_process_per_gpu_shard",
        "grant": "persistent_grant_process_per_gpu_shard_without_directions",
        "replay": "separate_replay_process",
        "endpoint": "separate_endpoint_process",
        "cross_route_shared_model_instance_forbidden": True,
        "prediction_route_persistent_model_allowed": True,
        "grant_route_separate_from_direction_bearing_prediction_route": True,
    }
    assert plan["prediction_execution"] == {
        "integrated_gradients_steps": 65,
        "ms_hvp_segments": 8,
        "response_batch_chunk_size": 16,
        "shard_count": 4,
        "physical_gpu_by_shard": {"0": 4, "1": 5, "2": 6, "3": 7},
        "jobs_serial_within_shard": True,
        "model_persistent_within_route_plan_phase_shard": True,
        "automatic_numerical_parameter_change_on_failure_forbidden": True,
    }


def test_every_endpoint_job_requires_a_prediction_commitment():
    plan = compile_execution_plan(*payloads(), repository_root=ROOT)
    for name in ("development_endpoint", "confirmation_endpoint"):
        assert all(job["requires_prediction_commitment"] for job in plan["queues"][name])


def test_grant_baseline_is_scheduled_only_with_official_cohort_semantics():
    plan = compile_execution_plan(*payloads(), repository_root=ROOT)
    for phase in ("development", "confirmation"):
        jobs = plan["queues"][f"{phase}_grant_cohort_prediction"]
        assert len(jobs) == 9
        assert {job["layer"] for job in jobs} == set(range(9))
        assert all(job["cohort_size"] == 6 for job in jobs)
        assert all(job["must_commit_before_phase_endpoints"] for job in jobs)
        assert all("not_per_row_classifier" in job["official_semantics"] for job in jobs)


def test_baseline_ready_plan_still_cannot_authorize_execution():
    challenge, universe, manifest, readiness, directions, decision, model = payloads()
    for entry in readiness["baselines"].values():
        if entry["required"]:
            entry["status"] = "READY"
    plan = compile_execution_plan(
        challenge, universe, manifest, readiness, directions, decision, model, repository_root=ROOT
    )
    assert plan["plan_gate"] == "PLAN_COMPILED_AWAITING_SCIENTIFIC_AUTHORIZATION"
    assert plan["execution_enabled"] is False


def test_mutated_site_hash_fails_closed():
    challenge, universe, manifest, readiness, directions, decision, model = payloads()
    manifest["prediction_sites"][0]["layer"] = 99
    with pytest.raises(ValueError, match="prediction sites hash mismatch"):
        compile_execution_plan(
            challenge, universe, manifest, readiness, directions, decision, model, repository_root=ROOT
        )


def test_reserve_cannot_become_executable():
    challenge, universe, manifest, readiness, directions, decision, model = payloads()
    manifest["unused_reserve"]["execution_forbidden"] = False
    with pytest.raises(ValueError, match="reserve execution"):
        compile_execution_plan(
            challenge, universe, manifest, readiness, directions, decision, model, repository_root=ROOT
        )


def test_readiness_from_another_protocol_cannot_be_reused():
    challenge, universe, manifest, readiness, directions, decision, model = payloads()
    readiness["protocol_id"] = "ANOTHER_PROTOCOL"
    with pytest.raises(ValueError, match="readiness protocol"):
        compile_execution_plan(
            challenge, universe, manifest, readiness, directions, decision, model, repository_root=ROOT
        )


def test_challenge_mutation_after_manifest_is_rejected():
    challenge, universe, manifest, readiness, directions, decision, model = payloads()
    challenge["endpoint_numerical_replay_protocol"]["absolute_tolerance"] = 2e-7
    with pytest.raises(ValueError, match="not bound to the challenge"):
        compile_execution_plan(
            challenge, universe, manifest, readiness, directions, decision, model, repository_root=ROOT
        )
