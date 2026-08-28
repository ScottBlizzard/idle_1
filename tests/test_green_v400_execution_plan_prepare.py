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
    return challenge, universe, manifest, readiness


def test_plan_compiles_all_routes_but_authorizes_nothing():
    plan = compile_execution_plan(*payloads(), repository_root=ROOT)
    assert plan["execution_enabled"] is False
    assert plan["untouched_rows_evaluated"] == 0
    assert plan["plan_gate"] == "PLAN_COMPILED_BLOCKED_BY_BASELINES"
    assert plan["queue_counts"] == {
        "development_prediction": 6 * 9,
        "development_endpoint": 6 * 9,
        "confirmation_prediction": 6 * 9,
        "confirmation_endpoint": 6 * 9,
        "endpoint_calibration": 3 * 9,
    }
    assert plan["gpu_policy"]["physical_gpu_indices"] == [4, 5, 6, 7]


def test_every_endpoint_job_requires_a_prediction_commitment():
    plan = compile_execution_plan(*payloads(), repository_root=ROOT)
    for name in ("development_endpoint", "confirmation_endpoint"):
        assert all(job["requires_prediction_commitment"] for job in plan["queues"][name])


def test_baseline_ready_plan_still_cannot_authorize_execution():
    challenge, universe, manifest, readiness = payloads()
    for entry in readiness["baselines"].values():
        if entry["required"]:
            entry["status"] = "READY"
    plan = compile_execution_plan(
        challenge, universe, manifest, readiness, repository_root=ROOT
    )
    assert plan["plan_gate"] == "PLAN_COMPILED_AWAITING_SCIENTIFIC_AUTHORIZATION"
    assert plan["execution_enabled"] is False


def test_mutated_site_hash_fails_closed():
    challenge, universe, manifest, readiness = payloads()
    manifest["prediction_sites"][0]["layer"] = 99
    with pytest.raises(ValueError, match="prediction sites hash mismatch"):
        compile_execution_plan(
            challenge, universe, manifest, readiness, repository_root=ROOT
        )


def test_reserve_cannot_become_executable():
    challenge, universe, manifest, readiness = payloads()
    manifest["unused_reserve"]["execution_forbidden"] = False
    with pytest.raises(ValueError, match="reserve execution"):
        compile_execution_plan(
            challenge, universe, manifest, readiness, repository_root=ROOT
        )


def test_readiness_from_another_protocol_cannot_be_reused():
    challenge, universe, manifest, readiness = payloads()
    readiness["protocol_id"] = "ANOTHER_PROTOCOL"
    with pytest.raises(ValueError, match="readiness protocol"):
        compile_execution_plan(
            challenge, universe, manifest, readiness, repository_root=ROOT
        )
