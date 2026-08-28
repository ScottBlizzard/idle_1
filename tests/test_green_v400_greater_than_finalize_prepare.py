import json
from pathlib import Path

from analysis.green_v400_finalize_sfc_prepare import finalize_prepare_manifest
from analysis.green_v400_execution_plan_prepare import compile_execution_plan
from analysis.green_v400_greater_than_universe_prepare import build_untouched_universe
from analysis.green_v400_silent_failure_protocol import validate_prepare_config
from tests.test_green_v400_greater_than_universe_prepare import FakeYearTokenizer


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE = ROOT / "configs" / "green_v400_greater_than_silent_failure_prepare.json"
UNIVERSE = ROOT / "configs" / "green_v400_greater_than_untouched_universe.json"


def small_universe_config():
    payload = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    payload["centuries"] = [18]
    payload["records_per_cell"] = 2
    payload["orientations_per_cell"] = {"up": 1, "down": 1}
    payload["role_nouns"] = {
        role: nouns[:1] for role, nouns in payload["role_nouns"].items()
    }
    return payload


def test_replication_challenge_passes_shared_static_protocol_validator():
    challenge = json.loads(CHALLENGE.read_text(encoding="utf-8"))
    assert validate_prepare_config(challenge) == []


def test_replication_manifest_expands_disjoint_prompt_roles_to_sites():
    challenge = json.loads(CHALLENGE.read_text(encoding="utf-8"))
    universe = build_untouched_universe(FakeYearTokenizer(), small_universe_config())
    manifest = finalize_prepare_manifest(challenge, universe)
    assert manifest["contains_scientific_outcome"] is False
    assert manifest["real_outcomes_authorized"] is False
    assert manifest["prediction_prompt_count"] == 8
    assert manifest["prediction_site_count"] == 72
    assert manifest["endpoint_calibration"]["site_count"] == 36
    assert manifest["unused_reserve"]["site_count"] == 36
    assert manifest["all_role_sets_disjoint"] is True
    assert set(manifest["direction_commitments"]) == {
        row["row_id"] for row in manifest["prediction_sites"]
    }


def test_replication_execution_plan_uses_its_own_blocked_readiness_registry():
    challenge = json.loads(CHALLENGE.read_text(encoding="utf-8"))
    universe = build_untouched_universe(FakeYearTokenizer(), small_universe_config())
    manifest = finalize_prepare_manifest(challenge, universe)
    readiness = json.loads(
        (ROOT / "configs" / "green_v400_greater_than_baseline_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    plan = compile_execution_plan(
        challenge, universe, manifest, readiness, repository_root=ROOT
    )
    assert plan["plan_gate"] == "PLAN_COMPILED_BLOCKED_BY_BASELINES"
    assert plan["execution_enabled"] is False
    assert plan["queue_counts"] == {
        "development_prediction": 4 * 9,
        "development_endpoint": 4 * 9,
        "confirmation_prediction": 4 * 9,
        "confirmation_endpoint": 4 * 9,
        "endpoint_calibration": 4 * 9,
    }
