import copy
import json
from pathlib import Path

import pytest

from analysis.green_v400_finalize_sfc_prepare import finalize_prepare_manifest
from analysis.green_v400_ioi_universe_prepare import build_untouched_universe
from analysis.green_v400_silent_failure_prepare import sha256_value
from tests.test_green_v400_ioi_universe_prepare import FakeTokenizer, small_config


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_CONFIG = ROOT / "configs" / "green_v400_silent_failure_challenge_prepare.json"


def challenge_config():
    return json.loads(CHALLENGE_CONFIG.read_text(encoding="utf-8"))


def universe():
    return build_untouched_universe(FakeTokenizer(), small_config())


def test_finalized_manifest_binds_disjoint_prediction_calibration_and_reserve_rows():
    result = finalize_prepare_manifest(challenge_config(), universe())
    prediction = set(result["row_ids"])
    calibration = {
        row["row_id"] for row in result["endpoint_calibration"]["sites"]
    }
    reserve = {row["row_id"] for row in result["unused_reserve"]["sites"]}
    assert len(prediction) == 12 * 9
    assert len(calibration) == 3 * 9
    assert len(reserve) == 3 * 9
    assert prediction.isdisjoint(calibration)
    assert prediction.isdisjoint(reserve)
    assert calibration.isdisjoint(reserve)
    assert result["contains_scientific_outcome"] is False
    assert result["real_outcomes_authorized"] is False
    assert result["site_definition"] == {
        "hook": "resid_post",
        "layers": list(range(9)),
    }


def test_site_identifier_binds_prompt_layer_and_hook():
    result = finalize_prepare_manifest(challenge_config(), universe())
    first = result["prediction_sites"][0]
    second = result["prediction_sites"][1]
    assert first["prompt_row_id"] == second["prompt_row_id"]
    assert first["layer"] != second["layer"]
    assert first["row_id"] != second["row_id"]


def test_universe_row_mutation_breaks_hash_binding():
    payload = universe()
    payload["rows"][0]["io_name"] = "tampered"
    with pytest.raises(ValueError, match="row hash mismatch"):
        finalize_prepare_manifest(challenge_config(), payload)


def test_model_weight_loading_flag_is_rejected_even_without_outcomes():
    payload = universe()
    payload["model_weights_loaded"] = True
    with pytest.raises(ValueError, match="tokenizer-only"):
        finalize_prepare_manifest(challenge_config(), payload)


def test_protocol_mismatch_is_rejected():
    payload = universe()
    payload["protocol_id"] = "DIFFERENT_PROTOCOL"
    with pytest.raises(ValueError, match="protocol identifiers"):
        finalize_prepare_manifest(challenge_config(), payload)


def test_overlapping_role_ids_are_rejected_with_valid_row_hash():
    payload = universe()
    development = next(row for row in payload["rows"] if row["role"] == "development")
    reserve = next(row for row in payload["rows"] if row["role"] == "unused_reserve")
    reserve["row_id"] = development["row_id"]
    payload["rows_sha256"] = sha256_value(payload["rows"])
    with pytest.raises(ValueError, match="must be disjoint"):
        finalize_prepare_manifest(challenge_config(), payload)


def test_duplicate_candidate_layer_is_rejected():
    config = challenge_config()
    config["candidate_population"]["layers"] = [0, 0, 1]
    with pytest.raises(ValueError, match="candidate layers"):
        finalize_prepare_manifest(config, universe())
