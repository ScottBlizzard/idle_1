import json
from pathlib import Path

import pytest

from analysis.green_v400_silent_failure_prepare import (
    build_prepare_manifest,
    canonical_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "green_v400_silent_failure_challenge_prepare.json"


def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_manifest_is_deterministic_and_contains_no_outcome():
    first = build_prepare_manifest(config(), ["row-b", "row-a"])
    second = build_prepare_manifest(config(), ["row-a", "row-b"])
    assert canonical_bytes(first) == canonical_bytes(second)
    assert first["row_ids"] == ["row-a", "row-b"]
    assert first["contains_scientific_outcome"] is False
    assert first["real_outcomes_authorized"] is False


def test_green_and_endpoint_commitments_are_disjoint():
    manifest = build_prepare_manifest(config(), ["row-a", "row-b"])
    green = {
        item
        for row in manifest["direction_commitments"].values()
        for item in row["green"]
    }
    endpoint = {
        item
        for row in manifest["direction_commitments"].values()
        for item in row["endpoint"]
    }
    assert green.isdisjoint(endpoint)


def test_commitments_change_by_row_and_ordinal():
    manifest = build_prepare_manifest(
        config(), ["row-a", "row-b"], green_direction_count=2
    )
    commitments = manifest["direction_commitments"]
    assert commitments["row-a"]["green"][0] != commitments["row-a"]["green"][1]
    assert commitments["row-a"]["green"][0] != commitments["row-b"]["green"][0]


@pytest.mark.parametrize("rows", [[], ["row-a", "row-a"], ["", "  "]])
def test_invalid_row_universe_is_rejected(rows):
    with pytest.raises(ValueError):
        build_prepare_manifest(config(), rows)


@pytest.mark.parametrize("green_count,endpoint_count", [(0, 1), (1, 0), (-1, 2)])
def test_nonpositive_direction_count_is_rejected(green_count, endpoint_count):
    with pytest.raises(ValueError):
        build_prepare_manifest(
            config(),
            ["row-a"],
            green_direction_count=green_count,
            endpoint_direction_count=endpoint_count,
        )
