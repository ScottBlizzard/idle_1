import hashlib
import json
import re
from pathlib import Path

import pytest

from analysis.green_v400_ioi_universe_prepare import (
    build_untouched_universe,
    validate_universe_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "green_v400_ioi_untouched_universe.json"


class FakeTokenizer:
    def __init__(self):
        self.ids = {}

    def encode(self, text, add_special_tokens=False):
        words = re.findall(r"[A-Za-z]+", text)
        ids = []
        for word in words:
            if word not in self.ids:
                self.ids[word] = int.from_bytes(
                    hashlib.sha256(word.encode()).digest()[:4], "big"
                )
            ids.append(self.ids[word])
        return ids


def small_config():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["rows_per_template"] = 3
    payload["names"] = payload["names"][:20]
    payload["places"] = payload["places"][:2]
    payload["items"] = payload["items"][:2]
    return payload


def test_universe_is_deterministic_unique_and_outcome_free():
    first = build_untouched_universe(FakeTokenizer(), small_config())
    second = build_untouched_universe(FakeTokenizer(), small_config())
    assert first["rows_sha256"] == second["rows_sha256"]
    assert first["row_count"] == 18
    assert len({row["row_id"] for row in first["rows"]}) == 18
    assert first["contains_scientific_outcome"] is False
    assert first["model_weights_loaded"] is False
    assert first["execution_authorized"] is False


def test_every_row_has_equal_clean_corrupt_length_and_expected_name_counts():
    universe = build_untouched_universe(FakeTokenizer(), small_config())
    for row in universe["rows"]:
        assert len(row["clean_token_ids"]) == len(row["corrupt_token_ids"])
        _, io_position, s_first, s_second = row["signature"]
        assert row["clean_token_ids"][io_position] == row["io_token_id"]
        assert row["clean_token_ids"][s_first] == row["s_token_id"]
        assert row["clean_token_ids"][s_second] == row["s_token_id"]


def test_historical_template_cannot_reenter():
    payload = small_config()
    payload["templates"][0]["text"] = payload["historical_template_forbidden"]
    with pytest.raises(ValueError):
        validate_universe_config(payload)


def test_execution_authorization_is_rejected():
    payload = small_config()
    payload["execution_authorized"] = True
    with pytest.raises(ValueError):
        validate_universe_config(payload)
