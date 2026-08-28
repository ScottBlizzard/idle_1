import json
import re
from pathlib import Path

import pytest

from analysis.green_v400_greater_than_universe_prepare import (
    build_untouched_universe,
    validate_universe_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "green_v400_greater_than_untouched_universe.json"


class FakeYearTokenizer:
    def __init__(self):
        self.pieces = {}
        self.reverse = {}

    def _id(self, piece):
        if piece not in self.pieces:
            value = len(self.pieces) + 1000
            self.pieces[piece] = value
            self.reverse[value] = piece
        return self.pieces[piece]

    def encode(self, text, add_special_tokens=False):
        if re.fullmatch(r"\d{2}", text):
            return [self._id(text)]
        if re.fullmatch(r" \d{2}", text):
            return [self._id(text)]
        if re.fullmatch(r" \d{4}", text):
            return [self._id(text[:3]), self._id(text[3:])]
        tokens = []
        cursor = 0
        for match in re.finditer(r" \d{4}| \d{2}|[A-Za-z]+|<\|endoftext\|>|[^\s]", text):
            if match.start() < cursor:
                continue
            piece = match.group(0)
            if re.fullmatch(r" \d{4}", piece):
                tokens.extend([self._id(piece[:3]), self._id(piece[3:])])
            else:
                tokens.append(self._id(piece))
            cursor = match.end()
        return tokens

    def decode(self, ids):
        return "".join(self.reverse[int(value)] for value in ids)


def small_config():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["centuries"] = [18]
    payload["records_per_cell"] = 2
    payload["orientations_per_cell"] = {"up": 1, "down": 1}
    payload["role_nouns"] = {
        role: nouns[:1] for role, nouns in payload["role_nouns"].items()
    }
    return payload


def test_universe_is_deterministic_disjoint_and_outcome_free():
    first = build_untouched_universe(FakeYearTokenizer(), small_config())
    second = build_untouched_universe(FakeYearTokenizer(), small_config())
    assert first["rows_sha256"] == second["rows_sha256"]
    assert first["row_count"] == 16
    assert len({row["row_id"] for row in first["rows"]}) == 16
    assert first["contains_scientific_outcome"] is False
    assert first["model_weights_loaded"] is False
    assert first["execution_authorized"] is False


def test_every_cell_has_both_orientations_and_exact_token_contract():
    artifact = build_untouched_universe(FakeYearTokenizer(), small_config())
    cells = {}
    for row in artifact["rows"]:
        key = (row["role"], row["noun"], row["century"], row["distance_bin"])
        cells.setdefault(key, set()).add(row["orientation"])
        assert len(row["clean_token_ids"]) == len(row["corrupt_token_ids"])
        differing = [
            index for index, pair in enumerate(zip(row["clean_token_ids"], row["corrupt_token_ids"]))
            if pair[0] != pair[1]
        ]
        assert differing == [row["site_position"]]
        assert row["final_position"] == len(row["clean_token_ids"]) - 1
    assert all(orientations == {"up", "down"} for orientations in cells.values())


def test_role_nouns_must_be_disjoint():
    payload = small_config()
    payload["role_nouns"]["confirmation"][0] = payload["role_nouns"]["development"][0]
    with pytest.raises(ValueError):
        validate_universe_config(payload)


def test_execution_authorization_and_endpoint_leakage_are_rejected():
    payload = small_config()
    payload["execution_authorized"] = True
    with pytest.raises(ValueError):
        validate_universe_config(payload)
    payload = small_config()
    payload["endpoint_contract"]["secondary_is_prediction_input"] = True
    with pytest.raises(ValueError):
        validate_universe_config(payload)
