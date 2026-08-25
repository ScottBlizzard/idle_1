"""Deterministic noun-separated record construction for GREEN v3.0.0."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Callable

from green_bridge_dataset import PairRecord, _candidate_pairs, cell_id
from green_bridge_spec import PROMPT, V200_DEVELOPMENT_GROUPS
from green_bridge_v300_spec import (
    CONFIRMATION_GROUPS,
    CONFIRMATION_NOUNS,
    DEVELOPMENT_GROUPS,
    DEVELOPMENT_NOUNS,
    NOUN_RANKS,
    ORIENTATIONS_PER_ROLE,
    RECORDS_PER_ROLE_PER_CELL,
    ROLES,
    V300_PAIR_SALT,
    V300_SPLIT_SALT,
    V300_SPLIT_SHA256,
    canonical_json,
    noun_rank,
    sha256_text,
)


V300_LITERAL_SPLIT_PAYLOAD = {
    "confirmation_groups": [
        {"century": 12, "noun": "warfare"},
        {"century": 14, "noun": "campaign"},
        {"century": 16, "noun": "campaign"},
        {"century": 14, "noun": "expedition"},
        {"century": 16, "noun": "expedition"},
        {"century": 12, "noun": "treaty"},
        {"century": 16, "noun": "treaty"},
    ],
    "confirmation_nouns": [
        {"noun": noun, "rank_key": NOUN_RANKS[noun]} for noun in CONFIRMATION_NOUNS
    ],
    "development_groups": [
        {"century": 12, "noun": "kingdom"},
        {"century": 16, "noun": "kingdom"},
        {"century": 12, "noun": "reign"},
        {"century": 14, "noun": "siege"},
        {"century": 16, "noun": "siege"},
    ],
    "development_nouns": [
        {"noun": noun, "rank_key": NOUN_RANKS[noun]} for noun in DEVELOPMENT_NOUNS
    ],
    "distance_bins": ["near", "far"],
    "orientations_per_role": {"down": 4, "up": 4},
    "records_per_role_per_cell": RECORDS_PER_ROLE_PER_CELL,
    "roles": ["transport", "joint"],
    "salt": V300_SPLIT_SALT,
    "schema": "green-bridge-v3.0.0-transport-split-v1",
    "source_split": "green-bridge-v2.0.0-unopened-confirmation",
}


def canonical_v300_split_payload() -> str:
    payload = build_green_bridge_v300_split()
    return canonical_json(payload)


def build_green_bridge_v300_split() -> dict:
    for noun, expected in NOUN_RANKS.items():
        if noun_rank(noun) != expected:
            raise AssertionError(f"noun rank mismatch: {noun}")
    payload = V300_LITERAL_SPLIT_PAYLOAD
    actual = sha256_text(canonical_json(payload))
    if actual != V300_SPLIT_SHA256:
        raise AssertionError(f"v3 split hash mismatch: {actual}")
    return payload


def _digest(kind: str, noun: str, century: int, distance: str, role: str, a: int, b: int) -> str:
    value = f"{V300_PAIR_SALT}|{kind}|{noun}|{century:02d}|{distance}|{role}|{a:02d}|{b:02d}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select_role(noun: str, century: int, distance: str, split: str, role: str,
                 excluded: set[tuple[int, int]], pair_allowed: Callable[[str, str], bool] | None) -> list[PairRecord]:
    ranked = []
    for a, b in _candidate_pairs(distance):
        if (a, b) in excluded:
            continue
        clean_a = PROMPT.format(noun=noun, cc=century, y=a)
        clean_b = PROMPT.format(noun=noun, cc=century, y=b)
        if pair_allowed is not None and not pair_allowed(clean_a, clean_b):
            continue
        ranked.append((_digest("pair", noun, century, distance, role, a, b),
                       _digest("orient", noun, century, distance, role, a, b), a, b))
    ranked.sort()
    quota = dict(ORIENTATIONS_PER_ROLE)
    selected = []
    for pair_digest, orientation_digest, a, b in ranked:
        preferred = "up" if int(orientation_digest[:2], 16) & 1 else "down"
        other = "down" if preferred == "up" else "up"
        orientation = preferred if quota[preferred] else other
        if not quota[orientation]:
            continue
        y, y_prime = (a, b) if orientation == "up" else (b, a)
        selected.append(PairRecord(
            population="v300_unopened_v200_confirmation", split=split, role=role,
            noun=noun, century=century, distance_bin=distance,
            cell_id=cell_id(noun, century, distance), item_index=len(selected),
            y=y, y_prime=y_prime,
            clean_prompt=PROMPT.format(noun=noun, cc=century, y=y),
            corrupt_prompt=PROMPT.format(noun=noun, cc=century, y=y_prime),
            pair_digest=pair_digest, orientation_digest=orientation_digest,
        ))
        quota[orientation] -= 1
        excluded.add((a, b))
        if not any(quota.values()):
            break
    if any(quota.values()):
        raise RuntimeError(f"v3 quota failure: {noun}/{century}/{distance}/{role}/{quota}")
    return selected


def build_green_bridge_v300_records(pair_allowed: Callable[[str, str], bool] | None = None) -> list[PairRecord]:
    build_green_bridge_v300_split()
    records = []
    for split, groups in (("development", DEVELOPMENT_GROUPS), ("confirmation", CONFIRMATION_GROUPS)):
        for noun, century in groups:
            for distance in ("near", "far"):
                excluded: set[tuple[int, int]] = set()
                for role in ROLES:
                    records.extend(_select_role(noun, century, distance, split, role, excluded, pair_allowed))
    verify_v300_contamination_firewall(records)
    return records


def verify_v300_contamination_firewall(records: list[PairRecord]) -> bool:
    dev_nouns = {row.noun for row in records if row.split == "development"}
    confirm_nouns = {row.noun for row in records if row.split == "confirmation"}
    if dev_nouns != set(DEVELOPMENT_NOUNS) or confirm_nouns != set(CONFIRMATION_NOUNS):
        raise AssertionError("v3 noun allocation mismatch")
    if dev_nouns & confirm_nouns:
        raise AssertionError("a noun crosses v3 phases")
    groups = {(row.noun, row.century) for row in records}
    if groups & set(V200_DEVELOPMENT_GROUPS):
        raise AssertionError("v2 development group contaminated v3")
    pairs_by_cell: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for row in records:
        key = (row.cell_id, row.split)
        pair = tuple(sorted((row.y, row.y_prime)))
        if pair in pairs_by_cell.setdefault(key, set()):
            raise AssertionError("transport/joint pairs are not disjoint")
        pairs_by_cell[key].add(pair)
    counts = {
        (split, role): sum(row.split == split and row.role == role for row in records)
        for split in ("development", "confirmation") for role in ROLES
    }
    if counts != {("development", "transport"): 80, ("development", "joint"): 80,
                  ("confirmation", "transport"): 112, ("confirmation", "joint"): 112}:
        raise AssertionError(f"v3 record counts mismatch: {counts}")
    return True


def v300_record_plan() -> dict:
    records = build_green_bridge_v300_records()
    return {
        "schema_version": "green-bridge-v3.0.0-record-plan-v1",
        "split_sha256": V300_SPLIT_SHA256,
        "records": [row.__dict__ | {"orientation": row.orientation} for row in records],
    }
