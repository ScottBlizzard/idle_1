"""Deterministic finite-population construction for the green-bridge run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Sequence

from green_bridge_spec import (
    DISTANCE_BINS,
    DONOR_CENTURIES,
    DONOR_NOUNS,
    EVALUATION_CENTURIES,
    EVALUATION_NOUNS,
    OUTPUT_ROOT,
    PROMPT,
    SALT,
    SUFFIX_MAX,
    SUFFIX_MIN,
    canonical_json,
    sha256_text,
    write_json_atomic,
)


@dataclass(frozen=True)
class PairRecord:
    population: str
    split: str
    role: str
    noun: str
    century: int
    distance_bin: str
    cell_id: str
    item_index: int
    y: int
    y_prime: int
    clean_prompt: str
    corrupt_prompt: str
    pair_digest: str
    orientation_digest: str

    @property
    def orientation(self) -> str:
        return "up" if self.y_prime > self.y else "down"


def _digest(kind: str, noun: str, century: int, bin_name: str, role: str, a: int, b: int) -> str:
    key = f"{SALT}|{kind}|{noun}|{century:02d}|{bin_name}|{role}|{a:02d}|{b:02d}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _candidate_pairs(bin_name: str) -> list[tuple[int, int]]:
    low, high = DISTANCE_BINS[bin_name]
    return [
        (a, b)
        for a in range(SUFFIX_MIN, SUFFIX_MAX + 1)
        for b in range(a + 1, SUFFIX_MAX + 1)
        if low <= b - a <= high
    ]


def is_development_cell(noun: str, century: int) -> bool:
    return EVALUATION_NOUNS.index(noun) % 3 == EVALUATION_CENTURIES.index(century)


def cell_id(noun: str, century: int, bin_name: str) -> str:
    return f"{noun}|{century:02d}|{bin_name}"


def _select_role(
    *,
    population: str,
    split: str,
    noun: str,
    century: int,
    bin_name: str,
    role: str,
    count_up: int,
    count_down: int,
    excluded: set[tuple[int, int]],
    pair_allowed: Callable[[str, str], bool] | None,
) -> list[PairRecord]:
    ranked = []
    for a, b in _candidate_pairs(bin_name):
        if (a, b) in excluded:
            continue
        clean_a = PROMPT.format(noun=noun, cc=century, y=a)
        clean_b = PROMPT.format(noun=noun, cc=century, y=b)
        if pair_allowed is not None and not pair_allowed(clean_a, clean_b):
            continue
        pair_hash = _digest("pair", noun, century, bin_name, role, a, b)
        orient_hash = _digest("orient", noun, century, bin_name, role, a, b)
        ranked.append((pair_hash, orient_hash, a, b))
    ranked.sort(key=lambda row: row[0])
    quotas = {"up": count_up, "down": count_down}
    selected: list[PairRecord] = []
    for pair_hash, orient_hash, a, b in ranked:
        preferred = "up" if int(orient_hash[:2], 16) & 1 else "down"
        orientation = preferred if quotas[preferred] else ("down" if preferred == "up" else "up")
        if quotas[orientation] == 0:
            continue
        y, y_prime = (a, b) if orientation == "up" else (b, a)
        selected.append(
            PairRecord(
                population=population,
                split=split,
                role=role,
                noun=noun,
                century=century,
                distance_bin=bin_name,
                cell_id=cell_id(noun, century, bin_name),
                item_index=len(selected),
                y=y,
                y_prime=y_prime,
                clean_prompt=PROMPT.format(noun=noun, cc=century, y=y),
                corrupt_prompt=PROMPT.format(noun=noun, cc=century, y=y_prime),
                pair_digest=pair_hash,
                orientation_digest=orient_hash,
            )
        )
        quotas[orientation] -= 1
        excluded.add((a, b))
        if not any(quotas.values()):
            break
    if any(quotas.values()):
        raise RuntimeError(
            f"quota failure for {noun}/{century}/{bin_name}/{role}: {quotas}"
        )
    return selected


def build_evaluation_records(
    pair_allowed: Callable[[str, str], bool] | None = None,
) -> list[PairRecord]:
    records: list[PairRecord] = []
    for noun in EVALUATION_NOUNS:
        for century in EVALUATION_CENTURIES:
            split = "development" if is_development_cell(noun, century) else "confirmation"
            for bin_name in DISTANCE_BINS:
                excluded: set[tuple[int, int]] = set()
                records.extend(
                    _select_role(
                        population="evaluation", split=split, noun=noun,
                        century=century, bin_name=bin_name, role="tensor",
                        count_up=4, count_down=4, excluded=excluded,
                        pair_allowed=pair_allowed,
                    )
                )
                records.extend(
                    _select_role(
                        population="evaluation", split=split, noun=noun,
                        century=century, bin_name=bin_name, role="energy",
                        count_up=4, count_down=4, excluded=excluded,
                        pair_allowed=pair_allowed,
                    )
                )
    return records


def build_donor_records(
    pair_allowed: Callable[[str, str], bool] | None = None,
) -> list[PairRecord]:
    records: list[PairRecord] = []
    for noun in DONOR_NOUNS:
        for century in DONOR_CENTURIES:
            for bin_name in DISTANCE_BINS:
                excluded: set[tuple[int, int]] = set()
                records.extend(
                    _select_role(
                        population="donor", split="donor", noun=noun,
                        century=century, bin_name=bin_name, role="basis",
                        count_up=2, count_down=2, excluded=excluded,
                        pair_allowed=pair_allowed,
                    )
                )
                records.extend(
                    _select_role(
                        population="donor", split="donor", noun=noun,
                        century=century, bin_name=bin_name, role="radius",
                        count_up=2, count_down=2, excluded=excluded,
                        pair_allowed=pair_allowed,
                    )
                )
    return records


def validate_plan(records: Sequence[PairRecord]) -> None:
    evaluation = [r for r in records if r.population == "evaluation"]
    donor = [r for r in records if r.population == "donor"]
    if len(evaluation) != 48 * 16:
        raise AssertionError(f"expected 768 evaluation records, got {len(evaluation)}")
    if len(donor) != 1024:
        raise AssertionError(f"expected 1024 donor records, got {len(donor)}")
    cells = sorted({r.cell_id for r in evaluation})
    if len(cells) != 48:
        raise AssertionError(f"expected 48 evaluation cells, got {len(cells)}")
    development = {r.cell_id for r in evaluation if r.split == "development"}
    confirmation = {r.cell_id for r in evaluation if r.split == "confirmation"}
    if len(development) != 16 or len(confirmation) != 32 or development & confirmation:
        raise AssertionError("development/confirmation cell split is invalid")
    for cid in cells:
        subset = [r for r in evaluation if r.cell_id == cid]
        for role in ("tensor", "energy"):
            role_rows = [r for r in subset if r.role == role]
            if len(role_rows) != 8:
                raise AssertionError(f"{cid}/{role} does not have eight items")
            if sum(r.orientation == "up" for r in role_rows) != 4:
                raise AssertionError(f"{cid}/{role} orientation quota failed")
        tensor_pairs = {tuple(sorted((r.y, r.y_prime))) for r in subset if r.role == "tensor"}
        energy_pairs = {tuple(sorted((r.y, r.y_prime))) for r in subset if r.role == "energy"}
        if tensor_pairs & energy_pairs:
            raise AssertionError(f"tensor/energy pair overlap in {cid}")


def plan_payload(records: Iterable[PairRecord]) -> dict:
    rows = [asdict(record) for record in records]
    return {
        "schema_version": "green-bridge-splits-v1",
        "records": rows,
        "records_sha256": sha256_text(canonical_json(rows)),
    }


def write_plan(path: Path, records: Sequence[PairRecord]) -> dict:
    validate_plan(records)
    payload = plan_payload(records)
    write_json_atomic(path, payload)
    return payload


class ConfirmationLock:
    """Prevent confirmation-record access until the analysis manifest is frozen."""

    def __init__(self, frozen_analysis_path: Path | None = None):
        self.path = frozen_analysis_path or OUTPUT_ROOT / "frozen_analysis.json"

    def assert_open(self) -> None:
        if not self.path.is_file():
            raise PermissionError(
                "confirmation is locked until frozen_analysis.json exists"
            )

    def select(self, records: Sequence[PairRecord]) -> list[PairRecord]:
        self.assert_open()
        return [record for record in records if record.split == "confirmation"]


def split_records(
    records: Sequence[PairRecord],
    split: str,
    *,
    confirmation_lock: ConfirmationLock | None = None,
) -> list[PairRecord]:
    if split == "confirmation":
        if confirmation_lock is None:
            raise PermissionError("confirmation selection requires a ConfirmationLock")
        return confirmation_lock.select(records)
    if split not in {"development", "donor"}:
        raise ValueError(f"unknown split: {split}")
    return [record for record in records if record.split == split]
