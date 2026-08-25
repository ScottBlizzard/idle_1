"""Deterministic finite-population construction for the green-bridge run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Sequence

from green_bridge_spec import (
    DISTANCE_BINS,
    DONOR_CENTURIES,
    DONOR_NOUNS,
    EVALUATION_CENTURIES,
    EVALUATION_NOUNS,
    HISTORICAL_V12_BASIS_SPEC,
    OUTPUT_ROOT,
    PROMPT,
    SALT,
    SUFFIX_MAX,
    SUFFIX_MIN,
    V200_CONFIRMATION_GROUPS,
    V200_DEVELOPMENT_GROUPS,
    V200_RESPLIT_SALT,
    V200_SPLIT_SHA256,
    canonical_json,
    sha256_text,
    write_json_atomic,
)

# Historical aliases are local to the archived v1.2 replay helpers below.
BASIS_V2_DONOR_CENTURIES = HISTORICAL_V12_BASIS_SPEC["donor_centuries"]
BASIS_V2_DONOR_NOUNS = HISTORICAL_V12_BASIS_SPEC["donor_nouns"]
BASIS_V2_DONOR_SELECTION_ORDER = HISTORICAL_V12_BASIS_SPEC["donor_selection_order"]
BASIS_V2_FIT_PAIRS = HISTORICAL_V12_BASIS_SPEC["fit_pairs"]
BASIS_V2_HOLDOUT_PAIRS = HISTORICAL_V12_BASIS_SPEC["holdout_pairs"]
BASIS_V2_RADIUS_PAIRS = HISTORICAL_V12_BASIS_SPEC["radius_pairs"]
BASIS_V2_SALT = HISTORICAL_V12_BASIS_SPEC["salt"]


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


def _digest_with_salt(
    salt: str,
    kind: str,
    noun: str,
    century: int,
    bin_name: str,
    role: str,
    a: int,
    b: int,
) -> str:
    key = f"{salt}|{kind}|{noun}|{century:02d}|{bin_name}|{role}|{a:02d}|{b:02d}"
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


def _v200_group_rank(noun: str, century: int) -> str:
    return hashlib.sha256(
        f"{V200_RESPLIT_SALT}|{noun}|{century:02d}".encode("utf-8")
    ).hexdigest()


def v200_split_payload() -> dict:
    def rows(groups):
        return [
            {"noun": noun, "century": century, "rank_key": _v200_group_rank(noun, century)}
            for noun, century in groups
        ]
    return {
        "schema": "green-bridge-v2.0.0-resplit-v1",
        "salt": V200_RESPLIT_SALT,
        "source_split": "green-bridge-v1.3.6-confirmation",
        "development_groups": rows(V200_DEVELOPMENT_GROUPS),
        "confirmation_groups": rows(V200_CONFIRMATION_GROUPS),
        "distance_bins": ["near", "far"],
        "roles": ["tensor", "energy"],
        "records_per_role_per_cell": 8,
    }


def build_green_bridge_v200_splits(
    pair_allowed: Callable[[str, str], bool] | None = None,
) -> tuple[list[PairRecord], dict]:
    """Resplit only the unopened v1.3.6 confirmation population."""
    original = build_evaluation_records(pair_allowed)
    exposed = {
        (row.noun, row.century) for row in original if row.split == "development"
    }
    source = [row for row in original if row.split == "confirmation"]
    development = set(V200_DEVELOPMENT_GROUPS)
    confirmation = set(V200_CONFIRMATION_GROUPS)
    source_groups = {(row.noun, row.century) for row in source}
    if development & confirmation or exposed & (development | confirmation):
        raise AssertionError("v2.0.0 split crosses the contamination firewall")
    if development | confirmation != source_groups:
        raise AssertionError("v2.0.0 split does not cover the unopened population")
    records = [
        replace(
            row,
            split=("development" if (row.noun, row.century) in development else "confirmation"),
        )
        for row in source
    ]
    dev_cells = {row.cell_id for row in records if row.split == "development"}
    confirm_cells = {row.cell_id for row in records if row.split == "confirmation"}
    if len(dev_cells) != 8 or len(confirm_cells) != 24:
        raise AssertionError("v2.0.0 cell counts are not 8 development and 24 confirmation")
    payload = v200_split_payload()
    actual = sha256_text(canonical_json(payload))
    payload["sha256"] = actual
    if actual != V200_SPLIT_SHA256:
        raise AssertionError(f"v2.0.0 split hash mismatch: {actual}")
    return records, payload


def build_legacy_donor_records(
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


def build_donor_records(
    pair_allowed: Callable[[str, str], bool] | None = None,
) -> list[PairRecord]:
    """Backward-compatible name for the immutable protocol-v1 donor plan."""
    return build_legacy_donor_records(pair_allowed)


def _select_basis_v2_role(
    *,
    noun: str,
    century: int,
    bin_name: str,
    role: str,
    count_up: int,
    count_down: int,
    used_suffixes: set[int],
    pair_allowed: Callable[[str, str], bool] | None,
) -> list[PairRecord]:
    ranked = []
    for a, b in _candidate_pairs(bin_name):
        if a in used_suffixes or b in used_suffixes:
            continue
        clean_a = PROMPT.format(noun=noun, cc=century, y=a)
        clean_b = PROMPT.format(noun=noun, cc=century, y=b)
        if pair_allowed is not None and not pair_allowed(clean_a, clean_b):
            continue
        pair_hash = _digest_with_salt(
            BASIS_V2_SALT, "pair", noun, century, bin_name, role, a, b
        )
        orient_hash = _digest_with_salt(
            BASIS_V2_SALT, "orient", noun, century, bin_name, role, a, b
        )
        ranked.append((pair_hash, orient_hash, a, b))
    ranked.sort(key=lambda row: row[0])
    quotas = {"up": count_up, "down": count_down}
    selected = []
    for pair_hash, orient_hash, a, b in ranked:
        if a in used_suffixes or b in used_suffixes:
            continue
        preferred = "up" if int(orient_hash[:2], 16) & 1 else "down"
        opposite = "down" if preferred == "up" else "up"
        orientation = preferred if quotas[preferred] else opposite
        if quotas[orientation] == 0:
            continue
        y, y_prime = (a, b) if orientation == "up" else (b, a)
        selected.append(PairRecord(
            population="donor_v2",
            split="donor",
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
        ))
        quotas[orientation] -= 1
        used_suffixes.update((a, b))
        if not any(quotas.values()):
            break
    if any(quotas.values()):
        raise RuntimeError(
            f"basis-v2 quota failure for {noun}/{century}/{bin_name}/{role}: {quotas}"
        )
    return selected


def build_basis_v2_donor_records(
    pair_allowed: Callable[[str, str], bool] | None = None,
) -> list[PairRecord]:
    records = []
    for noun in BASIS_V2_DONOR_NOUNS:
        for century in BASIS_V2_DONOR_CENTURIES:
            used_suffixes: set[int] = set()
            for bin_name, role, count_up, count_down in BASIS_V2_DONOR_SELECTION_ORDER:
                records.extend(_select_basis_v2_role(
                    noun=noun,
                    century=century,
                    bin_name=bin_name,
                    role=role,
                    count_up=count_up,
                    count_down=count_down,
                    used_suffixes=used_suffixes,
                    pair_allowed=pair_allowed,
                ))
            if len(used_suffixes) != 40:
                raise RuntimeError(
                    f"basis-v2 suffix quota failure for {noun}/{century}: {len(used_suffixes)}"
                )
    validate_basis_v2_plan(records)
    return records


def validate_evaluation_plan(records: Sequence[PairRecord]) -> None:
    evaluation = [row for row in records if row.population == "evaluation"]
    if len(evaluation) != 768:
        raise AssertionError(f"expected 768 evaluation records, got {len(evaluation)}")
    cells = sorted({row.cell_id for row in evaluation})
    if len(cells) != 48:
        raise AssertionError(f"expected 48 evaluation cells, got {len(cells)}")
    development = {row.cell_id for row in evaluation if row.split == "development"}
    confirmation = {row.cell_id for row in evaluation if row.split == "confirmation"}
    if len(development) != 16 or len(confirmation) != 32 or development & confirmation:
        raise AssertionError("development/confirmation cell split is invalid")
    for cid in cells:
        subset = [row for row in evaluation if row.cell_id == cid]
        for role in ("tensor", "energy"):
            role_rows = [row for row in subset if row.role == role]
            if len(role_rows) != 8 or sum(row.orientation == "up" for row in role_rows) != 4:
                raise AssertionError(f"{cid}/{role} quota failed")
        tensor_pairs = {tuple(sorted((row.y, row.y_prime))) for row in subset if row.role == "tensor"}
        energy_pairs = {tuple(sorted((row.y, row.y_prime))) for row in subset if row.role == "energy"}
        if tensor_pairs & energy_pairs:
            raise AssertionError(f"tensor/energy pair overlap in {cid}")


def validate_basis_v2_plan(records: Sequence[PairRecord]) -> None:
    rows = [row for row in records if row.population == "donor_v2"]
    expected_counts = {
        "basis_fit": BASIS_V2_FIT_PAIRS,
        "basis_holdout": BASIS_V2_HOLDOUT_PAIRS,
        "radius_v2": BASIS_V2_RADIUS_PAIRS,
    }
    counts = {role: sum(row.role == role for row in rows) for role in expected_counts}
    if counts != expected_counts or len(rows) != 1280:
        raise AssertionError(f"invalid basis-v2 role counts: {counts}")
    if set(BASIS_V2_DONOR_NOUNS) & set(EVALUATION_NOUNS):
        raise AssertionError("basis-v2/evaluation noun overlap")
    if set(BASIS_V2_DONOR_NOUNS) & set(DONOR_NOUNS):
        raise AssertionError("basis-v2/legacy donor noun overlap")
    prompts = [prompt for row in rows for prompt in (row.clean_prompt, row.corrupt_prompt)]
    if len(prompts) != 2560 or len(set(prompts)) != 2560:
        raise AssertionError("basis-v2 prompt keys are not unique")
    for noun in BASIS_V2_DONOR_NOUNS:
        for century in BASIS_V2_DONOR_CENTURIES:
            subset = [row for row in rows if row.noun == noun and row.century == century]
            suffixes = [suffix for row in subset for suffix in (row.y, row.y_prime)]
            if len(subset) != 20 or len(suffixes) != 40 or len(set(suffixes)) != 40:
                raise AssertionError(f"basis-v2 suffix reuse for {noun}/{century}")
            cursor = 0
            for bin_name, role, count_up, count_down in BASIS_V2_DONOR_SELECTION_ORDER:
                count = count_up + count_down
                group = subset[cursor:cursor + count]
                cursor += count
                if any(row.distance_bin != bin_name or row.role != role for row in group):
                    raise AssertionError(f"basis-v2 role order failed for {noun}/{century}")
                if sum(row.orientation == "up" for row in group) != count_up:
                    raise AssertionError(f"basis-v2 orientation quota failed for {noun}/{century}/{role}")


def basis_v2_plan_payload(records: Sequence[PairRecord]) -> dict:
    validate_basis_v2_plan(records)
    rows = [asdict(record) for record in records]
    pair_keys = [
        [row.noun, row.century, row.distance_bin, row.role, row.y, row.y_prime, row.pair_digest]
        for row in records
    ]
    prompt_keys = [
        [row.pair_digest, system, prompt]
        for row in records
        for system, prompt in (("clean", row.clean_prompt), ("corrupt", row.corrupt_prompt))
    ]
    by_role = {}
    for role in ("basis_fit", "basis_holdout", "radius_v2"):
        keys = [key for key, row in zip(pair_keys, records) if row.role == role]
        by_role[role + "_ordered_keys_sha256"] = sha256_text(canonical_json(keys))
    return {
        "schema_version": "green-bridge-donor-v2-plan-v1",
        "records": rows,
        "records_sha256": sha256_text(canonical_json(rows)),
        "counts": {
            role: sum(row.role == role for row in records)
            for role in ("basis_fit", "basis_holdout", "radius_v2")
        },
        "unique_prompt_count": len({prompt for row in records for prompt in (row.clean_prompt, row.corrupt_prompt)}),
        "prompt_overlap_count": 0,
        "legacy_noun_overlap_count": len(set(BASIS_V2_DONOR_NOUNS) & set(DONOR_NOUNS)),
        "evaluation_noun_overlap_count": len(set(BASIS_V2_DONOR_NOUNS) & set(EVALUATION_NOUNS)),
        "ordered_pair_keys": pair_keys,
        "ordered_prompt_keys": prompt_keys,
        "basis_v2_all_prompt_keys_sha256": sha256_text(canonical_json(prompt_keys)),
        **by_role,
    }


def write_basis_v2_plan(path: Path, records: Sequence[PairRecord]) -> dict:
    payload = basis_v2_plan_payload(records)
    write_json_atomic(path, payload)
    return payload


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
