"""Outcome-blind P13 cohort selection and exact contraction aggregation."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

from green_v410_protocol import PROTOCOL_ID, sha256_canonical


IOI_SELECTION_DOMAIN = b"GREEN-V410-P13-QUALIFICATION\0ioi\0"
GT_SELECTION_DOMAIN = b"GREEN-V410-P13-QUALIFICATION\0greater_than\0"
IOI_DIRECTION_DOMAIN = "GREEN_V410_P13_IOI_GREEN_PANEL"
GT_DIRECTION_DOMAIN = "GREEN_V410_P13_GT_GREEN_PANEL"
RESERVE_NOUNS = {"ceremony", "committee", "federation", "workshop"}
LAYERS = tuple(range(9))


def _prompt_id(row: Mapping[str, Any]) -> str:
    value = row.get("prompt_row_id", row.get("row_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("qualification prompt row ID is missing")
    return value


def _selection_digest(domain: bytes, prompt_row_id: str) -> str:
    return hashlib.sha256(domain + prompt_row_id.encode("utf-8")).hexdigest()


def _site(prompt_row_id: str, layer: int) -> dict[str, Any]:
    payload = {"prompt_row_id": prompt_row_id, "layer": layer, "hook": "resid_post"}
    return {"row_id": sha256_canonical(payload), **payload}


def select_ioi_qualification(prompt_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reserve = [
        row for row in prompt_rows
        if row.get("role") == "unused_reserve" and row.get("template_id") == "reserve_0"
    ]
    if len(reserve) != 64:
        raise ValueError(f"IOI reserve must contain exactly 64 prompts, got {len(reserve)}")
    ranked = sorted(
        ((_selection_digest(IOI_SELECTION_DOMAIN, _prompt_id(row)), _prompt_id(row), row)
         for row in reserve),
        key=lambda item: (item[0], item[1]),
    )
    selected = [dict(item[2]) for item in ranked[:12]]
    return _qualification_manifest("ioi", selected, IOI_DIRECTION_DOMAIN)


def select_gt_qualification(prompt_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reserve = [
        row for row in prompt_rows
        if row.get("role") == "unused_reserve" and row.get("noun") in RESERVE_NOUNS
    ]
    selected: list[dict[str, Any]] = []
    for century in (10, 18, 20):
        for distance in ("near", "far"):
            for orientation in ("up", "down"):
                candidates = [
                    row for row in reserve
                    if row.get("century") == century
                    and row.get("distance_bin") == distance
                    and row.get("orientation") == orientation
                ]
                if not candidates:
                    raise ValueError(
                        f"GT reserve stratum is empty: {century}/{distance}/{orientation}"
                    )
                ranked = sorted(
                    ((_selection_digest(GT_SELECTION_DOMAIN, _prompt_id(row)), _prompt_id(row), row)
                     for row in candidates),
                    key=lambda item: (item[0], item[1]),
                )
                selected.append(dict(ranked[0][2]))
    if len({_prompt_id(row) for row in selected}) != 12:
        raise ValueError("GT qualification prompt selection is not unique")
    return _qualification_manifest("greater_than", selected, GT_DIRECTION_DOMAIN)


def _qualification_manifest(
    task: str, selected_prompts: Sequence[Mapping[str, Any]], direction_domain: str
) -> dict[str, Any]:
    prompts = sorted((dict(row) for row in selected_prompts), key=_prompt_id)
    sites = [
        _site(_prompt_id(row), layer)
        for row in prompts
        for layer in LAYERS
    ]
    if len(prompts) != 12 or len(sites) != 108 or len({site["row_id"] for site in sites}) != 108:
        raise ValueError("qualification manifest counts are invalid")
    direction_rows = [
        {
            "site_row_id": site["row_id"],
            "direction_ordinal": ordinal,
            "direction_domain": direction_domain,
        }
        for site in sites
        for ordinal in range(8)
    ]
    manifest = {
        "schema_version": "green-v410-sfc-jwtec-p13-qualification-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": task,
        "selection_domain": (
            IOI_SELECTION_DOMAIN.decode("utf-8")
            if task == "ioi" else GT_SELECTION_DOMAIN.decode("utf-8")
        ),
        "direction_domain": direction_domain,
        "prompt_count": 12,
        "site_count": 108,
        "direction_count": 864,
        "selected_prompt_row_ids": [_prompt_id(row) for row in prompts],
        "selected_prompts_sha256": sha256_canonical(prompts),
        "sites": sites,
        "sites_sha256": sha256_canonical(sites),
        "direction_rows_sha256": sha256_canonical(direction_rows),
    }
    manifest["manifest_sha256"] = sha256_canonical(manifest)
    return manifest


def contraction_ratio(
    *,
    branch_centers: Mapping[str, Fraction],
    branch_intervals: Mapping[str, tuple[Fraction, Fraction]],
    relational_interval: tuple[Fraction, Fraction],
) -> Fraction:
    branches = ("PAT_J", "PAT_B", "TAR_J", "TAR_B")
    signs = {"PAT_J": 1, "PAT_B": -1, "TAR_J": -1, "TAR_B": 1}
    if set(branch_centers) != set(branches) or set(branch_intervals) != set(branches):
        raise ValueError("P13 branch panel mismatch")
    widths: dict[str, Fraction] = {}
    for branch in branches:
        center = branch_centers[branch]
        lower, upper = branch_intervals[branch]
        if lower > upper:
            raise ValueError("P13 branch interval is malformed")
        widths[branch] = max(abs(center - lower), abs(center - upper))
    relational_lower, relational_upper = relational_interval
    if relational_lower > relational_upper:
        raise ValueError("P13 relational interval is malformed")
    theta = sum((signs[branch] * branch_centers[branch] for branch in branches), Fraction(0))
    box = sum(widths.values(), Fraction(0))
    witness = max(abs(theta - relational_lower), abs(theta - relational_upper))
    if box == 0:
        if witness == 0:
            return Fraction(0)
        raise ValueError("INVALID_P13_ZERO_BOX_INCONSISTENT")
    return min(box, witness) / box


def aggregate_task_site_ratios(
    direction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_site: dict[str, dict[int, Fraction]] = {}
    for row in direction_rows:
        site_id = row.get("site_row_id")
        ordinal = row.get("direction_ordinal")
        ratio_pair = row.get("contraction_ratio")
        if not isinstance(site_id, str) or not site_id:
            raise ValueError("P13 site ID is malformed")
        if type(ordinal) is not int or ordinal not in range(8):
            raise ValueError("P13 direction ordinal is malformed")
        if (
            not isinstance(ratio_pair, (list, tuple)) or len(ratio_pair) != 2
            or any(type(value) is not int for value in ratio_pair)
        ):
            raise ValueError("P13 contraction ratio is malformed")
        ratio = Fraction(*ratio_pair)
        if ratio < 0 or ratio > 1:
            raise ValueError("P13 contraction ratio is outside [0,1]")
        if site_id in by_site and ordinal in by_site[site_id]:
            raise ValueError("P13 duplicate site-direction record")
        by_site.setdefault(site_id, {})[ordinal] = ratio
    if len(by_site) != 108:
        raise ValueError(f"P13 requires 108 sites, got {len(by_site)}")
    site_values = []
    for site_id in sorted(by_site):
        panel = by_site[site_id]
        if set(panel) != set(range(8)):
            raise ValueError("P13 site direction panel is incomplete")
        ratio = max(panel.values())
        site_values.append({
            "site_row_id": site_id,
            "contraction_ratio": [ratio.numerator, ratio.denominator],
        })
    ordered = sorted(Fraction(*row["contraction_ratio"]) for row in site_values)
    median = (ordered[53] + ordered[54]) / 2
    p90 = ordered[97]
    passed = median <= Fraction(1, 5) and p90 <= Fraction(1, 2)
    return {
        "site_values": site_values,
        "site_values_sha256": sha256_canonical(site_values),
        "median_ratio": [median.numerator, median.denominator],
        "p90_ratio": [p90.numerator, p90.denominator],
        "terminal_state": "PASS" if passed else "P13_FAIL",
    }

