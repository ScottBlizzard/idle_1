from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_p13 import (
    aggregate_task_site_ratios,
    contraction_ratio,
    select_gt_qualification,
    select_ioi_qualification,
)


def test_ioi_selection_is_exact_12_prompts_108_sites_864_directions():
    rows = [
        {
            "row_id": f"ioi-{index:03d}",
            "role": "unused_reserve",
            "template_id": "reserve_0",
        }
        for index in range(64)
    ]
    manifest = select_ioi_qualification(rows)
    assert manifest["prompt_count"] == 12
    assert manifest["site_count"] == 108
    assert manifest["direction_count"] == 864
    assert len(manifest["sites"]) == 108


def test_ioi_selection_rejects_wrong_reserve_count():
    rows = [
        {"row_id": f"ioi-{index}", "role": "unused_reserve", "template_id": "reserve_0"}
        for index in range(63)
    ]
    with pytest.raises(ValueError, match="exactly 64"):
        select_ioi_qualification(rows)


def test_gt_selection_covers_all_twelve_strata():
    rows = []
    counter = 0
    for noun in ("ceremony", "committee", "federation", "workshop"):
        for century in (10, 18, 20):
            for distance in ("near", "far"):
                for orientation in ("up", "down"):
                    rows.append({
                        "row_id": f"gt-{counter:03d}",
                        "role": "unused_reserve",
                        "noun": noun,
                        "century": century,
                        "distance_bin": distance,
                        "orientation": orientation,
                    })
                    counter += 1
    manifest = select_gt_qualification(rows)
    assert manifest["prompt_count"] == 12
    selected = [row for row in rows if row["row_id"] in manifest["selected_prompt_row_ids"]]
    assert len({(row["century"], row["distance_bin"], row["orientation"]) for row in selected}) == 12


def test_contraction_uses_joint_witness_minimum():
    centers = {
        "PAT_J": Fraction(5), "PAT_B": Fraction(2),
        "TAR_J": Fraction(4), "TAR_B": Fraction(2),
    }
    intervals = {
        branch: (center - 1, center + 1) for branch, center in centers.items()
    }
    # theta=1, box=4, witness radius=1/2, so rho=(1/2)/4=1/8.
    assert contraction_ratio(
        branch_centers=centers,
        branch_intervals=intervals,
        relational_interval=(Fraction(1, 2), Fraction(3, 2)),
    ) == Fraction(1, 8)


def test_contraction_zero_box_boundaries():
    centers = {branch: Fraction(0) for branch in ("PAT_J", "PAT_B", "TAR_J", "TAR_B")}
    intervals = {branch: (Fraction(0), Fraction(0)) for branch in centers}
    assert contraction_ratio(
        branch_centers=centers, branch_intervals=intervals,
        relational_interval=(Fraction(0), Fraction(0)),
    ) == 0
    with pytest.raises(ValueError, match="ZERO_BOX"):
        contraction_ratio(
            branch_centers=centers, branch_intervals=intervals,
            relational_interval=(Fraction(0), Fraction(1)),
        )


def task_rows(values: list[Fraction]) -> list[dict]:
    assert len(values) == 108
    return [
        {
            "site_row_id": f"site-{site:03d}",
            "direction_ordinal": ordinal,
            "contraction_ratio": [values[site].numerator, values[site].denominator],
        }
        for site in range(108)
        for ordinal in range(8)
    ]


def test_p13_equality_boundaries_pass():
    # 97 values at 1/5 and 11 at 1/2: median=1/5, nearest-rank p90=1/2.
    values = [Fraction(1, 5)] * 97 + [Fraction(1, 2)] * 11
    result = aggregate_task_site_ratios(task_rows(values))
    assert result["median_ratio"] == [1, 5]
    assert result["p90_ratio"] == [1, 2]
    assert result["terminal_state"] == "PASS"


def test_p13_p90_strictly_above_boundary_fails():
    values = [Fraction(1, 5)] * 97 + [Fraction(501, 1000)] * 11
    assert aggregate_task_site_ratios(task_rows(values))["terminal_state"] == "P13_FAIL"


def test_p13_requires_every_direction():
    values = [Fraction(1, 5)] * 108
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_task_site_ratios(task_rows(values)[:-1])

