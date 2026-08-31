from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_directions import (
    build_row_binding,
    generate_site_directions,
    qualification_master_seed,
    tensor_sha256,
)


def test_public_qualification_directions_are_deterministic():
    first = generate_site_directions("site-a", "GREEN_V410_P13_IOI_GREEN_PANEL")
    second = generate_site_directions("site-a", "GREEN_V410_P13_IOI_GREEN_PANEL")
    assert np.array_equal(first, second)
    assert first.dtype == np.dtype("<f4")
    assert first.shape == (8, 768)


def test_tasks_have_distinct_master_seeds_and_payloads():
    ioi_seed = qualification_master_seed("GREEN_V410_P13_IOI_GREEN_PANEL")
    gt_seed = qualification_master_seed("GREEN_V410_P13_GT_GREEN_PANEL")
    assert ioi_seed != gt_seed
    ioi = generate_site_directions("same-site", "GREEN_V410_P13_IOI_GREEN_PANEL")
    gt = generate_site_directions("same-site", "GREEN_V410_P13_GT_GREEN_PANEL")
    assert not np.array_equal(ioi, gt)


def test_float32_payload_norm_and_binding_are_exactly_checked():
    values = generate_site_directions("site-a", "GREEN_V410_P13_IOI_GREEN_PANEL")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    assert np.all(np.abs(norms - 0.001) <= 1e-8)
    binding = build_row_binding(
        task="ioi", site_row_id="site-a",
        direction_domain="GREEN_V410_P13_IOI_GREEN_PANEL", array=values,
    )
    assert binding["tensor_sha256"] == tensor_sha256(values)
    assert len(binding["binding_sha256"]) == 64


def test_unknown_direction_domain_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        qualification_master_seed("chosen-after-results")

