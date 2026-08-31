from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_causal_cone import build_causal_cone_plan, validate_causal_cone_plan


@pytest.mark.parametrize("layer", [0, 4, 8])
def test_full_cone_reaches_every_later_token_and_final_scalar(layer):
    plan = build_causal_cone_plan(site_layer=layer, patch_position=3, sequence_length=11)
    assert plan["outputs"] == ["PAT_J", "PAT_B", "TAR_J", "TAR_B", "PSI"]
    first = next(stage for stage in plan["stages"]
                 if stage["condition"] == "PAT" and stage["operation"] == "causal_attention")
    assert first["dynamic_token_rows"] == list(range(3, 11))
    assert {stage["block"] for stage in plan["stages"] if stage["block"] is not None} >= set(range(layer + 1, 12))


def test_earlier_patch_is_not_rewritten_as_final_token_patch():
    plan = build_causal_cone_plan(site_layer=8, patch_position=2, sequence_length=9)
    first_ln = next(stage for stage in plan["stages"]
                    if stage["condition"] == "PAT" and stage["operation"] == "ln1")
    assert first_ln["dynamic_token_rows"] == [2]
    assert plan["gate_position"] == 8
    assert plan["patch_position"] == 2


def test_matched_bypass_is_branch_specific_at_layer10():
    plan = build_causal_cone_plan(site_layer=8, patch_position=1, sequence_length=7)
    gate = [stage for stage in plan["stages"] if stage["block"] == 10]
    for condition in ("PAT", "TAR"):
        assert any(stage["condition"] == condition and stage["branch"] == "J"
                   and stage["operation"] == "selected_gate_live" for stage in gate)
        assert any(stage["condition"] == condition and stage["branch"] == "B"
                   and stage["operation"] == "selected_gate_anchor_t0" for stage in gate)


def test_cone_plan_rejects_old_or_downstream_site_layers():
    for layer in (-1, 9, 10):
        with pytest.raises(ValueError, match="0..8"):
            build_causal_cone_plan(site_layer=layer, patch_position=0, sequence_length=3)


def test_plan_hash_detects_dependency_tampering():
    plan = build_causal_cone_plan(site_layer=4, patch_position=2, sequence_length=10)
    changed = deepcopy(plan)
    changed["stages"][0]["dynamic_token_rows"] = [9]
    with pytest.raises(ValueError, match="self hash"):
        validate_causal_cone_plan(changed)

