from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_resource_plan import (
    TailShape, gpt2_small_tail_shape, plan_gpt2_tail_resources,
)


def test_gpt2_tail_formula_matches_shape_derived_count():
    plan = plan_gpt2_tail_resources(gpt2_small_tail_shape(12))
    assert plan.coefficient_terms_per_branch_cell == 7_112_448
    assert plan.coefficient_terms_four_branch_cell == 28_449_792
    assert plan.dense_arithmetic_mpfr_ops_lower_bound_per_precision_cell == 341_397_504


def test_frozen_one_hundred_million_op_cap_is_honestly_infeasible():
    plan = plan_gpt2_tail_resources(gpt2_small_tail_shape(12))
    assert plan.minimum_cells == 2
    assert plan.minimum_dense_arithmetic_mpfr_ops_lower_bound_per_precision_row_radius > 600_000_000
    assert plan.frozen_mpfr_ops_cap_per_row == 100_000_000
    assert plan.cap_infeasibility_proved is True


def test_resource_plan_is_monotone_in_sequence_length():
    short = plan_gpt2_tail_resources(gpt2_small_tail_shape(8))
    long = plan_gpt2_tail_resources(gpt2_small_tail_shape(16))
    assert long.coefficient_terms_per_branch_cell > short.coefficient_terms_per_branch_cell
    assert (long.minimum_dense_arithmetic_mpfr_ops_lower_bound_per_precision_row_radius
            > short.minimum_dense_arithmetic_mpfr_ops_lower_bound_per_precision_row_radius)


def test_resource_proof_assumptions_cannot_be_weakened_by_callers():
    with pytest.raises(ValueError, match="resource accounting policy"):
        plan_gpt2_tail_resources(gpt2_small_tail_shape(12), mpfr_ops_per_term_lower=1)
    with pytest.raises(ValueError, match="resource accounting policy"):
        plan_gpt2_tail_resources(gpt2_small_tail_shape(12), minimum_cells=3)


def test_resource_shape_rejects_inconsistent_heads():
    with pytest.raises(ValueError, match="head dimensions"):
        TailShape(768, 3072, 11, 64, 12, 10, 100)
