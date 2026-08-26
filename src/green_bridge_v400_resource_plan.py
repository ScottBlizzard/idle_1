"""Shape-derived, outcome-blind resource accounting for GREEN v4 programs."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TailShape:
    d_model: int
    d_mlp: int
    n_heads: int
    d_head: int
    sequence_length: int
    selected_gates: int
    contrast_width: int

    def __post_init__(self):
        values = asdict(self)
        if any(not isinstance(value, int) or value <= 0 for value in values.values()):
            raise ValueError("tail resource dimensions must be positive integers")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("attention head dimensions do not equal model width")


@dataclass(frozen=True)
class ResourcePlan:
    coefficient_terms_per_branch_cell: int
    coefficient_terms_four_branch_cell: int
    dense_arithmetic_mpfr_ops_lower_bound_per_precision_cell: int
    minimum_cells: int
    minimum_dense_arithmetic_mpfr_ops_lower_bound_per_precision_row_radius: int
    frozen_mpfr_ops_cap_per_row: int
    cap_infeasibility_proved: bool
    proof_assumptions: dict
    formula_version: str = "green-v400-gpt2-tail-resource-v2-lower-bound"

    def to_dict(self) -> dict:
        return asdict(self)


def plan_gpt2_tail_resources(shape: TailShape, *, mpfr_ops_per_term_lower: int = 12,
                             minimum_cells: int = 2,
                             frozen_mpfr_ops_cap_per_row: int = 100_000_000) -> ResourcePlan:
    if (mpfr_ops_per_term_lower != 12 or minimum_cells != 2
            or frozen_mpfr_ops_cap_per_row <= 0):
        raise ValueError("invalid resource accounting policy")
    # Block-10 computes only selected gates; block-11 remains dense at the
    # dependent final token.  Attention includes final-query score/value work.
    block10 = 2 * shape.d_model * shape.selected_gates
    block11_qkv = 3 * shape.d_model * shape.d_model
    block11_o = shape.d_model * shape.d_model
    block11_mlp = 2 * shape.d_model * shape.d_mlp
    attention = 2 * shape.d_model * shape.sequence_length
    contrast = shape.d_model  # 100-way contrast is precomposed and proved.
    per_branch = block10 + block11_qkv + block11_o + block11_mlp + attention + contrast
    four_branch = 4 * per_branch
    per_cell_ops = four_branch * mpfr_ops_per_term_lower
    per_radius = per_cell_ops * minimum_cells
    return ResourcePlan(
        per_branch, four_branch, per_cell_ops, minimum_cells, per_radius,
        frozen_mpfr_ops_cap_per_row, per_radius > frozen_mpfr_ops_cap_per_row,
        {"mpfr_ops_per_dense_coefficient_term_lower_bound": 12,
         "mandatory_initial_cells": 2,
         "scope": "one precision; dense coefficient arithmetic only",
         "cap_scope": "the cap covers the entire row across radii; one mandatory radius lower bound already exceeds it"},
    )


def gpt2_small_tail_shape(sequence_length: int) -> TailShape:
    return TailShape(768, 3072, 12, 64, int(sequence_length), 10, 100)
