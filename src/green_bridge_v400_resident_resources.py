"""Exact counted-arithmetic formulas for the native resident GPT-2 cell."""
from __future__ import annotations

from dataclasses import dataclass


PRIMITIVE_TAXONOMY = {
    "schema_version": "green-v400-directed-enclosure-arithmetic-v1",
    "included": ["mpfr_add/sub", "mpfr_mul", "mpfr_div/div_ui", "mpfr_sqrt",
                 "mpfr_tanh", "mpfr_exp"],
    "excluded": ["mpfr_set/copy", "mpfr_neg", "comparison", "serialization",
                 "synthetic fixture initialization div_2ui"],
}


def affine_jet2(input_width: int, output_width: int) -> int:
    return output_width * (12 * input_width - 4)


def layer_norm_jet2(width: int) -> int:
    return 122 * width + 198


def gelu_jet2(width: int) -> int:
    return 440 * width


def causal_attention_head_jet2(sequence_length: int, head_width: int) -> int:
    return 136 * sequence_length * head_width + 166 * sequence_length - 6 * head_width + 8


def gpt2_tail_jet2(d_model: int, d_mlp: int, sequence_length: int,
                   n_heads: int, d_head: int) -> int:
    return (
        3 * layer_norm_jet2(d_model)
        + 4 * affine_jet2(d_model, d_model)
        + n_heads * causal_attention_head_jet2(sequence_length, d_head)
        + affine_jet2(d_model, d_mlp)
        + gelu_jet2(d_mlp)
        + affine_jet2(d_mlp, d_model)
        + 2 * 6 * d_model
        + affine_jet2(d_model, 1)
    )


def gpt2_joint_witness_cell_jet2(d_model: int, d_mlp: int,
                                 sequence_length: int, n_heads: int,
                                 d_head: int, selected_gates: int) -> int:
    if n_heads * d_head != d_model:
        raise ValueError("attention dimensions do not close")
    condition = (
        2 * layer_norm_jet2(d_model)
        + 2 * affine_jet2(d_model, selected_gates)
        + 2 * gelu_jet2(selected_gates)
        + 6 * selected_gates
        + affine_jet2(selected_gates, d_model)
        + 6 * d_model
        + 2 * gpt2_tail_jet2(d_model, d_mlp, sequence_length, n_heads, d_head)
    )
    return 2 * condition + 18
