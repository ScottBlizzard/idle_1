from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import gmpy2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import affine_control_jet, add_jet, mul_jet
from green_bridge_v400_mpfr import mpfr_context
from green_bridge_v400_transformer_ops import (
    affine_map_jets, attention_head_jets, contrast_jet, gelu_erf_jet,
    gelu_new_jet, layernorm_jets, softmax_jets,
)


P = 256


def _affine(base, direction=1.0, lower=0.0, upper=0.0):
    return affine_control_jet(Interval.point(base, P), Interval.point(direction, P),
                              Interval.from_bounds(lower, upper, P))


def test_gelu_new_value_first_second_derivatives():
    x = _affine(0.0)
    y = gelu_new_jet(x, kappa=math.sqrt(2 / math.pi), lam=0.044715)
    assert y.value.contains(0.0)
    assert y.first.contains(0.5)
    assert y.second.contains(math.sqrt(2 / math.pi))


def test_exact_erf_gelu_when_selected():
    x = _affine(0.0)
    y = gelu_erf_jet(x)
    assert y.value.contains(0.0)
    assert y.first.contains(0.5)
    with mpfr_context(P):
        exact = gmpy2.sqrt(gmpy2.mpfr(2) / gmpy2.const_pi())
    assert y.second.lower <= exact <= y.second.upper


def test_layernorm_derivative_recurrence_against_symbolic_fixture():
    values = [_affine(1.0, 1.0), _affine(-1.0, 0.0)]
    normalized = layernorm_jets(values, epsilon=1e-5)
    assert normalized[0].value.lower > 0
    assert normalized[1].value.upper < 0
    assert normalized[0].first.lower <= 0 <= normalized[0].first.upper or normalized[0].first.upper > 0


def test_layernorm_shift_invariance_shared_dag():
    values = [_affine(1.0, 1.0), _affine(-2.0, 1.0), _affine(0.5, 1.0)]
    normalized = layernorm_jets(values, epsilon=1e-5)
    assert all(row.first.contains(0.0) for row in normalized)
    assert all(row.second.contains(0.0) for row in normalized)


def test_layernorm_positive_epsilon_margin():
    values = [_affine(2.0, 0.0), _affine(2.0, 0.0)]
    normalized = layernorm_jets(values, epsilon=1e-5)
    assert all(row.value.contains(0.0) for row in normalized)


def test_layernorm_zero_epsilon_rejected():
    with pytest.raises(ValueError):
        layernorm_jets([_affine(1.0), _affine(2.0)], epsilon=0.0)


def test_softmax_shift_invariance_shared_dag():
    scores = [_affine(1.0, 1.0), _affine(-1.0, 1.0), _affine(0.5, 1.0)]
    weights = softmax_jets(scores, pivot=0)
    assert all(weight.first.contains(0.0) for weight in weights)
    assert all(weight.second.contains(0.0) for weight in weights)


def test_softmax_fixed_pivot_extreme_scores():
    scores = [_affine(1000.0, 0.0), _affine(-1000.0, 0.0)]
    weights = softmax_jets(scores, pivot=0)
    assert weights[0].value.lower > 0.999
    assert weights[1].value.upper < 0.001


def test_softmax_first_second_derivative_identity():
    scores = [_affine(0.0, 1.0), _affine(0.0, -1.0)]
    weights = softmax_jets(scores, pivot=0)
    assert weights[0].value.contains(0.5)
    assert weights[0].first.contains(0.5)
    assert weights[0].second.contains(0.0)


def test_causal_mask_omits_masked_keys_exactly():
    queries = [[_affine(1.0, 0.0)], [_affine(1.0, 0.0)]]
    keys = [[_affine(1.0, 0.0)], [_affine(1.0, 0.0)]]
    values = [[_affine(2.0, 0.0)], [_affine(100.0, 0.0)]]
    output = attention_head_jets(queries, keys, values, causal=True)
    assert output[0][0].value.contains(2.0)
    assert output[0][0].value.upper < 3.0


def test_attention_two_token_one_head_analytic_fixture():
    queries = [[_affine(0.0, 0.0)], [_affine(0.0, 0.0)]]
    keys = [[_affine(0.0, 0.0)], [_affine(0.0, 0.0)]]
    values = [[_affine(0.0, 0.0)], [_affine(2.0, 0.0)]]
    output = attention_head_jets(queries, keys, values, causal=True)
    assert output[1][0].value.contains(1.0)


def test_attention_value_derivative_recurrence():
    queries = [[_affine(0.0, 0.0)], [_affine(0.0, 0.0)]]
    keys = [[_affine(0.0, 0.0)], [_affine(0.0, 0.0)]]
    values = [[_affine(0.0, 1.0)], [_affine(2.0, 3.0)]]
    output = attention_head_jets(queries, keys, values, causal=True)
    assert output[1][0].first.contains(2.0)


def test_residual_mlp_tiny_block_fixture():
    residual = [_affine(1.0, 1.0), _affine(-1.0, 0.0)]
    hidden = affine_map_jets([[1.0, -1.0]], residual, [0.0])
    activated = [gelu_new_jet(hidden[0], kappa=math.sqrt(2 / math.pi), lam=0.044715)]
    projected = affine_map_jets([[0.5], [-0.5]], activated, [0.0, 0.0])
    output = [add_jet(a, b) for a, b in zip(residual, projected)]
    assert len(output) == 2
    assert all(value.value.lower <= value.value.upper for value in output)


def test_final_layernorm_unembed_contrast_fixture():
    residual = [_affine(1.0, 0.0), _affine(-1.0, 0.0)]
    normalized = layernorm_jets(residual, epsilon=1e-5)
    logits = affine_map_jets([[1.0, 0.0], [0.0, 1.0]], normalized)
    contrast = contrast_jet(logits, [1.0, -1.0])
    assert contrast.value.lower > 0
