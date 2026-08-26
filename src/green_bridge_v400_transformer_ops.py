"""Certified smooth Transformer primitives for the GREEN v4 interval graph."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import gmpy2

from green_bridge_v400_interval import (
    Interval, erf_interval, exp_interval, inv_sqrt_interval, sqrt_interval,
    tanh_interval,
)
from green_bridge_v400_interval_jet import (
    CertifiedScalarPrimitive, Jet2, add_jet, compose_jet, constant_jet,
    mul_jet, reciprocal_jet, square_jet, sub_jet,
)
from green_bridge_v400_mpfr import ROUND_DOWN, ROUND_UP, mpfr_context


def _zero(precision: int) -> Jet2:
    return constant_jet(Interval.point(0, precision))


def _one(precision: int) -> Jet2:
    return constant_jet(Interval.point(1, precision))


def _constant(value, precision: int) -> Jet2:
    return constant_jet(Interval.point(value, precision))


def sum_jets(values: Iterable[Jet2], *, precision_bits: int) -> Jet2:
    level = list(values)
    if not level:
        return _zero(precision_bits)
    if any(value.precision_bits != precision_bits for value in level):
        raise ValueError("pairwise reduction precision mismatch")
    # Frozen balanced binary tree: deterministic and materially tighter than a
    # long left fold for Transformer affine and attention reductions.
    while len(level) > 1:
        next_level = [add_jet(level[index], level[index + 1])
                      for index in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def scale_jet(value: Jet2, scalar) -> Jet2:
    return mul_jet(value, _constant(scalar, value.precision_bits))


def exp_primitive() -> CertifiedScalarPrimitive:
    return CertifiedScalarPrimitive("exp", exp_interval, exp_interval, exp_interval)


def sqrt_primitive() -> CertifiedScalarPrimitive:
    def first(x: Interval):
        return Interval.point(0.5, x.precision_bits) * inv_sqrt_interval(x)
    def second(x: Interval):
        return (-Interval.point(0.25, x.precision_bits)
                * inv_sqrt_interval(x) / x)
    return CertifiedScalarPrimitive("sqrt", sqrt_interval, first, second)


def inv_sqrt_primitive() -> CertifiedScalarPrimitive:
    def first(x: Interval):
        return -Interval.point(0.5, x.precision_bits) * inv_sqrt_interval(x) / x
    def second(x: Interval):
        return Interval.point(0.75, x.precision_bits) * inv_sqrt_interval(x) / x.square()
    return CertifiedScalarPrimitive("inv_sqrt", inv_sqrt_interval, first, second)


def tanh_primitive() -> CertifiedScalarPrimitive:
    def first(x: Interval):
        t = tanh_interval(x)
        return Interval.point(1, x.precision_bits) - t.square()
    def second(x: Interval):
        t = tanh_interval(x)
        one = Interval.point(1, x.precision_bits)
        return -Interval.point(2, x.precision_bits) * t * (one - t.square())
    return CertifiedScalarPrimitive("tanh", tanh_interval, first, second)


def _pi_interval(precision: int) -> Interval:
    with mpfr_context(precision, ROUND_DOWN):
        lower = gmpy2.const_pi()
    with mpfr_context(precision, ROUND_UP):
        upper = gmpy2.const_pi()
    return Interval(lower, upper, precision)


def erf_primitive() -> CertifiedScalarPrimitive:
    def first(x: Interval):
        precision = x.precision_bits
        pi = _pi_interval(precision)
        coefficient = Interval.point(2, precision) * inv_sqrt_interval(pi)
        return coefficient * exp_interval(-x.square())
    def second(x: Interval):
        return -Interval.point(2, x.precision_bits) * x * first(x)
    return CertifiedScalarPrimitive("erf", erf_interval, first, second)


def sigmoid_jet(x: Jet2) -> Jet2:
    # sigmoid(x) = 1/(1+exp(-x)); using the exact shared expression keeps its
    # first and second derivatives sound through generic jet recurrences.
    negative = Jet2(-x.value, -x.first, -x.second)
    denominator = add_jet(_one(x.precision_bits), compose_jet(negative, exp_primitive()))
    return reciprocal_jet(denominator)


def gelu_new_jet(x: Jet2, *, kappa: float, lam: float) -> Jet2:
    """Frozen tanh-GELU expression with bit-exact runtime coefficients."""
    precision = x.precision_bits
    x2, x3 = mul_jet(x, x), mul_jet(mul_jet(x, x), x)
    u = scale_jet(add_jet(x, scale_jet(x3, lam)), kappa)
    tanh_u = compose_jet(u, tanh_primitive())
    return scale_jet(mul_jet(x, add_jet(_one(precision), tanh_u)), 0.5)


def gelu_erf_jet(x: Jet2) -> Jet2:
    precision = x.precision_bits
    pi = _pi_interval(precision)
    two = Interval.point(2, precision)
    sqrt_two = (two * Interval.point(1, precision))
    # A true sqrt(2) interval, rather than a decimal approximation.
    from green_bridge_v400_interval import sqrt_interval
    divisor = constant_jet(sqrt_interval(sqrt_two))
    erf_term = compose_jet(mul_jet(x, reciprocal_jet(divisor)), erf_primitive())
    return scale_jet(mul_jet(x, add_jet(_one(precision), erf_term)), 0.5)


def affine_map_jets(matrix: Sequence[Sequence], values: Sequence[Jet2],
                    bias: Sequence | None = None) -> list[Jet2]:
    if not values:
        raise ValueError("affine map needs at least one input")
    precision = values[0].precision_bits
    result = []
    for index, row in enumerate(matrix):
        if len(row) != len(values):
            raise ValueError("affine matrix width mismatch")
        terms = [scale_jet(value, weight) for weight, value in zip(row, values)]
        output = sum_jets(terms, precision_bits=precision)
        if bias is not None:
            output = add_jet(output, _constant(bias[index], precision))
        result.append(output)
    return result


def layernorm_jets(values: Sequence[Jet2], *, epsilon: float,
                   gamma: Sequence | None = None,
                   beta: Sequence | None = None) -> list[Jet2]:
    if not values:
        raise ValueError("LayerNorm requires a nonempty vector")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("LayerNorm epsilon must be finite and positive")
    precision, dimension = values[0].precision_bits, len(values)
    reciprocal_dimension = gmpy2.mpq(1, dimension)
    mean = scale_jet(sum_jets(values, precision_bits=precision), reciprocal_dimension)
    centered = [sub_jet(value, mean) for value in values]
    variance = scale_jet(
        sum_jets((square_jet(value) for value in centered), precision_bits=precision),
        reciprocal_dimension,
    )
    q = add_jet(variance, _constant(epsilon, precision))
    scale = compose_jet(q, inv_sqrt_primitive())
    gamma = [1.0] * dimension if gamma is None else list(gamma)
    beta = [0.0] * dimension if beta is None else list(beta)
    if len(gamma) != dimension or len(beta) != dimension:
        raise ValueError("LayerNorm affine parameter shape mismatch")
    return [add_jet(scale_jet(mul_jet(value, scale), g), _constant(b, precision))
            for value, g, b in zip(centered, gamma, beta)]


def dot_jets(left: Sequence[Jet2], right: Sequence[Jet2]) -> Jet2:
    if len(left) != len(right) or not left:
        raise ValueError("dot-product shape mismatch")
    return sum_jets((mul_jet(a, b) for a, b in zip(left, right)),
                    precision_bits=left[0].precision_bits)


def softmax_jets(scores: Sequence[Jet2], *, pivot: int = 0) -> list[Jet2]:
    if not scores or not 0 <= pivot < len(scores):
        raise ValueError("invalid softmax pivot")
    precision = scores[0].precision_bits
    shifted = [(_zero(precision) if index == pivot
                else sub_jet(score, scores[pivot]))
               for index, score in enumerate(scores)]
    exponentials = [(_one(precision) if index == pivot
                     else compose_jet(score, exp_primitive()))
                    for index, score in enumerate(shifted)]
    denominator = sum_jets(exponentials, precision_bits=precision)
    inverse = reciprocal_jet(denominator)
    return [mul_jet(value, inverse) for value in exponentials]


def attention_head_jets(queries: Sequence[Sequence[Jet2]],
                        keys: Sequence[Sequence[Jet2]],
                        values: Sequence[Sequence[Jet2]], *,
                        causal: bool = True) -> list[list[Jet2]]:
    if not queries or not keys or len(keys) != len(values):
        raise ValueError("attention shape mismatch")
    head_dim = len(queries[0])
    scaling = 1.0 / math.sqrt(head_dim)
    result = []
    for query_index, query in enumerate(queries):
        allowed = list(range(min(query_index + 1, len(keys)))) if causal else list(range(len(keys)))
        scores = [scale_jet(dot_jets(query, keys[index]), scaling) for index in allowed]
        weights = softmax_jets(scores, pivot=0)
        output = []
        for coordinate in range(len(values[0])):
            output.append(sum_jets(
                (mul_jet(weight, values[index][coordinate]) for weight, index in zip(weights, allowed)),
                precision_bits=query[0].precision_bits,
            ))
        result.append(output)
    return result


def contrast_jet(values: Sequence[Jet2], contrast: Sequence) -> Jet2:
    if len(values) != len(contrast) or not values:
        raise ValueError("contrast shape mismatch")
    return sum_jets((scale_jet(value, coefficient)
                     for value, coefficient in zip(values, contrast)),
                    precision_bits=values[0].precision_bits)


OPERATION_COVERAGE = {
    "affine": affine_map_jets,
    "gelu_new": gelu_new_jet,
    "gelu_erf": gelu_erf_jet,
    "layernorm": layernorm_jets,
    "softmax": softmax_jets,
    "attention": attention_head_jets,
    "contrast": contrast_jet,
    "sqrt": sqrt_primitive,
    "inv_sqrt": inv_sqrt_primitive,
}
