from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import sys

import gmpy2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr import (
    ROUND_DOWN, ROUND_UP, assert_precision_nesting, directed_fma,
    directed_pairwise_sum, exact_interval_from_ieee, exact_mpfr_from_ieee,
)


def test_ieee_float32_exact_import():
    value = np.float32(0.1)
    interval = exact_interval_from_ieee(value, precision_bits=256)
    assert interval.lower == interval.upper
    assert gmpy2.mpq(interval.lower) == gmpy2.mpq(*float(value).as_integer_ratio())


def test_ieee_float64_exact_import():
    for value in (float.fromhex("0x1.123456789abcdp-7"),
                  float.fromhex("0x1.123456789abcdp+700")):
        imported = exact_mpfr_from_ieee(value, precision_bits=256)
        assert gmpy2.mpq(imported) == gmpy2.mpq(*value.as_integer_ratio())


def test_ieee_subnormal_exact_import():
    value = float.fromhex("0x0.0000000000001p-1022")
    imported = exact_mpfr_from_ieee(value, precision_bits=256)
    assert imported > 0
    assert gmpy2.mpq(imported) == gmpy2.mpq(*value.as_integer_ratio())


def test_nan_inf_rejected():
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            exact_interval_from_ieee(value, precision_bits=256)


def test_directed_add_sub_contains_exact_rational():
    a = Interval.from_bounds("0.1", "0.1", 80)
    b = Interval.from_bounds("0.2", "0.2", 80)
    result = (a + b) - a
    exact = gmpy2.mpq(1, 5)
    assert gmpy2.mpq(result.lower) <= exact <= gmpy2.mpq(result.upper)


def test_directed_mul_all_sign_cases():
    cases = [((-2, -1), (3, 5), (-10, -3)),
             ((-2, 4), (-3, 5), (-12, 20)),
             ((1, 2), (3, 4), (3, 8))]
    for left, right, expected in cases:
        product = Interval.from_bounds(*left) * Interval.from_bounds(*right)
        assert product.lower <= expected[0]
        assert product.upper >= expected[1]


def test_reciprocal_rejects_zero_crossing():
    with pytest.raises(ZeroDivisionError):
        Interval.from_bounds(-1, 1).reciprocal()


def test_square_interval_crossing_zero():
    squared = Interval.from_bounds(-2, 3).square()
    assert squared.lower == 0
    assert squared.upper >= 9


def test_fma_pairwise_sum_outward():
    exact = gmpy2.mpq(1, 10) * gmpy2.mpq(1, 5) + gmpy2.mpq(1, 3)
    lower = directed_fma(gmpy2.mpq(1, 10), gmpy2.mpq(1, 5), gmpy2.mpq(1, 3),
                         precision_bits=32, rounding=ROUND_DOWN)
    upper = directed_fma(gmpy2.mpq(1, 10), gmpy2.mpq(1, 5), gmpy2.mpq(1, 3),
                         precision_bits=32, rounding=ROUND_UP)
    assert gmpy2.mpq(lower) <= exact <= gmpy2.mpq(upper)
    terms = [gmpy2.mpq(1, 3), gmpy2.mpq(-1, 7), gmpy2.mpq(1, 11)]
    exact_sum = sum(terms)
    lower_sum = directed_pairwise_sum(terms, precision_bits=32, rounding=ROUND_DOWN)
    upper_sum = directed_pairwise_sum(terms, precision_bits=32, rounding=ROUND_UP)
    assert gmpy2.mpq(lower_sum) <= exact_sum <= gmpy2.mpq(upper_sum)


def test_mpfr_384_512_nesting_scalar_primitives():
    low = Interval.from_bounds("0.123456789", "0.987654321", 384)
    high = Interval.from_bounds("0.123456789", "0.987654321", 512)
    assert_precision_nesting(low, high)
