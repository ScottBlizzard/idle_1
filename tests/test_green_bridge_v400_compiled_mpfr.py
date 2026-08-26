from __future__ import annotations

from fractions import Fraction
import os
from pathlib import Path
import sys

import numpy as np
import gmpy2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import (
    Interval, exp_interval, inv_sqrt_interval, sqrt_interval, tanh_interval,
)
from green_bridge_v400_interval_jet import Jet2
from green_bridge_v400_transformer_ops import affine_map_jets


def _backend():
    path = os.environ.get("GREEN_V400_MPFR_BACKEND")
    if not path:
        pytest.skip("compiled MPFR backend is not configured")
    return CompiledMPFRBackend(Path(path))


def _jet(center: float, radius: float, first: float, second: float, precision: int):
    return Jet2(
        Interval.from_bounds(center - radius, center + radius, precision),
        Interval.from_bounds(first - radius / 2, first + radius / 2, precision),
        Interval.from_bounds(second - radius / 4, second + radius / 4, precision),
    )


def _fraction(value) -> Fraction:
    rational = gmpy2.mpq(value)
    return Fraction(int(rational.numerator), int(rational.denominator))


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_affine_jet2_is_bit_identical_to_python_reference(precision):
    backend = _backend()
    weights = np.asarray([0.5, -1.25, 0.0, 2.0, -0.03125], dtype="<f4")
    bias = np.float32(0.125)
    values = [
        _jet(0.25, 2.0**(-10-index), -0.5 + index/8, 0.75-index/16, precision)
        for index in range(weights.size)
    ]
    reference = affine_map_jets([weights], values, [bias])[0]
    compiled = backend.affine_jet2(weights, bias, values, precision)
    for component in ("value", "first", "second"):
        interval = getattr(reference, component)
        assert backend.exact_fraction(compiled[component]["lower"]) == _fraction(interval.lower)
        assert backend.exact_fraction(compiled[component]["upper"]) == _fraction(interval.upper)


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_affine_accepts_non_ieee_mpfr_intermediates(precision):
    backend = _backend()
    seeds = [Interval.from_bounds(-0.3 + index/20, 0.2 + index/17, precision)
             for index in range(4)]
    values = [Jet2(exp_interval(seed), seed, exp_interval(-seed)) for seed in seeds]
    weights = np.asarray([0.1, -0.2, 0.3, -0.4], dtype="<f4")
    reference = affine_map_jets([weights], values, [np.float32(-0.0625)])[0]
    compiled = backend.affine_jet2(weights, np.float32(-0.0625), values, precision)
    for component in ("value", "first", "second"):
        interval = getattr(reference, component)
        assert backend.exact_fraction(compiled[component]["lower"]) == _fraction(interval.lower)
        assert backend.exact_fraction(compiled[component]["upper"]) == _fraction(interval.upper)


def test_compiled_affine_benchmark_is_deterministic():
    backend = _backend()
    first = backend.benchmark_affine_layer(384, 32, 8)
    second = backend.benchmark_affine_layer(384, 32, 8)
    assert first["checksum"] == second["checksum"]
    assert first["coefficient_terms"] == 256
    assert first["directed_mpfr_primitives"] == 3072
    assert first["elapsed_seconds"] > 0


@pytest.mark.parametrize("precision", [384, 512])
@pytest.mark.parametrize("operation,bounds,reference", [
    ("exp", (-0.7, 0.9), exp_interval),
    ("tanh", (-1.3, 0.4), tanh_interval),
    ("sqrt", (0.125, 3.75), sqrt_interval),
    ("inv_sqrt", (0.125, 3.75), inv_sqrt_interval),
])
def test_compiled_interval_primitives_are_bit_identical(
        precision, operation, bounds, reference):
    backend = _backend()
    interval = Interval.from_bounds(*bounds, precision)
    expected = reference(interval)
    actual = backend.interval_primitive(operation, interval)
    assert backend.exact_fraction(actual["lower"]) == _fraction(expected.lower)
    assert backend.exact_fraction(actual["upper"]) == _fraction(expected.upper)
