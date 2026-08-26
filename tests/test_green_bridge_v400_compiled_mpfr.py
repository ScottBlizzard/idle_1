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
from green_bridge_v400_transformer_ops import (
    affine_map_jets, attention_head_jets, gelu_new_jet, layernorm_jets,
)


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


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_gelu_new_jet_is_bit_identical(precision):
    backend = _backend()
    seed = Interval.from_bounds(-0.35, 0.42, precision)
    value = Jet2(exp_interval(seed),
                 Interval.from_bounds(-0.8, 1.1, precision),
                 tanh_interval(Interval.from_bounds(-0.4, 0.7, precision)))
    kappa = np.float32(np.sqrt(2.0 / np.pi))
    lam = np.float32(0.044715)
    expected = gelu_new_jet(value, kappa=float(kappa), lam=float(lam))
    actual = backend.gelu_new_jet2(value, kappa, lam)
    for component in ("value", "first", "second"):
        interval = getattr(expected, component)
        assert backend.exact_fraction(actual[component]["lower"]) == _fraction(interval.lower)
        assert backend.exact_fraction(actual[component]["upper"]) == _fraction(interval.upper)


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_layer_norm_jet_is_bit_identical(precision):
    backend = _backend()
    values = [
        _jet(-0.25 + index/7, 2.0**(-9-index), 0.1-index/13,
             -0.2+index/11, precision)
        for index in range(5)
    ]
    gamma = np.asarray([1.0, 0.5, -0.25, 1.25, 0.75], dtype="<f4")
    beta = np.asarray([0.0, -0.1, 0.2, 0.05, -0.075], dtype="<f4")
    epsilon = np.float32(1e-5)
    expected = layernorm_jets(values, epsilon=float(epsilon), gamma=gamma, beta=beta)
    actual = backend.layer_norm_jet2(values, epsilon, gamma, beta)["outputs"]
    for expected_jet, actual_jet in zip(expected, actual):
        for component in ("value", "first", "second"):
            interval = getattr(expected_jet, component)
            assert backend.exact_fraction(actual_jet[component]["lower"]) == _fraction(interval.lower)
            assert backend.exact_fraction(actual_jet[component]["upper"]) == _fraction(interval.upper)


def test_compiled_layer_norm_rejects_nonpositive_variance():
    backend = _backend()
    values = [_jet(1.0, 0.0, 0.0, 0.0, 384) for _ in range(3)]
    with pytest.raises(RuntimeError, match="status 5"):
        backend.layer_norm_jet2(
            values, np.float32(0.0), np.ones(3, dtype="<f4"), np.zeros(3, dtype="<f4")
        )


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_nonlinear_benchmarks_are_live(precision):
    backend = _backend()
    gelu = backend.benchmark_gelu(precision, 8)
    layer_norm = backend.benchmark_layer_norm(precision, 8, 2)
    attention = backend.benchmark_causal_attention(precision, 3, 2, 4)
    assert gelu["elapsed_seconds"] > 0 and gelu["jets_per_second"] > 0
    assert layer_norm["elapsed_seconds"] > 0 and layer_norm["vectors_per_second"] > 0
    assert attention["elapsed_seconds"] > 0 and attention["head_evaluations"] == 2
    assert len(gelu["checksum"]) == len(layer_norm["checksum"]) == len(attention["checksum"]) == 16


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_causal_attention_final_head_is_bit_identical(precision):
    backend = _backend()
    query = [_jet(-0.15 + coordinate/9, 2.0**(-10-coordinate),
                  0.2-coordinate/11, -0.1+coordinate/13, precision)
             for coordinate in range(3)]
    keys = [[_jet(-0.3 + token/8 + coordinate/17, 2.0**(-11-token-coordinate),
                  0.1+token/19-coordinate/23, -0.2+coordinate/29, precision)
             for coordinate in range(3)] for token in range(3)]
    values = [[_jet(0.25-token/7+coordinate/13, 2.0**(-12-token-coordinate),
                    -0.15+token/17+coordinate/31, 0.05-token/37, precision)
               for coordinate in range(3)] for token in range(3)]
    expected = attention_head_jets([query, query, query], keys, values, causal=True)[-1]
    actual = backend.causal_attention_final_head_jet2(query, keys, values, pivot=0)["outputs"]
    for expected_jet, actual_jet in zip(expected, actual):
        for component in ("value", "first", "second"):
            interval = getattr(expected_jet, component)
            assert backend.exact_fraction(actual_jet[component]["lower"]) == _fraction(interval.lower)
            assert backend.exact_fraction(actual_jet[component]["upper"]) == _fraction(interval.upper)
