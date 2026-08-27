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
from green_bridge_v400_final_contrast_fusion import fuse_final_contrast_exact
from green_bridge_v400_resident_resources import gpt2_joint_witness_cell_jet2
from green_bridge_v400_interval import (
    Interval, exp_interval, inv_sqrt_interval, sqrt_interval, tanh_interval,
)
from green_bridge_v400_interval_jet import Jet2, add_jet, constant_jet, sub_jet
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


def _decode_compiled_jet(backend, payload: dict, precision: int) -> Jet2:
    return Jet2(*(
        Interval.from_bounds(
            backend.exact_fraction(payload[component]["lower"]),
            backend.exact_fraction(payload[component]["upper"]), precision,
        )
        for component in ("value", "first", "second")
    ))


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


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_affine_accepts_noncontiguous_weight_column(precision):
    backend = _backend()
    matrix = np.arange(24, dtype="<f4").reshape(6, 4) / np.float32(17)
    weights = matrix[:, 2]
    assert not weights.flags.c_contiguous
    values = [_jet(-0.3 + index/9, 2.0**(-11-index),
                   0.2-index/13, -0.1+index/17, precision)
              for index in range(weights.size)]
    expected = affine_map_jets([weights], values, [np.float32(0.125)])[0]
    actual = backend.affine_jet2(weights, np.float32(0.125), values, precision)
    for component in ("value", "first", "second"):
        interval = getattr(expected, component)
        assert backend.exact_fraction(actual[component]["lower"]) == _fraction(interval.lower)
        assert backend.exact_fraction(actual[component]["upper"]) == _fraction(interval.upper)


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
def test_compiled_gelu_layer_matches_individual_exact_jets(precision):
    backend = _backend()
    values = [
        _jet(-0.5 + index / 9, 2.0**(-8-index), 0.2-index/11,
             -0.3+index/13, precision)
        for index in range(7)
    ]
    kappa = np.float32(np.sqrt(2.0 / np.pi))
    lam = np.float32(0.044715)
    batched = backend.gelu_new_layer_jet2(values, kappa, lam)["outputs"]
    individual = [backend.gelu_new_jet2(value, kappa, lam) for value in values]
    assert batched == individual


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
def test_compiled_exact_zero_endpoints_are_canonical(precision):
    backend = _backend()
    values = [constant_jet(Interval.point(value, precision))
              for value in (-0.5, -0.125, 0.25, 0.75)]
    actual = backend.layer_norm_jet2(
        values, np.float32(1e-5), np.ones(4, dtype="<f4"),
        np.zeros(4, dtype="<f4"),
    )["outputs"]
    for output in actual:
        for component in ("first", "second"):
            for endpoint in ("lower", "upper"):
                assert output[component][endpoint]["significand_hex"] == "0"
                assert output[component][endpoint]["exponent_2"] == 0


def test_python_decoder_short_circuits_noncanonical_zero_exponent():
    payload = {"significand_hex": "0", "exponent_2": -1073741823,
               "precision_bits": 384}
    assert CompiledMPFRBackend.exact_fraction(payload) == Fraction(0, 1)


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
def test_resident_joint_witness_cell_benchmark_is_live(precision):
    backend = _backend()
    result = backend.benchmark_gpt2_joint_witness_cell(
        precision, d_model=4, d_mlp=8, sequence_length=3,
        n_heads=2, d_head=2, selected_gates=2,
    )
    assert result["elapsed_seconds"] > 0 and result["cells_per_second"] > 0
    assert result["mpfr_primitive_count"] > 0
    assert result["mpfr_primitive_count"] == gpt2_joint_witness_cell_jet2(
        d_model=4, d_mlp=8, sequence_length=3,
        n_heads=2, d_head=2, selected_gates=2,
    )
    assert result["mpfr_primitives_per_second"] > 0
    assert result["dispatch_event_count"] == 81
    assert result["dispatch_trace_fnv1a_u64"] == "e0f23d0f4c4df894"
    assert len(result["checksum"]) == 16


@pytest.mark.parametrize("precision", [384, 512])
def test_packed_affine_layer_matches_individual_exact_columns(precision):
    backend = _backend()
    values = [_jet(-0.25 + index / 7, 2.0**(-9-index),
                   0.125-index/13, -0.0625+index/17, precision)
              for index in range(4)]
    weight = np.asarray([
        [0.5, -0.25, 0.125], [-0.75, 0.375, 0.25],
        [0.0625, -0.5, 0.875], [0.25, 0.75, -0.125],
    ], dtype="<f4")
    bias = np.asarray([0.125, -0.25, 0.5], dtype="<f4")
    packed = backend.packed_affine_layer_jet2(weight, bias, values)["outputs"]
    individual = [
        backend.affine_jet2(weight[:, output], bias[output], values, precision)
        for output in range(weight.shape[1])
    ]
    assert packed == individual


@pytest.mark.parametrize("precision", [384, 512])
def test_resident_jet_buffer_mlp_chain_matches_json_roundtrips(precision):
    backend = _backend()
    inputs = [_jet(-0.25 + index / 7, 2.0**(-9-index),
                   0.125-index/13, -0.0625+index/17, precision)
              for index in range(4)]
    first_weight = (np.arange(32, dtype="<f4").reshape(4, 8) - 13) / np.float32(31)
    first_bias = (np.arange(8, dtype="<f4") - 3) / np.float32(29)
    second_weight = (np.arange(24, dtype="<f4").reshape(8, 3) - 9) / np.float32(37)
    second_bias = np.asarray([0.125, -0.25, 0.5], dtype="<f4")
    kappa = np.float32(np.sqrt(2.0 / np.pi))
    lam = np.float32(0.044715)
    first_json = backend.packed_affine_layer_jet2(
        first_weight, first_bias, inputs
    )["outputs"]
    first_jets = [_decode_compiled_jet(backend, item, precision) for item in first_json]
    gelu_json = backend.gelu_new_layer_jet2(first_jets, kappa, lam)["outputs"]
    gelu_jets = [_decode_compiled_jet(backend, item, precision) for item in gelu_json]
    expected = backend.packed_affine_layer_jet2(
        second_weight, second_bias, gelu_jets
    )["outputs"]
    buffers = []
    try:
        buffers.append(backend.resident_jet_buffer(inputs))
        buffers.append(backend.resident_packed_affine_layer_jet2(
            buffers[-1], first_weight, first_bias
        ))
        buffers.append(backend.resident_gelu_new_layer_jet2(
            buffers[-1], kappa, lam
        ))
        buffers.append(backend.resident_packed_affine_layer_jet2(
            buffers[-1], second_weight, second_bias
        ))
        assert [buffer.width for buffer in buffers] == [4, 8, 8, 3]
        assert backend.export_resident_jet_buffer(buffers[-1])["outputs"] == expected
    finally:
        for buffer in reversed(buffers):
            buffer.close()


@pytest.mark.parametrize("precision", [384, 512])
def test_resident_f32_constant_import_is_exact_and_has_zero_derivatives(precision):
    backend = _backend()
    values = np.asarray([-0.0, 0.125, -0.75, 1.0 / 3.0], dtype="<f4")
    with backend.resident_f32_constant_buffer(values, precision) as buffer:
        outputs = backend.export_resident_jet_buffer(buffer)["outputs"]
    expected = [constant_jet(Interval.point(float(value), precision)) for value in values]
    for actual, reference in zip(outputs, expected):
        decoded = _decode_compiled_jet(backend, actual, precision)
        assert decoded == reference


@pytest.mark.parametrize("precision", [384, 512])
def test_resident_layer_norm_matches_existing_compiled_path(precision):
    backend = _backend()
    inputs = [_jet(-0.25 + index / 7, 2.0**(-9-index),
                   0.125-index/13, -0.0625+index/17, precision)
              for index in range(5)]
    epsilon = np.float32(1.0e-5)
    gamma = np.asarray([1.0, -0.5, 0.25, 1.5, -0.75], dtype="<f4")
    beta = np.asarray([0.0, 0.125, -0.25, 0.5, -0.0625], dtype="<f4")
    expected = backend.layer_norm_jet2(inputs, epsilon, gamma, beta)["outputs"]
    buffers = []
    try:
        buffers.append(backend.resident_jet_buffer(inputs))
        buffers.append(backend.resident_layer_norm_jet2(
            buffers[-1], epsilon, gamma, beta
        ))
        assert backend.export_resident_jet_buffer(buffers[-1])["outputs"] == expected
    finally:
        for buffer in reversed(buffers):
            buffer.close()


@pytest.mark.parametrize("precision", [384, 512])
def test_resident_add_and_sub_match_python_reference(precision):
    backend = _backend()
    left = [_jet(-0.2 + index/9, 2.0**(-10-index),
                 0.1-index/17, -0.05+index/19, precision)
            for index in range(4)]
    right = [_jet(0.3-index/11, 2.0**(-11-index),
                  -0.12+index/23, 0.07-index/29, precision)
             for index in range(4)]
    expected_add = [add_jet(x, y) for x, y in zip(left, right)]
    expected_sub = [sub_jet(x, y) for x, y in zip(left, right)]
    buffers = []
    try:
        buffers.extend((backend.resident_jet_buffer(left),
                        backend.resident_jet_buffer(right)))
        buffers.append(backend.resident_add_jet2(buffers[0], buffers[1]))
        buffers.append(backend.resident_sub_jet2(buffers[0], buffers[1]))
        for resident, expected in zip(buffers[2:], (expected_add, expected_sub)):
            payload = backend.export_resident_jet_buffer(resident)["outputs"]
            for actual, reference in zip(payload, expected):
                for component in ("value", "first", "second"):
                    interval = getattr(reference, component)
                    assert backend.exact_fraction(actual[component]["lower"]) == _fraction(interval.lower)
                    assert backend.exact_fraction(actual[component]["upper"]) == _fraction(interval.upper)
    finally:
        for buffer in reversed(buffers):
            buffer.close()


def test_resident_binary_rejects_width_mismatch():
    backend = _backend()
    left = backend.resident_jet_buffer([_jet(0.0, 0.01, 0.0, 0.0, 384)])
    right = backend.resident_jet_buffer([
        _jet(0.0, 0.01, 0.0, 0.0, 384), _jet(0.1, 0.01, 0.0, 0.0, 384)
    ])
    try:
        with pytest.raises(ValueError, match="buffer mismatch"):
            backend.resident_add_jet2(left, right)
    finally:
        right.close()
        left.close()


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


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_causal_attention_all_heads_matches_individual_heads(precision):
    backend = _backend()
    n_heads, head_dim, sequence_length = 2, 3, 3
    d_model = n_heads * head_dim
    query = [_jet(-0.2 + coordinate/13, 2.0**(-10-coordinate),
                  0.15-coordinate/17, -0.08+coordinate/19, precision)
             for coordinate in range(d_model)]
    keys = [[_jet(-0.3 + token/11 + coordinate/23, 2.0**(-11-token-coordinate),
                  0.12+token/29-coordinate/31, -0.18+coordinate/37, precision)
             for coordinate in range(d_model)] for token in range(sequence_length)]
    values = [[_jet(0.22-token/13+coordinate/17, 2.0**(-12-token-coordinate),
                    -0.11+token/19+coordinate/41, 0.04-token/43, precision)
               for coordinate in range(d_model)] for token in range(sequence_length)]
    batched = backend.causal_attention_final_all_heads_jet2(
        query, keys, values, n_heads, pivot=0
    )["outputs"]
    individual = []
    for head in range(n_heads):
        start, stop = head * head_dim, (head + 1) * head_dim
        individual.extend(backend.causal_attention_final_head_jet2(
            query[start:stop], [row[start:stop] for row in keys],
            [row[start:stop] for row in values], pivot=0,
        )["outputs"])
    assert batched == individual


@pytest.mark.parametrize("precision", [384, 512])
def test_compiled_final_contrast_is_bit_identical_to_exact_fusion(precision):
    backend = _backend()
    values = [_jet(-0.2 + index/7, 2.0**(-10-index),
                   0.1-index/13, -0.05+index/17, precision)
              for index in range(4)]
    unembed = np.asarray([
        [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7],
        [-0.3, 0.2, 0.1, -0.6, 0.5, -0.4, 0.7],
        [0.25, -0.125, 0.375, -0.5, 0.625, -0.75, 0.875],
        [-0.05, 0.15, -0.25, 0.35, -0.45, 0.55, -0.65],
    ], dtype="<f4")
    bias = np.asarray([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07], dtype="<f4")
    suffix_ids = np.asarray([0, 3, 6], dtype="<i8")
    coefficients = np.asarray([0.1, -0.3, 0.2], dtype="<f8")

    def exact(value):
        fraction = Fraction.from_float(float(value))
        return gmpy2.mpq(fraction.numerator, fraction.denominator)

    fused_weights = [
        sum((exact(coefficients[index]) * exact(unembed[coordinate, token])
             for index, token in enumerate(suffix_ids)), gmpy2.mpq(0))
        for coordinate in range(unembed.shape[0])
    ]
    fused_bias = sum((exact(coefficients[index]) * exact(bias[token])
                      for index, token in enumerate(suffix_ids)), gmpy2.mpq(0))
    expected = affine_map_jets([fused_weights], values, [fused_bias])[0]
    actual = backend.final_contrast_jet2(
        values, unembed, bias, suffix_ids, coefficients
    )
    fused_actual = backend.fused_contrast_jet2(
        values, fuse_final_contrast_exact(
            unembed, bias, suffix_ids, coefficients
        ).payload(),
    )
    assert fused_actual == actual
    for component in ("value", "first", "second"):
        interval = getattr(expected, component)
        assert backend.exact_fraction(actual[component]["lower"]) == _fraction(interval.lower)
        assert backend.exact_fraction(actual[component]["upper"]) == _fraction(interval.upper)
