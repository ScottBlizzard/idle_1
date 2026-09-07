"""Regression for nonfinite-to-zero corruption and overflow-safe softmax jets."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys

import gmpy2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend, _exact_fraction
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import affine_control_jet, constant_jet
from green_bridge_v400_mpfr_tensor_executor import _decode_interval, _decode_jet


@pytest.fixture
def backend():
    path = os.environ.get("GREEN_V400_MPFR_BACKEND")
    if not path:
        pytest.skip("compiled MPFR backend is not configured")
    return CompiledMPFRBackend(Path(path))


def attention(backend, query, keys, values, pivot=0):
    with ExitStack() as stack:
        q, k, v = [stack.enter_context(backend.resident_jet_buffer(items))
                   for items in ([query], keys, values)]
        result = stack.enter_context(backend.resident_causal_attention_all_heads_jet2(
            q, k, v, len(keys), 1, 1, pivot))
        raw = backend.export_resident_jet_buffer(result)["outputs"][0]
    legacy = backend.causal_attention_final_head_jet2(
        [query], [[x] for x in keys], [[x] for x in values], pivot)["outputs"][0]
    assert legacy == raw
    return _decode_jet(raw, query.precision_bits)


@pytest.mark.parametrize("precision", [384, 512])
@pytest.mark.parametrize("pivot", [0, 1])
def test_overflowing_scores_constant_one_is_not_zero(backend, precision, pivot):
    point = lambda x: constant_jet(Interval.point(x, precision))
    result = attention(backend, point(1000000), [point(0), point(1000000)],
                       [point(1), point(1)], pivot)
    assert result.value.lower <= 1 <= result.value.upper
    assert result.value.lower > gmpy2.mpq(999, 1000)
    assert result.value.upper < gmpy2.mpq(1001, 1000)
    for component in (result.first, result.second):
        assert component.lower == component.upper == 0


@pytest.mark.parametrize("wide", [False, True])
def test_stable_softmax_encloses_independent_derivatives_and_nests(backend, wide):
    results = []
    for precision in (384, 512):
        domain = Interval.from_bounds(-1, 1, precision)
        affine = lambda a, b: affine_control_jet(
            Interval.point(a, precision), Interval.point(b, precision), domain)
        point = lambda x: constant_jet(Interval.point(x, precision))
        results.append(attention(backend, affine(80, 1000 if wide else 1),
            [point(0), point(1), point(-0.5)],
            [affine(1, 0.25), affine(-2, 0.5), affine(3, -0.125)]))
    for field in ("value", "first", "second"):
        low, high = [getattr(result, field) for result in results]
        assert low.lower <= high.lower <= high.upper <= low.upper
    with gmpy2.context(precision=1024):
        for raw_t in (-1, -0.5, 0, 0.5, 1):
            t = gmpy2.mpfr(raw_t)
            slope = 1000 if wide else 1
            keys = [gmpy2.mpfr(x) for x in (0, 1, -0.5)]
            scores = [(80 + slope * t) * k for k in keys]
            shift = max(scores)
            exps = [gmpy2.exp(s - shift) for s in scores]
            probabilities = [x / sum(exps) for x in exps]
            ds = [slope * k for k in keys]
            mean = sum(p * d for p, d in zip(probabilities, ds))
            variance = sum(p * (d - mean)**2 for p, d in zip(probabilities, ds))
            dp = [p * (d - mean) for p, d in zip(probabilities, ds)]
            ddp = [p * ((d - mean)**2 - variance) for p, d in zip(probabilities, ds)]
            values = [a + b * t for a, b in ((1, .25), (-2, .5), (3, -.125))]
            dv = [gmpy2.mpfr(x) for x in (.25, .5, -.125)]
            reference = [sum(p * v for p, v in zip(probabilities, values)),
                         sum(p1*v + p*v1 for p1, v, p, v1 in zip(dp, values, probabilities, dv)),
                         sum(p2*v + 2*p1*v1 for p2, v, p1, v1 in zip(ddp, values, dp, dv))]
            for result in results:
                for field, value in zip(("value", "first", "second"), reference):
                    interval = getattr(result, field)
                    assert interval.lower <= gmpy2.mpq(value) <= interval.upper


def test_native_nonfinite_export_is_explicit_and_both_decoders_reject(backend):
    raw = backend.interval_primitive("exp", Interval.point(10**12, 384))
    assert raw["upper"] == {"nonfinite": "infinity"}
    with pytest.raises(ValueError, match="NONFINITE_MPFR_ENDPOINT"):
        _exact_fraction(raw["upper"])
    with pytest.raises(ValueError, match="NONFINITE_MPFR_ENDPOINT"):
        _decode_interval(raw, 384)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_resident_nonfinite_input_fails_closed(backend, bad):
    with pytest.raises(RuntimeError):
        backend.resident_f32_constant_buffer(np.asarray([bad], dtype=np.float32), 384)
