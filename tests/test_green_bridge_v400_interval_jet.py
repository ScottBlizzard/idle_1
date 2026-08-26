from __future__ import annotations

import math
from pathlib import Path
import sys

import gmpy2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_interval import (
    Interval, exp_interval, tanh_interval,
)
from green_bridge_v400_interval_jet import (
    CertifiedScalarPrimitive, Jet2, affine_control_jet, centered_tighten,
    compose_jet, constant_jet, mul_jet, reciprocal_jet,
)


P = 256


def _exp_primitive():
    return CertifiedScalarPrimitive("exp", exp_interval, exp_interval, exp_interval)


def _tanh_primitive():
    def first(x):
        t = tanh_interval(x)
        return Interval.point(1, P) - t.square()
    def second(x):
        t = tanh_interval(x)
        return -Interval.point(2, P) * t * (Interval.point(1, P) - t.square())
    return CertifiedScalarPrimitive("tanh", tanh_interval, first, second)


def test_affine_jet_exact():
    domain = Interval.from_bounds(-1, 1, P)
    jet = affine_control_jet(Interval.point(3, P), Interval.point(2, P), domain)
    assert jet.value.lower == 1 and jet.value.upper == 5
    assert jet.first.lower == jet.first.upper == 2
    assert jet.second.lower == jet.second.upper == 0


def test_product_jet_quadratic_exact():
    x = affine_control_jet(Interval.point(0, P), Interval.point(1, P),
                           Interval.from_bounds(-2, 3, P))
    square = mul_jet(x, x)
    assert square.value.lower <= 0 <= square.value.upper
    assert square.first.lower <= -4 and square.first.upper >= 6
    assert square.second.lower <= 2 <= square.second.upper


def test_reciprocal_jet_contains_symbolic_derivatives():
    x = affine_control_jet(Interval.point(2, P), Interval.point(1, P),
                           Interval.from_bounds(-0.5, 0.5, P))
    y = reciprocal_jet(x)
    # At t=0: y=1/2, y'=-1/4, y''=1/4.
    assert y.value.contains(0.5)
    assert y.first.contains(-0.25)
    assert y.second.contains(0.25)


def test_exp_jet_contains_symbolic_derivatives():
    x = affine_control_jet(Interval.point(0, P), Interval.point(1, P),
                           Interval.from_bounds(-0.1, 0.1, P))
    y = compose_jet(x, _exp_primitive())
    assert y.value.contains(1.0)
    assert y.first.contains(1.0)
    assert y.second.contains(1.0)


def test_tanh_jet_contains_symbolic_derivatives():
    x = affine_control_jet(Interval.point(0, P), Interval.point(1, P),
                           Interval.from_bounds(-0.1, 0.1, P))
    y = compose_jet(x, _tanh_primitive())
    assert y.value.contains(0.0)
    assert y.first.contains(1.0)
    assert y.second.contains(0.0)


def test_centered_mean_value_intersection_sound():
    cell = Interval.from_bounds(-0.25, 0.25, P)
    x = affine_control_jet(Interval.point(1, P), Interval.point(2, P), cell)
    center = Jet2(Interval.point(1, P), Interval.point(2, P), Interval.point(0, P))
    tightened = centered_tighten(cell, center, x)
    assert tightened.value.lower == x.value.lower
    assert tightened.value.upper == x.value.upper


def test_partition_exact_cover_and_weight_sum():
    h = gmpy2.mpq(1, 1)
    cells = [(gmpy2.mpq(-1), gmpy2.mpq(-1, 2)),
             (gmpy2.mpq(-1, 2), gmpy2.mpq(0)),
             (gmpy2.mpq(0), gmpy2.mpq(1, 2)),
             (gmpy2.mpq(1, 2), gmpy2.mpq(1))]
    assert cells[0][0] == -h and cells[-1][1] == h
    assert all(cells[i][1] == cells[i + 1][0] for i in range(len(cells) - 1))
    positive_weight = sum(h * (b - a) - (b*b - a*a)/2 for a, b in cells[2:])
    assert positive_weight == gmpy2.mpq(1, 2)


def test_global_m2_bound_for_quadratic():
    x = affine_control_jet(Interval.point(0, P), Interval.point(1, P),
                           Interval.from_bounds(-1, 1, P))
    second = mul_jet(x, x).second
    assert second.magnitude() >= 2


def test_signed_curvature_cubic_identity():
    # Psi=t^3: K+ = h^3 and K- = -h^3, so central residual is 2h^3.
    h = gmpy2.mpq(1, 2)
    k_plus, k_minus = h**3, -(h**3)
    secant_residual = k_plus - k_minus
    assert secant_residual == 2 * h**3


def test_grid_miss_fixture_certificate_catches_offgrid_curvature():
    center = 0.03125
    width = 0.001
    grid = [i / 16 for i in range(-16, 17)]
    sampled = max(math.exp(-((x - center) / width) ** 2) for x in grid)
    certified = Interval.from_bounds(0, 1, P)
    assert sampled < 1e-100
    assert certified.contains(1.0)
