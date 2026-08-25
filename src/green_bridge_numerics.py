"""Frozen Richardson numerical-error propagation for the GREEN bridge.

This module is deliberately model-free.  It converts duplicate TransformerLens
endpoint noise plus full-versus-half finite-difference discrepancies into the
gate- and item-level uncertainty bounds frozen by the GPTPRO Gate-04 decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from matched_bypass_gate import GateJet
from green_bridge_spec import AD_ROUTE_GAMMA


def round_up(value: float) -> float:
    value64 = np.float64(value)
    if np.isnan(value64):
        raise ValueError("cannot outward-round NaN")
    return float(np.nextafter(value64, np.inf))


def round_down(value: float) -> float:
    value64 = np.float64(value)
    if np.isnan(value64):
        raise ValueError("cannot outward-round NaN")
    return float(np.nextafter(value64, -np.inf))


def add_up(*values: float) -> float:
    total = np.float64(0.0)
    for value in values:
        if not np.isfinite(value) or value < 0:
            raise ValueError("add_up expects finite nonnegative values")
        total = np.nextafter(total + np.float64(value), np.inf)
    return float(total)


def multiply_up(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right) or left < 0 or right < 0:
        raise ValueError("multiply_up expects finite nonnegative values")
    return round_up(np.float64(left) * np.float64(right))


def norm_up(value) -> float:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("norm input must be finite")
    return round_up(np.linalg.norm(array))


def subtract_down(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("subtract_down expects finite values")
    return round_down(np.float64(left) - np.float64(right))


@dataclass(frozen=True)
class GateNumericalBounds:
    eta_G: float
    eta_C: float
    eta_J: float
    eta_H: float
    epsilon_G: float
    epsilon_C: float
    epsilon_delta_H: np.ndarray
    inverse_admissible: bool
    A_max: np.ndarray
    epsilon_A: np.ndarray
    epsilon_P: np.ndarray
    epsilon_P_F: float


@dataclass(frozen=True)
class ScaleNumericalBoundsV200:
    epsilon_G: float
    epsilon_C: float
    epsilon_J: float
    epsilon_delta_H: np.ndarray
    A_max: np.ndarray
    epsilon_A: np.ndarray
    epsilon_P: np.ndarray
    epsilon_P_F: float
    inverse_admissible: bool


@dataclass(frozen=True)
class DyadicEnclosureV200:
    coarse: ScaleNumericalBoundsV200
    fine: ScaleNumericalBoundsV200
    final_epsilon_G: float
    final_epsilon_C: float
    final_epsilon_J: float
    final_epsilon_delta_H: np.ndarray
    final_A_max: np.ndarray
    final_epsilon_A: np.ndarray
    final_epsilon_P: np.ndarray
    final_epsilon_P_F: float
    final_inverse_admissible: bool
    overlap_G: bool
    overlap_C: bool
    overlap_J: bool
    overlap_delta_H: np.ndarray


@dataclass(frozen=True)
class ADRouteCertificateV200:
    forward: GateJet
    reverse: GateJet
    reference: GateJet
    route_difference_G: float
    route_difference_C: float
    route_difference_J: float
    route_difference_delta_H: np.ndarray
    route_radius_G: float
    route_radius_C: float
    route_radius_J: float
    route_radius_delta_H: np.ndarray
    route_pass_G: bool
    route_pass_C: bool
    route_pass_J: bool
    route_pass_delta_H: np.ndarray

    @property
    def passed(self) -> bool:
        return bool(
            self.route_pass_G and self.route_pass_C and self.route_pass_J
            and np.all(self.route_pass_delta_H)
        )


@dataclass(frozen=True)
class ADCertifiedEnclosureV200:
    fine_jet: GateJet
    ad_reference: GateJet
    epsilon_G: float
    epsilon_C: float
    epsilon_J: float
    epsilon_delta_H: np.ndarray
    inverse_lower_bound: float
    inverse_admissible: bool
    A_max: np.ndarray
    epsilon_A: np.ndarray
    epsilon_P: np.ndarray
    epsilon_P_F: float


def _jet_average(left: GateJet, right: GateJet) -> GateJet:
    return GateJet(*[
        0.5 * (np.asarray(getattr(left, name), dtype=np.float64)
               + np.asarray(getattr(right, name), dtype=np.float64))
        for name in ("G", "C", "J_path", "H_path", "H_control")
    ])


def _route_scalar(left, right) -> tuple[float, float, bool]:
    difference = norm_up(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    scale = max(1.0, norm_up(left), norm_up(right))
    guard = round_up(2.0 * AD_ROUTE_GAMMA * scale)
    radius = round_up(difference / 2.0 + AD_ROUTE_GAMMA * scale)
    return difference, radius, bool(difference <= guard)


def ad_route_certificate_v200(forward: GateJet, reverse: GateJet) -> ADRouteCertificateV200:
    _validate_jet_v200(forward, "forward")
    _validate_jet_v200(reverse, "reverse")
    reference = _jet_average(forward, reverse)
    dG, rG, pG = _route_scalar(forward.G, reverse.G)
    dC, rC, pC = _route_scalar(forward.C, reverse.C)
    dJ, rJ, pJ = _route_scalar(forward.J_path, reverse.J_path)
    fd = np.asarray(forward.H_path) - np.asarray(forward.H_control)
    rd = np.asarray(reverse.H_path) - np.asarray(reverse.H_control)
    differences, radii, passes = [], [], []
    for left, right in zip(fd, rd):
        difference, radius, passed = _route_scalar(left, right)
        differences.append(difference); radii.append(radius); passes.append(passed)
    return ADRouteCertificateV200(
        forward, reverse, reference, dG, dC, dJ,
        np.asarray(differences), rG, rC, rJ, np.asarray(radii),
        pG, pC, pJ, np.asarray(passes, dtype=bool),
    )


def ad_certified_enclosure_v200(
    fine_jet: GateJet,
    certificate: ADRouteCertificateV200,
    *,
    epsilon_y: float,
    fine_h_x: float,
    fine_h_z: float,
) -> ADCertifiedEnclosureV200:
    """Literal dual-route AD-certified fine-Richardson contract."""
    _validate_jet_v200(fine_jet, "fine_jet")
    if epsilon_y < 0 or fine_h_x <= 0 or fine_h_z <= 0:
        raise ValueError("endpoint error and fine radii are invalid")
    ad = certificate.reference
    eta_G = 3.0 * epsilon_y / fine_h_z
    eta_C = 64.0 * epsilon_y / (3.0 * fine_h_z * fine_h_z)
    eta_J = 3.0 * epsilon_y / fine_h_x
    eta_H = 17.0 * epsilon_y / (3.0 * fine_h_x * fine_h_z)
    fine_delta = np.asarray(fine_jet.H_path) - np.asarray(fine_jet.H_control)
    ad_delta = np.asarray(ad.H_path) - np.asarray(ad.H_control)
    epsilon_G = add_up(norm_up(np.asarray(fine_jet.G) - np.asarray(ad.G)), certificate.route_radius_G, 10.0 * eta_G)
    epsilon_C = add_up(norm_up(np.asarray(fine_jet.C) - np.asarray(ad.C)), certificate.route_radius_C, 10.0 * eta_C)
    epsilon_J = add_up(norm_up(np.asarray(fine_jet.J_path) - np.asarray(ad.J_path)), certificate.route_radius_J, math.sqrt(500.0) * eta_J)
    epsilon_delta_H = np.asarray([
        add_up(norm_up(fine_delta[i] - ad_delta[i]), certificate.route_radius_delta_H[i], 20.0 * eta_H)
        for i in range(5)
    ])
    C_norm_nearest = float(np.linalg.norm(np.asarray(fine_jet.C, dtype=np.float64)))
    c_low = subtract_down(C_norm_nearest, epsilon_C)
    inverse = bool(c_low > 0.0)
    if inverse:
        C_norm = round_down(C_norm_nearest)
        G_norm = norm_up(fine_jet.G)
        A_max = np.asarray([
            round_up(add_up(norm_up(fine_delta[i]), epsilon_delta_H[i]) / c_low)
            for i in range(5)
        ])
        epsilon_A = np.asarray([
            round_up(add_up(epsilon_delta_H[i], multiply_up(A_max[i], epsilon_C)) / C_norm)
            for i in range(5)
        ])
        epsilon_P = np.asarray([
            add_up(multiply_up(epsilon_G, A_max[i]), multiply_up(G_norm, epsilon_A[i]))
            for i in range(5)
        ])
        epsilon_P_F = norm_up(epsilon_P)
    else:
        A_max = np.full(5, np.inf)
        epsilon_A = np.full(5, np.inf)
        epsilon_P = np.full(5, np.inf)
        epsilon_P_F = math.inf
    return ADCertifiedEnclosureV200(
        fine_jet, ad, epsilon_G, epsilon_C, epsilon_J, epsilon_delta_H,
        c_low, inverse, A_max, epsilon_A, epsilon_P, epsilon_P_F,
    )


def ad_matched_bypass_compatibility_v200(
    certificate: ADRouteCertificateV200,
    whitebox_A: np.ndarray,
    *,
    epsilon_wb: float = 1e-10,
) -> dict:
    whitebox = np.asarray(whitebox_A, dtype=np.float64)
    if whitebox.shape != (5,):
        raise ValueError("whitebox coordinates must have shape [5]")
    ad = certificate.reference
    delta = np.asarray(ad.H_path) - np.asarray(ad.H_control)
    residuals = np.asarray([norm_up(delta[i] - whitebox[i] * np.asarray(ad.C)) for i in range(5)])
    C_norm = norm_up(ad.C)
    bounds = np.asarray([
        add_up(
            certificate.route_radius_delta_H[i],
            multiply_up(abs(float(whitebox[i])), certificate.route_radius_C),
            multiply_up(epsilon_wb, add_up(C_norm, certificate.route_radius_C)),
        ) for i in range(5)
    ])
    ratios = np.asarray([compatibility_ratio(float(q), float(b)) for q, b in zip(residuals, bounds)])
    return {"passed": bool(np.all(residuals <= np.nextafter(bounds, np.inf))),
            "residuals": residuals, "bounds": bounds, "ratios": ratios,
            "max_ratio": float(np.max(ratios))}


def richardson_numerical_bounds(
    rich: GateJet,
    half: GateJet,
    *,
    epsilon_y: float,
    h1: float,
    h2: float,
) -> GateNumericalBounds:
    """Apply the frozen closed-form Richardson propagation to one gate."""
    if epsilon_y < 0 or h1 <= 0 or h2 <= 0:
        raise ValueError("epsilon_y must be nonnegative and radii must be positive")

    for name, value in (
        ("rich.H_path", rich.H_path),
        ("rich.H_control", rich.H_control),
        ("rich.J_path", rich.J_path),
        ("half.H_path", half.H_path),
        ("half.H_control", half.H_control),
        ("half.J_path", half.J_path),
    ):
        if np.asarray(value).shape != (5, 100):
            raise ValueError(f"{name} must have shape [5,100]")

    k = int(np.asarray(rich.G).size)
    if k != 100:
        raise ValueError(f"the frozen output dimension is 100, got {k}")
    root_k = math.sqrt(k)
    eta_G = 3.0 * epsilon_y / h2
    eta_C = 64.0 * epsilon_y / (3.0 * h2 * h2)
    eta_J = 3.0 * epsilon_y / h1
    eta_H = 17.0 * epsilon_y / (3.0 * h1 * h2)

    rich_G = np.asarray(rich.G, dtype=np.float64)
    half_G = np.asarray(half.G, dtype=np.float64)
    rich_C = np.asarray(rich.C, dtype=np.float64)
    half_C = np.asarray(half.C, dtype=np.float64)
    rich_delta_H = np.asarray(rich.H_path, dtype=np.float64) - np.asarray(
        rich.H_control, dtype=np.float64
    )
    half_delta_H = np.asarray(half.H_path, dtype=np.float64) - np.asarray(
        half.H_control, dtype=np.float64
    )

    epsilon_G = float(np.linalg.norm(rich_G - half_G) + root_k * eta_G)
    epsilon_C = float(np.linalg.norm(rich_C - half_C) + root_k * eta_C)
    epsilon_delta_H = np.linalg.norm(rich_delta_H - half_delta_H, axis=1)
    epsilon_delta_H = epsilon_delta_H + 2.0 * root_k * eta_H

    C_norm = float(np.linalg.norm(rich_C))
    inverse_admissible = C_norm > epsilon_C
    probe_frame_dim = int(rich_delta_H.shape[0])
    if inverse_admissible:
        A_max = (
            np.linalg.norm(rich_delta_H, axis=1) + epsilon_delta_H
        ) / (C_norm - epsilon_C)
        epsilon_A = (epsilon_delta_H + A_max * epsilon_C) / C_norm
        epsilon_P = epsilon_G * A_max + float(np.linalg.norm(rich_G)) * epsilon_A
        epsilon_P_F = float(np.linalg.norm(epsilon_P))
    else:
        A_max = np.full(probe_frame_dim, np.inf, dtype=np.float64)
        epsilon_A = np.full(probe_frame_dim, np.inf, dtype=np.float64)
        epsilon_P = np.full(probe_frame_dim, np.inf, dtype=np.float64)
        epsilon_P_F = math.inf

    return GateNumericalBounds(
        eta_G=float(eta_G),
        eta_C=float(eta_C),
        eta_J=float(eta_J),
        eta_H=float(eta_H),
        epsilon_G=epsilon_G,
        epsilon_C=epsilon_C,
        epsilon_delta_H=np.asarray(epsilon_delta_H, dtype=np.float64),
        inverse_admissible=inverse_admissible,
        A_max=np.asarray(A_max, dtype=np.float64),
        epsilon_A=np.asarray(epsilon_A, dtype=np.float64),
        epsilon_P=np.asarray(epsilon_P, dtype=np.float64),
        epsilon_P_F=epsilon_P_F,
    )


def active_contraction_bound(
    contrast_norm: float,
    delta_norm: float,
    epsilon_P_F: float,
) -> float:
    return float(contrast_norm * delta_norm * epsilon_P_F)


def active_envelope_contraction_bound(
    contrast_norm: float,
    projected_direction_norm: float,
    physical_direction_norm: float,
    epsilon_P_F: float,
    gate_response_norm: float,
    epsilon_G: float,
    envelope_residual_absolute: float,
) -> float:
    """Frozen v1.3 ambient-operator error including envelope residual."""
    return multiply_up(
        contrast_norm,
        add_up(
            multiply_up(projected_direction_norm, epsilon_P_F),
            multiply_up(
                multiply_up(add_up(gate_response_norm, epsilon_G), physical_direction_norm),
                envelope_residual_absolute,
            ),
        ),
    )


def certified_null_bound(
    contrast_norm: float,
    delta_norm: float,
    gate_response_norm: float,
    epsilon_G: float,
    whitebox_A_norm: float,
) -> float:
    return multiply_up(
        multiply_up(multiply_up(contrast_norm, delta_norm), add_up(gate_response_norm, epsilon_G)),
        whitebox_A_norm,
    )


def sum_item_error_bounds(bounds) -> float:
    return float(sum(float(value) for value in bounds))


def cell_error_bound(target_item_bounds, patched_item_bounds) -> float:
    target = np.asarray(tuple(target_item_bounds), dtype=np.float64)
    patched = np.asarray(tuple(patched_item_bounds), dtype=np.float64)
    if target.shape != patched.shape or target.size == 0:
        raise ValueError("target and patched item bounds must be nonempty and paired")
    return float(np.mean(target + patched))


def _validate_jet_v200(jet: GateJet, name: str) -> None:
    expected = {
        "G": (100,), "C": (100,), "J_path": (5, 100),
        "H_path": (5, 100), "H_control": (5, 100),
    }
    for field, shape in expected.items():
        value = np.asarray(getattr(jet, field), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name}.{field} must be finite with shape {shape}")


def _identification_bounds_v200(
    jet: GateJet,
    epsilon_G: float,
    epsilon_C: float,
    epsilon_delta_H: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, float]:
    C_norm = float(np.linalg.norm(jet.C))
    delta_H = np.asarray(jet.H_path, dtype=np.float64) - np.asarray(
        jet.H_control, dtype=np.float64
    )
    inverse = bool(C_norm > epsilon_C)
    if not inverse:
        inf = np.full(5, np.inf, dtype=np.float64)
        return False, inf, inf.copy(), inf.copy(), math.inf
    A_max = (np.linalg.norm(delta_H, axis=1) + epsilon_delta_H) / (
        C_norm - epsilon_C
    )
    epsilon_A = (epsilon_delta_H + A_max * epsilon_C) / C_norm
    epsilon_P = epsilon_G * A_max + float(np.linalg.norm(jet.G)) * epsilon_A
    return True, A_max, epsilon_A, epsilon_P, float(np.linalg.norm(epsilon_P))


def richardson_pair_bounds_v200(
    rich: GateJet,
    small: GateJet,
    *,
    epsilon_y: float,
    h1: float,
    h2: float,
) -> ScaleNumericalBoundsV200:
    """Bound one Richardson pair using only endpoint noise and scale change."""
    _validate_jet_v200(rich, "rich")
    _validate_jet_v200(small, "small")
    if epsilon_y < 0 or h1 <= 0 or h2 <= 0:
        raise ValueError("epsilon_y must be nonnegative and radii must be positive")
    root_k = 10.0
    eta_G = 3.0 * epsilon_y / h2
    eta_C = 64.0 * epsilon_y / (3.0 * h2 * h2)
    eta_J = 3.0 * epsilon_y / h1
    eta_H = 17.0 * epsilon_y / (3.0 * h1 * h2)
    rich_delta = np.asarray(rich.H_path) - np.asarray(rich.H_control)
    small_delta = np.asarray(small.H_path) - np.asarray(small.H_control)
    epsilon_G = float(np.linalg.norm(np.asarray(rich.G) - np.asarray(small.G)) + root_k * eta_G)
    epsilon_C = float(np.linalg.norm(np.asarray(rich.C) - np.asarray(small.C)) + root_k * eta_C)
    epsilon_J = float(
        np.linalg.norm(np.asarray(rich.J_path) - np.asarray(small.J_path))
        + math.sqrt(500.0) * eta_J
    )
    epsilon_delta_H = np.linalg.norm(rich_delta - small_delta, axis=1) + 2.0 * root_k * eta_H
    inverse, A_max, epsilon_A, epsilon_P, epsilon_P_F = _identification_bounds_v200(
        rich, epsilon_G, epsilon_C, epsilon_delta_H
    )
    return ScaleNumericalBoundsV200(
        epsilon_G=epsilon_G,
        epsilon_C=epsilon_C,
        epsilon_J=epsilon_J,
        epsilon_delta_H=np.asarray(epsilon_delta_H, dtype=np.float64),
        A_max=A_max,
        epsilon_A=epsilon_A,
        epsilon_P=epsilon_P,
        epsilon_P_F=epsilon_P_F,
        inverse_admissible=inverse,
    )


def dyadic_enclosure_v200(
    coarse_jet: GateJet,
    fine_jet: GateJet,
    coarse: ScaleNumericalBoundsV200,
    fine: ScaleNumericalBoundsV200,
) -> DyadicEnclosureV200:
    """Intersect coarse/fine error balls and center the final ball on fine."""
    _validate_jet_v200(coarse_jet, "coarse_jet")
    _validate_jet_v200(fine_jet, "fine_jet")
    delta_coarse = np.asarray(coarse_jet.H_path) - np.asarray(coarse_jet.H_control)
    delta_fine = np.asarray(fine_jet.H_path) - np.asarray(fine_jet.H_control)
    dG = float(np.linalg.norm(np.asarray(fine_jet.G) - np.asarray(coarse_jet.G)))
    dC = float(np.linalg.norm(np.asarray(fine_jet.C) - np.asarray(coarse_jet.C)))
    dJ = float(np.linalg.norm(np.asarray(fine_jet.J_path) - np.asarray(coarse_jet.J_path)))
    dH = np.linalg.norm(delta_fine - delta_coarse, axis=1)
    overlap_G = dG <= math.nextafter(fine.epsilon_G + coarse.epsilon_G, math.inf)
    overlap_C = dC <= math.nextafter(fine.epsilon_C + coarse.epsilon_C, math.inf)
    overlap_J = dJ <= math.nextafter(fine.epsilon_J + coarse.epsilon_J, math.inf)
    overlap_H = dH <= np.nextafter(fine.epsilon_delta_H + coarse.epsilon_delta_H, np.inf)
    final_G = min(fine.epsilon_G, dG + coarse.epsilon_G)
    final_C = min(fine.epsilon_C, dC + coarse.epsilon_C)
    final_J = min(fine.epsilon_J, dJ + coarse.epsilon_J)
    final_H = np.minimum(fine.epsilon_delta_H, dH + coarse.epsilon_delta_H)
    inverse, A_max, epsilon_A, epsilon_P, epsilon_P_F = _identification_bounds_v200(
        fine_jet, final_G, final_C, final_H
    )
    return DyadicEnclosureV200(
        coarse=coarse, fine=fine,
        final_epsilon_G=final_G, final_epsilon_C=final_C,
        final_epsilon_J=final_J, final_epsilon_delta_H=final_H,
        final_A_max=A_max, final_epsilon_A=epsilon_A,
        final_epsilon_P=epsilon_P, final_epsilon_P_F=epsilon_P_F,
        final_inverse_admissible=inverse,
        overlap_G=bool(overlap_G), overlap_C=bool(overlap_C),
        overlap_J=bool(overlap_J), overlap_delta_H=np.asarray(overlap_H, dtype=bool),
    )


def compatibility_ratio(residual: float, bound: float) -> float:
    if np.isnan(residual) or np.isnan(bound) or residual < 0.0 or bound < 0.0:
        raise ValueError("residual and bound must be nonnegative")
    if bound == 0.0:
        return 0.0 if residual == 0.0 else math.inf
    return float(residual / bound)


_compatibility_ratio = compatibility_ratio


def _active_enclosure_value(enclosure, name: str):
    if hasattr(enclosure, name):
        return getattr(enclosure, name)
    return getattr(enclosure, "final_" + name)


def factorization_compatibility_v200(identification, jet: GateJet, enclosure) -> dict:
    delta_H = np.asarray(jet.H_path) - np.asarray(jet.H_control)
    residuals = np.asarray([
        norm_up(delta_H[i] - identification.A[i] * np.asarray(jet.C))
        for i in range(5)
    ])
    eps_h = _active_enclosure_value(enclosure, "epsilon_delta_H")
    eps_a = _active_enclosure_value(enclosure, "epsilon_A")
    a_max = _active_enclosure_value(enclosure, "A_max")
    eps_c = _active_enclosure_value(enclosure, "epsilon_C")
    C_norm = norm_up(jet.C)
    bounds = np.asarray([
        add_up(eps_h[i], multiply_up(eps_a[i], C_norm), multiply_up(a_max[i], eps_c))
        for i in range(5)
    ])
    ratios = np.array([_compatibility_ratio(float(q), float(b)) for q, b in zip(residuals, bounds)])
    bounds = np.nextafter(bounds, np.inf)
    passed = bool(np.all(residuals <= bounds))
    return {"passed": passed, "residuals": residuals, "bounds": bounds,
            "ratios": ratios, "max_ratio": float(np.max(ratios))}


def whitebox_compatibility_v200(
    identification,
    whitebox_A: np.ndarray,
    enclosure,
    *,
    epsilon_wb: float = 1e-10,
) -> dict:
    whitebox = np.asarray(whitebox_A, dtype=np.float64)
    if whitebox.shape != (5,) or epsilon_wb < 0:
        raise ValueError("whitebox coordinates must have shape [5]")
    residuals = np.nextafter(np.abs(np.asarray(identification.A) - whitebox), np.inf)
    bounds = np.asarray([
        add_up(value, epsilon_wb)
        for value in _active_enclosure_value(enclosure, "epsilon_A")
    ])
    ratios = np.array([_compatibility_ratio(float(q), float(b)) for q, b in zip(residuals, bounds)])
    return {"passed": bool(np.all(residuals <= np.nextafter(bounds, np.inf))),
            "residuals": residuals, "bounds": bounds, "ratios": ratios,
            "max_ratio": float(np.max(ratios))}


def whitebox_factorization_compatibility_v200(
    jet: GateJet,
    whitebox_A: np.ndarray,
    enclosure,
    *,
    epsilon_wb: float = 1e-10,
) -> dict:
    whitebox = np.asarray(whitebox_A, dtype=np.float64)
    delta_H = np.asarray(jet.H_path) - np.asarray(jet.H_control)
    residuals = np.asarray([
        norm_up(delta_H[i] - whitebox[i] * np.asarray(jet.C))
        for i in range(5)
    ])
    eps_h = _active_enclosure_value(enclosure, "epsilon_delta_H")
    eps_c = _active_enclosure_value(enclosure, "epsilon_C")
    C_plus = add_up(norm_up(jet.C), eps_c)
    bounds = np.asarray([
        add_up(
            eps_h[i], multiply_up(abs(float(whitebox[i])), eps_c),
            multiply_up(epsilon_wb, C_plus),
        ) for i in range(5)
    ])
    ratios = np.array([_compatibility_ratio(float(q), float(b)) for q, b in zip(residuals, bounds)])
    return {"passed": bool(np.all(residuals <= np.nextafter(bounds, np.inf))),
            "residuals": residuals, "bounds": bounds, "ratios": ratios,
            "max_ratio": float(np.max(ratios))}


def shift_null_compatibility_v200(
    estimate: float,
    epsilon_A: float,
    *,
    epsilon_wb: float = 1e-10,
) -> dict:
    bound = add_up(float(epsilon_A), float(epsilon_wb))
    residual = round_up(abs(float(estimate)))
    return {"passed": residual <= math.nextafter(bound, math.inf),
            "residual": residual, "bound": bound,
            "ratio": _compatibility_ratio(residual, bound)}


def unresolved_gate_contraction_bound_v200(
    contrast: np.ndarray,
    gate_response: np.ndarray,
    epsilon_G: float,
    whitebox_gradient: np.ndarray,
    physical_direction: np.ndarray,
    *,
    epsilon_g_wb: float = math.sqrt(768.0) * 1e-10,
) -> float:
    ell = np.asarray(contrast, dtype=np.float64)
    G = np.asarray(gate_response, dtype=np.float64)
    g = np.asarray(whitebox_gradient, dtype=np.float64)
    v = np.asarray(physical_direction, dtype=np.float64)
    if ell.shape != G.shape or g.shape != v.shape:
        raise ValueError("unresolved-bound vector shapes disagree")
    left = add_up(abs(float(ell @ G)), multiply_up(norm_up(ell), epsilon_G))
    right = add_up(abs(float(g @ v)), multiply_up(epsilon_g_wb, norm_up(v)))
    return multiply_up(left, right)


def minkowski_sum_interval(*intervals) -> tuple[float, float]:
    lower = upper = 0.0
    for interval in intervals:
        lo, hi = map(float, interval)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError("intervals must be finite and ordered")
        lower = round_down(np.float64(lower) + np.float64(lo))
        upper = round_up(np.float64(upper) + np.float64(hi))
    return lower, upper


def subtract_intervals(left, right) -> tuple[float, float]:
    l0, l1 = map(float, left)
    r0, r1 = map(float, right)
    if l0 > l1 or r0 > r1:
        raise ValueError("intervals must be ordered")
    return round_down(np.float64(l0) - np.float64(r1)), round_up(np.float64(l1) - np.float64(r0))


def absolute_value_interval(interval) -> tuple[float, float]:
    lower, upper = map(float, interval)
    if lower > upper:
        raise ValueError("interval must be ordered")
    high = round_up(max(abs(lower), abs(upper)))
    return (0.0, high) if lower <= 0 <= upper else (round_down(min(abs(lower), abs(upper))), high)


def worst_case_interval_rmse(actual, intervals) -> float:
    y = np.asarray(actual, dtype=np.float64)
    bounds = np.asarray(intervals, dtype=np.float64)
    if bounds.shape != (len(y), 2) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("intervals must have shape [N,2]")
    squared = np.maximum((y - bounds[:, 0]) ** 2, (y - bounds[:, 1]) ** 2)
    return round_up(np.sqrt(round_up(np.mean(np.nextafter(squared, np.inf)))))


def robust_interval_auc_lower_bound(labels, intervals) -> float:
    labels = np.asarray(labels, dtype=bool)
    bounds = np.asarray(intervals, dtype=np.float64)
    if bounds.shape != (len(labels), 2) or not labels.any() or labels.all():
        raise ValueError("robust AUC requires ordered intervals and both classes")
    positives = bounds[labels]
    negatives = bounds[~labels]
    score = 0.0
    for positive in positives:
        for negative in negatives:
            if positive[0] > negative[1]:
                score += 1.0
            elif positive[0] == negative[1]:
                score += 0.5
    return float(score / (len(positives) * len(negatives)))
