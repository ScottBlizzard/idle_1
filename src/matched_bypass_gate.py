"""Numerical matched-bypass gate identification independent of model hooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _array(x, name: str, ndim: int | None = None) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def richardson(full: np.ndarray, half: np.ndarray) -> np.ndarray:
    full = _array(full, "full")
    half = _array(half, "half")
    if full.shape != half.shape:
        raise ValueError("full and half estimates must have equal shape")
    return (4.0 * half - full) / 3.0


@dataclass(frozen=True)
class GateJet:
    G: np.ndarray
    C: np.ndarray
    J_path: np.ndarray
    H_path: np.ndarray
    H_control: np.ndarray


@dataclass(frozen=True)
class GateIdentification:
    A: np.ndarray
    P: np.ndarray
    D: np.ndarray
    delta_H: np.ndarray
    factorization_residual: float
    curvature_norm: float
    gate_response_norm: float


def identify_gate(jet: GateJet, *, curvature_floor: float = 0.0) -> GateIdentification:
    """Apply the exact matched-bypass inverse for one actual scalar gate.

    Shapes are ``G,C:[k]`` and ``J_path,H_path,H_control:[r1,k]``.  No
    regularization or pseudo-inverse is used because every upstream coordinate
    has an explicit axis measurement.
    """
    G = _array(jet.G, "G", 1)
    C = _array(jet.C, "C", 1)
    J = _array(jet.J_path, "J_path", 2)
    HP = _array(jet.H_path, "H_path", 2)
    HC = _array(jet.H_control, "H_control", 2)
    if G.shape != C.shape or J.shape != HP.shape or HP.shape != HC.shape:
        raise ValueError("gate-jet arrays have incompatible shapes")
    if J.shape[1] != len(G):
        raise ValueError("response output dimensions disagree")
    if curvature_floor < 0:
        raise ValueError("curvature_floor must be nonnegative")
    curvature_sq = float(C @ C)
    curvature_norm = float(np.sqrt(curvature_sq))
    if curvature_norm <= curvature_floor:
        raise ValueError("FAIL_CURVATURE: matched-bypass inverse is undefined")
    delta_H = HP - HC
    A = delta_H @ C / curvature_sq
    P = A[:, None] * G[None, :]
    D = J - P
    fitted = A[:, None] * C[None, :]
    denominator = max(float(np.linalg.norm(delta_H)), 1e-12)
    residual = float(np.linalg.norm(delta_H - fitted) / denominator)
    return GateIdentification(
        A=A,
        P=P,
        D=D,
        delta_H=delta_H,
        factorization_residual=residual,
        curvature_norm=curvature_norm,
        gate_response_norm=float(np.linalg.norm(G)),
    )


def _centered_first(evaluate: Callable[[float, float], np.ndarray], hx: float, axis: str):
    if axis == "x":
        return (evaluate(hx, 0.0) - evaluate(-hx, 0.0)) / (2.0 * hx)
    if axis == "z":
        return (evaluate(0.0, hx) - evaluate(0.0, -hx)) / (2.0 * hx)
    raise ValueError(axis)


def finite_gate_jet(
    path_evaluators: list[Callable[[float, float], np.ndarray]],
    control_evaluators: list[Callable[[float, float], np.ndarray]],
    *,
    radius_x: float,
    radius_z: float,
) -> GateJet:
    """Estimate one gate jet from scalar-axis path/control evaluators.

    Each list entry corresponds to one residual basis axis.  Evaluators must use
    identical centers, residual bypasses, gate writes, and downstream maps.
    """
    if radius_x <= 0 or radius_z <= 0:
        raise ValueError("finite-difference radii must be positive")
    if not path_evaluators or len(path_evaluators) != len(control_evaluators):
        raise ValueError("path/control evaluators must be nonempty and paired")
    base = _array(path_evaluators[0](0.0, 0.0), "base", 1)
    z_plus = _array(path_evaluators[0](0.0, radius_z), "z_plus", 1)
    z_minus = _array(path_evaluators[0](0.0, -radius_z), "z_minus", 1)
    G = (z_plus - z_minus) / (2.0 * radius_z)
    C = (z_plus - 2.0 * base + z_minus) / radius_z**2
    J_rows, HP_rows, HC_rows = [], [], []
    for path, control in zip(path_evaluators, control_evaluators):
        J_rows.append(_centered_first(path, radius_x, "x"))
        corners = {}
        control_corners = {}
        for sx in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners[sx, sz] = _array(path(sx * radius_x, sz * radius_z), "path")
                control_corners[sx, sz] = _array(
                    control(sx * radius_x, sz * radius_z), "control"
                )
        denominator = 4.0 * radius_x * radius_z
        HP_rows.append(
            (corners[1.0, 1.0] - corners[1.0, -1.0]
             - corners[-1.0, 1.0] + corners[-1.0, -1.0]) / denominator
        )
        HC_rows.append(
            (control_corners[1.0, 1.0] - control_corners[1.0, -1.0]
             - control_corners[-1.0, 1.0] + control_corners[-1.0, -1.0])
            / denominator
        )
    return GateJet(G, C, np.stack(J_rows), np.stack(HP_rows), np.stack(HC_rows))


def extrapolate_gate_jet(full: GateJet, half: GateJet) -> GateJet:
    return GateJet(
        G=richardson(full.G, half.G),
        C=richardson(full.C, half.C),
        J_path=richardson(full.J_path, half.J_path),
        H_path=richardson(full.H_path, half.H_path),
        H_control=richardson(full.H_control, half.H_control),
    )


def symmetric_relative_change(a: np.ndarray, b: np.ndarray, floor: float = 1e-12) -> float:
    a = _array(a, "a")
    b = _array(b, "b")
    return float(2.0 * np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), floor))


def cosine(a: np.ndarray, b: np.ndarray, floor: float = 1e-12) -> float:
    a = _array(a, "a").reshape(-1)
    b = _array(b, "b").reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > floor else 0.0


def expected_tensor_calls(
    probe_frame_dim: int = 5,
    n_gates: int = 10,
    n_radii: int = 2,
    n_systems: int = 2,
) -> int:
    if probe_frame_dim <= 0:
        raise ValueError("probe_frame_dim must be positive")
    per_gate_radius_system = 2 + 10 * probe_frame_dim
    return n_systems * (n_gates * n_radii * per_gate_radius_system + 1)


def reconstruct_cotangent(Q: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
    """Reconstruct the ambient cotangent without an ambient operator matrix."""
    frame = _array(Q, "Q", 2)
    coefficients = _array(A_hat, "A_hat", 1)
    if frame.shape[1] != coefficients.size:
        raise ValueError("frame and cotangent-coordinate dimensions disagree")
    return frame @ coefficients


def operator_action(G_hat: np.ndarray, g_hat: np.ndarray, v: np.ndarray) -> np.ndarray:
    response = _array(G_hat, "G_hat", 1)
    cotangent = _array(g_hat, "g_hat", 1)
    direction = _array(v, "v", 1)
    if cotangent.shape != direction.shape:
        raise ValueError("cotangent and physical direction dimensions disagree")
    return response * float(cotangent @ direction)


def operator_frobenius_norm(G_hat: np.ndarray, g_hat: np.ndarray) -> float:
    response = _array(G_hat, "G_hat", 1)
    cotangent = _array(g_hat, "g_hat", 1)
    return float(np.linalg.norm(response) * np.linalg.norm(cotangent))


def operator_inner_product(
    G1: np.ndarray, g1: np.ndarray, G2: np.ndarray, g2: np.ndarray
) -> float:
    response1 = _array(G1, "G1", 1)
    response2 = _array(G2, "G2", 1)
    cotangent1 = _array(g1, "g1", 1)
    cotangent2 = _array(g2, "g2", 1)
    if response1.shape != response2.shape or cotangent1.shape != cotangent2.shape:
        raise ValueError("operator factor dimensions disagree")
    return float((response1 @ response2) * (cotangent1 @ cotangent2))


def direct_bypass_in_common_frame(
    direct_gate_coordinates: np.ndarray,
    gate_frame: np.ndarray,
    common_frame: np.ndarray,
) -> np.ndarray:
    direct = _array(direct_gate_coordinates, "direct_gate_coordinates", 2)
    gate = _array(gate_frame, "gate_frame", 2)
    common = _array(common_frame, "common_frame", 2)
    if direct.shape[1] != gate.shape[1] or gate.shape[0] != common.shape[0]:
        raise ValueError("direct bypass and frame dimensions disagree")
    return direct @ gate.T @ common
