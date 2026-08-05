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
    residual_rank = int(rich_delta_H.shape[0])
    if inverse_admissible:
        A_max = (
            np.linalg.norm(rich_delta_H, axis=1) + epsilon_delta_H
        ) / (C_norm - epsilon_C)
        epsilon_A = (epsilon_delta_H + A_max * epsilon_C) / C_norm
        epsilon_P = epsilon_G * A_max + float(np.linalg.norm(rich_G)) * epsilon_A
        epsilon_P_F = float(np.linalg.norm(epsilon_P))
    else:
        A_max = np.full(residual_rank, np.inf, dtype=np.float64)
        epsilon_A = np.full(residual_rank, np.inf, dtype=np.float64)
        epsilon_P = np.full(residual_rank, np.inf, dtype=np.float64)
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


def certified_null_bound(
    contrast_norm: float,
    delta_norm: float,
    gate_response_norm: float,
    epsilon_G: float,
    whitebox_A_norm: float,
) -> float:
    return float(
        contrast_norm
        * delta_norm
        * (gate_response_norm + epsilon_G)
        * whitebox_A_norm
    )


def sum_item_error_bounds(bounds) -> float:
    return float(sum(float(value) for value in bounds))


def cell_error_bound(target_item_bounds, patched_item_bounds) -> float:
    target = np.asarray(tuple(target_item_bounds), dtype=np.float64)
    patched = np.asarray(tuple(patched_item_bounds), dtype=np.float64)
    if target.shape != patched.shape or target.size == 0:
        raise ValueError("target and patched item bounds must be nonempty and paired")
    return float(np.mean(target + patched))
