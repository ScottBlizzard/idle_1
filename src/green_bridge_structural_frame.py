"""Deterministic LayerNorm structural-envelope probe frames for protocol v1.3.

This module is deliberately endpoint-blind: it depends only on raw residual
anchors and frozen model parameters, never on responses, targets, or inverse
estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.linalg import qr
from threadpoolctl import threadpool_info, threadpool_limits

from green_bridge_spec import (
    ALL_GATE_FRAME_DIM,
    COMMON_FRAME_DIM,
    FIRST_ORDER_COEFFICIENT_SEED,
    FIRST_ORDER_RESIDUAL_DIRECTIONS,
    PROBE_FRAME_DIM,
    RESIDUAL_RADIUS_MULTIPLIER,
)


@dataclass(frozen=True)
class FrameExtension:
    frame: np.ndarray
    extension: np.ndarray
    extension_source: str


def _vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a vector, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def center_residual(residual) -> np.ndarray:
    value = _vector(residual, "residual")
    return value - np.mean(value, dtype=np.float64)


def normalize_atom(atom) -> np.ndarray:
    value = _vector(atom, "atom")
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0.0 else np.zeros_like(value)


def _canonicalize_columns(frame: np.ndarray) -> np.ndarray:
    result = np.asarray(frame, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return np.ascontiguousarray(result)


def _common_raw_atoms(resid_tar, resid_pat, resid_cor) -> np.ndarray:
    values = [_vector(x, name) for x, name in (
        (resid_tar, "resid_tar"), (resid_pat, "resid_pat"),
        (resid_cor, "resid_cor"),
    )]
    dimension = values[0].shape[0]
    if any(value.shape != (dimension,) for value in values):
        raise ValueError("the three residual anchors must have equal shape")
    atoms = [np.ones(dimension, dtype=np.float64) / np.sqrt(float(dimension))]
    atoms.extend(normalize_atom(center_residual(value)) for value in values)
    return np.column_stack(atoms)


def canonical_common_frame(resid_tar, resid_pat, resid_cor) -> np.ndarray:
    """Return the canonical unpivoted economic QR of the ordered raw atoms."""
    raw = _common_raw_atoms(resid_tar, resid_pat, resid_cor)

    def construct() -> np.ndarray:
        with threadpool_limits(limits=1, user_api="blas"):
            q, _ = qr(
                raw, mode="economic", pivoting=False, overwrite_a=False,
                check_finite=True,
            )
        return _canonicalize_columns(q)

    first = construct()
    second = construct()
    if not np.array_equal(first, second):
        raise RuntimeError("common-frame QR is not bitwise repeatable")
    if first.shape != (raw.shape[0], COMMON_FRAME_DIM):
        raise RuntimeError(f"invalid common-frame shape {first.shape}")
    return first


def _twice_residualized(frame: np.ndarray, atom: np.ndarray) -> np.ndarray:
    result = atom - frame @ (frame.T @ atom)
    return result - frame @ (frame.T @ result)


def extend_frame_with_atom(
    frame, atom, *, return_metadata: bool = False
) -> np.ndarray | FrameExtension:
    """Append an atom using frozen two-pass MGS and deterministic completion."""
    base = np.asarray(frame, dtype=np.float64)
    value = normalize_atom(atom)
    if base.ndim != 2 or base.shape[0] != value.shape[0]:
        raise ValueError("frame and atom dimensions disagree")
    if not np.isfinite(base).all():
        raise ValueError("frame contains NaN or infinity")
    residual = _twice_residualized(base, value)
    source = "structural_atom"
    norm = float(np.linalg.norm(residual))
    if norm == 0.0:
        source = "deterministic_standard_basis_completion"
        residual = None
        for coordinate in range(base.shape[0]):
            candidate = np.zeros(base.shape[0], dtype=np.float64)
            candidate[coordinate] = 1.0
            candidate = _twice_residualized(base, candidate)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > 0.5:
                residual = candidate / candidate_norm
                break
        if residual is None:
            raise ValueError("no deterministic standard-basis completion exists")
    else:
        residual = residual / norm
    residual = _canonicalize_columns(residual[:, None])[:, 0]
    extended = np.ascontiguousarray(np.column_stack((base, residual)))
    metadata = FrameExtension(extended, residual, source)
    return metadata if return_metadata else metadata.frame


def layernorm_gate_atom(ln_scale, mlp_input_weight, gate: int | None = None) -> np.ndarray:
    gamma = _vector(ln_scale, "ln_scale")
    weight = np.asarray(mlp_input_weight, dtype=np.float64)
    if weight.ndim == 2:
        if gate is None or not 0 <= int(gate) < weight.shape[1]:
            raise ValueError("a valid actual MLP gate coordinate is required")
        weight = weight[:, int(gate)]
    weight = _vector(weight, "mlp_input_weight")
    if weight.shape != gamma.shape:
        raise ValueError("LayerNorm scale and MLP input weight shapes disagree")
    return gamma * weight


def canonical_gate_frame(common_frame, gate_atom, *, return_metadata: bool = False):
    result = extend_frame_with_atom(
        common_frame, gate_atom, return_metadata=return_metadata
    )
    frame = result.frame if isinstance(result, FrameExtension) else result
    if frame.shape[1] != PROBE_FRAME_DIM:
        raise ValueError(f"gate frame must have {PROBE_FRAME_DIM} columns")
    return result


def canonical_all_gate_frame(
    common_frame, gate_atoms, *, return_metadata: bool = False
):
    frame = np.asarray(common_frame, dtype=np.float64)
    metadata: list[FrameExtension] = []
    for atom in gate_atoms:
        extension = extend_frame_with_atom(frame, atom, return_metadata=True)
        metadata.append(extension)
        frame = extension.frame
    if frame.shape[1] != ALL_GATE_FRAME_DIM:
        raise ValueError(f"all-gate frame must have {ALL_GATE_FRAME_DIM} columns")
    return (frame, metadata) if return_metadata else frame


def frame_containment_metrics(frame, raw_atoms) -> dict[str, float]:
    q = np.asarray(frame, dtype=np.float64)
    atoms = np.asarray(raw_atoms, dtype=np.float64)
    if atoms.ndim == 1:
        atoms = atoms[:, None]
    if q.ndim != 2 or atoms.ndim != 2 or q.shape[0] != atoms.shape[0]:
        raise ValueError("frame and raw atoms must be compatible matrices")
    residual = atoms - q @ (q.T @ atoms)
    identity = np.eye(q.shape[1], dtype=np.float64)
    return {
        "orthogonal_max_abs": float(np.max(np.abs(q.T @ q - identity))),
        "atom_residual_relative": float(
            np.linalg.norm(residual) / max(float(np.linalg.norm(atoms)), 1e-12)
        ),
        "atom_residual_frobenius": float(np.linalg.norm(residual)),
    }


def frame_sha256(frame) -> str:
    value = np.ascontiguousarray(np.asarray(frame, dtype=np.float64))
    return hashlib.sha256(value.tobytes()).hexdigest()


def first_order_coefficient_directions() -> np.ndarray:
    """Return the frozen 250-by-14 PCG64 coefficient design."""
    rng = np.random.Generator(np.random.PCG64(FIRST_ORDER_COEFFICIENT_SEED))
    directions = [
        np.eye(ALL_GATE_FRAME_DIM, dtype=np.float64)[index]
        for index in range(ALL_GATE_FRAME_DIM)
    ]
    while len(directions) < FIRST_ORDER_RESIDUAL_DIRECTIONS:
        value = rng.standard_normal(ALL_GATE_FRAME_DIM)
        value /= np.linalg.norm(value)
        first = int(np.flatnonzero(np.abs(value) > 0.0)[0])
        if value[first] < 0.0:
            value = -value
        if any(abs(float(value @ old)) > 0.999999 for old in directions):
            continue
        directions.append(value)
    return np.ascontiguousarray(np.stack(directions), dtype=np.float64)


def residual_radius(resid_tar, resid_pat, resid_cor) -> dict[str, float | bool]:
    tar = _vector(resid_tar, "resid_tar")
    pat = _vector(resid_pat, "resid_pat")
    cor = _vector(resid_cor, "resid_cor")
    if tar.shape != pat.shape or pat.shape != cor.shape:
        raise ValueError("the three residual anchors must have equal shape")
    chord_rms = float(np.linalg.norm(tar - cor) / np.sqrt(float(tar.size)))
    radius = RESIDUAL_RADIUS_MULTIPLIER * chord_rms
    scale = float(np.median([
        np.sqrt(np.mean(value * value, dtype=np.float64))
        for value in (tar, pat, cor)
    ]))
    floor = 2.0 ** -10 * scale
    return {
        "h_x": radius,
        "half_h_x": 0.5 * radius,
        "chord_rms": chord_rms,
        "residual_scale": scale,
        "floor": floor,
        "floor_pass": radius >= floor,
    }


def target_physical_vector(resid_tar, resid_cor, h_x: float) -> np.ndarray:
    chord = _vector(resid_tar, "resid_tar") - _vector(resid_cor, "resid_cor")
    norm = float(np.linalg.norm(chord))
    if h_x < 0.0:
        raise ValueError("h_x must be nonnegative")
    return np.zeros_like(chord) if norm == 0.0 else h_x * chord / norm


def blas_threadpools() -> list[dict]:
    """Expose audit metadata without making it part of frame construction."""
    return threadpool_info()
