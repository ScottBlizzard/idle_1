"""Estimators for local interventional response signatures (IRS).

IRS compares how a patched computation and its clean target respond to the same
activation perturbations.  It is a local functional diagnostic, not a structural
circuit-identification certificate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_array(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return x


def isotropic_probes(
    n_items: int,
    n_probes: int,
    dimension: int,
    radius: float,
    rng: np.random.RandomState,
    *,
    shared_across_items: bool = False,
) -> np.ndarray:
    """Draw normalized Rademacher probes with shape [item, probe, dimension]."""
    if min(n_items, n_probes, dimension) < 1 or radius <= 0:
        raise ValueError("probe counts, dimension, and radius must be positive")
    leading = 1 if shared_across_items else n_items
    signs = rng.choice((-1.0, 1.0), size=(leading, n_probes, dimension))
    probes = radius * signs / np.sqrt(float(dimension))
    if shared_across_items:
        probes = np.repeat(probes, n_items, axis=0)
    return probes


def reference_chord_probes(
    centers: np.ndarray,
    reference: np.ndarray,
    n_probes: int,
    interpolation: float,
    nearest_pool: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Choose forward probes toward randomly selected nearby reference states.

    The returned deltas have shape [item, probe, dimension].  Chord interpolation
    is not assumed to remain on a nonlinear manifold; callers must audit endpoints.
    """
    centers = _finite_array(centers, "centers")
    reference = _finite_array(reference, "reference")
    if centers.ndim != 2 or reference.ndim != 2 or centers.shape[1] != reference.shape[1]:
        raise ValueError("centers and reference must be 2D arrays with equal width")
    if n_probes < 1 or nearest_pool < 1 or not (0 < interpolation <= 1):
        raise ValueError("invalid chord-probe configuration")
    pool_size = min(nearest_pool, len(reference))
    deltas = np.empty((len(centers), n_probes, centers.shape[1]), dtype=np.float64)
    ref_sq = np.einsum("ij,ij->i", reference, reference)
    for i, center in enumerate(centers):
        d2 = np.maximum(
            np.dot(center, center) - 2.0 * reference @ center + ref_sq,
            0.0,
        )
        pool = np.argpartition(d2, pool_size - 1)[:pool_size]
        nonself = pool[d2[pool] > 1e-20]
        if len(nonself):
            pool = nonself
        chosen = rng.choice(pool, size=n_probes, replace=len(pool) < n_probes)
        deltas[i] = interpolation * (reference[chosen] - center)
    if np.any(np.linalg.norm(deltas, axis=-1) <= 0):
        raise ValueError("reference probes contain a zero-length delta")
    return deltas


def forward_signature(
    center_outputs: np.ndarray,
    perturbed_outputs: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """Compute directional forward differences per unit activation displacement."""
    center_outputs = _finite_array(center_outputs, "center_outputs")
    perturbed_outputs = _finite_array(perturbed_outputs, "perturbed_outputs")
    deltas = _finite_array(deltas, "deltas")
    if center_outputs.ndim == 1:
        center_outputs = center_outputs[:, None]
    if perturbed_outputs.ndim == 2:
        perturbed_outputs = perturbed_outputs[:, :, None]
    if deltas.ndim != 3 or perturbed_outputs.ndim != 3:
        raise ValueError("perturbed_outputs and deltas must have item/probe axes")
    if perturbed_outputs.shape[:2] != deltas.shape[:2]:
        raise ValueError("perturbed_outputs and deltas disagree on item/probe shape")
    if center_outputs.shape != (len(deltas), perturbed_outputs.shape[2]):
        raise ValueError("center_outputs has incompatible shape")
    lengths = np.linalg.norm(deltas, axis=-1)
    if np.any(lengths <= 0):
        raise ValueError("all deltas must have positive length")
    return (perturbed_outputs - center_outputs[:, None, :]) / lengths[:, :, None]


def symmetric_signature(
    plus_outputs: np.ndarray,
    minus_outputs: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """Compute central directional differences for endpoints center +/- delta."""
    plus_outputs = _finite_array(plus_outputs, "plus_outputs")
    minus_outputs = _finite_array(minus_outputs, "minus_outputs")
    deltas = _finite_array(deltas, "deltas")
    if plus_outputs.ndim == 2:
        plus_outputs = plus_outputs[:, :, None]
    if minus_outputs.ndim == 2:
        minus_outputs = minus_outputs[:, :, None]
    if plus_outputs.shape != minus_outputs.shape:
        raise ValueError("plus_outputs and minus_outputs must have equal shape")
    if plus_outputs.shape[:2] != deltas.shape[:2]:
        raise ValueError("outputs and deltas disagree on item/probe shape")
    lengths = np.linalg.norm(deltas, axis=-1)
    if np.any(lengths <= 0):
        raise ValueError("all deltas must have positive length")
    return (plus_outputs - minus_outputs) / (2.0 * lengths[:, :, None])


@dataclass(frozen=True)
class IRSComparison:
    rmse: float
    normalized_rmse: float
    mean_cosine: float
    per_item_rmse: np.ndarray
    per_item_normalized_rmse: np.ndarray
    per_item_target_rms: np.ndarray
    normalization_floor_active: np.ndarray
    normalization_floor_fraction: float


def compare_signatures(
    patched: np.ndarray,
    target: np.ndarray,
    *,
    normalization_floor: float = 1e-8,
) -> IRSComparison:
    """Compare paired signatures using probe-mean, output-summed energy.

    The output coordinate axis is summed, not silently averaged.  Consequently,
    replicating a scalar output ``k`` times multiplies squared energy by ``k``.
    Scalar-output historical results are unchanged.  The returned floor mask
    makes ill-conditioned normalized scores auditable.
    """
    patched = _finite_array(patched, "patched")
    target = _finite_array(target, "target")
    if patched.shape != target.shape or patched.ndim != 3:
        raise ValueError("signatures must have equal [item, probe, output] shape")
    if normalization_floor <= 0:
        raise ValueError("normalization_floor must be positive")
    diff = patched - target
    per_item_rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=2), axis=1))
    target_rms = np.sqrt(np.mean(np.sum(target ** 2, axis=2), axis=1))
    floor_active = target_rms < normalization_floor
    per_item_normalized = per_item_rmse / np.maximum(target_rms, normalization_floor)
    patch_flat = patched.reshape(len(patched), -1)
    target_flat = target.reshape(len(target), -1)
    denom = np.linalg.norm(patch_flat, axis=1) * np.linalg.norm(target_flat, axis=1)
    cosine = np.divide(
        np.einsum("ij,ij->i", patch_flat, target_flat),
        denom,
        out=np.zeros(len(patched), dtype=np.float64),
        where=denom > normalization_floor,
    )
    return IRSComparison(
        rmse=float(np.sqrt(np.mean(np.sum(diff ** 2, axis=2)))),
        normalized_rmse=float(np.mean(per_item_normalized)),
        mean_cosine=float(np.mean(cosine)),
        per_item_rmse=per_item_rmse,
        per_item_normalized_rmse=per_item_normalized,
        per_item_target_rms=target_rms,
        normalization_floor_active=floor_active,
        normalization_floor_fraction=float(np.mean(floor_active)),
    )
