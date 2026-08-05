"""Model-free rank-five donor-projector and radius audits."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class BasisAuditError(RuntimeError):
    gate: str
    detail: str

    def __str__(self) -> str:
        return f"{self.gate}: {self.detail}"


@dataclass(frozen=True)
class CanonicalBasis:
    U: np.ndarray
    singular: np.ndarray
    projector: np.ndarray
    threadpools: list[dict]
    orthogonal_max_abs: float
    repeated_svd_bitwise_equal: bool


def matrix_sha256(matrix: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
    return hashlib.sha256(value.tobytes()).hexdigest()


def canonical_rank_basis(chords: np.ndarray, *, rank: int = 5) -> CanonicalBasis:
    from scipy.linalg import svd
    from threadpoolctl import threadpool_info, threadpool_limits

    raw = np.asarray(chords)
    if raw.dtype != np.float64:
        raise TypeError(f"chord matrix dtype must be float64, got {raw.dtype}")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 768:
        raise ValueError(f"invalid chord matrix shape: {matrix.shape}")
    if matrix.shape[0] <= rank:
        raise ValueError("chord matrix has too few rows")
    if rank != 5:
        raise ValueError("the protocol-v1.2 residual rank is exactly five")
    if not np.isfinite(matrix).all():
        raise ValueError("chord matrix contains NaN or infinity")

    def one_svd():
        _, singular, vt = svd(
            matrix,
            full_matrices=False,
            lapack_driver="gesvd",
            overwrite_a=False,
            check_finite=True,
        )
        basis = vt[:rank].T.copy()
        for column in range(rank):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0:
                basis[:, column] *= -1.0
        return basis, singular

    with threadpool_limits(limits=1, user_api="blas"):
        pools = [
            dict(row) for row in threadpool_info()
            if row.get("user_api") == "blas"
        ]
        if not pools:
            raise BasisAuditError(
                "08A_BASIS_THREAD_CONTRACT",
                "no loaded BLAS runtime was introspectable",
            )
        if any(int(row.get("num_threads", -1)) != 1 for row in pools):
            raise BasisAuditError(
                "08A_BASIS_THREAD_CONTRACT",
                repr(pools),
            )
        basis_1, singular_1 = one_svd()
        basis_2, singular_2 = one_svd()

    repeated = bool(
        np.array_equal(singular_1, singular_2)
        and np.array_equal(basis_1, basis_2)
    )
    if not repeated:
        raise BasisAuditError(
            "08A_BASIS_THREAD_CONTRACT",
            "repeated canonical SVD was not bitwise equal",
        )
    orthogonal_error = float(np.max(
        np.abs(basis_1.T @ basis_1 - np.eye(rank, dtype=np.float64))
    ))
    if orthogonal_error > 5e-13:
        raise BasisAuditError(
            "08A_BASIS_THREAD_CONTRACT",
            f"basis orthogonality error {orthogonal_error}",
        )
    projector = basis_1 @ basis_1.T
    return CanonicalBasis(
        U=basis_1,
        singular=singular_1,
        projector=projector,
        threadpools=pools,
        orthogonal_max_abs=orthogonal_error,
        repeated_svd_bitwise_equal=repeated,
    )


def principal_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (768, 5) or right.shape != (768, 5):
        raise ValueError("principal-angle bases must have shape [768,5]")
    smallest = float(np.linalg.svd(left.T @ right, compute_uv=False).min())
    return float(math.degrees(math.acos(float(np.clip(smallest, -1.0, 1.0)))))


def holdout_efficiency(
    holdout_chords: np.ndarray,
    fit_basis: np.ndarray,
    holdout_singular: np.ndarray,
) -> float:
    matrix = np.asarray(holdout_chords, dtype=np.float64)
    captured = float(np.linalg.norm(matrix @ fit_basis, ord="fro") ** 2)
    optimal = float(np.sum(np.asarray(holdout_singular, dtype=np.float64)[:5] ** 2))
    return captured / optimal


def spectrum_passes(singular: np.ndarray, *, gap_min: float = 1.10) -> bool:
    values = np.asarray(singular, dtype=np.float64)
    if values.ndim != 1 or values.size < 6 or values[0] <= 0 or values[5] <= 0:
        return False
    return bool(values[4] / values[5] >= gap_min and values[4] / values[0] >= 1e-4)


def angle_passes(angle_degrees: float) -> bool:
    return bool(float(angle_degrees) <= 15.0)


def efficiency_passes(efficiency: float) -> bool:
    return bool(float(efficiency) >= 0.90)


def bootstrap_q95_passes(angle_degrees: float) -> bool:
    return bool(float(angle_degrees) <= 15.0)


def leave_one_noun_audit(
    fit_chords: np.ndarray,
    fit_nouns: np.ndarray,
    full_basis: np.ndarray,
    ordered_nouns,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    bases, angles, floors = [], {}, {}
    nouns = np.asarray(fit_nouns)
    for noun in ordered_nouns:
        result = canonical_rank_basis(fit_chords[nouns != noun], rank=5)
        floor = float(result.singular[4] / result.singular[0])
        angle = principal_angle_degrees(full_basis, result.U)
        bases.append(result.U)
        floors[str(noun)] = floor
        angles[str(noun)] = angle
        if floor < 1e-4 or not angle_passes(angle):
            raise BasisAuditError(
                "08F_BASIS_LEAVE_ONE_NOUN",
                f"leave-{noun}: floor={floor}, angle={angle}",
            )
    return np.stack(bases), angles, floors


def noun_cluster_bootstrap(
    fit_chords: np.ndarray,
    fit_nouns: np.ndarray,
    full_basis: np.ndarray,
    ordered_nouns,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    seed = int.from_bytes(
        hashlib.sha256(
            b"idle1-gt-bridge-basis-v2-20260805:noun-bootstrap"
        ).digest()[:8],
        "big",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = rng.integers(0, 16, size=(256, 16), endpoint=False)
    nouns = np.asarray(fit_nouns)
    blocks = [np.asarray(fit_chords[nouns == noun], dtype=np.float64) for noun in ordered_nouns]
    if any(block.shape != (32, 768) for block in blocks):
        raise ValueError("each bootstrap noun block must have shape [32,768]")
    angles = np.empty(256, dtype=np.float64)
    floor_failures = 0
    for index, noun_indices in enumerate(sampled):
        matrix = np.concatenate([blocks[int(noun_index)] for noun_index in noun_indices], axis=0)
        result = canonical_rank_basis(matrix, rank=5)
        if result.singular[4] / result.singular[0] < 1e-4:
            floor_failures += 1
            angles[index] = 90.0
        else:
            angles[index] = principal_angle_degrees(full_basis, result.U)
    q95 = float(np.quantile(angles, 0.95, method="higher"))
    if not bootstrap_q95_passes(q95):
        raise BasisAuditError(
            "08G_BASIS_NOUN_BOOTSTRAP",
            f"q95_higher={q95}, floor_failures={floor_failures}",
        )
    return sampled, angles, floor_failures, q95


def fit_rank5_basis(
    fit_chords: np.ndarray,
    holdout_chords: np.ndarray,
    fit_nouns: np.ndarray,
    ordered_nouns,
) -> dict:
    fit = canonical_rank_basis(fit_chords, rank=5)
    holdout = canonical_rank_basis(holdout_chords, rank=5)
    fit_gap = float(fit.singular[4] / fit.singular[5])
    fit_floor = float(fit.singular[4] / fit.singular[0])
    if not spectrum_passes(fit.singular):
        raise BasisAuditError(
            "08B_BASIS_FIT_SPECTRUM",
            f"sigma5/sigma6={fit_gap}, sigma5/sigma1={fit_floor}",
        )
    holdout_gap = float(holdout.singular[4] / holdout.singular[5])
    holdout_floor = float(holdout.singular[4] / holdout.singular[0])
    if not spectrum_passes(holdout.singular):
        raise BasisAuditError(
            "08C_BASIS_HOLDOUT_SPECTRUM",
            f"sigma5/sigma6={holdout_gap}, sigma5/sigma1={holdout_floor}",
        )
    angle = principal_angle_degrees(fit.U, holdout.U)
    if not angle_passes(angle):
        raise BasisAuditError("08D_BASIS_FIT_HOLDOUT_ANGLE", f"angle={angle}")
    efficiency = holdout_efficiency(holdout_chords, fit.U, holdout.singular)
    if not efficiency_passes(efficiency):
        raise BasisAuditError("08E_BASIS_HOLDOUT_ENERGY", f"efficiency={efficiency}")
    leave_bases, leave_angles, leave_floors = leave_one_noun_audit(
        fit_chords, fit_nouns, fit.U, ordered_nouns
    )
    sampled, bootstrap_angles, floor_failures, q95 = noun_cluster_bootstrap(
        fit_chords, fit_nouns, fit.U, ordered_nouns
    )
    return {
        "U": fit.U,
        "projector": fit.projector,
        "singular_fit": fit.singular,
        "U_holdout": holdout.U,
        "singular_holdout": holdout.singular,
        "leave_one_bases": leave_bases,
        "leave_one_angles": leave_angles,
        "leave_one_floors": leave_floors,
        "sampled_noun_indices": sampled,
        "bootstrap_angles": bootstrap_angles,
        "bootstrap_floor_failures": floor_failures,
        "bootstrap_q95": q95,
        "audit": {
            "rank": 5,
            "fit": {"sigma5_over_sigma6": fit_gap, "sigma5_over_sigma1": fit_floor},
            "holdout": {"sigma5_over_sigma6": holdout_gap, "sigma5_over_sigma1": holdout_floor},
            "fit_holdout_angle_degrees": angle,
            "holdout_efficiency": efficiency,
            "leave_one_noun_angles_degrees": leave_angles,
            "leave_one_noun_rank_floors": leave_floors,
            "bootstrap": {
                "replicates": 256,
                "quantile": 0.95,
                "quantile_method": "higher",
                "q95_higher_degrees": q95,
                "rank_floor_failures": floor_failures,
                "sampled_noun_indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(),
            },
            "threadpools": fit.threadpools,
            "repeated_svd_bitwise_equal": fit.repeated_svd_bitwise_equal,
            "orthogonal_max_abs": fit.orthogonal_max_abs,
        },
    }


def construct_rank5_radii(
    radius_chords: np.ndarray,
    radius_clean_pre: np.ndarray,
    radius_corrupt_pre: np.ndarray,
    radius_rms_anchor: np.ndarray,
    radius_nouns: np.ndarray,
    basis: np.ndarray,
    ordered_nouns,
) -> dict:
    projected = np.asarray(radius_chords, dtype=np.float64) @ basis
    sigma_x = float(np.median(np.linalg.norm(projected, axis=1) / math.sqrt(5.0)))
    h1 = 0.20 * sigma_x
    residual_floor = 2.0**-10 * float(np.median(radius_rms_anchor))
    if h1 < residual_floor:
        raise BasisAuditError("09_RADIUS_FLOOR", f"h1={h1} < {residual_floor}")
    clean = np.asarray(radius_clean_pre, dtype=np.float64)
    corrupt = np.asarray(radius_corrupt_pre, dtype=np.float64)
    pooled = np.concatenate([clean, corrupt], axis=0)
    medians = np.median(pooled, axis=0)
    mad = np.median(np.abs(pooled - medians), axis=0)
    gate_sigma = np.maximum(1.4826 * mad, np.median(np.abs(clean - corrupt), axis=0))
    h2 = 0.20 * gate_sigma
    gate_floor = 2.0**-10 * np.maximum(1.0, np.median(np.abs(pooled), axis=0))
    if np.any(h2 < gate_floor):
        raise BasisAuditError(
            "09_RADIUS_FLOOR",
            f"gate slots {np.flatnonzero(h2 < gate_floor).tolist()}",
        )
    nouns = np.asarray(radius_nouns)
    changes = {}
    for noun in ordered_nouns:
        keep = nouns != noun
        projected_leave = np.asarray(radius_chords, dtype=np.float64)[keep] @ basis
        sx = float(np.median(np.linalg.norm(projected_leave, axis=1) / math.sqrt(5.0)))
        cp, xp = clean[keep], corrupt[keep]
        pp = np.concatenate([cp, xp], axis=0)
        pm = np.median(pp, axis=0)
        gs = np.maximum(
            1.4826 * np.median(np.abs(pp - pm), axis=0),
            np.median(np.abs(cp - xp), axis=0),
        )
        change = max(
            abs(sx - sigma_x) / sigma_x,
            float(np.max(np.abs(gs - gate_sigma) / gate_sigma)),
        )
        changes[str(noun)] = change
        if change > 0.20:
            raise BasisAuditError("09_RADIUS_STABILITY", f"leave-{noun} change={change}")
    return {
        "sigma_x": sigma_x,
        "h1": h1,
        "h2": h2.tolist(),
        "residual_floor": residual_floor,
        "gate_floor": gate_floor.tolist(),
        "leave_one_radius_change": changes,
    }
