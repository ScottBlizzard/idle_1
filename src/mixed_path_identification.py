"""Fixed-basis mixed-path identification for the ASG-RDAG theory gate.

This module implements only the restricted estimator licensed by
``analysis/GPTPRO_THEORY_PACKAGE_20260805.md``.  It does not turn arbitrary
transformer block-output Hessians into structural path effects.  Structural
recovery is refused when a design is rank deficient or a declared gate has
unknown/degenerate curvature.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np


class IdentificationError(ValueError):
    """Raised when the declared design cannot identify the requested object."""


class CurvatureError(IdentificationError):
    """Raised when the known gate-curvature inverse is undefined or unstable."""


class RankDeficiencyError(IdentificationError):
    """Raised instead of returning a ridge-regularized pseudo-certificate."""


def _finite_array(x: np.ndarray, name: str, *, ndim: int | None = None) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return out


def _positive_vector(x: np.ndarray, name: str, length: int) -> np.ndarray:
    out = _finite_array(x, name, ndim=1)
    if len(out) != length:
        raise ValueError(f"{name} must have length {length}; got {len(out)}")
    if np.any(out <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return out


def numerical_rank_threshold(matrix: np.ndarray) -> float:
    """Protocol rank threshold: 1e3 * eps * max(m, d) * sigma_max."""
    matrix = _finite_array(matrix, "matrix", ndim=2)
    singular = np.linalg.svd(matrix, compute_uv=False)
    sigma_max = float(singular[0]) if len(singular) else 0.0
    return 1e3 * np.finfo(np.float64).eps * max(matrix.shape) * sigma_max


@dataclass(frozen=True)
class DesignDiagnostics:
    n_rows: int
    dimension: int
    rank: int
    required_rank: int
    singular_values: np.ndarray
    covariance_eigenvalues: np.ndarray
    lambda_min: float
    lambda_max: float
    condition_number: float
    effective_rank: float
    rank_threshold: float

    @property
    def full_rank(self) -> bool:
        return self.rank == self.required_rank


def design_diagnostics(directions: np.ndarray) -> DesignDiagnostics:
    """Return rank and empirical second-moment diagnostics for a probe design."""
    directions = _finite_array(directions, "directions", ndim=2)
    n_rows, dimension = directions.shape
    if n_rows < 1 or dimension < 1:
        raise ValueError("directions must be nonempty")
    singular = np.linalg.svd(directions, compute_uv=False)
    threshold = numerical_rank_threshold(directions)
    rank = int(np.sum(singular > threshold))
    moment = directions.T @ directions / float(n_rows)
    eigenvalues = np.linalg.eigvalsh(moment)
    lambda_min = float(eigenvalues[0])
    lambda_max = float(eigenvalues[-1])
    condition = (
        float(lambda_max / lambda_min)
        if lambda_min > max(threshold * threshold / max(n_rows, 1), 0.0)
        else float("inf")
    )
    effective_rank = (
        float(np.trace(moment) / lambda_max) if lambda_max > 0 else 0.0
    )
    return DesignDiagnostics(
        n_rows=n_rows,
        dimension=dimension,
        rank=rank,
        required_rank=dimension,
        singular_values=singular,
        covariance_eigenvalues=eigenvalues,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        condition_number=condition,
        effective_rank=effective_rank,
        rank_threshold=threshold,
    )


def require_full_rank(directions: np.ndarray, name: str) -> DesignDiagnostics:
    diagnostics = design_diagnostics(directions)
    if not diagnostics.full_rank:
        raise RankDeficiencyError(
            f"{name} rank {diagnostics.rank} < required "
            f"{diagnostics.required_rank}; ridge fallback is prohibited"
        )
    return diagnostics


def cartesian_kronecker_design(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Return rows ``v (x) u`` for every Cartesian pair in U then V order."""
    U = _finite_array(U, "U", ndim=2)
    V = _finite_array(V, "V", ndim=2)
    return np.stack([np.kron(v, u) for u, v in product(U, V)], axis=0)


def paired_kronecker_design(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Return rows ``v_n (x) u_n`` for paired, non-Cartesian probes."""
    U = _finite_array(U, "U", ndim=2)
    V = _finite_array(V, "V", ndim=2)
    if len(U) != len(V):
        raise ValueError("paired U and V must contain the same number of rows")
    return np.stack([np.kron(v, u) for u, v in zip(U, V)], axis=0)


def curvature_ratio(
    psi_prime: np.ndarray,
    psi_second: np.ndarray,
    *,
    curvature_floor: float = 0.0,
    ratio_ceiling: float | None = None,
) -> np.ndarray:
    """Compute the architecture-derived ``psi' / psi''`` correction.

    ``curvature_floor`` and ``ratio_ceiling`` must be frozen from development
    data before confirmatory use.  No estimate is returned on failure.
    """
    prime = _finite_array(psi_prime, "psi_prime", ndim=1)
    second = _finite_array(psi_second, "psi_second", ndim=1)
    if prime.shape != second.shape:
        raise ValueError("psi_prime and psi_second must have equal shape")
    if curvature_floor < 0:
        raise ValueError("curvature_floor must be nonnegative")
    if np.any(np.abs(second) <= curvature_floor):
        raise CurvatureError(
            "FAIL_CURVATURE: a declared gate has zero or sub-threshold curvature"
        )
    rho = prime / second
    if not np.isfinite(rho).all():
        raise CurvatureError("FAIL_CURVATURE: curvature ratio is nonfinite")
    if ratio_ceiling is not None:
        if ratio_ceiling <= 0:
            raise ValueError("ratio_ceiling must be positive")
        if np.any(np.abs(rho) > ratio_ceiling):
            raise CurvatureError("FAIL_CURVATURE: curvature ratio exceeds ceiling")
    return rho


def central_first_response(
    plus_outputs: np.ndarray,
    minus_outputs: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Central first response with output shape ``[direction, output]``."""
    plus = _finite_array(plus_outputs, "plus_outputs", ndim=2)
    minus = _finite_array(minus_outputs, "minus_outputs", ndim=2)
    if plus.shape != minus.shape:
        raise ValueError("plus_outputs and minus_outputs must have equal shape")
    radius = _positive_vector(radii, "radii", len(plus))
    return (plus - minus) / (2.0 * radius[:, None])


def mixed_four_corner_response(
    y_pp: np.ndarray,
    y_pm: np.ndarray,
    y_mp: np.ndarray,
    y_mm: np.ndarray,
    radius1: np.ndarray,
    radius2: np.ndarray,
) -> np.ndarray:
    """Four-point mixed response in exact ``++ - +- - -+ + --`` order."""
    pp = _finite_array(y_pp, "y_pp", ndim=3)
    pm = _finite_array(y_pm, "y_pm", ndim=3)
    mp = _finite_array(y_mp, "y_mp", ndim=3)
    mm = _finite_array(y_mm, "y_mm", ndim=3)
    if not (pp.shape == pm.shape == mp.shape == mm.shape):
        raise ValueError("all mixed corners must have equal [m1, m2, k] shape")
    r = _positive_vector(radius1, "radius1", pp.shape[0])
    t = _positive_vector(radius2, "radius2", pp.shape[1])
    denominator = 4.0 * r[:, None, None] * t[None, :, None]
    return (pp - pm - mp + mm) / denominator


def mixed_response_from_corner_array(
    corners: np.ndarray,
    radius1: np.ndarray,
    radius2: np.ndarray,
) -> np.ndarray:
    """Decode ``[m1,m2,sign1,sign2,k]`` with sign order ``[-1,+1]``."""
    corners = _finite_array(corners, "corners", ndim=5)
    if corners.shape[2:4] != (2, 2):
        raise ValueError("corner sign axes must both have length two")
    return mixed_four_corner_response(
        corners[:, :, 1, 1, :],
        corners[:, :, 1, 0, :],
        corners[:, :, 0, 1, :],
        corners[:, :, 0, 0, :],
        radius1,
        radius2,
    )


@dataclass(frozen=True)
class StructuralEstimate:
    J1: np.ndarray
    J2: np.ndarray
    H: np.ndarray
    P: np.ndarray
    D: np.ndarray
    G: np.ndarray
    U_diagnostics: DesignDiagnostics
    V_diagnostics: DesignDiagnostics
    W_diagnostics: DesignDiagnostics


def fit_paired_mixed_tensor(
    U_pairs: np.ndarray,
    V_pairs: np.ndarray,
    Y12_pairs: np.ndarray,
) -> tuple[np.ndarray, DesignDiagnostics]:
    """Fit ``H`` from arbitrary paired mixed probes when Kronecker-complete.

    The returned tensor has shape ``[output, M1, M2]``.  Full marginal ranks do
    not suffice; this function explicitly refuses a deficient paired Kronecker
    design.
    """
    U = _finite_array(U_pairs, "U_pairs", ndim=2)
    V = _finite_array(V_pairs, "V_pairs", ndim=2)
    Y = _finite_array(Y12_pairs, "Y12_pairs", ndim=2)
    if len(U) != len(V) or len(U) != len(Y):
        raise ValueError("paired directions and responses must share a row count")
    W = paired_kronecker_design(U, V)
    diagnostics = require_full_rank(W, "paired Kronecker design W")
    coefficient = np.linalg.lstsq(W, Y, rcond=None)[0]
    H = np.stack(
        [
            coefficient[:, alpha].reshape(U.shape[1], V.shape[1], order="F")
            for alpha in range(Y.shape[1])
        ],
        axis=0,
    )
    return H, diagnostics


def structural_inverse(
    J1: np.ndarray,
    J2: np.ndarray,
    H: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(P, D, G)`` under the ASG-RDAG gate assumptions."""
    J1 = _finite_array(J1, "J1", ndim=2)
    J2 = _finite_array(J2, "J2", ndim=2)
    H = _finite_array(H, "H", ndim=3)
    rho = _finite_array(rho, "rho", ndim=1)
    if H.shape != (J1.shape[0], J1.shape[1], J2.shape[1]):
        raise ValueError("H shape must be [output, M1, M2]")
    if len(rho) != J2.shape[1]:
        raise ValueError("rho length must equal the M2 dimension")
    P = H * rho[None, None, :]
    G = J2.copy()
    D = J1 - P.sum(axis=2)
    return P, D, G


def fit_cartesian_structural_tensors(
    U: np.ndarray,
    V: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    Y12: np.ndarray,
    rho: np.ndarray,
) -> StructuralEstimate:
    """Fit the complete Cartesian design using unregularized SVD least squares."""
    U = _finite_array(U, "U", ndim=2)
    V = _finite_array(V, "V", ndim=2)
    Y1 = _finite_array(Y1, "Y1", ndim=2)
    Y2 = _finite_array(Y2, "Y2", ndim=2)
    Y12 = _finite_array(Y12, "Y12", ndim=3)
    if Y1.shape[0] != len(U) or Y2.shape[0] != len(V):
        raise ValueError("first-response rows must match their design rows")
    if Y1.shape[1] != Y2.shape[1]:
        raise ValueError("Y1 and Y2 must share the output dimension")
    if Y12.shape != (len(U), len(V), Y1.shape[1]):
        raise ValueError("Y12 must have shape [m1, m2, output]")
    u_diag = require_full_rank(U, "U")
    v_diag = require_full_rank(V, "V")
    W = cartesian_kronecker_design(U, V)
    w_diag = require_full_rank(W, "Cartesian Kronecker design W")

    J1 = np.linalg.lstsq(U, Y1, rcond=None)[0].T
    J2 = np.linalg.lstsq(V, Y2, rcond=None)[0].T
    H = np.empty((Y1.shape[1], U.shape[1], V.shape[1]), dtype=np.float64)
    for alpha in range(Y1.shape[1]):
        left_solution = np.linalg.lstsq(U, Y12[:, :, alpha], rcond=None)[0]
        H[alpha] = np.linalg.lstsq(V, left_solution.T, rcond=None)[0].T
    P, D, G = structural_inverse(J1, J2, H, rho)
    return StructuralEstimate(J1, J2, H, P, D, G, u_diag, v_diag, w_diag)


def active_edge_inverse(
    G: np.ndarray,
    P: np.ndarray,
    psi_prime: np.ndarray,
    *,
    visibility_floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover visible ``C`` columns and upstream ``A`` rows gate by gate."""
    G = _finite_array(G, "G", ndim=2)
    P = _finite_array(P, "P", ndim=3)
    prime = _finite_array(psi_prime, "psi_prime", ndim=1)
    if P.shape[0] != G.shape[0] or P.shape[2] != G.shape[1]:
        raise ValueError("P and G dimensions disagree")
    if len(prime) != G.shape[1]:
        raise ValueError("psi_prime length differs from the gate dimension")
    if visibility_floor < 0:
        raise ValueError("visibility_floor must be nonnegative")
    if np.any(np.abs(prime) <= visibility_floor):
        raise IdentificationError("a gate has zero or sub-threshold psi_prime")
    C = G / prime[None, :]
    A = np.empty((G.shape[1], P.shape[1]), dtype=np.float64)
    for j in range(G.shape[1]):
        denom = float(G[:, j] @ G[:, j])
        if denom <= visibility_floor * visibility_floor:
            raise IdentificationError(f"gate {j} is output-null or sub-threshold")
        A[j] = (G[:, j] @ P[:, :, j]) / denom
    return A, C


def factorization_residuals(
    G: np.ndarray,
    P: np.ndarray,
    *,
    numerical_floor: float = 1e-12,
) -> np.ndarray:
    """Return the necessary ASG rank-one slice residual for every gate."""
    G = _finite_array(G, "G", ndim=2)
    P = _finite_array(P, "P", ndim=3)
    if numerical_floor <= 0:
        raise ValueError("numerical_floor must be positive")
    if P.shape[0] != G.shape[0] or P.shape[2] != G.shape[1]:
        raise ValueError("P and G dimensions disagree")
    residual = np.empty(G.shape[1], dtype=np.float64)
    for j in range(G.shape[1]):
        denom_g = float(G[:, j] @ G[:, j])
        if denom_g > numerical_floor**2:
            a = (G[:, j] @ P[:, :, j]) / denom_g
            fitted = G[:, j, None] * a[None, :]
        else:
            fitted = np.zeros_like(P[:, :, j])
        residual[j] = np.linalg.norm(P[:, :, j] - fitted) / max(
            np.linalg.norm(P[:, :, j]), numerical_floor
        )
    return residual


def local_path_effect(
    P: np.ndarray,
    upstream_displacement: np.ndarray,
    output_contrast: np.ndarray | None = None,
) -> np.ndarray | float:
    """Compute the absolute-unit local path effect ``sum_j(P_j) delta``."""
    P = _finite_array(P, "P", ndim=3)
    delta = _finite_array(upstream_displacement, "upstream_displacement", ndim=1)
    if len(delta) != P.shape[1]:
        raise ValueError("upstream displacement has incompatible dimension")
    path_vector = P.sum(axis=2) @ delta
    if output_contrast is None:
        return path_vector
    contrast = _finite_array(output_contrast, "output_contrast", ndim=1)
    if len(contrast) != P.shape[0]:
        raise ValueError("output contrast has incompatible dimension")
    return float(contrast @ path_vector)


def structural_response_energy(
    delta_J1: np.ndarray,
    delta_J2: np.ndarray,
    delta_P: np.ndarray,
    U_eval: np.ndarray,
    V_eval: np.ndarray,
    *,
    cartesian: bool = False,
) -> tuple[float, np.ndarray]:
    """Return absolute Frobenius-compatible energy and per-pair values.

    With ``cartesian=False``, rows of ``U_eval`` and ``V_eval`` are paired.
    With ``cartesian=True``, every product pair is enumerated.
    """
    J1 = _finite_array(delta_J1, "delta_J1", ndim=2)
    J2 = _finite_array(delta_J2, "delta_J2", ndim=2)
    P = _finite_array(delta_P, "delta_P", ndim=3)
    U = _finite_array(U_eval, "U_eval", ndim=2)
    V = _finite_array(V_eval, "V_eval", ndim=2)
    if P.shape != (J1.shape[0], J1.shape[1], J2.shape[1]):
        raise ValueError("delta_P shape must be [output, M1, M2]")
    if U.shape[1] != J1.shape[1] or V.shape[1] != J2.shape[1]:
        raise ValueError("evaluation probes have incompatible dimensions")
    if cartesian:
        pairs = list(product(range(len(U)), range(len(V))))
    else:
        if len(U) != len(V):
            raise ValueError("paired evaluation sets must have equal row counts")
        pairs = [(n, n) for n in range(len(U))]
    values = np.empty(len(pairs), dtype=np.float64)
    for n, (i, j) in enumerate(pairs):
        u, v = U[i], V[j]
        z1 = J1 @ u
        z2 = J2 @ v
        z12 = np.einsum("aij,i,j->a", P, u, v)
        values[n] = z1 @ z1 + z2 @ z2 + z12 @ z12
    return float(values.mean()), values


def serfling_finite_population_factor(population_size: int, sample_size: int) -> float:
    if population_size < 1 or sample_size < 1 or sample_size > population_size:
        raise ValueError("require 1 <= sample_size <= population_size")
    if sample_size <= population_size / 2:
        return 1.0 - (sample_size - 1.0) / population_size
    return (1.0 - sample_size / population_size) * (1.0 + 1.0 / sample_size)


def serfling_radius(
    lower: float,
    upper: float,
    population_size: int,
    sample_size: int,
    delta: float,
) -> float:
    """Two-sided Hoeffding-Serfling radius from the binding theory package."""
    if not np.isfinite([lower, upper]).all() or upper < lower:
        raise ValueError("invalid finite-population range")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    rho = serfling_finite_population_factor(population_size, sample_size)
    return float(
        (upper - lower)
        * np.sqrt(rho * np.log(2.0 / delta) / (2.0 * sample_size))
    )


def bounded_covariance_operator_radius(
    dimension: int,
    norm_bound: float,
    sample_size: int,
    delta: float,
) -> float:
    """The 1/4-net i.i.d. covariance radius in Theorem 3, equation (4.34)."""
    if dimension < 1 or sample_size < 1 or norm_bound <= 0:
        raise ValueError("dimension, sample_size, and norm_bound must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    return float(
        2.0
        * norm_bound**2
        * np.sqrt(
            (np.log(2.0 / delta) + dimension * np.log(9.0))
            / (2.0 * sample_size)
        )
    )


def without_replacement_covariance_operator_radius(
    dimension: int,
    norm_bound: float,
    population_size: int,
    sample_size: int,
    delta: float,
) -> float:
    """Conditional finite-pool covariance radius in equation (4.48)."""
    if dimension < 1 or norm_bound <= 0:
        raise ValueError("dimension and norm_bound must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    rho = serfling_finite_population_factor(population_size, sample_size)
    return float(
        2.0
        * norm_bound**2
        * np.sqrt(
            rho
            * (np.log(2.0 / delta) + dimension * np.log(9.0))
            / (2.0 * sample_size)
        )
    )


def validate_center_replay(
    center_outputs: np.ndarray,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> np.ndarray:
    """Require repeated unperturbed evaluations to replay the same center."""
    centers = _finite_array(center_outputs, "center_outputs", ndim=2)
    reference = centers[0]
    if not np.allclose(centers, reference[None, :], atol=atol, rtol=rtol):
        max_gap = float(np.max(np.abs(centers - reference[None, :])))
        raise IdentificationError(f"center replay failed; maximum gap={max_gap:.6g}")
    return reference.copy()


def evaluate_cartesian_corners(
    function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    U: np.ndarray,
    V: np.ndarray,
    radius1: np.ndarray,
    radius2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate first and mixed corners for a deterministic analytic function."""
    U = _finite_array(U, "U", ndim=2)
    V = _finite_array(V, "V", ndim=2)
    r = _positive_vector(radius1, "radius1", len(U))
    t = _positive_vector(radius2, "radius2", len(V))
    center1 = np.zeros(U.shape[1], dtype=np.float64)
    center2 = np.zeros(V.shape[1], dtype=np.float64)
    first1_plus = np.stack([function(r[a] * U[a], center2) for a in range(len(U))])
    first1_minus = np.stack([function(-r[a] * U[a], center2) for a in range(len(U))])
    first2_plus = np.stack([function(center1, t[b] * V[b]) for b in range(len(V))])
    first2_minus = np.stack([function(center1, -t[b] * V[b]) for b in range(len(V))])
    Y1 = central_first_response(first1_plus, first1_minus, r)
    Y2 = central_first_response(first2_plus, first2_minus, t)
    corner = np.empty((len(U), len(V), 2, 2, len(function(center1, center2))))
    for a in range(len(U)):
        for b in range(len(V)):
            for si, sign1 in enumerate((-1.0, 1.0)):
                for sj, sign2 in enumerate((-1.0, 1.0)):
                    corner[a, b, si, sj] = function(
                        sign1 * r[a] * U[a], sign2 * t[b] * V[b]
                    )
    Y12 = mixed_response_from_corner_array(corner, r, t)
    return Y1, Y2, Y12
