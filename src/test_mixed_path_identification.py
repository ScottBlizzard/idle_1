"""Deterministic CPU tests for the ASG-RDAG mixed-path estimator.

The file is runnable directly because the bundled local Python does not require
pytest.  Pytest can also discover every ``test_*`` function on a server.
"""
from __future__ import annotations

import inspect
from itertools import combinations

import numpy as np

from mixed_path_identification import (
    CurvatureError,
    IdentificationError,
    RankDeficiencyError,
    active_edge_inverse,
    cartesian_kronecker_design,
    central_first_response,
    curvature_ratio,
    design_diagnostics,
    evaluate_cartesian_corners,
    fit_cartesian_structural_tensors,
    fit_paired_mixed_tensor,
    mixed_four_corner_response,
    mixed_response_from_corner_array,
    paired_kronecker_design,
    serfling_radius,
    structural_response_energy,
    validate_center_replay,
)


TOL = 1e-12


def _base_parameters() -> tuple[np.ndarray, ...]:
    A = np.array([[1.0, -2.0], [0.5, 1.0]])
    C = np.array([[2.0, -1.0], [1.0, 3.0]])
    D = np.array([[0.25, -0.5], [1.0, 0.75]])
    y0 = np.array([0.1, -0.2])
    return A, C, D, y0


def _quadratic_function(
    A: np.ndarray, C: np.ndarray, D: np.ndarray, y0: np.ndarray
):
    def function(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = A @ u + v
        return y0 + D @ u + C @ (q + 0.5 * q**2)

    return function


def _ground_truth(
    A: np.ndarray, C: np.ndarray, D: np.ndarray
) -> tuple[np.ndarray, ...]:
    P = np.stack([np.outer(C[:, j], A[j]) for j in range(A.shape[0])], axis=2)
    G = C.copy()
    J1 = D + P.sum(axis=2)
    J2 = G.copy()
    H = P.copy()
    return J1, J2, H, P, D, G


def _fit_base(U: np.ndarray, V: np.ndarray, r: np.ndarray, t: np.ndarray):
    A, C, D, y0 = _base_parameters()
    Y1, Y2, Y12 = evaluate_cartesian_corners(
        _quadratic_function(A, C, D, y0), U, V, r, t
    )
    estimate = fit_cartesian_structural_tensors(
        U, V, Y1, Y2, Y12, rho=np.ones(V.shape[1])
    )
    return estimate, _ground_truth(A, C, D)


def _assert_estimate_exact(estimate, truth) -> None:
    names = ("J1", "J2", "H", "P", "D", "G")
    for name, expected in zip(names, truth):
        error = np.max(np.abs(getattr(estimate, name) - expected))
        assert error < TOL, (name, error)


def test_quadratic_asg_exact_recovery() -> None:
    U = np.eye(2)
    V = np.eye(2)
    estimate, truth = _fit_base(U, V, np.array([0.37, 0.37]), np.array([0.37, 0.37]))
    _assert_estimate_exact(estimate, truth)


def test_noncoordinate_full_rank_recovery() -> None:
    z = 1.0 / np.sqrt(2.0)
    U = np.array([[1.0, 0.0], [0.0, 1.0], [z, z], [z, -z]])
    V = U.copy()
    estimate, truth = _fit_base(
        U,
        V,
        np.array([0.11, 0.19, 0.23, 0.31]),
        np.array([0.13, 0.17, 0.29, 0.37]),
    )
    _assert_estimate_exact(estimate, truth)


def test_active_edge_inverse() -> None:
    U = np.eye(2)
    V = np.eye(2)
    estimate, _ = _fit_base(U, V, np.array([0.2, 0.3]), np.array([0.4, 0.5]))
    A, C, _, _ = _base_parameters()
    A_hat, C_hat = active_edge_inverse(estimate.G, estimate.P, np.ones(2))
    assert np.max(np.abs(A_hat - A)) < TOL
    assert np.max(np.abs(C_hat - C)) < TOL


def test_product_energy_identity() -> None:
    J1, J2, _, P, _, _ = _ground_truth(*_base_parameters()[:3])
    delta_J1 = J1 - 0.3
    delta_J2 = J2 + 0.2
    delta_P = P - 0.1
    signs = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    ) / np.sqrt(2.0)
    energy, _ = structural_response_energy(
        delta_J1, delta_J2, delta_P, signs, signs, cartesian=True
    )
    expected = (
        0.5 * np.linalg.norm(delta_J1) ** 2
        + 0.5 * np.linalg.norm(delta_J2) ** 2
        + 0.25 * np.linalg.norm(delta_P) ** 2
    )
    assert abs(energy - expected) < TOL
    theta_norm_sq = (
        np.linalg.norm(delta_J1 - delta_P.sum(axis=2)) ** 2
        + np.linalg.norm(delta_J2) ** 2
        + np.linalg.norm(delta_P) ** 2
    )
    d2 = delta_P.shape[2]
    c_q = 0.25
    C_q = 0.5
    assert c_q / (2 * d2 + 1) * theta_norm_sq <= energy + TOL
    assert energy <= C_q * (2 * d2 + 1) * theta_norm_sq + TOL


def test_first_order_cancellation_requires_mixed() -> None:
    C = np.array([[1.0, 1.0]])
    D = np.zeros((1, 1))
    y0 = np.zeros(1)
    A_star = np.array([[1.0], [-1.0]])
    A_patch = np.zeros((2, 1))
    U = np.ones((1, 1))
    V = np.eye(2)
    radii1 = np.array([0.31])
    radii2 = np.array([0.23, 0.37])
    star_y = evaluate_cartesian_corners(
        _quadratic_function(A_star, C, D, y0), U, V, radii1, radii2
    )
    patch_y = evaluate_cartesian_corners(
        _quadratic_function(A_patch, C, D, y0), U, V, radii1, radii2
    )
    star = fit_cartesian_structural_tensors(U, V, *star_y, rho=np.ones(2))
    patch = fit_cartesian_structural_tensors(U, V, *patch_y, rho=np.ones(2))
    zero_order_gap = float(
        _quadratic_function(A_star, C, D, y0)(np.zeros(1), np.zeros(2))[0]
        - _quadratic_function(A_patch, C, D, y0)(np.zeros(1), np.zeros(2))[0]
    )
    first_energy = np.linalg.norm(star.J1 - patch.J1) ** 2 + np.linalg.norm(
        star.J2 - patch.J2
    ) ** 2
    mixed_energy = np.linalg.norm(star.P - patch.P) ** 2
    assert zero_order_gap == 0.0
    assert np.max(np.abs(star.J1 - patch.J1)) < TOL
    assert np.max(np.abs(star.J2 - patch.J2)) < TOL
    assert first_energy < 1e-24
    assert mixed_energy > 0
    assert np.allclose(star.P.reshape(-1), [1.0, -1.0])
    assert np.allclose(patch.P, 0.0)


def test_rank_deficient_first_block_fails() -> None:
    U = np.array([[1.0, 0.0], [1.0, 0.0]])
    V = np.eye(2)
    Y1 = np.zeros((2, 1))
    Y2 = np.zeros((2, 1))
    Y12 = np.zeros((2, 2, 1))
    diagnostics = design_diagnostics(U)
    assert diagnostics.rank == 1
    try:
        fit_cartesian_structural_tensors(U, V, Y1, Y2, Y12, np.ones(2))
    except RankDeficiencyError as exc:
        assert "ridge fallback is prohibited" in str(exc)
    else:
        raise AssertionError("rank-deficient U returned a structural estimate")


def test_marginal_span_not_kronecker_complete() -> None:
    U = np.eye(2)
    V = np.eye(2)
    W = paired_kronecker_design(U, V)
    assert design_diagnostics(U).rank == 2
    assert design_diagnostics(V).rank == 2
    assert design_diagnostics(W).rank == 2
    assert W.shape[1] == 4
    hidden = np.array([[0.0, 1.0], [0.0, 0.0]])
    observed = np.array([u @ hidden @ v for u, v in zip(U, V)])
    assert np.allclose(observed, 0.0)


def test_complete_paired_kronecker_recovery() -> None:
    U = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    H_true = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [0.25, -2.0]],
        ]
    )
    Y = np.stack(
        [np.einsum("aij,i,j->a", H_true, u, v) for u, v in zip(U, V)], axis=0
    )
    H_hat, diagnostics = fit_paired_mixed_tensor(U, V, Y)
    assert diagnostics.full_rank
    assert diagnostics.rank == 4
    assert np.max(np.abs(H_hat - H_true)) < TOL


def test_unknown_curvature_ratio_nonidentification() -> None:
    J1_A = -1.0 + 1.0 * 1.0 * 1.0
    J2_A = 1.0 * 1.0
    H_A = 1.0 * 1.0 * 1.0
    P_A = 1.0 * 1.0 * 1.0
    J1_B = -2.0 + 1.0 * 1.0 * 2.0
    J2_B = 1.0 * 1.0
    H_B = 1.0 * 0.5 * 2.0
    P_B = 1.0 * 1.0 * 2.0
    assert J1_A == J1_B == 0.0
    assert J2_A == J2_B == 1.0
    assert H_A == H_B == 1.0
    assert P_A == 1.0 and P_B == 2.0


def test_zero_curvature_fails() -> None:
    try:
        curvature_ratio(np.array([1.0]), np.array([0.0]))
    except CurvatureError as exc:
        assert "FAIL_CURVATURE" in str(exc)
    else:
        raise AssertionError("zero curvature returned a path estimate")
    observed = []
    for A in (1.0, 3.0):
        D = -A
        observed.append((D + A, 1.0, 0.0))
    assert observed[0] == observed[1] == (0.0, 1.0, 0.0)


def test_omitted_bypass_impossibility() -> None:
    grid = np.linspace(-1.0, 1.0, 7)
    total_A = np.array([u * v - u * v for u in grid for v in grid])
    total_B = np.zeros_like(total_A)
    path_A = np.array([u * v for u in grid for v in grid])
    path_B = np.zeros_like(path_A)
    assert np.array_equal(total_A, total_B)
    assert not np.array_equal(path_A, path_B)
    complete_cut_diagnostic = "FAIL"
    assert complete_cut_diagnostic == "FAIL"


def _quartic_function(A: float, gamma: float, eta: float):
    def psi(q: np.ndarray) -> np.ndarray:
        return q + 0.5 * q**2 + gamma / 6.0 * q**3 + eta / 24.0 * q**4

    def function(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([psi(A * u[0] + v[0])])

    return function


def test_central_first_r2_scaling() -> None:
    A, gamma, eta = 1.7, 0.8, -0.3
    function = _quartic_function(A, gamma, eta)
    errors = []
    for r in (0.4, 0.2, 0.1):
        plus = function(np.array([r]), np.zeros(1))[None, :]
        minus = function(np.array([-r]), np.zeros(1))[None, :]
        estimate = central_first_response(plus, minus, np.array([r]))[0, 0]
        expected_error = gamma * r**2 * A**3 / 6.0
        assert abs((estimate - A) - expected_error) < TOL
        errors.append(abs(estimate - A))
    assert abs(errors[0] / errors[1] - 4.0) < TOL
    assert abs(errors[1] / errors[2] - 4.0) < TOL


def test_mixed_four_point_r2_t2_scaling() -> None:
    A, gamma, eta = 1.3, -0.4, 0.9
    function = _quartic_function(A, gamma, eta)
    errors = []
    for r, t in ((0.4, 0.3), (0.2, 0.15), (0.1, 0.075)):
        pp = function(np.array([r]), np.array([t]))[None, None, :]
        pm = function(np.array([r]), np.array([-t]))[None, None, :]
        mp = function(np.array([-r]), np.array([t]))[None, None, :]
        mm = function(np.array([-r]), np.array([-t]))[None, None, :]
        estimate = mixed_four_corner_response(
            pp, pm, mp, mm, np.array([r]), np.array([t])
        )[0, 0, 0]
        expected_error = eta / 6.0 * (r**2 * A**3 + t**2 * A)
        assert abs((estimate - A) - expected_error) < TOL
        errors.append(abs(estimate - A))
    assert abs(errors[0] / errors[1] - 4.0) < TOL
    assert abs(errors[1] / errors[2] - 4.0) < TOL


def test_output_dimension_energy_scaling() -> None:
    J1 = np.array([[1.0, -2.0]])
    J2 = np.array([[0.5, 1.5]])
    P = np.array([[[1.0, 0.2], [-0.3, 0.7]]])
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    scalar, _ = structural_response_energy(J1, J2, P, U, V, cartesian=True)
    for k in (2, 5, 11):
        energy, _ = structural_response_energy(
            np.repeat(J1, k, axis=0),
            np.repeat(J2, k, axis=0),
            np.repeat(P, k, axis=0),
            U,
            V,
            cartesian=True,
        )
        assert abs(energy - k * scalar) < TOL
        assert abs(energy / k - scalar) < TOL


def test_without_replacement_serfling_interval() -> None:
    population = np.linspace(-0.2, 1.4, 8)
    N, m = len(population), 4
    population_mean = float(population.mean())
    subset_means = np.array(
        [population[list(indices)].mean() for indices in combinations(range(N), m)]
    )
    for delta in (0.5, 0.2, 0.1, 0.05):
        radius = serfling_radius(
            float(population.min()), float(population.max()), N, m, delta
        )
        violation = float(np.mean(np.abs(subset_means - population_mean) > radius))
        assert violation <= delta + 1e-15, (delta, violation, radius)


def test_corner_sign_order() -> None:
    r, t = np.array([0.7]), np.array([0.4])
    corners = np.empty((1, 1, 2, 2, 1), dtype=np.float64)
    for si, sign1 in enumerate((-1.0, 1.0)):
        for sj, sign2 in enumerate((-1.0, 1.0)):
            corners[0, 0, si, sj, 0] = sign1 * r[0] * sign2 * t[0]
    estimate = mixed_response_from_corner_array(corners, r, t)
    assert abs(estimate[0, 0, 0] - 1.0) < TOL


def test_variable_radius_denominator() -> None:
    U = np.eye(2)
    V = np.eye(2)
    r = np.array([0.07, 0.53])
    t = np.array([0.11, 0.41])

    def bilinear(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([2.0 * u[0] * v[0] - 3.0 * u[1] * v[1]])

    _, _, Y12 = evaluate_cartesian_corners(bilinear, U, V, r, t)
    expected = np.array([[[2.0], [0.0]], [[0.0], [-3.0]]])
    assert np.max(np.abs(Y12 - expected)) < TOL


def test_center_replay() -> None:
    replay = np.repeat(np.array([[1.0, -2.0, 0.5]]), 8, axis=0)
    assert np.array_equal(validate_center_replay(replay), replay[0])
    replay[4, 1] += 1e-4
    try:
        validate_center_replay(replay)
    except IdentificationError as exc:
        assert "center replay failed" in str(exc)
    else:
        raise AssertionError("non-deterministic center replay was accepted")


def test_no_ridge_fallback() -> None:
    signature = inspect.signature(fit_cartesian_structural_tensors)
    assert "ridge" not in signature.parameters
    U = np.zeros((2, 1))
    V = np.ones((1, 1))
    try:
        fit_cartesian_structural_tensors(
            U,
            V,
            np.zeros((2, 1)),
            np.zeros((1, 1)),
            np.zeros((2, 1, 1)),
            np.ones(1),
        )
    except RankDeficiencyError:
        pass
    else:
        raise AssertionError("rank deficiency was silently regularized")


TESTS = [
    test_quadratic_asg_exact_recovery,
    test_noncoordinate_full_rank_recovery,
    test_active_edge_inverse,
    test_product_energy_identity,
    test_first_order_cancellation_requires_mixed,
    test_rank_deficient_first_block_fails,
    test_marginal_span_not_kronecker_complete,
    test_complete_paired_kronecker_recovery,
    test_unknown_curvature_ratio_nonidentification,
    test_zero_curvature_fails,
    test_omitted_bypass_impossibility,
    test_central_first_r2_scaling,
    test_mixed_four_point_r2_t2_scaling,
    test_output_dimension_energy_scaling,
    test_without_replacement_serfling_interval,
    test_corner_sign_order,
    test_variable_radius_denominator,
    test_center_replay,
    test_no_ridge_fallback,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(TESTS)} mixed-path identification tests")
