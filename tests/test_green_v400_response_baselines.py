import pytest
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_response_baselines import (
    batched_response_effects,
    calibrate_integrated_gradients_grid,
    compare_batched_response_fields,
    compare_response_fields,
    exact_finite_response,
    first_order_response,
    hvp_second_order_response,
    integrated_gradients_response,
    ms_hvp_response,
    response_effects,
)


DTYPE = torch.float64


class ElementwiseBatchResponse:
    supports_batch = True

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, x):
        return self.scale * (torch.sin(x) + 0.2 * x**3).sum(dim=1)


def scalar_version(scale=1.0):
    return lambda x: scale * (torch.sin(x) + 0.2 * x**3).sum()


@pytest.mark.parametrize(
    "method", ["exact", "first_order", "integrated_gradients", "hvp", "ms_hvp"]
)
def test_vectorized_methods_match_scalar_independent_evaluations(method):
    center = torch.tensor([0.2, -0.3, 0.5], dtype=DTYPE)
    directions = torch.tensor(
        [[0.1, 0.4, -0.2], [-0.2, 0.5, 0.3], [0.05, -0.1, 0.2]],
        dtype=DTYPE,
    )
    scalar = response_effects(
        method,
        scalar_version(),
        center,
        directions,
        integrated_gradients_steps=17,
        ms_hvp_segments=8,
    )
    batched = batched_response_effects(
        method,
        ElementwiseBatchResponse(),
        center,
        directions,
        integrated_gradients_steps=17,
        ms_hvp_segments=8,
    )
    torch.testing.assert_close(batched, scalar, rtol=1e-12, atol=1e-12)


def test_batched_field_comparison_matches_scalar_field_comparison():
    center = torch.tensor([0.1, -0.2], dtype=DTYPE)
    directions = torch.tensor([[0.2, 0.3], [-0.1, 0.4]], dtype=DTYPE)
    scalar = compare_response_fields(
        "integrated_gradients",
        scalar_version(1.0),
        scalar_version(1.1),
        center,
        directions,
        integrated_gradients_steps=13,
    )
    batched = compare_batched_response_fields(
        "integrated_gradients",
        ElementwiseBatchResponse(1.0),
        ElementwiseBatchResponse(1.1),
        center,
        directions,
        integrated_gradients_steps=13,
    )
    torch.testing.assert_close(batched.discrepancies, scalar.discrepancies)
    assert batched.rmse == pytest.approx(scalar.rmse)


def test_batched_ig_chunking_preserves_values():
    center = torch.tensor([0.1, -0.2], dtype=DTYPE)
    directions = torch.tensor([[0.2, 0.3], [-0.1, 0.4]], dtype=DTYPE)
    unchunked = batched_response_effects(
        "integrated_gradients",
        ElementwiseBatchResponse(),
        center,
        directions,
        integrated_gradients_steps=13,
        batch_chunk_size=1000,
    )
    chunked = batched_response_effects(
        "integrated_gradients",
        ElementwiseBatchResponse(),
        center,
        directions,
        integrated_gradients_steps=13,
        batch_chunk_size=3,
    )
    torch.testing.assert_close(chunked, unchunked, rtol=1e-12, atol=1e-12)


def test_all_baselines_are_exact_for_linear_response():
    center = torch.tensor([0.2, -0.3], dtype=DTYPE)
    directions = torch.tensor([[0.1, 0.4], [-0.2, 0.5]], dtype=DTYPE)
    weight = torch.tensor([2.0, -3.0], dtype=DTYPE)
    response = lambda x: torch.dot(weight, x) + 4.0
    expected = directions @ weight
    for method in ("exact", "first_order", "integrated_gradients", "hvp", "ms_hvp"):
        actual = response_effects(method, response, center, directions, integrated_gradients_steps=9)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_hvp_and_integrated_gradients_are_exact_for_quadratic_response():
    center = torch.tensor([0.3, -0.2], dtype=DTYPE)
    directions = torch.tensor([[0.4, 0.1], [-0.2, 0.5]], dtype=DTYPE)
    matrix = torch.tensor([[2.0, 0.5], [0.5, 3.0]], dtype=DTYPE)
    response = lambda x: 0.5 * torch.dot(x, matrix @ x)
    exact = exact_finite_response(response, center, directions)
    torch.testing.assert_close(
        hvp_second_order_response(response, center, directions), exact, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        integrated_gradients_response(response, center, directions, steps=17),
        exact,
        rtol=1e-12,
        atol=1e-12,
    )


def test_second_order_improves_over_first_order_for_cubic_near_center():
    center = torch.tensor([0.7], dtype=DTYPE)
    directions = torch.tensor([[0.1], [-0.1]], dtype=DTYPE)
    response = lambda x: x[0] ** 3
    exact = exact_finite_response(response, center, directions)
    first = first_order_response(response, center, directions)
    second = hvp_second_order_response(response, center, directions)
    assert torch.linalg.vector_norm(second - exact) < torch.linalg.vector_norm(first - exact)


def test_ms_hvp_improves_over_single_point_hvp_for_quartic_finite_step():
    center = torch.tensor([0.7], dtype=DTYPE)
    directions = torch.tensor([[0.8], [-0.5]], dtype=DTYPE)
    response = lambda x: x[0] ** 4 + 0.3 * x[0] ** 3
    exact = exact_finite_response(response, center, directions)
    single = hvp_second_order_response(response, center, directions)
    multi = ms_hvp_response(response, center, directions, segments=16)
    assert torch.linalg.vector_norm(multi - exact) < torch.linalg.vector_norm(single - exact)


def test_ms_hvp_rejects_one_segment_instead_of_aliasing_single_hvp():
    center = torch.tensor([0.0], dtype=DTYPE)
    directions = torch.ones((1, 1), dtype=DTYPE)
    with pytest.raises(ValueError, match="at least two"):
        ms_hvp_response(lambda x: x[0] ** 2, center, directions, segments=1)


def test_identical_fields_have_zero_discrepancy():
    center = torch.tensor([0.2, -0.1], dtype=DTYPE)
    directions = torch.tensor([[0.1, 0.2], [0.3, -0.4]], dtype=DTYPE)
    response = lambda x: torch.sin(x[0]) + x[1] ** 2
    result = compare_response_fields("hvp", response, response, center, directions)
    assert result.rmse == pytest.approx(0.0, abs=1e-15)
    assert result.normalized_rmse == pytest.approx(0.0, abs=1e-15)


def test_integrated_gradients_converges_to_exact_cubic_finite_response():
    center = torch.tensor([0.4], dtype=DTYPE)
    directions = torch.tensor([[0.6], [-0.3]], dtype=DTYPE)
    response = lambda x: x[0] ** 3 - 0.2 * x[0]
    exact = exact_finite_response(response, center, directions)
    integrated = integrated_gradients_response(response, center, directions, steps=1025)
    torch.testing.assert_close(integrated, exact, rtol=2e-6, atol=2e-7)


def test_ig_grid_calibration_selects_first_passing_resolution():
    center = torch.tensor([0.4], dtype=DTYPE)
    directions = torch.tensor([[0.6], [-0.3]], dtype=DTYPE)
    response = lambda x: x[0] ** 3 - 0.2 * x[0]
    calibration = calibrate_integrated_gradients_grid(
        response,
        lambda x: 1.2 * response(x),
        center,
        directions,
        grid=(3, 9, 33, 129),
        absolute_tolerance=1e-5,
        relative_tolerance=0.0,
    )
    assert calibration.converged is True
    assert calibration.selected_steps in calibration.grid
    selected_index = calibration.grid.index(calibration.selected_steps)
    assert calibration.records[selected_index]["passed"] is True
    assert all(not row["passed"] for row in calibration.records[:selected_index])


def test_ig_successive_grid_rule_does_not_call_float_difference_gap_convergence():
    center = torch.tensor([0.4], dtype=DTYPE)
    directions = torch.tensor([[0.6], [-0.3]], dtype=DTYPE)
    response = lambda x: x[0] ** 3 - 0.2 * x[0]
    calibration = calibrate_integrated_gradients_grid(
        response,
        lambda x: 1.2 * response(x),
        center,
        directions,
        grid=(3, 9, 33, 129),
        absolute_tolerance=1e-5,
        relative_tolerance=0.0,
        selection_rule="successive_grid_stability",
    )
    assert calibration.selection_rule == "successive_grid_stability"
    assert calibration.records[0]["passed"] is False
    assert calibration.records[0]["max_absolute_successive_grid_difference"] is None


def test_ig_grid_calibration_fails_closed_on_malformed_grid():
    center = torch.tensor([0.0], dtype=DTYPE)
    directions = torch.ones((1, 1), dtype=DTYPE)
    with pytest.raises(ValueError, match="unique, increasing"):
        calibrate_integrated_gradients_grid(
            lambda x: x[0], lambda x: x[0], center, directions, grid=(9, 3)
        )


def test_invalid_panel_and_unknown_method_fail_closed():
    center = torch.tensor([0.0], dtype=DTYPE)
    with pytest.raises(ValueError):
        exact_finite_response(lambda x: x[0], center, torch.empty((0, 1), dtype=DTYPE))
    with pytest.raises(ValueError, match="unsupported baseline"):
        response_effects("not_a_method", lambda x: x[0], center, torch.ones((1, 1), dtype=DTYPE))
