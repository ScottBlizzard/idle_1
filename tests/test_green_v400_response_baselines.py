import pytest
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_response_baselines import (
    compare_response_fields,
    exact_finite_response,
    first_order_response,
    hvp_second_order_response,
    integrated_gradients_response,
    response_effects,
)


DTYPE = torch.float64


def test_all_baselines_are_exact_for_linear_response():
    center = torch.tensor([0.2, -0.3], dtype=DTYPE)
    directions = torch.tensor([[0.1, 0.4], [-0.2, 0.5]], dtype=DTYPE)
    weight = torch.tensor([2.0, -3.0], dtype=DTYPE)
    response = lambda x: torch.dot(weight, x) + 4.0
    expected = directions @ weight
    for method in ("exact", "first_order", "integrated_gradients", "hvp"):
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


def test_invalid_panel_and_unknown_method_fail_closed():
    center = torch.tensor([0.0], dtype=DTYPE)
    with pytest.raises(ValueError):
        exact_finite_response(lambda x: x[0], center, torch.empty((0, 1), dtype=DTYPE))
    with pytest.raises(ValueError, match="unsupported baseline"):
        response_effects("not_a_method", lambda x: x[0], center, torch.ones((1, 1), dtype=DTYPE))
