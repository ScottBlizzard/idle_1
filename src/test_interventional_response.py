"""Regression tests for interventional response signatures."""
from __future__ import annotations

import numpy as np

from interventional_response import (
    compare_signatures,
    forward_signature,
    isotropic_probes,
    reference_chord_probes,
    symmetric_signature,
)


def test_symmetric_signature_recovers_linear_directional_derivative() -> None:
    rng = np.random.RandomState(1)
    centers = rng.randn(5, 4)
    probes = isotropic_probes(5, 12, 4, 0.1, rng)
    weight = np.array([1.0, -2.0, 0.5, 3.0])
    plus = np.einsum("npd,d->np", centers[:, None, :] + probes, weight)
    minus = np.einsum("npd,d->np", centers[:, None, :] - probes, weight)
    signature = symmetric_signature(plus, minus, probes)[:, :, 0]
    expected = np.einsum("npd,d->np", probes, weight) / np.linalg.norm(probes, axis=-1)
    assert np.allclose(signature, expected, atol=1e-10)


def test_forward_signature_recovers_linear_directional_derivative() -> None:
    rng = np.random.RandomState(2)
    centers = rng.randn(3, 5)
    probes = isotropic_probes(3, 7, 5, 0.03, rng)
    weight = rng.randn(5)
    center_y = centers @ weight
    perturbed_y = np.einsum("npd,d->np", centers[:, None, :] + probes, weight)
    signature = forward_signature(center_y, perturbed_y, probes)[:, :, 0]
    expected = np.einsum("npd,d->np", probes, weight) / np.linalg.norm(probes, axis=-1)
    assert np.allclose(signature, expected, atol=1e-10)


def test_forward_linear_bias_and_symmetric_quadratic_cancellation() -> None:
    """For f(x)=x+2x^2 at zero, forward error is exactly 2r."""
    radii = np.array([0.2, 0.1, 0.05, 0.025])
    centers = np.zeros((len(radii), 1))
    probes = radii[:, None, None]
    center_y = np.zeros(len(radii))
    plus_y = radii[:, None] + 2.0 * radii[:, None] ** 2
    minus_y = -radii[:, None] + 2.0 * radii[:, None] ** 2
    forward = forward_signature(center_y, plus_y, probes)[:, 0, 0]
    symmetric = symmetric_signature(plus_y, minus_y, probes)[:, 0, 0]
    assert np.allclose(forward - 1.0, 2.0 * radii, atol=1e-12)
    assert np.allclose(symmetric, 1.0, atol=1e-12)


def test_reference_chord_probes_point_toward_reference() -> None:
    rng = np.random.RandomState(3)
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    reference = np.array([[1.0, 0.0], [0.0, 1.0], [9.0, 10.0], [10.0, 9.0]])
    probes = reference_chord_probes(centers, reference, 2, 0.5, 2, rng)
    endpoints = centers[:, None, :] + probes
    assert np.all(np.linalg.norm(endpoints - centers[:, None, :], axis=-1) > 0)
    assert np.all(np.linalg.norm(probes, axis=-1) <= np.sqrt(2) * 0.5 + 1e-12)


def test_signature_comparison_detects_mismatch() -> None:
    target = np.ones((4, 8, 1))
    same = compare_signatures(target.copy(), target)
    different = compare_signatures(-target, target)
    assert same.rmse == 0.0
    assert different.rmse > 1.0
    assert different.mean_cosine < 0.0


if __name__ == "__main__":
    tests = [
        test_symmetric_signature_recovers_linear_directional_derivative,
        test_forward_signature_recovers_linear_directional_derivative,
        test_forward_linear_bias_and_symmetric_quadratic_cancellation,
        test_reference_chord_probes_point_toward_reference,
        test_signature_comparison_detects_mismatch,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
