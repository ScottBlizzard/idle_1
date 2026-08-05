"""Numerical regression tests for the July-audit cross-fitted scorer."""
from __future__ import annotations

import numpy as np

from validity_crossfit import CrossFitSiteReference


def test_held_out_knn_keeps_nearest_neighbor() -> None:
    fit = np.array([[0.0], [10.0], [20.0], [30.0]])
    cal = np.array([[1.0], [11.0], [21.0], [29.0]])
    ref = CrossFitSiteReference(fit, cal, knn_k=1, proj_rank=1)
    raw = ref.raw_metrics(np.array([[0.1]]))
    assert np.allclose(raw["knn"], [0.1], atol=1e-7), raw["knn"]


def test_self_query_explicitly_skips_zero_neighbor() -> None:
    fit = np.array([[0.0], [10.0], [20.0], [30.0]])
    cal = np.array([[1.0], [11.0], [21.0], [29.0]])
    ref = CrossFitSiteReference(fit, cal, knn_k=1, proj_rank=1)
    raw = ref.raw_metrics(fit, exclude_self=True)
    assert np.all(raw["knn"] > 0), raw["knn"]


def test_rank_is_bounded_by_observed_support() -> None:
    rng = np.random.RandomState(3)
    latent_fit = rng.randn(20, 3)
    latent_cal = rng.randn(10, 3)
    map_ = rng.randn(3, 40)
    ref = CrossFitSiteReference(latent_fit @ map_, latent_cal @ map_, proj_rank=32)
    assert ref.selected_rank <= 3, ref.diagnostics()


def test_cross_fit_scoring_is_finite_and_separates_shift() -> None:
    rng = np.random.RandomState(7)
    fit = rng.randn(160, 12)
    cal = rng.randn(80, 12)
    heldout = rng.randn(40, 12)
    shifted = rng.randn(40, 12) + 5.0
    ref = CrossFitSiteReference(fit, cal, knn_k=8, proj_rank=8)
    same = ref.score(heldout)
    other = ref.score(shifted)
    for values in [*same.values(), *other.values()]:
        assert np.isfinite(values).all()
    assert other["overlap_ecdf"].mean() < same["overlap_ecdf"].mean()
    assert other["overlap_conformal"].mean() < same["overlap_conformal"].mean()


def test_composite_conformal_uses_disjoint_normalization_and_calibration() -> None:
    rng = np.random.RandomState(11)
    fit = rng.randn(80, 6)
    cal = rng.randn(40, 6)
    ref = CrossFitSiteReference(fit, cal, knn_k=4, proj_rank=4)
    assert len(ref.composite_normalization_ref) == 20
    assert len(ref.composite_calibration_ref) == 20
    assert not np.shares_memory(
        ref.composite_normalization_ref, ref.composite_calibration_ref
    )
    scores = ref.score(rng.randn(7, 6))
    grid = 1.0 / (len(ref.composite_calibration_ref) + 1.0)
    scaled = scores["overlap_conformal"] / grid
    assert np.allclose(scaled, np.round(scaled))


def test_composite_conformal_empirical_false_alarm_control() -> None:
    # Marginal split-conformal coverage is the guarantee.  Pool independent
    # repetitions to keep this regression test stable without a huge sample.
    p_values = []
    for seed in range(80):
        rng = np.random.RandomState(1000 + seed)
        fit = rng.randn(100, 5)
        cal = rng.randn(60, 5)
        query = rng.randn(10, 5)
        ref = CrossFitSiteReference(fit, cal, knn_k=5, proj_rank=4)
        p_values.extend(ref.score(query)["overlap_conformal"].tolist())
    false_alarm = np.mean(np.asarray(p_values) <= 0.1)
    assert false_alarm <= 0.13, false_alarm


if __name__ == "__main__":
    tests = [
        test_held_out_knn_keeps_nearest_neighbor,
        test_self_query_explicitly_skips_zero_neighbor,
        test_rank_is_bounded_by_observed_support,
        test_cross_fit_scoring_is_finite_and_separates_shift,
        test_composite_conformal_uses_disjoint_normalization_and_calibration,
        test_composite_conformal_empirical_false_alarm_control,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
