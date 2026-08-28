import pytest

from analysis.green_v400_transition_gate import (
    SimultaneousBinInterval,
    corrected_regime_separation_gate,
)


def test_corrected_gate_accepts_established_low_and_high_regimes():
    assert corrected_regime_separation_gate(
        low_bins=[SimultaneousBinInterval(0.01, 0.19)],
        high_bins=[SimultaneousBinInterval(0.81, 0.95)],
    )


def test_low_lcb_below_threshold_is_not_enough():
    assert not corrected_regime_separation_gate(
        low_bins=[SimultaneousBinInterval(0.01, 0.70)],
        high_bins=[SimultaneousBinInterval(0.81, 0.95)],
    )


def test_high_ucb_above_threshold_is_not_enough():
    assert not corrected_regime_separation_gate(
        low_bins=[SimultaneousBinInterval(0.01, 0.19)],
        high_bins=[SimultaneousBinInterval(0.40, 0.95)],
    )


@pytest.mark.parametrize(
    "lower,upper",
    [(-0.01, 0.2), (0.3, 0.2), (0.2, 1.01)],
)
def test_interval_validation_rejects_invalid_probability_bounds(lower, upper):
    with pytest.raises(ValueError):
        SimultaneousBinInterval(lower, upper)


def test_empty_complete_bin_family_is_rejected():
    with pytest.raises(ValueError):
        corrected_regime_separation_gate(
            low_bins=[],
            high_bins=[SimultaneousBinInterval(0.81, 0.95)],
        )
