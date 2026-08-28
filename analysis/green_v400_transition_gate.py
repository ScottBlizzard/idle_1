"""Outcome-blind GREEN v4 transition gate helpers.

This module encodes only the direction of the simultaneous confidence gates.
It does not load, compute, or authorize any real GREEN outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SimultaneousBinInterval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower <= self.upper <= 1.0):
            raise ValueError("confidence interval must satisfy 0 <= lower <= upper <= 1")


def corrected_regime_separation_gate(
    low_bins: Iterable[SimultaneousBinInterval],
    high_bins: Iterable[SimultaneousBinInterval],
    *,
    low_probability_max: float = 0.20,
    high_probability_min: float = 0.80,
) -> bool:
    """Return whether simultaneous intervals establish low/high separation.

    Low success is established with an *upper* confidence bound. High success
    is established with a lower confidence bound. Using an LCB for both sides
    would make the low-regime clause directionally invalid.
    """

    if not (0.0 <= low_probability_max < high_probability_min <= 1.0):
        raise ValueError("probability thresholds must be ordered in [0, 1]")

    low = tuple(low_bins)
    high = tuple(high_bins)
    if not low or not high:
        raise ValueError("at least one low and one high complete bin are required")

    low_established = any(interval.upper <= low_probability_max for interval in low)
    high_established = any(interval.lower >= high_probability_min for interval in high)
    return low_established and high_established

