from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_final_contrast_fusion import fuse_final_contrast_exact


def test_exact_final_contrast_fusion_matches_direct_rational_evaluation():
    unembed = np.asarray([[0.5, -0.25, 0.125], [0.75, 0.5, -1.0]], dtype="<f4")
    bias = np.asarray([0.25, -0.5, 0.75], dtype="<f4")
    suffix = np.asarray([2, 0], dtype="<i8")
    coefficients = np.asarray([0.5, -0.25], dtype="<f8")
    fusion = fuse_final_contrast_exact(unembed, bias, suffix, coefficients)
    residual = [Fraction(3, 8), Fraction(-5, 16)]
    fused = sum((weight * value for weight, value in zip(fusion.weights, residual)), fusion.bias)
    direct = sum(
        Fraction(float(coefficients[index])) * (
            sum(residual[coordinate] * Fraction(float(unembed[coordinate, token]))
                for coordinate in range(2)) + Fraction(float(bias[token]))
        )
        for index, token in enumerate(suffix)
    )
    assert fused == direct
    assert len(fusion.semantic_hash()) == 64


def test_exact_fusion_rejects_silent_dtype_conversion_and_binds_inputs():
    unembed = np.zeros((2, 3), dtype="<f4")
    bias = np.zeros(3, dtype="<f4")
    suffix = np.asarray([0, 2], dtype="<i8")
    coefficients = np.asarray([1.0, -0.5], dtype="<f8")
    first = fuse_final_contrast_exact(unembed, bias, suffix, coefficients)
    changed = unembed.copy(); changed[0, 0] = np.float32(0.25)
    second = fuse_final_contrast_exact(changed, bias, suffix, coefficients)
    assert first.semantic_hash() != second.semantic_hash()
    with pytest.raises(ValueError, match="canonical"):
        fuse_final_contrast_exact(unembed.astype("<f8"), bias, suffix, coefficients)
    with pytest.raises(ValueError, match="finite"):
        bad = unembed.copy(); bad[0, 0] = np.nan
        fuse_final_contrast_exact(bad, bias, suffix, coefficients)
    with pytest.raises(ValueError, match="shapes or suffix"):
        fuse_final_contrast_exact(unembed, bias, np.asarray([0, 0], dtype="<i8"), coefficients)


def test_gpt2_width_fusion_is_deterministic():
    rng = np.random.default_rng(31)
    unembed = rng.standard_normal((768, 100)).astype("<f4")
    bias = rng.standard_normal(100).astype("<f4")
    suffix = np.arange(100, dtype="<i8")
    coefficients = (np.where(np.arange(100) % 2, -1.0, 1.0) / 50.0).astype("<f8")
    first = fuse_final_contrast_exact(unembed, bias, suffix, coefficients)
    second = fuse_final_contrast_exact(unembed, bias, suffix, coefficients)
    assert first.semantic_hash() == second.semantic_hash()
    assert len(first.weights) == 768
