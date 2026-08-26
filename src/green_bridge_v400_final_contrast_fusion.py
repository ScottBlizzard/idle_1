"""Exact dyadic precomposition of the fixed suffix-logit contrast."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

import numpy as np

from green_bridge_v400_schemas import canonical_json, sha256_canonical


def _exact_ieee_fraction(value) -> Fraction:
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def _balanced_sum(values: list[Fraction]) -> Fraction:
    if not values:
        return Fraction(0)
    current = values
    while len(current) > 1:
        current = [
            current[index] + current[index + 1]
            for index in range(0, len(current) - 1, 2)
        ] + ([current[-1]] if len(current) % 2 else [])
    return current[0]


def _dyadic_payload(value: Fraction) -> dict:
    denominator = value.denominator
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("fused contrast coefficient is not dyadic")
    exponent = -(denominator.bit_length() - 1)
    numerator = value.numerator
    if numerator == 0:
        return {"significand": "0", "exponent_2": 0}
    while numerator % 2 == 0:
        numerator //= 2
        exponent += 1
    return {"significand": str(numerator), "exponent_2": exponent}


@dataclass(frozen=True)
class ExactFinalContrastFusion:
    weights: tuple[Fraction, ...]
    bias: Fraction
    input_closure_canonical_json: str

    def payload(self) -> dict:
        return {
            "schema_version": "green-v400-exact-final-contrast-fusion-v1",
            "weights": [_dyadic_payload(value) for value in self.weights],
            "bias": _dyadic_payload(self.bias),
            "d_model": len(self.weights),
            "input_closure": json.loads(self.input_closure_canonical_json),
        }

    def semantic_hash(self) -> str:
        return sha256_canonical(self.payload())


def fuse_final_contrast_exact(unembed, bias, suffix_ids,
                              coefficients) -> ExactFinalContrastFusion:
    inputs = [np.asarray(value) for value in (unembed, bias, suffix_ids, coefficients)]
    expected_dtypes = [np.dtype("<f4"), np.dtype("<f4"), np.dtype("<i8"), np.dtype("<f8")]
    if any(value.dtype != dtype for value, dtype in zip(inputs, expected_dtypes)):
        raise ValueError("final-contrast fusion requires canonical f32/f32/i64/f64 inputs")
    if any(not value.flags.c_contiguous for value in inputs):
        raise ValueError("final-contrast fusion inputs must already be C-contiguous")
    unembed, bias, suffix_ids, coefficients = inputs
    if (unembed.ndim != 2 or unembed.shape[0] == 0 or bias.ndim != 1 or suffix_ids.ndim != 1
            or coefficients.ndim != 1):
        raise ValueError("final-contrast fusion inputs have noncanonical ranks")
    if (not np.isfinite(unembed).all() or not np.isfinite(bias).all()
            or not np.isfinite(coefficients).all()):
        raise ValueError("final-contrast fusion inputs must be finite")
    if (unembed.ndim != 2 or unembed.shape[1] != bias.size
            or suffix_ids.size == 0 or suffix_ids.size != coefficients.size
            or np.any(suffix_ids < 0) or np.any(suffix_ids >= bias.size)
            or len(set(int(value) for value in suffix_ids)) != suffix_ids.size):
        raise ValueError("invalid final-contrast fusion shapes or suffix ids")
    exact_coefficients = [_exact_ieee_fraction(value) for value in coefficients]
    weights = []
    for coordinate in range(unembed.shape[0]):
        weights.append(_balanced_sum([
            coefficient * _exact_ieee_fraction(unembed[coordinate, token_id])
            for coefficient, token_id in zip(exact_coefficients, suffix_ids)
        ]))
    fused_bias = _balanced_sum([
        coefficient * _exact_ieee_fraction(bias[token_id])
        for coefficient, token_id in zip(exact_coefficients, suffix_ids)
    ])
    names = ("unembed.W_U_full", "unembed.b_U_full", "unembed.suffix_ids",
             "contrast.coefficients")
    closure = {}
    for name, array in zip(names, (unembed, bias, suffix_ids, coefficients)):
        prefix = canonical_json({
            "dtype": array.dtype.str, "shape": list(array.shape),
            "byte_order": "<", "layout": "C",
        }).encode("ascii") + b"\0"
        closure[name] = {
            "dtype": array.dtype.str, "shape": list(array.shape),
            "semantic_sha256": hashlib.sha256(prefix + array.tobytes()).hexdigest(),
        }
    return ExactFinalContrastFusion(tuple(weights), fused_bias, canonical_json(closure))
