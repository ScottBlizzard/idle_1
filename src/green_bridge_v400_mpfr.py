"""MPFR contexts and bit-exact IEEE import for GREEN v4 certificates."""
from __future__ import annotations

from contextlib import contextmanager
import math
import platform
import struct
import sys
from typing import Iterator

import gmpy2


ROUND_DOWN = gmpy2.RoundDown
ROUND_UP = gmpy2.RoundUp
ROUND_NEAREST = gmpy2.RoundToNearest


@contextmanager
def mpfr_context(precision_bits: int, rounding=ROUND_NEAREST) -> Iterator[None]:
    if precision_bits < 2:
        raise ValueError("MPFR precision must be at least two bits")
    context = gmpy2.get_context().copy()
    context.precision = int(precision_bits)
    context.round = rounding
    context.trap_invalid = True
    context.trap_divzero = True
    context.trap_overflow = True
    with gmpy2.context(context):
        yield


def _exact_ratio(value) -> tuple[int, int]:
    if isinstance(value, bool):
        return int(value), 1
    if isinstance(value, int):
        return value, 1
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, float):
        raise TypeError(f"unsupported IEEE value type: {type(value)!r}")
    if not math.isfinite(value):
        raise ValueError("NaN and infinity are forbidden")
    return value.as_integer_ratio()


def exact_mpfr_from_ieee(value, *, precision_bits: int) -> gmpy2.mpfr:
    """Decode a finite Python/NumPy IEEE value as its exact dyadic rational."""
    if hasattr(value, "item"):
        value = value.item()
    numerator, denominator = _exact_ratio(value)
    rational = gmpy2.mpq(numerator, denominator)
    if isinstance(value, (int, bool)):
        required = max(2, abs(numerator).bit_length())
    else:
        # Exponent magnitude does not consume MPFR significand precision.  Strip
        # every power of two and count only the odd IEEE significand bits.
        odd = abs(numerator)
        while odd and odd % 2 == 0:
            odd //= 2
        required = max(2, odd.bit_length())
    if precision_bits < required:
        raise ValueError(f"precision {precision_bits} cannot import value exactly; need {required}")
    with mpfr_context(precision_bits, ROUND_NEAREST):
        result = gmpy2.mpfr(rational)
    if gmpy2.mpq(result) != rational:
        raise RuntimeError("IEEE import was not exact")
    return result


def exact_interval_from_ieee(value, *, precision_bits: int):
    from green_bridge_v400_interval import Interval
    exact = exact_mpfr_from_ieee(value, precision_bits=precision_bits)
    return Interval(exact, exact, precision_bits)


def directed_binary(name: str, left, right, *, precision_bits: int, rounding):
    with mpfr_context(precision_bits, rounding):
        a, b = gmpy2.mpfr(left), gmpy2.mpfr(right)
        if name == "add":
            return a + b
        if name == "sub":
            return a - b
        if name == "mul":
            return a * b
        if name == "div":
            return a / b
        if name == "fma":
            raise ValueError("directed_binary fma requires three arguments")
    raise ValueError(f"unknown binary operation {name}")


def directed_fma(a, b, c, *, precision_bits: int, rounding):
    with mpfr_context(precision_bits, rounding):
        return gmpy2.fma(gmpy2.mpfr(a), gmpy2.mpfr(b), gmpy2.mpfr(c))


def directed_pairwise_sum(values, *, precision_bits: int, rounding):
    """Sum in a fixed balanced tree under one directed MPFR rounding mode."""
    level = list(values)
    if not level:
        return gmpy2.mpfr(0)
    while len(level) > 1:
        next_level = []
        for index in range(0, len(level) - 1, 2):
            next_level.append(directed_binary(
                "add", level[index], level[index + 1],
                precision_bits=precision_bits, rounding=rounding,
            ))
        if len(level) & 1:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def rounding_environment_manifest() -> dict:
    return {
        "schema_version": "green-v400-rounding-environment-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "gmpy2": gmpy2.version(),
        "mpfr": gmpy2.mpfr_version(),
        "gmp": gmpy2.mp_version(),
        "mpc": gmpy2.mpc_version(),
        "rounding_modes": {
            "lower": str(ROUND_DOWN), "upper": str(ROUND_UP),
            "nearest": str(ROUND_NEAREST),
        },
        "gpu_used_for_certificate": False,
    }


def assert_precision_nesting(low_precision_record, high_precision_record) -> None:
    """Require every high-precision interval to be inside its official peer."""
    if isinstance(low_precision_record, dict):
        if set(low_precision_record) != set(high_precision_record):
            raise AssertionError("precision records have different keys")
        for key in low_precision_record:
            assert_precision_nesting(low_precision_record[key], high_precision_record[key])
        return
    if isinstance(low_precision_record, (list, tuple)):
        if len(low_precision_record) != len(high_precision_record):
            raise AssertionError("precision records have different lengths")
        for low, high in zip(low_precision_record, high_precision_record):
            assert_precision_nesting(low, high)
        return
    if not (low_precision_record.lower <= high_precision_record.lower
            and high_precision_record.upper <= low_precision_record.upper):
        raise AssertionError("higher-precision interval is not nested")
