"""Immutable outward MPFR intervals used by GREEN v4."""
from __future__ import annotations

from dataclasses import dataclass
import math

import gmpy2

from green_bridge_v400_mpfr import (
    ROUND_DOWN, ROUND_NEAREST, ROUND_UP, directed_binary, exact_mpfr_from_ieee,
    mpfr_context,
)


class EmptyIntersection(ValueError):
    pass


def _digits(value: gmpy2.mpfr) -> dict:
    significand, exponent, precision = value.digits(16)
    return {"significand_hex": significand, "exponent": exponent, "precision": precision}


@dataclass(frozen=True)
class Interval:
    lower: gmpy2.mpfr
    upper: gmpy2.mpfr
    precision_bits: int

    def __post_init__(self):
        if not (gmpy2.is_finite(self.lower) and gmpy2.is_finite(self.upper)):
            raise ValueError("interval endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        if self.precision_bits < 2:
            raise ValueError("invalid interval precision")

    @classmethod
    def point(cls, value, precision_bits: int = 256) -> "Interval":
        if isinstance(value, gmpy2.mpfr):
            rational = gmpy2.mpq(value)
            with mpfr_context(precision_bits, ROUND_DOWN):
                lower = gmpy2.mpfr(rational)
            with mpfr_context(precision_bits, ROUND_UP):
                upper = gmpy2.mpfr(rational)
            return cls(lower, upper, precision_bits)
        if isinstance(value, (int, bool, float)) or hasattr(value, "item"):
            exact = exact_mpfr_from_ieee(value, precision_bits=precision_bits)
            return cls(exact, exact, precision_bits)
        rational = gmpy2.mpq(str(value))
        with mpfr_context(precision_bits, ROUND_DOWN):
            lower = gmpy2.mpfr(rational)
        with mpfr_context(precision_bits, ROUND_UP):
            upper = gmpy2.mpfr(rational)
        return cls(lower, upper, precision_bits)

    @classmethod
    def from_bounds(cls, lower, upper, precision_bits: int = 256) -> "Interval":
        def convert(value, rounding):
            if isinstance(value, gmpy2.mpfr):
                source = gmpy2.mpq(value)
            elif isinstance(value, (int, bool, float)) or hasattr(value, "item"):
                numerator, denominator = (value.item() if hasattr(value, "item") else value).as_integer_ratio() if not isinstance(value, (int, bool)) else (int(value), 1)
                source = gmpy2.mpq(numerator, denominator)
            else:
                source = gmpy2.mpq(str(value))
            with mpfr_context(precision_bits, rounding):
                return gmpy2.mpfr(source)
        return cls(convert(lower, ROUND_DOWN), convert(upper, ROUND_UP), precision_bits)

    def _coerce(self, other) -> "Interval":
        if isinstance(other, Interval):
            if other.precision_bits == self.precision_bits:
                return other
            return Interval.from_bounds(other.lower, other.upper, self.precision_bits)
        return Interval.point(other, self.precision_bits)

    def __add__(self, other):
        other = self._coerce(other)
        return Interval(
            directed_binary("add", self.lower, other.lower, precision_bits=self.precision_bits, rounding=ROUND_DOWN),
            directed_binary("add", self.upper, other.upper, precision_bits=self.precision_bits, rounding=ROUND_UP),
            self.precision_bits,
        )

    __radd__ = __add__

    def __neg__(self):
        with mpfr_context(self.precision_bits, ROUND_NEAREST):
            lower = gmpy2.mpfr(-gmpy2.mpq(self.upper))
            upper = gmpy2.mpfr(-gmpy2.mpq(self.lower))
        return Interval(lower, upper, self.precision_bits)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        lowers, uppers = [], []
        for left in (self.lower, self.upper):
            for right in (other.lower, other.upper):
                lowers.append(directed_binary("mul", left, right, precision_bits=self.precision_bits, rounding=ROUND_DOWN))
                uppers.append(directed_binary("mul", left, right, precision_bits=self.precision_bits, rounding=ROUND_UP))
        return Interval(min(lowers), max(uppers), self.precision_bits)

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lower <= 0 <= self.upper:
            raise ZeroDivisionError("interval reciprocal crosses zero")
        lower = directed_binary("div", 1, self.upper, precision_bits=self.precision_bits, rounding=ROUND_DOWN)
        upper = directed_binary("div", 1, self.lower, precision_bits=self.precision_bits, rounding=ROUND_UP)
        return Interval(min(lower, upper), max(lower, upper), self.precision_bits)

    def __truediv__(self, other):
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def square(self):
        zero = gmpy2.mpfr(0)
        products_down = [directed_binary("mul", endpoint, endpoint, precision_bits=self.precision_bits, rounding=ROUND_DOWN) for endpoint in (self.lower, self.upper)]
        products_up = [directed_binary("mul", endpoint, endpoint, precision_bits=self.precision_bits, rounding=ROUND_UP) for endpoint in (self.lower, self.upper)]
        if self.lower <= 0 <= self.upper:
            return Interval(zero, max(products_up), self.precision_bits)
        return Interval(min(products_down), max(products_up), self.precision_bits)

    def intersect(self, other) -> "Interval":
        other = self._coerce(other)
        lower, upper = max(self.lower, other.lower), min(self.upper, other.upper)
        if lower > upper:
            raise EmptyIntersection("intervals are disjoint")
        return Interval(lower, upper, self.precision_bits)

    def hull(self, other) -> "Interval":
        other = self._coerce(other)
        return Interval(min(self.lower, other.lower), max(self.upper, other.upper), self.precision_bits)

    def width(self):
        return directed_binary("sub", self.upper, self.lower, precision_bits=self.precision_bits, rounding=ROUND_UP)

    def midpoint(self):
        with mpfr_context(self.precision_bits, ROUND_NEAREST):
            return (self.lower + self.upper) / 2

    def radius(self):
        return directed_binary("div", self.width(), 2, precision_bits=self.precision_bits, rounding=ROUND_UP)

    def magnitude(self):
        with mpfr_context(self.precision_bits, ROUND_UP):
            return max(abs(self.lower), abs(self.upper))

    def contains(self, value) -> bool:
        if isinstance(value, gmpy2.mpfr):
            exact = value
        else:
            exact = exact_mpfr_from_ieee(value, precision_bits=self.precision_bits)
        return self.lower <= exact <= self.upper

    def canonical(self) -> dict:
        return {
            "lower": _digits(self.lower), "upper": _digits(self.upper),
            "precision_bits": self.precision_bits,
        }


def _monotone(interval: Interval, function) -> Interval:
    with mpfr_context(interval.precision_bits, ROUND_DOWN):
        lower = function(interval.lower)
    with mpfr_context(interval.precision_bits, ROUND_UP):
        upper = function(interval.upper)
    return Interval(lower, upper, interval.precision_bits)


def exp_interval(interval: Interval) -> Interval:
    return _monotone(interval, gmpy2.exp)


def log_interval(interval: Interval) -> Interval:
    if interval.lower <= 0:
        raise ValueError("log interval must be strictly positive")
    return _monotone(interval, gmpy2.log)


def sqrt_interval(interval: Interval) -> Interval:
    if interval.lower < 0:
        raise ValueError("sqrt interval must be nonnegative")
    return _monotone(interval, gmpy2.sqrt)


def inv_sqrt_interval(interval: Interval) -> Interval:
    if interval.lower <= 0:
        raise ValueError("inverse sqrt interval must be strictly positive")
    return sqrt_interval(interval).reciprocal()


def tanh_interval(interval: Interval) -> Interval:
    return _monotone(interval, gmpy2.tanh)


def erf_interval(interval: Interval) -> Interval:
    return _monotone(interval, gmpy2.erf)
