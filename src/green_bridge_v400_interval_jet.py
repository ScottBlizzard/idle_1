"""Value/first/second outward interval jets for one real control variable."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

from green_bridge_v400_interval import Interval


@dataclass(frozen=True)
class Jet2:
    value: Interval
    first: Interval
    second: Interval

    def __post_init__(self):
        precisions = {self.value.precision_bits, self.first.precision_bits, self.second.precision_bits}
        if len(precisions) != 1:
            raise ValueError("jet components have different precision")

    @property
    def precision_bits(self) -> int:
        return self.value.precision_bits


@dataclass(frozen=True)
class JetTensor:
    """Immutable tensor view whose entries retain shared Jet2 identities."""
    flat: tuple[Jet2, ...]
    shape: tuple[int, ...]

    def __post_init__(self):
        size = 1
        for dimension in self.shape:
            if dimension < 0:
                raise ValueError("negative tensor dimension")
            size *= dimension
        if size != len(self.flat):
            raise ValueError("JetTensor shape does not match storage")

    @classmethod
    def from_flat(cls, values: Sequence[Jet2], shape: Sequence[int]) -> "JetTensor":
        return cls(tuple(values), tuple(int(value) for value in shape))

    def reshape(self, shape: Sequence[int]) -> "JetTensor":
        return JetTensor(self.flat, tuple(int(value) for value in shape))

    def __iter__(self) -> Iterator[Jet2]:
        return iter(self.flat)


@dataclass(frozen=True)
class CertifiedScalarPrimitive:
    name: str
    value_interval: Callable[[Interval], Interval]
    first_interval: Callable[[Interval], Interval]
    second_interval: Callable[[Interval], Interval]


def constant_jet(x: Interval) -> Jet2:
    zero = Interval.point(0, x.precision_bits)
    return Jet2(x, zero, zero)


def affine_control_jet(base: Interval, direction: Interval, domain: Interval) -> Jet2:
    if not (base.precision_bits == direction.precision_bits == domain.precision_bits):
        raise ValueError("affine control precision mismatch")
    zero = Interval.point(0, base.precision_bits)
    return Jet2(base + domain * direction, direction, zero)


def add_jet(x: Jet2, y: Jet2) -> Jet2:
    return Jet2(x.value + y.value, x.first + y.first, x.second + y.second)


def sub_jet(x: Jet2, y: Jet2) -> Jet2:
    return Jet2(x.value - y.value, x.first - y.first, x.second - y.second)


def mul_jet(x: Jet2, y: Jet2) -> Jet2:
    two = Interval.point(2, x.precision_bits)
    return Jet2(
        x.value * y.value,
        x.first * y.value + x.value * y.first,
        x.second * y.value + two * x.first * y.first + x.value * y.second,
    )


def square_jet(x: Jet2) -> Jet2:
    """Dependency-aware square; unlike x*x its value is never spuriously negative."""
    two = Interval.point(2, x.precision_bits)
    return Jet2(
        x.value.square(),
        two * x.value * x.first,
        two * (x.first.square() + x.value * x.second),
    )


def reciprocal_jet(x: Jet2) -> Jet2:
    inverse = x.value.reciprocal()
    inverse2, inverse3 = inverse.square(), inverse.square() * inverse
    two = Interval.point(2, x.precision_bits)
    return Jet2(
        inverse,
        -x.first * inverse2,
        two * x.first.square() * inverse3 - x.second * inverse2,
    )


def compose_jet(x: Jet2, primitive: CertifiedScalarPrimitive) -> Jet2:
    value = primitive.value_interval(x.value)
    first_factor = primitive.first_interval(x.value)
    second_factor = primitive.second_interval(x.value)
    return Jet2(
        value,
        first_factor * x.first,
        second_factor * x.first.square() + first_factor * x.second,
    )


def centered_tighten(cell: Interval, center_jet: Jet2, cell_jet: Jet2) -> Jet2:
    midpoint = Interval.point(cell.midpoint(), cell.precision_bits)
    delta = cell - midpoint
    value_mean = center_jet.value + delta * cell_jet.first
    first_mean = center_jet.first + delta * cell_jet.second
    return Jet2(
        cell_jet.value.intersect(value_mean),
        cell_jet.first.intersect(first_mean),
        cell_jet.second,
    )
