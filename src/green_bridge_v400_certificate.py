"""Partitioned signed-curvature Joint Witness certificates."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
from typing import Iterable

import gmpy2

from green_bridge_v400_interval import EmptyIntersection, Interval
from green_bridge_v400_relational_graph import RelationalGraph


@dataclass(frozen=True)
class DyadicCell:
    lower: Fraction
    upper: Fraction
    depth: int = 0

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError("invalid dyadic cell")

    def midpoint(self) -> Fraction:
        return (self.lower + self.upper) / 2

    def bisect(self) -> tuple["DyadicCell", "DyadicCell"]:
        midpoint = self.midpoint()
        return (DyadicCell(self.lower, midpoint, self.depth + 1),
                DyadicCell(midpoint, self.upper, self.depth + 1))

    def interval(self, precision_bits: int) -> Interval:
        return Interval.from_bounds(str(self.lower), str(self.upper), precision_bits)


@dataclass(frozen=True)
class CellCertificate:
    cell: DyadicCell
    value: Interval
    first: Interval
    second: Interval


@dataclass(frozen=True)
class EndpointCertificate:
    h: Fraction
    negative: Interval
    center: Interval
    positive: Interval
    slope: Interval


@dataclass(frozen=True)
class CurvatureCertificate:
    positive: Interval
    negative: Interval
    secant: Interval
    m2: Interval


@dataclass(frozen=True)
class EndpointErrorCertificate:
    positive_residual: Interval
    negative_residual: Interval
    epsilon_psi: gmpy2.mpfr


@dataclass(frozen=True)
class JointWitnessCertificate:
    status: str
    h: Fraction
    witness_interval: Interval | None
    endpoint: EndpointCertificate | None
    curvature: CurvatureCertificate | None
    endpoint_error: EndpointErrorCertificate | None
    cells: tuple[CellCertificate, ...]


def certify_cell(graph: RelationalGraph, cell: DyadicCell,
                 precision_bits: int) -> CellCertificate:
    local = RelationalGraph(graph.nodes, graph.output_id, precision_bits)
    jet = local.evaluate(cell.interval(precision_bits))
    return CellCertificate(cell, jet.value, jet.first, jet.second)


def _point_interval(value: Fraction, precision_bits: int) -> Interval:
    return Interval.from_bounds(str(value), str(value), precision_bits)


def certify_endpoints_and_slope(graph: RelationalGraph, h: Fraction,
                                precision_bits: int) -> EndpointCertificate:
    if h <= 0:
        raise ValueError("witness radius must be positive")
    local = RelationalGraph(graph.nodes, graph.output_id, precision_bits)
    negative = local.evaluate(_point_interval(-h, precision_bits)).value
    center_jet = local.evaluate(_point_interval(Fraction(0), precision_bits))
    positive = local.evaluate(_point_interval(h, precision_bits)).value
    return EndpointCertificate(h, negative, center_jet.value, positive, center_jet.first)


def compute_m2(cell_certificates: list[CellCertificate]) -> Interval:
    if not cell_certificates:
        raise ValueError("M2 needs at least one cell")
    precision = cell_certificates[0].second.precision_bits
    maximum = max(cell.second.magnitude() for cell in cell_certificates)
    return Interval.from_bounds(0, maximum, precision)


def _weight(cell: DyadicCell, h: Fraction) -> Fraction:
    a, b = cell.lower, cell.upper
    if a >= 0:
        return h * (b - a) - (b*b - a*a) / 2
    if b <= 0:
        return h * (b - a) + (b*b - a*a) / 2
    raise ValueError("curvature cell crosses zero")


def integrate_signed_curvature(cell_certificates: list[CellCertificate],
                               h: Fraction) -> CurvatureCertificate:
    if not cell_certificates:
        raise ValueError("curvature integration needs cells")
    precision = cell_certificates[0].second.precision_bits
    positive = Interval.point(0, precision)
    negative = Interval.point(0, precision)
    for certificate in cell_certificates:
        if certificate.cell.lower < -h or certificate.cell.upper > h:
            raise ValueError("curvature cell outside witness radius")
        weight = Interval.point(gmpy2.mpq(_weight(certificate.cell, h).numerator,
                                          _weight(certificate.cell, h).denominator), precision)
        contribution = weight * certificate.second
        if certificate.cell.lower >= 0:
            positive = positive + contribution
        elif certificate.cell.upper <= 0:
            negative = negative + contribution
        else:
            raise ValueError("partition must split at zero")
    return CurvatureCertificate(positive, negative, positive - negative,
                                compute_m2(cell_certificates))


def compute_epsilon_psi(endpoint: EndpointCertificate,
                        curvature: CurvatureCertificate) -> EndpointErrorCertificate:
    precision = endpoint.center.precision_bits
    h = Interval.point(gmpy2.mpq(endpoint.h.numerator, endpoint.h.denominator), precision)
    direct_positive = endpoint.positive - endpoint.center - h * endpoint.slope
    direct_negative = endpoint.negative - endpoint.center + h * endpoint.slope
    try:
        positive = direct_positive.intersect(curvature.positive)
        negative = direct_negative.intersect(curvature.negative)
    except EmptyIntersection as error:
        raise RuntimeError("CERTIFICATE_IMPLEMENTATION_INVALID") from error
    epsilon = max(positive.magnitude(), negative.magnitude())
    return EndpointErrorCertificate(positive, negative, epsilon)


def witness_interval(endpoint: EndpointCertificate,
                     curvature: CurvatureCertificate,
                     endpoint_error: EndpointErrorCertificate | None = None) -> Interval:
    precision = endpoint.center.precision_bits
    h = Interval.point(gmpy2.mpq(endpoint.h.numerator, endpoint.h.denominator), precision)
    two_h = Interval.point(2, precision) * h
    secant = (endpoint.positive - endpoint.negative) / two_h
    secant_remainder = curvature.secant if endpoint_error is None else (
        endpoint_error.positive_residual - endpoint_error.negative_residual
    )
    # central secant = Psi'(0) + K_sec/(2h)
    return secant - secant_remainder / two_h


def certify_joint_witness(row_spec, plan) -> JointWitnessCertificate:
    if not getattr(row_spec, "split", None) == "synthetic":
        if not getattr(plan, "execution_authorized", False):
            raise RuntimeError("REAL_ROW_CERTIFICATE_UNAUTHORIZED")
    graph = getattr(row_spec, "graph", None)
    if graph is None:
        from green_bridge_v400_relational_graph import extract_joint_witness_graph
        graph = extract_joint_witness_graph(row_spec)
    h = getattr(plan, "h", Fraction(1))
    precision = getattr(plan, "precision_bits", graph.precision_bits)
    if getattr(plan, "max_cells", 2) < 2:
        return JointWitnessCertificate("RESOURCE_INCONCLUSIVE", h, None, None,
                                       None, None, ())
    cells = [DyadicCell(-h, Fraction(0)), DyadicCell(Fraction(0), h)]
    certificates = [certify_cell(graph, cell, precision) for cell in cells]
    endpoint = certify_endpoints_and_slope(graph, h, precision)
    curvature = integrate_signed_curvature(certificates, h)
    error = compute_epsilon_psi(endpoint, curvature)
    interval = witness_interval(endpoint, curvature, error)
    return JointWitnessCertificate("CERTIFIED", h, interval, endpoint, curvature,
                                   error, tuple(certificates))
