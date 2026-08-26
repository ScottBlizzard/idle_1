"""Partitioned signed-curvature Joint Witness certificates."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
from typing import Iterable

import gmpy2

from green_bridge_v400_interval import EmptyIntersection, Interval
from green_bridge_v400_mpfr import ROUND_UP, directed_binary
from green_bridge_v400_relational_graph import RelationalGraph
from green_bridge_v400_schemas import CertificatePlan


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
class RadiusCertificate:
    h: Fraction
    official_witness: Interval
    audit_witness: Interval
    endpoint: EndpointCertificate
    curvature: CurvatureCertificate
    endpoint_error: EndpointErrorCertificate
    official_cells: tuple[CellCertificate, ...]
    audit_cells: tuple[CellCertificate, ...]
    audit_nested: bool


@dataclass(frozen=True)
class JointWitnessCertificate:
    status: str
    h: Fraction
    witness_interval: Interval | None
    endpoint: EndpointCertificate | None
    curvature: CurvatureCertificate | None
    endpoint_error: EndpointErrorCertificate | None
    cells: tuple[CellCertificate, ...]
    radii: tuple[RadiusCertificate, ...] = ()
    audit_nested: bool = False


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


def _interval_nested(inner: Interval, outer: Interval) -> bool:
    return outer.lower <= inner.lower <= inner.upper <= outer.upper


def _cell_tolerance(certificate: CellCertificate, absolute: Fraction,
                    relative: Fraction) -> gmpy2.mpfr:
    scale = max(gmpy2.mpfr(1), certificate.second.magnitude())
    # The binding protocol requires both the absolute and relative width
    # criteria to hold.  Therefore the effective scalar threshold is the
    # smaller limit, not the more permissive one.
    return min(gmpy2.mpfr(absolute.numerator) / absolute.denominator,
               (gmpy2.mpfr(relative.numerator) / relative.denominator) * scale)


def _cell_priority(certificate: CellCertificate, h: Fraction) -> gmpy2.mpfr:
    """Return w(J) * wid(Q_J) with an exact curvature-kernel weight."""
    weight = _weight(certificate.cell, h)
    return directed_binary(
        "mul",
        gmpy2.mpq(weight.numerator, weight.denominator),
        certificate.second.width(),
        precision_bits=certificate.second.precision_bits,
        rounding=ROUND_UP,
    )


def _hex_fraction(value: str) -> Fraction:
    numerator, denominator = float.fromhex(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def _validate_partition(certificates: list[CellCertificate], h: Fraction) -> None:
    if not certificates:
        raise RuntimeError("CERTIFICATE_EMPTY_PARTITION")
    ordered = sorted(certificates, key=lambda item: item.cell.lower)
    if ordered[0].cell.lower != -h or ordered[-1].cell.upper != h:
        raise RuntimeError("CERTIFICATE_PARTITION_DOES_NOT_COVER_RADIUS")
    if not any(item.cell.upper == 0 for item in ordered):
        raise RuntimeError("CERTIFICATE_PARTITION_MISSING_ZERO_SPLIT")
    for left, right in zip(ordered, ordered[1:]):
        if left.cell.upper != right.cell.lower:
            raise RuntimeError("CERTIFICATE_PARTITION_GAP_OR_OVERLAP")


def _adaptive_cells(graph: RelationalGraph, h: Fraction, precision_bits: int,
                    plan: CertificatePlan) -> list[CellCertificate] | None:
    absolute = _hex_fraction(plan.absolute_width_tolerance)
    relative = _hex_fraction(plan.relative_width_tolerance)
    pending: list[tuple[gmpy2.mpfr, Fraction, int, CellCertificate]] = []
    for cell in (DyadicCell(-h, Fraction(0)), DyadicCell(Fraction(0), h)):
        certificate = certify_cell(graph, cell, precision_bits)
        heapq.heappush(pending, (-_cell_priority(certificate, h), cell.lower,
                                 cell.depth, certificate))
    accepted: list[CellCertificate] = []
    while pending:
        _, _, _, certificate = heapq.heappop(pending)
        cell = certificate.cell
        if certificate.second.width() <= _cell_tolerance(certificate, absolute, relative):
            accepted.append(certificate)
            continue
        if cell.depth >= plan.max_depth:
            return None
        if len(accepted) + len(pending) + 2 > plan.max_cells:
            return None
        left, right = cell.bisect()
        for child in (left, right):
            child_certificate = certify_cell(graph, child, precision_bits)
            heapq.heappush(
                pending,
                (-_cell_priority(child_certificate, h), child.lower,
                 child.depth, child_certificate),
            )
    accepted.sort(key=lambda item: item.cell.lower)
    _validate_partition(accepted, h)
    return accepted


def _audit_same_partition(graph: RelationalGraph,
                          official: list[CellCertificate],
                          precision_bits: int) -> list[CellCertificate]:
    return [certify_cell(graph, certificate.cell, precision_bits)
            for certificate in official]


def _radius_certificate(graph: RelationalGraph, h: Fraction,
                        plan: CertificatePlan) -> RadiusCertificate | None:
    official = _adaptive_cells(graph, h, plan.official_precision_bits, plan)
    if official is None:
        return None
    audit = _audit_same_partition(graph, official, plan.audit_precision_bits)
    for low, high in zip(official, audit):
        if not all((_interval_nested(high.value, low.value),
                    _interval_nested(high.first, low.first),
                    _interval_nested(high.second, low.second))):
            raise RuntimeError("CERTIFICATE_PRECISION_NESTING_INVALID")

    endpoint = certify_endpoints_and_slope(graph, h, plan.official_precision_bits)
    audit_endpoint = certify_endpoints_and_slope(graph, h, plan.audit_precision_bits)
    if not all(_interval_nested(high, low) for high, low in (
        (audit_endpoint.negative, endpoint.negative),
        (audit_endpoint.center, endpoint.center),
        (audit_endpoint.positive, endpoint.positive),
        (audit_endpoint.slope, endpoint.slope),
    )):
        raise RuntimeError("CERTIFICATE_ENDPOINT_NESTING_INVALID")

    curvature = integrate_signed_curvature(official, h)
    audit_curvature = integrate_signed_curvature(audit, h)
    error = compute_epsilon_psi(endpoint, curvature)
    audit_error = compute_epsilon_psi(audit_endpoint, audit_curvature)
    witness = witness_interval(endpoint, curvature, error)
    audit_witness = witness_interval(audit_endpoint, audit_curvature, audit_error)
    if not _interval_nested(audit_witness, witness):
        raise RuntimeError("CERTIFICATE_WITNESS_NESTING_INVALID")
    return RadiusCertificate(h, witness, audit_witness, endpoint, curvature,
                             error, tuple(official), tuple(audit), True)


def certify_joint_witness(row_spec, plan) -> JointWitnessCertificate:
    if not isinstance(plan, CertificatePlan):
        raise TypeError("certificate execution requires a validated CertificatePlan")
    if plan.row_hash != getattr(row_spec, "row_hash", None):
        raise ValueError("certificate plan row hash mismatch")
    if not getattr(row_spec, "split", None) == "synthetic":
        if not plan.execution_authorized:
            raise RuntimeError("REAL_ROW_CERTIFICATE_UNAUTHORIZED")
    graph = getattr(row_spec, "graph", None)
    if graph is None:
        from green_bridge_v400_relational_graph import extract_joint_witness_graph
        graph = extract_joint_witness_graph(row_spec)
    radius_certificates: list[RadiusCertificate] = []
    intersection: Interval | None = None
    for radius in plan.radii:
        h = radius.as_fraction()
        result = _radius_certificate(graph, h, plan)
        if result is None:
            return JointWitnessCertificate("RESOURCE_INCONCLUSIVE", h, None, None,
                                           None, None, (), tuple(radius_certificates), False)
        radius_certificates.append(result)
        try:
            intersection = (result.official_witness if intersection is None
                            else intersection.intersect(result.official_witness))
        except EmptyIntersection:
            return JointWitnessCertificate("RADIUS_INTERSECTION_EMPTY", h, None,
                                           result.endpoint, result.curvature,
                                           result.endpoint_error,
                                           result.official_cells,
                                           tuple(radius_certificates), True)
    final = radius_certificates[-1]
    return JointWitnessCertificate("CERTIFIED", final.h, intersection,
                                   final.endpoint, final.curvature,
                                   final.endpoint_error, final.official_cells,
                                   tuple(radius_certificates), True)
