"""Partitioned signed-curvature Joint Witness certificates."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
import json
from pathlib import Path
from typing import Iterable

import gmpy2

from green_bridge_v400_interval import EmptyIntersection, Interval
from green_bridge_v400_relational_graph import RelationalGraph
from green_bridge_v400_schemas import CertificatePlan, JointWitnessRowSpec
from green_bridge_v400_schemas import sha256_canonical


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
    audit_endpoint: EndpointCertificate
    curvature: CurvatureCertificate
    audit_curvature: CurvatureCertificate
    endpoint_error: EndpointErrorCertificate
    audit_endpoint_error: EndpointErrorCertificate
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
    row_hash: str = ""
    audit_witness_interval: Interval | None = None
    resource_reason: str | None = None


def _exact_mpfr_payload(value) -> list[int]:
    exact = gmpy2.mpq(value)
    return [int(exact.numerator), int(exact.denominator)]


def _interval_payload(interval: Interval | None) -> dict | None:
    if interval is None:
        return None
    return {
        "precision_bits": interval.precision_bits,
        "lower": _exact_mpfr_payload(interval.lower),
        "upper": _exact_mpfr_payload(interval.upper),
    }


def _cell_payload(certificate: CellCertificate) -> dict:
    return {
        "cell": {
            "lower": [certificate.cell.lower.numerator, certificate.cell.lower.denominator],
            "upper": [certificate.cell.upper.numerator, certificate.cell.upper.denominator],
            "depth": certificate.cell.depth,
        },
        "value": _interval_payload(certificate.value),
        "first": _interval_payload(certificate.first),
        "second": _interval_payload(certificate.second),
    }


def _endpoint_payload(certificate: EndpointCertificate | None) -> dict | None:
    if certificate is None:
        return None
    return {
        "h": [certificate.h.numerator, certificate.h.denominator],
        "negative": _interval_payload(certificate.negative),
        "center": _interval_payload(certificate.center),
        "positive": _interval_payload(certificate.positive),
        "slope": _interval_payload(certificate.slope),
    }


def _curvature_payload(certificate: CurvatureCertificate | None) -> dict | None:
    if certificate is None:
        return None
    return {
        "positive": _interval_payload(certificate.positive),
        "negative": _interval_payload(certificate.negative),
        "secant": _interval_payload(certificate.secant),
        "m2": _interval_payload(certificate.m2),
    }


def _endpoint_error_payload(
    certificate: EndpointErrorCertificate | None,
) -> dict | None:
    if certificate is None:
        return None
    return {
        "positive_residual": _interval_payload(certificate.positive_residual),
        "negative_residual": _interval_payload(certificate.negative_residual),
        "epsilon_psi": _exact_mpfr_payload(certificate.epsilon_psi),
    }


def joint_witness_certificate_payload(
    certificate: JointWitnessCertificate, *, row_spec: JointWitnessRowSpec,
    plan: CertificatePlan,
) -> dict:
    """Serialize the current certificate object for synthetic rows only.

    The current dataclasses do not yet carry the binding per-operation rounding
    and runtime provenance, so this payload must not be represented as the final
    formal certificate artifact.
    """
    if not isinstance(row_spec, JointWitnessRowSpec) or not isinstance(plan, CertificatePlan):
        raise TypeError("certificate serialization requires validated row and plan schemas")
    if row_spec.split != "synthetic" or plan.execution_authorized:
        raise RuntimeError("REAL_CERTIFICATE_SERIALIZATION_UNAUTHORIZED")
    if plan.row_hash != row_spec.row_hash or certificate.row_hash != row_spec.row_hash:
        raise ValueError("certificate serialization identity mismatch")
    radii = []
    for radius in certificate.radii:
        radii.append({
            "h": [radius.h.numerator, radius.h.denominator],
            "official_witness": _interval_payload(radius.official_witness),
            "audit_witness": _interval_payload(radius.audit_witness),
            "endpoint": _endpoint_payload(radius.endpoint),
            "audit_endpoint": _endpoint_payload(radius.audit_endpoint),
            "curvature": _curvature_payload(radius.curvature),
            "audit_curvature": _curvature_payload(radius.audit_curvature),
            "endpoint_error": _endpoint_error_payload(radius.endpoint_error),
            "audit_endpoint_error": _endpoint_error_payload(radius.audit_endpoint_error),
            "official_cells": [_cell_payload(cell) for cell in radius.official_cells],
            "audit_cells": [_cell_payload(cell) for cell in radius.audit_cells],
            "audit_nested": radius.audit_nested,
        })
    payload = {
        "schema_version": "green-v400-joint-witness-certificate-v1",
        "serialization_scope": "current_in_memory_certificate_object_only",
        "binding_component_accounting_complete": False,
        "row_hash": row_spec.row_hash,
        "row_spec_semantic_hash": sha256_canonical(row_spec),
        "certificate_plan_semantic_hash": sha256_canonical(plan),
        "certificate_plan": plan.to_dict(),
        "contains_scientific_outcome": False,
        "computation_status": certificate.status,
        "resource_reason": certificate.resource_reason,
        "h": [certificate.h.numerator, certificate.h.denominator],
        "witness_interval": _interval_payload(certificate.witness_interval),
        "audit_witness_interval": _interval_payload(certificate.audit_witness_interval),
        "endpoint": _endpoint_payload(certificate.endpoint),
        "curvature": _curvature_payload(certificate.curvature),
        "endpoint_error": _endpoint_error_payload(certificate.endpoint_error),
        "cells": [_cell_payload(cell) for cell in certificate.cells],
        "radii": radii,
        "audit_nested": certificate.audit_nested,
    }
    payload["certificate_semantic_hash"] = sha256_canonical(payload)
    return payload


def write_joint_witness_certificate(
    path: Path, certificate: JointWitnessCertificate, *,
    row_spec: JointWitnessRowSpec, plan: CertificatePlan,
) -> dict:
    """Write one immutable canonical certificate file and return its payload."""
    output = Path(path)
    if output.exists():
        raise FileExistsError("certificate output is immutable")
    payload = joint_witness_certificate_payload(
        certificate, row_spec=row_spec, plan=plan,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return payload


def certify_cell(graph: RelationalGraph, cell: DyadicCell,
                 precision_bits: int) -> CellCertificate:
    domain = cell.interval(precision_bits)
    if isinstance(graph, RelationalGraph):
        local = RelationalGraph(graph.nodes, graph.output_id, precision_bits)
        jet = local.evaluate(domain)
    else:
        evaluate_interval = getattr(graph, "evaluate_interval", None)
        if not callable(evaluate_interval):
            raise TypeError("certificate evaluator must provide evaluate_interval")
        jet = evaluate_interval(domain)
        if jet.precision_bits != precision_bits:
            raise RuntimeError("certificate evaluator precision mismatch")
    return CellCertificate(cell, jet.value, jet.first, jet.second)


def _point_interval(value: Fraction, precision_bits: int) -> Interval:
    return Interval.from_bounds(str(value), str(value), precision_bits)


def certify_endpoints_and_slope(graph: RelationalGraph, h: Fraction,
                                precision_bits: int) -> EndpointCertificate:
    if h <= 0:
        raise ValueError("witness radius must be positive")
    if isinstance(graph, RelationalGraph):
        evaluator = RelationalGraph(graph.nodes, graph.output_id, precision_bits).evaluate
    else:
        evaluator = getattr(graph, "evaluate_interval", None)
        if not callable(evaluator):
            raise TypeError("certificate evaluator must provide evaluate_interval")
    negative_jet = evaluator(_point_interval(-h, precision_bits))
    center_jet = evaluator(_point_interval(Fraction(0), precision_bits))
    positive_jet = evaluator(_point_interval(h, precision_bits))
    if any(jet.precision_bits != precision_bits
           for jet in (negative_jet, center_jet, positive_jet)):
        raise RuntimeError("certificate evaluator precision mismatch")
    negative = negative_jet.value
    positive = positive_jet.value
    return EndpointCertificate(h, negative, center_jet.value, positive, center_jet.first)


def certify_adaptive_cells(evaluator, h: Fraction, precision_bits: int,
                           plan: CertificatePlan) -> list[CellCertificate] | None:
    """Public outcome-agnostic adaptive partition entry for interval evaluators."""
    if not isinstance(plan, CertificatePlan):
        raise TypeError("adaptive certificate execution requires CertificatePlan")
    if getattr(evaluator, "contains_scientific_outcome", None) is not False:
        raise RuntimeError("ADAPTIVE_EVALUATOR_OUTCOME_BOUNDARY_MISSING")
    if getattr(evaluator, "certificate_row_hash", None) != plan.row_hash:
        raise RuntimeError("ADAPTIVE_EVALUATOR_PLAN_IDENTITY_MISMATCH")
    return _adaptive_cells(evaluator, h, precision_bits, plan)


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


def _cell_tolerance_met(certificate: CellCertificate, absolute: Fraction,
                        relative: Fraction) -> bool:
    """Compare both frozen tolerances as exact rationals, independent of context."""
    width = gmpy2.mpq(certificate.second.width())
    scale = max(gmpy2.mpq(1), gmpy2.mpq(certificate.second.magnitude()))
    absolute_limit = gmpy2.mpq(absolute.numerator, absolute.denominator)
    relative_limit = gmpy2.mpq(relative.numerator, relative.denominator) * scale
    return width <= absolute_limit and width <= relative_limit


def _cell_priority(certificate: CellCertificate, h: Fraction) -> gmpy2.mpq:
    """Return w(J) * wid(Q_J) with an exact curvature-kernel weight."""
    weight = _weight(certificate.cell, h)
    return (
        gmpy2.mpq(weight.numerator, weight.denominator)
        * gmpy2.mpq(certificate.second.width())
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


def _adaptive_cells_with_reason(
    graph: RelationalGraph, h: Fraction, precision_bits: int,
    plan: CertificatePlan,
) -> tuple[list[CellCertificate] | None, str | None]:
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
        if _cell_tolerance_met(certificate, absolute, relative):
            accepted.append(certificate)
            continue
        if cell.depth >= plan.max_depth:
            return None, "MAX_DEPTH_REACHED"
        if len(accepted) + len(pending) + 2 > plan.max_cells:
            return None, "MAX_CELLS_REACHED"
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
    return accepted, None


def _adaptive_cells(graph: RelationalGraph, h: Fraction, precision_bits: int,
                    plan: CertificatePlan) -> list[CellCertificate] | None:
    return _adaptive_cells_with_reason(graph, h, precision_bits, plan)[0]


def _audit_same_partition(graph: RelationalGraph,
                          official: list[CellCertificate],
                          precision_bits: int) -> list[CellCertificate]:
    return [certify_cell(graph, certificate.cell, precision_bits)
            for certificate in official]


def _radius_certificate(graph: RelationalGraph, h: Fraction,
                        plan: CertificatePlan) -> tuple[RadiusCertificate | None, str | None]:
    official, resource_reason = _adaptive_cells_with_reason(
        graph, h, plan.official_precision_bits, plan
    )
    if official is None:
        return None, resource_reason
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
    if not all(_interval_nested(high, low) for high, low in (
        (audit_curvature.positive, curvature.positive),
        (audit_curvature.negative, curvature.negative),
        (audit_curvature.secant, curvature.secant),
        (audit_curvature.m2, curvature.m2),
        (audit_error.positive_residual, error.positive_residual),
        (audit_error.negative_residual, error.negative_residual),
    )) or gmpy2.mpq(audit_error.epsilon_psi) > gmpy2.mpq(error.epsilon_psi):
        raise RuntimeError("CERTIFICATE_COMPONENT_NESTING_INVALID")
    witness = witness_interval(endpoint, curvature, error)
    audit_witness = witness_interval(audit_endpoint, audit_curvature, audit_error)
    if not _interval_nested(audit_witness, witness):
        raise RuntimeError("CERTIFICATE_WITNESS_NESTING_INVALID")
    return RadiusCertificate(
        h, witness, audit_witness, endpoint, audit_endpoint, curvature,
        audit_curvature, error, audit_error, tuple(official), tuple(audit), True,
    ), None


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
    audit_intersection: Interval | None = None
    for radius in plan.radii:
        h = radius.as_fraction()
        result, resource_reason = _radius_certificate(graph, h, plan)
        if result is None:
            return JointWitnessCertificate(
                "RESOURCE_INCONCLUSIVE", h, None, None, None, None, (),
                tuple(radius_certificates), False, row_spec.row_hash, None,
                resource_reason,
            )
        radius_certificates.append(result)
        try:
            intersection = (result.official_witness if intersection is None
                            else intersection.intersect(result.official_witness))
            audit_intersection = (
                result.audit_witness if audit_intersection is None
                else audit_intersection.intersect(result.audit_witness)
            )
        except EmptyIntersection:
            raise RuntimeError("CERTIFICATE_IMPLEMENTATION_INVALID")
        if not _interval_nested(audit_intersection, intersection):
            raise RuntimeError("CERTIFICATE_CROSS_RADIUS_NESTING_INVALID")
    final = radius_certificates[-1]
    return JointWitnessCertificate(
        "INTERVAL_COMPUTED", final.h, intersection, final.endpoint,
        final.curvature, final.endpoint_error, final.official_cells,
        tuple(radius_certificates), True, row_spec.row_hash,
        audit_intersection, None,
    )
