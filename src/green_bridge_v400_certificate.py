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
from green_bridge_v400_schemas import (
    AnytimeCellState, CertificatePlan, JointWitnessRowSpec,
    MonotoneAnytimeCertificateState, RESOURCE_REASONS, canonical_json,
    sha256_canonical,
)


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
    accounting: "CurvatureComponentAccounting | None" = None


@dataclass(frozen=True)
class CurvatureComponentAccounting:
    """Exact-rational audit split for signed curvature integration.

    The weight term includes outward multiplication of an exact rational weight
    by the certified cell interval.  The summation term is then isolated with an
    exact directed-shadow sum of those already-rounded products.
    """

    positive_midpoint: gmpy2.mpq
    negative_midpoint: gmpy2.mpq
    positive_radius: gmpy2.mpq
    negative_radius: gmpy2.mpq
    positive_weight_rounding: gmpy2.mpq
    negative_weight_rounding: gmpy2.mpq
    positive_summation_rounding: gmpy2.mpq
    negative_summation_rounding: gmpy2.mpq


@dataclass(frozen=True)
class EndpointErrorCertificate:
    positive_residual: Interval
    negative_residual: Interval
    epsilon_psi: gmpy2.mpfr
    accounting: "EndpointErrorComponentAccounting | None" = None


@dataclass(frozen=True)
class EndpointErrorComponentAccounting:
    positive_endpoint_evaluation_radius: gmpy2.mpq
    negative_endpoint_evaluation_radius: gmpy2.mpq
    center_evaluation_radius: gmpy2.mpq
    slope_evaluation_contribution: gmpy2.mpq
    positive_direct_residual_center: gmpy2.mpq
    negative_direct_residual_center: gmpy2.mpq
    positive_direct_residual_radius: gmpy2.mpq
    negative_direct_residual_radius: gmpy2.mpq
    positive_direct_arithmetic_rounding: gmpy2.mpq
    negative_direct_arithmetic_rounding: gmpy2.mpq
    positive_curvature_midpoint: gmpy2.mpq
    negative_curvature_midpoint: gmpy2.mpq
    positive_curvature_radius: gmpy2.mpq
    negative_curvature_radius: gmpy2.mpq
    positive_weight_rounding: gmpy2.mpq
    negative_weight_rounding: gmpy2.mpq
    positive_summation_rounding: gmpy2.mpq
    negative_summation_rounding: gmpy2.mpq
    input_import_contribution: gmpy2.mpq
    graph_reduction_contribution: gmpy2.mpq
    subdivision_contribution: gmpy2.mpq


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


class AnytimeEvaluationFailure(RuntimeError):
    """A failed sibling attempt requiring reconciliation with the admission ledger."""

    def __init__(self, prior_state_semantic_hash: str, original: Exception):
        super().__init__(f"ANYTIME_EVALUATION_FAILED_AFTER_ADMISSION_ATTEMPT: {original}")
        self.prior_state_semantic_hash = prior_state_semantic_hash
        self.logical_evaluations = 2
        self.maximum_new_native_admissions = 2
        self.original = original


def _exact_mpfr_payload(value) -> list[int]:
    exact = gmpy2.mpq(value)
    return [int(exact.numerator), int(exact.denominator)]


def _exact_midpoint(interval: Interval) -> gmpy2.mpq:
    return (gmpy2.mpq(interval.lower) + gmpy2.mpq(interval.upper)) / 2


def _exact_radius(interval: Interval) -> gmpy2.mpq:
    return (gmpy2.mpq(interval.upper) - gmpy2.mpq(interval.lower)) / 2


def _interval_payload(interval: Interval | None) -> dict | None:
    if interval is None:
        return None
    return {
        "precision_bits": interval.precision_bits,
        "lower": _exact_mpfr_payload(interval.lower),
        "upper": _exact_mpfr_payload(interval.upper),
    }


def _interval_from_payload(payload: dict) -> Interval:
    precision = int(payload["precision_bits"])
    lower = Fraction(int(payload["lower"][0]), int(payload["lower"][1]))
    upper = Fraction(int(payload["upper"][0]), int(payload["upper"][1]))
    return Interval.from_bounds(lower, upper, precision)


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
    payload = {
        "positive": _interval_payload(certificate.positive),
        "negative": _interval_payload(certificate.negative),
        "secant": _interval_payload(certificate.secant),
        "m2": _interval_payload(certificate.m2),
    }
    if certificate.accounting is not None:
        accounting = certificate.accounting
        payload["component_accounting"] = {
            "positive_midpoint": _exact_mpfr_payload(accounting.positive_midpoint),
            "negative_midpoint": _exact_mpfr_payload(accounting.negative_midpoint),
            "positive_radius": _exact_mpfr_payload(accounting.positive_radius),
            "negative_radius": _exact_mpfr_payload(accounting.negative_radius),
            "positive_weight_rounding": _exact_mpfr_payload(
                accounting.positive_weight_rounding
            ),
            "negative_weight_rounding": _exact_mpfr_payload(
                accounting.negative_weight_rounding
            ),
            "positive_summation_rounding": _exact_mpfr_payload(
                accounting.positive_summation_rounding
            ),
            "negative_summation_rounding": _exact_mpfr_payload(
                accounting.negative_summation_rounding
            ),
        }
    return payload


def _endpoint_error_payload(
    certificate: EndpointErrorCertificate | None,
) -> dict | None:
    if certificate is None:
        return None
    payload = {
        "positive_residual": _interval_payload(certificate.positive_residual),
        "negative_residual": _interval_payload(certificate.negative_residual),
        "epsilon_psi": _exact_mpfr_payload(certificate.epsilon_psi),
    }
    if certificate.accounting is not None:
        accounting = certificate.accounting
        payload["component_accounting"] = {
            field: _exact_mpfr_payload(getattr(accounting, field))
            for field in accounting.__dataclass_fields__
        }
        payload["runtime_parity"] = {
            "available": False,
            "discrepancy": None,
            "diagnostic_only": True,
            "included_in_epsilon_psi": False,
        }
    return payload


def joint_witness_certificate_payload(
    certificate: JointWitnessCertificate, *, row_spec: JointWitnessRowSpec,
    plan: CertificatePlan,
) -> dict:
    """Serialize one component-accounted certificate for synthetic rows only.

    Backend/commit provenance and process-tree resources belong to the enclosing
    immutable run artifact, so this object alone is not a final launch artifact.
    """
    if not isinstance(row_spec, JointWitnessRowSpec) or not isinstance(plan, CertificatePlan):
        raise TypeError("certificate serialization requires validated row and plan schemas")
    if row_spec.split != "synthetic" or plan.execution_authorized:
        raise RuntimeError("REAL_CERTIFICATE_SERIALIZATION_UNAUTHORIZED")
    if plan.row_hash != row_spec.row_hash or certificate.row_hash != row_spec.row_hash:
        raise ValueError("certificate serialization identity mismatch")
    radii = []
    for radius in certificate.radii:
        if radius.endpoint_error.accounting is None or radius.audit_endpoint_error.accounting is None:
            raise RuntimeError("CERTIFICATE_COMPONENT_ACCOUNTING_MISSING")
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
        "schema_version": "green-v400-joint-witness-certificate-v2",
        "serialization_scope": "component_accounted_synthetic_certificate_only",
        "binding_component_accounting_complete": True,
        "binding_final_artifact_complete": False,
        "final_artifact_missing_scopes": [
            "backend_environment_commit_provenance",
            "process_tree_resource_record",
        ],
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


def _certify_cell_pair(graph, cells: tuple[DyadicCell, DyadicCell],
                       precision_bits: int) -> tuple[CellCertificate, CellCertificate]:
    evaluate_pair = getattr(graph, "evaluate_interval_pair", None)
    if not callable(evaluate_pair):
        return tuple(certify_cell(graph, cell, precision_bits) for cell in cells)
    domains = tuple(cell.interval(precision_bits) for cell in cells)
    jets = evaluate_pair(domains)
    if len(jets) != 2 or any(jet.precision_bits != precision_bits for jet in jets):
        raise RuntimeError("certificate pair evaluator precision/count mismatch")
    return tuple(
        CellCertificate(cell, jet.value, jet.first, jet.second)
        for cell, jet in zip(cells, jets)
    )


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


def _exact_scaled_bounds(interval: Interval, weight: Fraction) -> tuple[gmpy2.mpq, gmpy2.mpq]:
    rational_weight = gmpy2.mpq(weight.numerator, weight.denominator)
    products = (
        gmpy2.mpq(interval.lower) * rational_weight,
        gmpy2.mpq(interval.upper) * rational_weight,
    )
    return min(products), max(products)


def _shadow_radius(bounds: list[tuple[gmpy2.mpq, gmpy2.mpq]]) -> gmpy2.mpq:
    lower = sum((item[0] for item in bounds), gmpy2.mpq(0))
    upper = sum((item[1] for item in bounds), gmpy2.mpq(0))
    return (upper - lower) / 2


def _shadow_midpoint(bounds: list[tuple[gmpy2.mpq, gmpy2.mpq]]) -> gmpy2.mpq:
    lower = sum((item[0] for item in bounds), gmpy2.mpq(0))
    upper = sum((item[1] for item in bounds), gmpy2.mpq(0))
    return (lower + upper) / 2


def integrate_signed_curvature(cell_certificates: list[CellCertificate],
                               h: Fraction) -> CurvatureCertificate:
    if not cell_certificates:
        raise ValueError("curvature integration needs cells")
    precision = cell_certificates[0].second.precision_bits
    positive = Interval.point(0, precision)
    negative = Interval.point(0, precision)
    exact_positive: list[tuple[gmpy2.mpq, gmpy2.mpq]] = []
    exact_negative: list[tuple[gmpy2.mpq, gmpy2.mpq]] = []
    product_positive: list[tuple[gmpy2.mpq, gmpy2.mpq]] = []
    product_negative: list[tuple[gmpy2.mpq, gmpy2.mpq]] = []
    for certificate in cell_certificates:
        if certificate.cell.lower < -h or certificate.cell.upper > h:
            raise ValueError("curvature cell outside witness radius")
        exact_weight = _weight(certificate.cell, h)
        weight = Interval.point(gmpy2.mpq(exact_weight.numerator,
                                          exact_weight.denominator), precision)
        contribution = weight * certificate.second
        exact_bounds = _exact_scaled_bounds(certificate.second, exact_weight)
        product_bounds = (gmpy2.mpq(contribution.lower), gmpy2.mpq(contribution.upper))
        if certificate.cell.lower >= 0:
            positive = positive + contribution
            exact_positive.append(exact_bounds)
            product_positive.append(product_bounds)
        elif certificate.cell.upper <= 0:
            negative = negative + contribution
            exact_negative.append(exact_bounds)
            product_negative.append(product_bounds)
        else:
            raise ValueError("partition must split at zero")

    positive_weight_rounding = (
        _shadow_radius(product_positive) - _shadow_radius(exact_positive)
    )
    negative_weight_rounding = (
        _shadow_radius(product_negative) - _shadow_radius(exact_negative)
    )
    positive_summation_rounding = _exact_radius(positive) - _shadow_radius(product_positive)
    negative_summation_rounding = _exact_radius(negative) - _shadow_radius(product_negative)
    contributions = (
        positive_weight_rounding, negative_weight_rounding,
        positive_summation_rounding, negative_summation_rounding,
    )
    if any(value < 0 for value in contributions):
        raise RuntimeError("CERTIFICATE_COMPONENT_ACCOUNTING_INVALID")
    accounting = CurvatureComponentAccounting(
        _shadow_midpoint(product_positive),
        _shadow_midpoint(product_negative),
        _exact_radius(positive),
        _exact_radius(negative),
        positive_weight_rounding,
        negative_weight_rounding,
        positive_summation_rounding,
        negative_summation_rounding,
    )
    return CurvatureCertificate(
        positive, negative, positive - negative,
        compute_m2(cell_certificates), accounting,
    )


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
    if curvature.accounting is None:
        raise RuntimeError("CERTIFICATE_COMPONENT_ACCOUNTING_MISSING")
    positive_eval_radius = _exact_radius(endpoint.positive)
    negative_eval_radius = _exact_radius(endpoint.negative)
    center_eval_radius = _exact_radius(endpoint.center)
    rational_h = gmpy2.mpq(endpoint.h.numerator, endpoint.h.denominator)
    slope_eval_contribution = abs(rational_h) * _exact_radius(endpoint.slope)
    direct_positive_radius = _exact_radius(direct_positive)
    direct_negative_radius = _exact_radius(direct_negative)
    positive_arithmetic = direct_positive_radius - (
        positive_eval_radius + center_eval_radius + slope_eval_contribution
    )
    negative_arithmetic = direct_negative_radius - (
        negative_eval_radius + center_eval_radius + slope_eval_contribution
    )
    if positive_arithmetic < 0 or negative_arithmetic < 0:
        raise RuntimeError("CERTIFICATE_COMPONENT_ACCOUNTING_INVALID")
    curve = curvature.accounting
    accounting = EndpointErrorComponentAccounting(
        positive_eval_radius,
        negative_eval_radius,
        center_eval_radius,
        slope_eval_contribution,
        _exact_midpoint(endpoint.positive) - _exact_midpoint(endpoint.center)
        - rational_h * _exact_midpoint(endpoint.slope),
        _exact_midpoint(endpoint.negative) - _exact_midpoint(endpoint.center)
        + rational_h * _exact_midpoint(endpoint.slope),
        direct_positive_radius,
        direct_negative_radius,
        positive_arithmetic,
        negative_arithmetic,
        curve.positive_midpoint,
        curve.negative_midpoint,
        curve.positive_radius,
        curve.negative_radius,
        curve.positive_weight_rounding,
        curve.negative_weight_rounding,
        curve.positive_summation_rounding,
        curve.negative_summation_rounding,
        gmpy2.mpq(0),
        gmpy2.mpq(0),
        gmpy2.mpq(0),
    )
    return EndpointErrorCertificate(positive, negative, epsilon, accounting)


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
        * (gmpy2.mpq(certificate.second.upper)
           - gmpy2.mpq(certificate.second.lower))
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


def _rational_payload(value) -> tuple[int, int]:
    exact = gmpy2.mpq(value)
    return int(exact.numerator), int(exact.denominator)


def _jet_payload(certificate: CellCertificate) -> dict:
    return {
        "value": _interval_payload(certificate.value),
        "first": _interval_payload(certificate.first),
        "second": _interval_payload(certificate.second),
    }


def _cell_from_anytime_state(leaf: AnytimeCellState) -> CellCertificate:
    cell = DyadicCell(
        Fraction(*leaf.lower), Fraction(*leaf.upper), leaf.depth,
    )
    return CellCertificate(
        cell,
        _interval_from_payload(leaf.jet_payload["value"]),
        _interval_from_payload(leaf.jet_payload["first"]),
        _interval_from_payload(leaf.jet_payload["second"]),
    )


def _cache_entry_semantic_hash(evaluator_identity_sha256: str,
                               certificate: CellCertificate) -> str:
    payload = {
        "schema_version": "green-v400-anytime-cache-entry-identity-v1",
        "evaluator_identity_sha256": evaluator_identity_sha256,
        "precision_bits": certificate.second.precision_bits,
        "lower": list(_rational_payload(certificate.cell.lower)),
        "upper": list(_rational_payload(certificate.cell.upper)),
        "jet_semantic_hash": sha256_canonical(_jet_payload(certificate)),
    }
    return sha256_canonical(payload)


def _anytime_leaf(certificate: CellCertificate, h: Fraction,
                   evaluator_identity_sha256: str, result_source: str) -> AnytimeCellState:
    jet_payload = _jet_payload(certificate)
    return AnytimeCellState(
        "green-v400-anytime-cell-state-v1",
        evaluator_identity_sha256,
        certificate.second.precision_bits,
        _rational_payload(certificate.cell.lower),
        _rational_payload(certificate.cell.upper),
        certificate.cell.depth,
        _rational_payload(_cell_priority(certificate, h)),
        jet_payload,
        sha256_canonical(jet_payload),
        result_source,
        _cache_entry_semantic_hash(evaluator_identity_sha256, certificate),
    )


def _normalize_anytime_metadata(metadata: object, count: int) -> tuple[str, ...]:
    if metadata is None:
        return ("COMPUTED",) * count
    if not isinstance(metadata, (list, tuple)) or len(metadata) != count:
        raise RuntimeError("ANYTIME_EVALUATION_METADATA_INVALID")
    sources = []
    for item in metadata:
        if not isinstance(item, dict) or set(item) != {"result_source"}:
            raise RuntimeError("ANYTIME_EVALUATION_METADATA_INVALID")
        source = item["result_source"]
        if source not in {"COMPUTED", "EXACT_CACHE_HIT"}:
            raise RuntimeError("ANYTIME_EVALUATION_METADATA_INVALID")
        sources.append(source)
    return tuple(sources)


def _evaluate_anytime_domains(evaluator, domains: tuple[Interval, ...]):
    """Evaluate synthetic domains and return immutable accounting metadata.

    A synthetic fixture may expose ``evaluate_interval_pair_with_metadata`` or
    ``evaluate_interval_with_metadata``.  Existing evaluators require no change
    and are conservatively treated as admitted, completed native dispatches.
    """
    if len(domains) == 2:
        method = getattr(evaluator, "evaluate_interval_pair_with_metadata", None)
        if callable(method):
            result = method(domains)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("ANYTIME_PAIR_EVALUATION_INVALID")
            jets, metadata = result
            jets = tuple(jets)
            sources = _normalize_anytime_metadata(metadata, 2)
        else:
            method = getattr(evaluator, "evaluate_interval_pair", None)
            if callable(method):
                jets = tuple(method(domains))
            else:
                jets = tuple(evaluator.evaluate_interval(domain) for domain in domains)
            sources = ("COMPUTED", "COMPUTED")
    else:
        jets = []
        sources = []
        metadata_method = getattr(evaluator, "evaluate_interval_with_metadata", None)
        for domain in domains:
            if callable(metadata_method):
                result = metadata_method(domain)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise RuntimeError("ANYTIME_EVALUATION_INVALID")
                jet, metadata = result
                source = _normalize_anytime_metadata((metadata,), 1)[0]
            else:
                jet = evaluator.evaluate_interval(domain)
                source = "COMPUTED"
            jets.append(jet)
            sources.append(source)
        jets, sources = tuple(jets), tuple(sources)
    if len(jets) != len(domains) or any(
        jet.precision_bits != domain.precision_bits
        for jet, domain in zip(jets, domains)
    ):
        raise RuntimeError("ANYTIME_EVALUATOR_PRECISION_OR_COUNT_INVALID")
    cache_hits = sum(source == "EXACT_CACHE_HIT" for source in sources)
    accounting = {
        "logical_evaluations": len(domains),
        "admitted_native_dispatches": len(domains) - cache_hits,
        "completed_native_dispatches": len(domains) - cache_hits,
        "exact_cache_hits": cache_hits,
    }
    return jets, sources, accounting


def _add_anytime_accounting(*items: dict) -> dict:
    fields = (
        "logical_evaluations", "admitted_native_dispatches",
        "completed_native_dispatches", "exact_cache_hits",
    )
    return {field: sum(int(item[field]) for item in items) for field in fields}


def _endpoint_from_anytime_payload(payload: dict) -> EndpointCertificate:
    return EndpointCertificate(
        Fraction(*payload["h"]),
        _interval_from_payload(payload["negative"]),
        _interval_from_payload(payload["center"]),
        _interval_from_payload(payload["positive"]),
        _interval_from_payload(payload["slope"]),
    )


def _checked_anytime_intersection(left: Interval, right: Interval,
                                  quantity: str) -> Interval:
    try:
        return left.intersect(right)
    except EmptyIntersection as error:
        raise RuntimeError(
            f"CERTIFICATE_IMPLEMENTATION_INVALID:{quantity}_EMPTY_INTERSECTION"
        ) from error


def _direct_endpoint_residuals(endpoint: EndpointCertificate) -> tuple[Interval, Interval]:
    precision = endpoint.center.precision_bits
    h = Interval.point(
        gmpy2.mpq(endpoint.h.numerator, endpoint.h.denominator), precision,
    )
    return (
        endpoint.positive - endpoint.center - h * endpoint.slope,
        endpoint.negative - endpoint.center + h * endpoint.slope,
    )


def _witness_from_residuals(endpoint: EndpointCertificate,
                            curvature: CurvatureCertificate,
                            positive: Interval, negative: Interval) -> Interval:
    error = EndpointErrorCertificate(
        positive, negative, max(positive.magnitude(), negative.magnitude()), None,
    )
    return witness_interval(endpoint, curvature, error)


def _raw_curvature_accounting_payload(curvature: CurvatureCertificate) -> dict:
    payload = _curvature_payload(curvature)
    accounting = payload.get("component_accounting") if payload else None
    if accounting is None:
        raise RuntimeError("CERTIFICATE_COMPONENT_ACCOUNTING_MISSING")
    return accounting


def _build_anytime_state(*, plan: CertificatePlan, resource_lock_semantic_hash: str,
                         evaluator_identity_sha256: str, h: Fraction,
                         precision_bits: int, phase: str, checkpoint_index: int,
                         parent_state_semantic_hash: str, accounting: dict,
                         leaves: tuple[AnytimeCellState, ...],
                         endpoint: EndpointCertificate,
                         raw_curvature: CurvatureCertificate,
                         monotone_curvature: tuple[Interval, Interval],
                         monotone_residuals: tuple[Interval, Interval],
                         raw_witness: Interval, monotone_witness: Interval,
                         computation_status: str = "PROVISIONAL",
                         resource_reason: str | None = None,
                         ) -> MonotoneAnytimeCertificateState:
    return MonotoneAnytimeCertificateState(
        "green-v400-monotone-anytime-state-v1",
        "outcome_blind_synthetic_only",
        plan.row_hash,
        sha256_canonical(plan),
        resource_lock_semantic_hash,
        evaluator_identity_sha256,
        _rational_payload(h),
        precision_bits,
        phase,
        checkpoint_index,
        parent_state_semantic_hash,
        accounting["logical_evaluations"],
        accounting["admitted_native_dispatches"],
        accounting["completed_native_dispatches"],
        accounting["exact_cache_hits"],
        leaves,
        _endpoint_payload(endpoint),
        _interval_payload(raw_curvature.positive),
        _interval_payload(raw_curvature.negative),
        _raw_curvature_accounting_payload(raw_curvature),
        _interval_payload(monotone_curvature[0]),
        _interval_payload(monotone_curvature[1]),
        _interval_payload(monotone_residuals[0]),
        _interval_payload(monotone_residuals[1]),
        _interval_payload(raw_witness),
        _interval_payload(monotone_witness),
        computation_status,
        resource_reason,
        False,
    )


def initialize_monotone_anytime_state(
    evaluator, h: Fraction, precision_bits: int, plan: CertificatePlan, *,
    resource_lock_semantic_hash: str,
) -> MonotoneAnytimeCertificateState:
    """Create one outcome-blind synthetic checkpoint from the zero-split cells."""
    if not isinstance(plan, CertificatePlan):
        raise TypeError("anytime initialization requires CertificatePlan")
    if getattr(evaluator, "contains_scientific_outcome", None) is not False:
        raise RuntimeError("ANYTIME_EVALUATOR_OUTCOME_BOUNDARY_MISSING")
    if getattr(evaluator, "synthetic_only", None) is not True:
        raise RuntimeError("ANYTIME_REAL_EXECUTION_UNAUTHORIZED")
    if getattr(evaluator, "certificate_row_hash", None) != plan.row_hash:
        raise RuntimeError("ANYTIME_EVALUATOR_PLAN_IDENTITY_MISMATCH")
    evaluator_identity = getattr(evaluator, "evaluator_identity_sha256", None)
    if (not isinstance(evaluator_identity, str) or len(evaluator_identity) != 64
            or any(character not in "0123456789abcdef" for character in evaluator_identity)):
        raise RuntimeError("ANYTIME_EVALUATOR_IDENTITY_INVALID")
    if h <= 0 or h not in {radius.as_fraction() for radius in plan.radii}:
        raise ValueError("anytime radius is not frozen in the certificate plan")
    if precision_bits != plan.official_precision_bits:
        raise ValueError("anytime initialization is restricted to official precision")
    if (not isinstance(resource_lock_semantic_hash, str)
            or len(resource_lock_semantic_hash) != 64
            or any(character not in "0123456789abcdef"
                   for character in resource_lock_semantic_hash)):
        raise ValueError("anytime resource lock identity is invalid")

    initial_cells = (DyadicCell(-h, Fraction(0)), DyadicCell(Fraction(0), h))
    domains = tuple(cell.interval(precision_bits) for cell in initial_cells)
    jets, sources, cell_accounting = _evaluate_anytime_domains(evaluator, domains)
    certificates = tuple(
        CellCertificate(cell, jet.value, jet.first, jet.second)
        for cell, jet in zip(initial_cells, jets)
    )
    _validate_partition(list(certificates), h)
    endpoint_domains = tuple(
        _point_interval(value, precision_bits) for value in (-h, Fraction(0), h)
    )
    endpoint_jets, _, endpoint_accounting = _evaluate_anytime_domains(
        evaluator, endpoint_domains,
    )
    endpoint = EndpointCertificate(
        h, endpoint_jets[0].value, endpoint_jets[1].value,
        endpoint_jets[2].value, endpoint_jets[1].first,
    )
    raw = integrate_signed_curvature(list(certificates), h)
    direct_positive, direct_negative = _direct_endpoint_residuals(endpoint)
    residuals = (
        _checked_anytime_intersection(direct_positive, raw.positive, "POSITIVE_RESIDUAL"),
        _checked_anytime_intersection(direct_negative, raw.negative, "NEGATIVE_RESIDUAL"),
    )
    witness = _witness_from_residuals(endpoint, raw, *residuals)
    leaves = tuple(
        _anytime_leaf(certificate, h, evaluator_identity, source)
        for certificate, source in zip(certificates, sources)
    )
    return _build_anytime_state(
        plan=plan,
        resource_lock_semantic_hash=resource_lock_semantic_hash,
        evaluator_identity_sha256=evaluator_identity,
        h=h,
        precision_bits=precision_bits,
        phase="SYNTHETIC_OFFICIAL",
        checkpoint_index=0,
        parent_state_semantic_hash="0" * 64,
        accounting=_add_anytime_accounting(cell_accounting, endpoint_accounting),
        leaves=leaves,
        endpoint=endpoint,
        raw_curvature=raw,
        monotone_curvature=(raw.positive, raw.negative),
        monotone_residuals=residuals,
        raw_witness=witness,
        monotone_witness=witness,
    )


def _resource_anytime_state(state: MonotoneAnytimeCertificateState,
                             reason: str, *, accounting_delta: dict | None = None,
                             ) -> MonotoneAnytimeCertificateState:
    if reason not in RESOURCE_REASONS:
        raise ValueError("unknown frozen anytime resource reason")
    accounting_delta = accounting_delta or {
        "logical_evaluations": 0,
        "admitted_native_dispatches": 0,
        "completed_native_dispatches": 0,
        "exact_cache_hits": 0,
    }
    expected_fields = {
        "logical_evaluations", "admitted_native_dispatches",
        "completed_native_dispatches", "exact_cache_hits",
    }
    if (set(accounting_delta) != expected_fields
            or any(type(value) is not int or value < 0
                   for value in accounting_delta.values())
            or accounting_delta["logical_evaluations"] != (
                accounting_delta["admitted_native_dispatches"]
                + accounting_delta["exact_cache_hits"])
            or accounting_delta["completed_native_dispatches"]
            > accounting_delta["admitted_native_dispatches"]):
        raise ValueError("failed anytime admission accounting is invalid")
    # Preserve the raw accounting byte-for-byte; this transition performs no arithmetic.
    return MonotoneAnytimeCertificateState.from_dict(state.to_dict() | {
        "checkpoint_index": state.checkpoint_index + 1,
        "parent_state_semantic_hash": state.semantic_hash(),
        "logical_evaluations": (
            state.logical_evaluations + accounting_delta["logical_evaluations"]
        ),
        "admitted_native_dispatches": (
            state.admitted_native_dispatches
            + accounting_delta["admitted_native_dispatches"]
        ),
        "completed_native_dispatches": (
            state.completed_native_dispatches
            + accounting_delta["completed_native_dispatches"]
        ),
        "exact_cache_hits": state.exact_cache_hits + accounting_delta["exact_cache_hits"],
        "computation_status": "RESOURCE_INCONCLUSIVE",
        "resource_reason": reason,
    })


def transition_anytime_resource_failure_after_admission(
    state: MonotoneAnytimeCertificateState, *,
    logical_evaluations: int, admitted_native_dispatches: int,
    completed_native_dispatches: int, exact_cache_hits: int,
    resource_reason: str,
) -> MonotoneAnytimeCertificateState:
    """Record a supervisor-classified failure without refunding admitted work.

    Process death cannot update an in-process object.  The external supervisor
    must apply this pure transition to the last hash-verified checkpoint using
    its admission ledger before publishing ``RESOURCE_INCONCLUSIVE``.
    """
    if not isinstance(state, MonotoneAnytimeCertificateState):
        raise TypeError("failed admission transition requires validated anytime state")
    state.assert_integrity()
    if state.computation_status != "PROVISIONAL":
        raise RuntimeError("ANYTIME_STATE_IS_TERMINAL")
    return _resource_anytime_state(
        state, resource_reason, accounting_delta={
            "logical_evaluations": logical_evaluations,
            "admitted_native_dispatches": admitted_native_dispatches,
            "completed_native_dispatches": completed_native_dispatches,
            "exact_cache_hits": exact_cache_hits,
        },
    )


def advance_monotone_anytime_state(
    state: MonotoneAnytimeCertificateState, evaluator, plan: CertificatePlan,
) -> MonotoneAnytimeCertificateState:
    """Atomically replace exactly one frozen-priority parent by two children."""
    if not isinstance(state, MonotoneAnytimeCertificateState):
        raise TypeError("anytime refinement requires validated state")
    state.assert_integrity()
    if state.computation_status != "PROVISIONAL":
        raise RuntimeError("ANYTIME_STATE_IS_TERMINAL")
    if (not isinstance(plan, CertificatePlan)
            or sha256_canonical(plan) != state.certificate_plan_semantic_hash
            or plan.row_hash != state.row_hash):
        raise RuntimeError("ANYTIME_STATE_PLAN_IDENTITY_MISMATCH")
    if (getattr(evaluator, "contains_scientific_outcome", None) is not False
            or getattr(evaluator, "synthetic_only", None) is not True):
        raise RuntimeError("ANYTIME_REAL_EXECUTION_UNAUTHORIZED")
    if (getattr(evaluator, "certificate_row_hash", None) != state.row_hash
            or getattr(evaluator, "evaluator_identity_sha256", None)
            != state.evaluator_identity_sha256):
        raise RuntimeError("ANYTIME_EVALUATOR_IDENTITY_MISMATCH")

    h = Fraction(*state.radius)
    certificates = [_cell_from_anytime_state(leaf) for leaf in state.leaves]
    _validate_partition(certificates, h)
    absolute = _hex_fraction(plan.absolute_width_tolerance)
    relative = _hex_fraction(plan.relative_width_tolerance)
    unresolved = [
        certificate for certificate in certificates
        if not _cell_tolerance_met(certificate, absolute, relative)
    ]
    if not unresolved:
        raise RuntimeError("ANYTIME_PARTITION_TOLERANCES_MET")
    parent = min(
        unresolved,
        key=lambda certificate: (
            -_cell_priority(certificate, h), certificate.cell.lower,
            certificate.cell.depth,
        ),
    )
    if parent.cell.depth >= plan.max_depth:
        return _resource_anytime_state(state, "MAX_DEPTH_REACHED")
    if len(certificates) + 1 > plan.max_cells:
        return _resource_anytime_state(
            state, "MAX_FINAL_LEAVES_PER_RADIUS_REACHED",
        )

    children = parent.cell.bisect()
    domains = tuple(cell.interval(state.precision_bits) for cell in children)
    # No state is mutated before both sibling evaluations and validations succeed.
    try:
        jets, sources, step_accounting = _evaluate_anytime_domains(evaluator, domains)
    except Exception as error:
        raise AnytimeEvaluationFailure(state.semantic_hash(), error) from error
    child_certificates = tuple(
        CellCertificate(cell, jet.value, jet.first, jet.second)
        for cell, jet in zip(children, jets)
    )
    next_certificates = [
        certificate for certificate in certificates if certificate.cell != parent.cell
    ] + list(child_certificates)
    next_certificates.sort(key=lambda certificate: certificate.cell.lower)
    _validate_partition(next_certificates, h)
    raw = integrate_signed_curvature(next_certificates, h)

    previous_curvature = (
        _interval_from_payload(state.monotone_curvature_positive),
        _interval_from_payload(state.monotone_curvature_negative),
    )
    monotone_curvature = (
        _checked_anytime_intersection(previous_curvature[0], raw.positive,
                                      "POSITIVE_CURVATURE"),
        _checked_anytime_intersection(previous_curvature[1], raw.negative,
                                      "NEGATIVE_CURVATURE"),
    )
    endpoint = _endpoint_from_anytime_payload(state.endpoint_payload)
    direct = _direct_endpoint_residuals(endpoint)
    previous_residuals = (
        _interval_from_payload(state.monotone_residual_positive),
        _interval_from_payload(state.monotone_residual_negative),
    )
    raw_residuals = (
        _checked_anytime_intersection(direct[0], raw.positive,
                                      "RAW_POSITIVE_RESIDUAL"),
        _checked_anytime_intersection(direct[1], raw.negative,
                                      "RAW_NEGATIVE_RESIDUAL"),
    )
    monotone_residuals = tuple(
        _checked_anytime_intersection(
            previous,
            _checked_anytime_intersection(bound, curvature, name),
            f"MONOTONE_{name}",
        )
        for previous, bound, curvature, name in (
            (previous_residuals[0], direct[0], monotone_curvature[0],
             "POSITIVE_RESIDUAL"),
            (previous_residuals[1], direct[1], monotone_curvature[1],
             "NEGATIVE_RESIDUAL"),
        )
    )
    raw_witness = _witness_from_residuals(endpoint, raw, *raw_residuals)
    candidate_witness = _witness_from_residuals(
        endpoint, raw, *monotone_residuals,
    )
    monotone_witness = _checked_anytime_intersection(
        _interval_from_payload(state.monotone_witness),
        candidate_witness,
        "WITNESS",
    )

    old_leaf_by_cell = {
        (Fraction(*leaf.lower), Fraction(*leaf.upper), leaf.depth): leaf
        for leaf in state.leaves
    }
    new_leaves = []
    child_source_by_cell = {
        child.cell: source for child, source in zip(child_certificates, sources)
    }
    for certificate in next_certificates:
        old = old_leaf_by_cell.get((
            certificate.cell.lower, certificate.cell.upper, certificate.cell.depth,
        ))
        new_leaves.append(old if old is not None else _anytime_leaf(
            certificate, h, state.evaluator_identity_sha256,
            child_source_by_cell[certificate.cell],
        ))
    total_accounting = _add_anytime_accounting({
        "logical_evaluations": state.logical_evaluations,
        "admitted_native_dispatches": state.admitted_native_dispatches,
        "completed_native_dispatches": state.completed_native_dispatches,
        "exact_cache_hits": state.exact_cache_hits,
    }, step_accounting)
    return _build_anytime_state(
        plan=plan,
        resource_lock_semantic_hash=state.resource_lock_semantic_hash,
        evaluator_identity_sha256=state.evaluator_identity_sha256,
        h=h,
        precision_bits=state.precision_bits,
        phase=state.phase,
        checkpoint_index=state.checkpoint_index + 1,
        parent_state_semantic_hash=state.semantic_hash(),
        accounting=total_accounting,
        leaves=tuple(new_leaves),
        endpoint=endpoint,
        raw_curvature=raw,
        monotone_curvature=monotone_curvature,
        monotone_residuals=monotone_residuals,
        raw_witness=raw_witness,
        monotone_witness=monotone_witness,
    )


def monotone_anytime_state_payload(
    state: MonotoneAnytimeCertificateState,
) -> dict:
    payload = state.to_dict()
    return {
        "state": payload,
        "state_semantic_hash": state.semantic_hash(),
    }


def serialize_monotone_anytime_state(
    state: MonotoneAnytimeCertificateState,
) -> str:
    return canonical_json(monotone_anytime_state_payload(state))


def restore_monotone_anytime_state(payload) -> MonotoneAnytimeCertificateState:
    if isinstance(payload, (str, bytes, bytearray)):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or set(payload) != {
        "state", "state_semantic_hash",
    }:
        raise ValueError("anytime checkpoint envelope field mismatch")
    state = MonotoneAnytimeCertificateState.from_dict(payload["state"])
    if payload["state_semantic_hash"] != state.semantic_hash():
        raise ValueError("anytime checkpoint semantic hash mismatch")
    return state


def _lift_interval_precision(interval: Interval, precision_bits: int) -> Interval:
    """Import exact MPFR endpoints into another precision without float transit."""
    return Interval.from_bounds(
        gmpy2.mpq(interval.lower), gmpy2.mpq(interval.upper), precision_bits,
    )


def _validate_anytime_frozen_partition_audit_inputs(
    state: MonotoneAnytimeCertificateState, evaluator, plan: CertificatePlan,
) -> None:
    if not isinstance(state, MonotoneAnytimeCertificateState):
        raise TypeError("anytime audit replay requires a validated official state")
    if not isinstance(plan, CertificatePlan):
        raise TypeError("anytime audit replay requires CertificatePlan")
    state.assert_integrity()
    if (state.phase != "SYNTHETIC_OFFICIAL"
            or state.computation_status != "PROVISIONAL"
            or state.precision_bits != plan.official_precision_bits):
        raise RuntimeError("ANYTIME_AUDIT_REQUIRES_COMPLETE_PROVISIONAL_OFFICIAL_STATE")
    if (state.row_hash != plan.row_hash
            or state.certificate_plan_semantic_hash != sha256_canonical(plan)):
        raise RuntimeError("ANYTIME_AUDIT_PLAN_IDENTITY_MISMATCH")
    if plan.audit_precision_bits <= plan.official_precision_bits:
        raise RuntimeError("ANYTIME_AUDIT_PRECISION_LADDER_INVALID")
    if (getattr(evaluator, "contains_scientific_outcome", None) is not False
            or getattr(evaluator, "synthetic_only", None) is not True):
        raise RuntimeError("ANYTIME_AUDIT_REAL_EXECUTION_UNAUTHORIZED")
    if (getattr(evaluator, "certificate_row_hash", None) != state.row_hash
            or getattr(evaluator, "evaluator_identity_sha256", None)
            != state.evaluator_identity_sha256):
        raise RuntimeError("ANYTIME_AUDIT_EVALUATOR_IDENTITY_MISMATCH")


def _legacy_final_partition_intersection_audit(
    state: MonotoneAnytimeCertificateState, evaluator, plan: CertificatePlan,
) -> dict:
    """Replay one frozen official partition at audit precision.

    Exactly the final official leaves and the three endpoint domains are
    evaluated.  No audit-precision adaptive queue exists on this path.  Raw
    audit quantities must nest in their corresponding raw official quantities;
    tightened audit quantities are sound nonempty intersections with the
    official monotone checkpoint and must nest there as well.
    """
    _validate_anytime_frozen_partition_audit_inputs(state, evaluator, plan)
    h = Fraction(*state.radius)
    audit_precision = plan.audit_precision_bits
    official_cells = tuple(_cell_from_anytime_state(leaf) for leaf in state.leaves)
    domains = tuple(cell.cell.interval(audit_precision) for cell in official_cells)
    audit_jets, sources, cell_accounting = _evaluate_anytime_domains(
        evaluator, domains,
    )
    audit_cells = tuple(
        CellCertificate(cell.cell, jet.value, jet.first, jet.second)
        for cell, jet in zip(official_cells, audit_jets)
    )
    cell_rows = []
    for official, audit, source in zip(official_cells, audit_cells, sources):
        components = {}
        for name in ("value", "first", "second"):
            low = getattr(official, name)
            high = getattr(audit, name)
            if not _interval_nested(high, low):
                raise RuntimeError(
                    f"ANYTIME_AUDIT_CELL_{name.upper()}_NESTING_INVALID"
                )
            components[name] = {
                "official": _interval_payload(low),
                "audit": _interval_payload(high),
                "audit_inside_official": True,
            }
        cell_rows.append({
            "lower": _rational_payload(official.cell.lower),
            "upper": _rational_payload(official.cell.upper),
            "depth": official.cell.depth,
            "result_source": source,
            "components": components,
        })

    endpoint_domains = tuple(
        _point_interval(value, audit_precision)
        for value in (-h, Fraction(0), h)
    )
    endpoint_jets, _, endpoint_accounting = _evaluate_anytime_domains(
        evaluator, endpoint_domains,
    )
    audit_endpoint = EndpointCertificate(
        h, endpoint_jets[0].value, endpoint_jets[1].value,
        endpoint_jets[2].value, endpoint_jets[1].first,
    )
    official_endpoint = _endpoint_from_anytime_payload(state.endpoint_payload)
    endpoint_rows = {}
    for name in ("negative", "center", "positive", "slope"):
        low = getattr(official_endpoint, name)
        high = getattr(audit_endpoint, name)
        if not _interval_nested(high, low):
            raise RuntimeError(
                f"ANYTIME_AUDIT_ENDPOINT_{name.upper()}_NESTING_INVALID"
            )
        endpoint_rows[name] = {
            "official": _interval_payload(low),
            "audit": _interval_payload(high),
            "audit_inside_official": True,
        }

    official_raw_curvature = (
        _interval_from_payload(state.raw_curvature_positive),
        _interval_from_payload(state.raw_curvature_negative),
    )
    audit_raw_curvature_certificate = integrate_signed_curvature(
        list(audit_cells), h,
    )
    audit_raw_curvature = (
        audit_raw_curvature_certificate.positive,
        audit_raw_curvature_certificate.negative,
    )
    for name, high, low in zip(
        ("POSITIVE", "NEGATIVE"), audit_raw_curvature,
        official_raw_curvature,
    ):
        if not _interval_nested(high, low):
            raise RuntimeError(f"ANYTIME_AUDIT_RAW_{name}_CURVATURE_NESTING_INVALID")

    official_direct = _direct_endpoint_residuals(official_endpoint)
    official_raw_residuals = tuple(
        _checked_anytime_intersection(direct, curvature, f"OFFICIAL_RAW_{name}")
        for direct, curvature, name in zip(
            official_direct, official_raw_curvature, ("POSITIVE", "NEGATIVE"),
        )
    )
    audit_direct = _direct_endpoint_residuals(audit_endpoint)
    audit_raw_residuals = tuple(
        _checked_anytime_intersection(direct, curvature, f"AUDIT_RAW_{name}")
        for direct, curvature, name in zip(
            audit_direct, audit_raw_curvature, ("POSITIVE", "NEGATIVE"),
        )
    )
    for name, high, low in zip(
        ("POSITIVE", "NEGATIVE"), audit_raw_residuals,
        official_raw_residuals,
    ):
        if not _interval_nested(high, low):
            raise RuntimeError(f"ANYTIME_AUDIT_RAW_{name}_RESIDUAL_NESTING_INVALID")

    official_monotone_curvature = (
        _interval_from_payload(state.monotone_curvature_positive),
        _interval_from_payload(state.monotone_curvature_negative),
    )
    audit_monotone_curvature = tuple(
        _checked_anytime_intersection(
            raw, _lift_interval_precision(official, audit_precision),
            f"AUDIT_MONOTONE_{name}_CURVATURE",
        )
        for raw, official, name in zip(
            audit_raw_curvature, official_monotone_curvature,
            ("POSITIVE", "NEGATIVE"),
        )
    )
    official_monotone_residuals = (
        _interval_from_payload(state.monotone_residual_positive),
        _interval_from_payload(state.monotone_residual_negative),
    )
    audit_monotone_residuals = tuple(
        _checked_anytime_intersection(
            _checked_anytime_intersection(
                direct, curvature, f"AUDIT_TIGHT_{name}_RESIDUAL"
            ),
            _lift_interval_precision(official, audit_precision),
            f"AUDIT_MONOTONE_{name}_RESIDUAL",
        )
        for direct, curvature, official, name in zip(
            audit_direct, audit_monotone_curvature,
            official_monotone_residuals, ("POSITIVE", "NEGATIVE"),
        )
    )
    official_raw_witness = _interval_from_payload(state.raw_witness)
    audit_raw_witness = _witness_from_residuals(
        audit_endpoint, audit_raw_curvature_certificate, *audit_raw_residuals,
    )
    if not _interval_nested(audit_raw_witness, official_raw_witness):
        raise RuntimeError("ANYTIME_AUDIT_RAW_WITNESS_NESTING_INVALID")
    official_monotone_witness = _interval_from_payload(state.monotone_witness)
    audit_monotone_witness = _checked_anytime_intersection(
        _witness_from_residuals(
            audit_endpoint, audit_raw_curvature_certificate,
            *audit_monotone_residuals,
        ),
        _lift_interval_precision(official_monotone_witness, audit_precision),
        "AUDIT_MONOTONE_WITNESS",
    )
    if not _interval_nested(audit_monotone_witness, official_monotone_witness):
        raise RuntimeError("ANYTIME_AUDIT_MONOTONE_WITNESS_NESTING_INVALID")

    accounting = _add_anytime_accounting(cell_accounting, endpoint_accounting)
    payload = {
        "schema_version": "green-v400-anytime-frozen-partition-audit-v1",
        "execution_scope": "outcome_blind_synthetic_only",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "official_state_semantic_hash": state.semantic_hash(),
        "certificate_plan_semantic_hash": sha256_canonical(plan),
        "evaluator_identity_sha256": state.evaluator_identity_sha256,
        "radius": _rational_payload(h),
        "official_precision_bits": plan.official_precision_bits,
        "audit_precision_bits": audit_precision,
        "same_frozen_partition": True,
        "independent_audit_adaptive_queue": False,
        "cells": cell_rows,
        "endpoints": endpoint_rows,
        "raw_curvature": {
            name: {
                "official": _interval_payload(low),
                "audit": _interval_payload(high),
                "audit_inside_official": True,
            }
            for name, low, high in zip(
                ("positive", "negative"), official_raw_curvature,
                audit_raw_curvature,
            )
        },
        "monotone_curvature": {
            name: {
                "official": _interval_payload(low),
                "audit": _interval_payload(high),
                "audit_inside_official": True,
            }
            for name, low, high in zip(
                ("positive", "negative"), official_monotone_curvature,
                audit_monotone_curvature,
            )
        },
        "raw_residual": {
            name: {
                "official": _interval_payload(low),
                "audit": _interval_payload(high),
                "audit_inside_official": True,
            }
            for name, low, high in zip(
                ("positive", "negative"), official_raw_residuals,
                audit_raw_residuals,
            )
        },
        "monotone_residual": {
            name: {
                "official": _interval_payload(low),
                "audit": _interval_payload(high),
                "audit_inside_official": True,
            }
            for name, low, high in zip(
                ("positive", "negative"), official_monotone_residuals,
                audit_monotone_residuals,
            )
        },
        "raw_witness": {
            "official": _interval_payload(official_raw_witness),
            "audit": _interval_payload(audit_raw_witness),
            "audit_inside_official": True,
        },
        "monotone_witness": {
            "official": _interval_payload(official_monotone_witness),
            "audit": _interval_payload(audit_monotone_witness),
            "audit_inside_official": True,
        },
        "accounting": accounting,
    }
    return payload | {"report_semantic_hash": sha256_canonical(payload)}


def _legacy_final_partitions_intersection_audit(
    states: Iterable[MonotoneAnytimeCertificateState], evaluator,
    plan: CertificatePlan,
) -> dict:
    """Audit all frozen radii phase-major, then verify prefix intersections."""
    states = tuple(states)
    expected_radii = tuple(radius.as_fraction() for radius in plan.radii)
    if (len(states) != len(expected_radii)
            or tuple(Fraction(*state.radius) for state in states) != expected_radii):
        raise RuntimeError("ANYTIME_AUDIT_FROZEN_RADIUS_ORDER_INVALID")
    # Validate every official state before admitting the first 512-bit dispatch.
    for state in states:
        _validate_anytime_frozen_partition_audit_inputs(state, evaluator, plan)
    radius_reports = tuple(
        _legacy_final_partition_intersection_audit(state, evaluator, plan)
        for state in states
    )
    official_intersection = None
    audit_intersection = None
    prefix_rows = []
    for state, report in zip(states, radius_reports):
        official = _interval_from_payload(state.monotone_witness)
        audit = _interval_from_payload(report["monotone_witness"]["audit"])
        official_intersection = (
            official if official_intersection is None
            else _checked_anytime_intersection(
                official_intersection, official, "OFFICIAL_CROSS_RADIUS_WITNESS"
            )
        )
        audit_intersection = (
            audit if audit_intersection is None
            else _checked_anytime_intersection(
                audit_intersection, audit, "AUDIT_CROSS_RADIUS_WITNESS"
            )
        )
        if not _interval_nested(audit_intersection, official_intersection):
            raise RuntimeError("ANYTIME_AUDIT_CROSS_RADIUS_NESTING_INVALID")
        prefix_rows.append({
            "radius": list(state.radius),
            "official_intersection": _interval_payload(official_intersection),
            "audit_intersection": _interval_payload(audit_intersection),
            "audit_inside_official": True,
        })
    accounting = _add_anytime_accounting(*(
        report["accounting"] for report in radius_reports
    ))
    payload = {
        "schema_version": "green-v400-anytime-frozen-partitions-audit-v1",
        "execution_scope": "outcome_blind_synthetic_only",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "phase_major_all_official_before_audit": True,
        "radius_reports": list(radius_reports),
        "cross_radius_prefix_intersections": prefix_rows,
        "accounting": accounting,
    }
    return payload | {"report_semantic_hash": sha256_canonical(payload)}


def _validated_anytime_checkpoint_history(
    history: Iterable[MonotoneAnytimeCertificateState], evaluator,
    plan: CertificatePlan,
) -> tuple[MonotoneAnytimeCertificateState, ...]:
    history = tuple(history)
    if not history:
        raise ValueError("anytime audit checkpoint history is empty")
    for index, state in enumerate(history):
        _validate_anytime_frozen_partition_audit_inputs(state, evaluator, plan)
        if state.checkpoint_index != index:
            raise RuntimeError("ANYTIME_AUDIT_CHECKPOINT_INDEX_INVALID")
        if index == 0:
            if state.parent_state_semantic_hash != "0" * 64 or len(state.leaves) != 2:
                raise RuntimeError("ANYTIME_AUDIT_INITIAL_CHECKPOINT_INVALID")
            continue
        previous = history[index - 1]
        if (state.parent_state_semantic_hash != previous.semantic_hash()
                or len(state.leaves) != len(previous.leaves) + 1
                or state.resource_lock_semantic_hash
                != previous.resource_lock_semantic_hash):
            raise RuntimeError("ANYTIME_AUDIT_CHECKPOINT_CHAIN_INVALID")
        previous_cells = {
            (Fraction(*leaf.lower), Fraction(*leaf.upper), leaf.depth)
            for leaf in previous.leaves
        }
        current_cells = {
            (Fraction(*leaf.lower), Fraction(*leaf.upper), leaf.depth)
            for leaf in state.leaves
        }
        removed = previous_cells - current_cells
        added = current_cells - previous_cells
        if len(removed) != 1 or len(added) != 2:
            raise RuntimeError("ANYTIME_AUDIT_SPLIT_HISTORY_INVALID")
        lower, upper, depth = next(iter(removed))
        expected = {
            (child.lower, child.upper, child.depth)
            for child in DyadicCell(lower, upper, depth).bisect()
        }
        if added != expected:
            raise RuntimeError("ANYTIME_AUDIT_NON_DYADIC_HISTORY_INVALID")
    return history


def _require_audit_nested(high: Interval, low: Interval, quantity: str) -> None:
    if not _interval_nested(high, low):
        raise RuntimeError(f"ANYTIME_AUDIT_{quantity}_NESTING_INVALID")


def audit_monotone_anytime_checkpoint_history(
    history: Iterable[MonotoneAnytimeCertificateState], evaluator,
    plan: CertificatePlan,
) -> dict:
    """Independently replay one complete official split history at 512 bits.

    Every unique cell admitted by the 384-bit history is evaluated exactly once
    at audit precision.  The audit monotone recurrence uses only audit-precision
    raw quantities; official intervals are read solely by the subsequent
    fail-closed containment checks.
    """
    history = _validated_anytime_checkpoint_history(history, evaluator, plan)
    final_state = history[-1]
    h = Fraction(*final_state.radius)
    audit_precision = plan.audit_precision_bits
    endpoint_domains = tuple(
        _point_interval(value, audit_precision)
        for value in (-h, Fraction(0), h)
    )
    endpoint_jets, _, endpoint_accounting = _evaluate_anytime_domains(
        evaluator, endpoint_domains,
    )
    audit_endpoint = EndpointCertificate(
        h, endpoint_jets[0].value, endpoint_jets[1].value,
        endpoint_jets[2].value, endpoint_jets[1].first,
    )
    official_endpoint = _endpoint_from_anytime_payload(
        history[0].endpoint_payload
    )
    endpoint_rows = {}
    for name in ("negative", "center", "positive", "slope"):
        low = getattr(official_endpoint, name)
        high = getattr(audit_endpoint, name)
        _require_audit_nested(high, low, f"ENDPOINT_{name.upper()}")
        endpoint_rows[name] = {
            "official": _interval_payload(low),
            "audit": _interval_payload(high),
            "audit_inside_official": True,
        }

    audit_cells_by_key: dict[tuple[Fraction, Fraction, int], CellCertificate] = {}
    cell_sources: dict[tuple[Fraction, Fraction, int], str] = {}
    cell_accounting_rows = []
    prior_keys: set[tuple[Fraction, Fraction, int]] = set()
    audit_monotone_curvature = None
    audit_monotone_residuals = None
    audit_monotone_witness = None
    checkpoint_reports = []
    for checkpoint, state in enumerate(history):
        official_cells = tuple(_cell_from_anytime_state(leaf) for leaf in state.leaves)
        current_keys = {
            (cell.cell.lower, cell.cell.upper, cell.cell.depth)
            for cell in official_cells
        }
        added_keys = current_keys - prior_keys
        if checkpoint == 0:
            expected_new = 2
        else:
            expected_new = 2
        if len(added_keys) != expected_new:
            raise RuntimeError("ANYTIME_AUDIT_UNIQUE_CELL_REPLAY_INVALID")
        added_cells = tuple(
            cell for cell in official_cells
            if (cell.cell.lower, cell.cell.upper, cell.cell.depth) in added_keys
        )
        jets, sources, accounting = _evaluate_anytime_domains(
            evaluator,
            tuple(cell.cell.interval(audit_precision) for cell in added_cells),
        )
        cell_accounting_rows.append(accounting)
        for official, jet, source in zip(added_cells, jets, sources):
            key = (official.cell.lower, official.cell.upper, official.cell.depth)
            audit_cells_by_key[key] = CellCertificate(
                official.cell, jet.value, jet.first, jet.second,
            )
            cell_sources[key] = source
        audit_cells = tuple(audit_cells_by_key[
            (official.cell.lower, official.cell.upper, official.cell.depth)
        ] for official in official_cells)
        cell_rows = []
        for official, audit in zip(official_cells, audit_cells):
            components = {}
            for name in ("value", "first", "second"):
                low, high = getattr(official, name), getattr(audit, name)
                _require_audit_nested(
                    high, low, f"CHECKPOINT_{checkpoint}_CELL_{name.upper()}",
                )
                components[name] = {
                    "official": _interval_payload(low),
                    "audit": _interval_payload(high),
                    "audit_inside_official": True,
                }
            key = (official.cell.lower, official.cell.upper, official.cell.depth)
            cell_rows.append({
                "lower": _rational_payload(official.cell.lower),
                "upper": _rational_payload(official.cell.upper),
                "depth": official.cell.depth,
                "result_source": cell_sources[key],
                "components": components,
            })

        audit_raw_certificate = integrate_signed_curvature(list(audit_cells), h)
        audit_raw_curvature = (
            audit_raw_certificate.positive, audit_raw_certificate.negative,
        )
        audit_direct = _direct_endpoint_residuals(audit_endpoint)
        audit_raw_residuals = tuple(
            _checked_anytime_intersection(
                direct, curvature, f"AUDIT_CHECKPOINT_{checkpoint}_RAW_{name}"
            )
            for direct, curvature, name in zip(
                audit_direct, audit_raw_curvature, ("POSITIVE", "NEGATIVE"),
            )
        )
        audit_raw_witness = _witness_from_residuals(
            audit_endpoint, audit_raw_certificate, *audit_raw_residuals,
        )
        if checkpoint == 0:
            audit_monotone_curvature = audit_raw_curvature
            audit_monotone_residuals = audit_raw_residuals
            audit_monotone_witness = audit_raw_witness
        else:
            assert audit_monotone_curvature is not None
            assert audit_monotone_residuals is not None
            assert audit_monotone_witness is not None
            audit_monotone_curvature = tuple(
                _checked_anytime_intersection(
                    previous, raw,
                    f"AUDIT_CHECKPOINT_{checkpoint}_MONOTONE_{name}_CURVATURE",
                )
                for previous, raw, name in zip(
                    audit_monotone_curvature, audit_raw_curvature,
                    ("POSITIVE", "NEGATIVE"),
                )
            )
            audit_monotone_residuals = tuple(
                _checked_anytime_intersection(
                    previous,
                    _checked_anytime_intersection(
                        direct, curvature,
                        f"AUDIT_CHECKPOINT_{checkpoint}_TIGHT_{name}_RESIDUAL",
                    ),
                    f"AUDIT_CHECKPOINT_{checkpoint}_MONOTONE_{name}_RESIDUAL",
                )
                for previous, direct, curvature, name in zip(
                    audit_monotone_residuals, audit_direct,
                    audit_monotone_curvature, ("POSITIVE", "NEGATIVE"),
                )
            )
            candidate_witness = _witness_from_residuals(
                audit_endpoint, audit_raw_certificate,
                *audit_monotone_residuals,
            )
            audit_monotone_witness = _checked_anytime_intersection(
                audit_monotone_witness, candidate_witness,
                f"AUDIT_CHECKPOINT_{checkpoint}_MONOTONE_WITNESS",
            )

        official_raw_curvature = (
            _interval_from_payload(state.raw_curvature_positive),
            _interval_from_payload(state.raw_curvature_negative),
        )
        official_monotone_curvature = (
            _interval_from_payload(state.monotone_curvature_positive),
            _interval_from_payload(state.monotone_curvature_negative),
        )
        official_direct = _direct_endpoint_residuals(
            _endpoint_from_anytime_payload(state.endpoint_payload)
        )
        official_raw_residuals = tuple(
            _checked_anytime_intersection(
                direct, curvature, f"OFFICIAL_CHECKPOINT_{checkpoint}_RAW_{name}"
            )
            for direct, curvature, name in zip(
                official_direct, official_raw_curvature,
                ("POSITIVE", "NEGATIVE"),
            )
        )
        official_monotone_residuals = (
            _interval_from_payload(state.monotone_residual_positive),
            _interval_from_payload(state.monotone_residual_negative),
        )
        official_raw_witness = _interval_from_payload(state.raw_witness)
        official_monotone_witness = _interval_from_payload(state.monotone_witness)
        for name, high, low in zip(
            ("POSITIVE", "NEGATIVE"), audit_raw_curvature,
            official_raw_curvature,
        ):
            _require_audit_nested(
                high, low, f"CHECKPOINT_{checkpoint}_RAW_{name}_CURVATURE",
            )
        for name, high, low in zip(
            ("POSITIVE", "NEGATIVE"), audit_monotone_curvature,
            official_monotone_curvature,
        ):
            _require_audit_nested(
                high, low, f"CHECKPOINT_{checkpoint}_MONOTONE_{name}_CURVATURE",
            )
        for name, high, low in zip(
            ("POSITIVE", "NEGATIVE"), audit_raw_residuals,
            official_raw_residuals,
        ):
            _require_audit_nested(
                high, low, f"CHECKPOINT_{checkpoint}_RAW_{name}_RESIDUAL",
            )
        for name, high, low in zip(
            ("POSITIVE", "NEGATIVE"), audit_monotone_residuals,
            official_monotone_residuals,
        ):
            _require_audit_nested(
                high, low, f"CHECKPOINT_{checkpoint}_MONOTONE_{name}_RESIDUAL",
            )
        _require_audit_nested(
            audit_raw_witness, official_raw_witness,
            f"CHECKPOINT_{checkpoint}_RAW_WITNESS",
        )
        _require_audit_nested(
            audit_monotone_witness, official_monotone_witness,
            f"CHECKPOINT_{checkpoint}_MONOTONE_WITNESS",
        )

        def paired_rows(names, official_values, audit_values):
            return {
                name: {
                    "official": _interval_payload(low),
                    "audit": _interval_payload(high),
                    "audit_inside_official": True,
                }
                for name, low, high in zip(names, official_values, audit_values)
            }

        checkpoint_reports.append({
            "checkpoint_index": checkpoint,
            "official_state_semantic_hash": state.semantic_hash(),
            "partition_semantic_hash": sha256_canonical([
                {"lower": list(leaf.lower), "upper": list(leaf.upper),
                 "depth": leaf.depth} for leaf in state.leaves
            ]),
            "cells": cell_rows,
            "raw_curvature": paired_rows(
                ("positive", "negative"), official_raw_curvature,
                audit_raw_curvature,
            ),
            "monotone_curvature": paired_rows(
                ("positive", "negative"), official_monotone_curvature,
                audit_monotone_curvature,
            ),
            "raw_residual": paired_rows(
                ("positive", "negative"), official_raw_residuals,
                audit_raw_residuals,
            ),
            "monotone_residual": paired_rows(
                ("positive", "negative"), official_monotone_residuals,
                audit_monotone_residuals,
            ),
            "raw_witness": {
                "official": _interval_payload(official_raw_witness),
                "audit": _interval_payload(audit_raw_witness),
                "audit_inside_official": True,
            },
            "monotone_witness": {
                "official": _interval_payload(official_monotone_witness),
                "audit": _interval_payload(audit_monotone_witness),
                "audit_inside_official": True,
            },
        })
        prior_keys = current_keys

    accounting = _add_anytime_accounting(
        *cell_accounting_rows, endpoint_accounting,
    )
    expected_cells = 2 * len(final_state.leaves) - 2
    if (accounting["logical_evaluations"] != expected_cells + 3
            or accounting["exact_cache_hits"] != 0):
        raise RuntimeError("ANYTIME_AUDIT_FULL_HISTORY_ACCOUNTING_INVALID")
    final = checkpoint_reports[-1]
    payload = {
        "schema_version": "green-v400-anytime-full-history-audit-v1",
        "execution_scope": "outcome_blind_synthetic_only",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "certificate_plan_semantic_hash": sha256_canonical(plan),
        "resource_lock_semantic_hash": final_state.resource_lock_semantic_hash,
        "evaluator_identity_sha256": final_state.evaluator_identity_sha256,
        "radius": _rational_payload(h),
        "official_precision_bits": plan.official_precision_bits,
        "audit_precision_bits": audit_precision,
        "complete_split_history_replayed": True,
        "audit_recurrence_uses_official_intervals": False,
        "unique_history_cell_count": expected_cells,
        "checkpoint_reports": checkpoint_reports,
        "cells": final["cells"],
        "endpoints": endpoint_rows,
        "raw_curvature": final["raw_curvature"],
        "monotone_curvature": final["monotone_curvature"],
        "raw_residual": final["raw_residual"],
        "monotone_residual": final["monotone_residual"],
        "raw_witness": final["raw_witness"],
        "monotone_witness": final["monotone_witness"],
        "accounting": accounting,
    }
    return payload | {"report_semantic_hash": sha256_canonical(payload)}


def audit_monotone_anytime_checkpoint_histories(
    histories: Iterable[Iterable[MonotoneAnytimeCertificateState]], evaluator,
    plan: CertificatePlan,
) -> dict:
    """Validate all 384 histories, then independently replay every 512 history."""
    histories = tuple(tuple(history) for history in histories)
    expected_radii = tuple(radius.as_fraction() for radius in plan.radii)
    if (len(histories) != len(expected_radii)
            or tuple(Fraction(*history[-1].radius) for history in histories)
            != expected_radii):
        raise RuntimeError("ANYTIME_AUDIT_FROZEN_RADIUS_ORDER_INVALID")
    for history in histories:
        _validated_anytime_checkpoint_history(history, evaluator, plan)
    radius_reports = tuple(
        audit_monotone_anytime_checkpoint_history(history, evaluator, plan)
        for history in histories
    )
    official_intersection = None
    audit_intersection = None
    prefix_rows = []
    for history, report in zip(histories, radius_reports):
        official = _interval_from_payload(history[-1].monotone_witness)
        audit = _interval_from_payload(report["monotone_witness"]["audit"])
        official_intersection = (
            official if official_intersection is None else
            _checked_anytime_intersection(
                official_intersection, official, "OFFICIAL_CROSS_RADIUS_WITNESS"
            )
        )
        audit_intersection = (
            audit if audit_intersection is None else
            _checked_anytime_intersection(
                audit_intersection, audit, "AUDIT_CROSS_RADIUS_WITNESS"
            )
        )
        _require_audit_nested(
            audit_intersection, official_intersection, "CROSS_RADIUS_WITNESS",
        )
        prefix_rows.append({
            "radius": list(history[-1].radius),
            "official_intersection": _interval_payload(official_intersection),
            "audit_intersection": _interval_payload(audit_intersection),
            "audit_inside_official": True,
        })
    accounting = _add_anytime_accounting(*(
        report["accounting"] for report in radius_reports
    ))
    payload = {
        "schema_version": "green-v400-anytime-full-histories-audit-v1",
        "execution_scope": "outcome_blind_synthetic_only",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "phase_major_all_official_before_audit": True,
        "complete_split_histories_replayed": True,
        "audit_recurrence_uses_official_intervals": False,
        "radius_reports": list(radius_reports),
        "cross_radius_prefix_intersections": prefix_rows,
        "accounting": accounting,
    }
    return payload | {"report_semantic_hash": sha256_canonical(payload)}


def _adaptive_cells_with_reason(
    graph: RelationalGraph, h: Fraction, precision_bits: int,
    plan: CertificatePlan,
) -> tuple[list[CellCertificate] | None, str | None]:
    absolute = _hex_fraction(plan.absolute_width_tolerance)
    relative = _hex_fraction(plan.relative_width_tolerance)
    pending: list[tuple[gmpy2.mpfr, Fraction, int, CellCertificate]] = []
    initial_cells = (DyadicCell(-h, Fraction(0)), DyadicCell(Fraction(0), h))
    for cell, certificate in zip(
        initial_cells, _certify_cell_pair(graph, initial_cells, precision_bits)
    ):
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
            return None, "MAX_FINAL_LEAVES_PER_RADIUS_REACHED"
        left, right = cell.bisect()
        children = (left, right)
        for child, child_certificate in zip(
            children, _certify_cell_pair(graph, children, precision_bits)
        ):
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
