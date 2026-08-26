from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import gmpy2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_certificate import (
    CellCertificate, CurvatureCertificate, DyadicCell, EndpointCertificate, certify_cell,
    certify_endpoints_and_slope, certify_joint_witness, compute_epsilon_psi,
    compute_m2, integrate_signed_curvature, witness_interval,
)
import green_bridge_v400_certificate as certificate_module
from green_bridge_v400_interval import Interval
from green_bridge_v400_relational_graph import GraphNode, RelationalGraph
from green_bridge_v400_schemas import CertificatePlan, Dyadic, JointWitnessRowSpec


P = 256


def _polynomial_graph(power: int) -> RelationalGraph:
    nodes = {
        "t": GraphNode("t", "affine_control", params={"base": 0, "direction": 1},
                       provenance="control", depends_on_t=True)
    }
    output = "t"
    for index in range(2, power + 1):
        node_id = f"p{index}"
        nodes[node_id] = GraphNode(node_id, "mul", (output, "t"),
                                   provenance=f"power{index}", depends_on_t=True)
        output = node_id
    return RelationalGraph(nodes, output, P)


def _certificate_parts(power: int, h=Fraction(1)):
    graph = _polynomial_graph(power)
    cells = [DyadicCell(-h, Fraction(0)), DyadicCell(Fraction(0), h)]
    certified = [certify_cell(graph, cell, P) for cell in cells]
    endpoint = certify_endpoints_and_slope(graph, h, P)
    curvature = integrate_signed_curvature(certified, h)
    error = compute_epsilon_psi(endpoint, curvature)
    return graph, certified, endpoint, curvature, error


def _plan(row_hash: str, *, radii=(Dyadic(1, 0),), max_depth=8,
          max_cells=1024) -> CertificatePlan:
    return CertificatePlan(
        "green-v400-certificate-plan-v1", row_hash, tuple(radii),
        "[-h,0],[0,h]", "curvature-weighted width priority dyadic bisection",
        "0x1p-80", "0x1p-40", max_depth, max_cells, 384, 512, (), False,
    )


def test_width_stop_requires_absolute_and_relative_tolerances():
    cell = DyadicCell(Fraction(-1), Fraction(0))
    certificate = CellCertificate(
        cell,
        Interval.point(0, P),
        Interval.point(0, P),
        Interval.from_bounds(-1, 1, P),
    )
    absolute = Fraction(1, 2**80)
    relative = Fraction(1, 2**40)
    tolerance = certificate_module._cell_tolerance(certificate, absolute, relative)
    assert tolerance == gmpy2.mpfr(absolute.numerator) / absolute.denominator


def test_subdivision_uses_curvature_weighted_width_priority(monkeypatch):
    calls: list[DyadicCell] = []

    def fake_certify_cell(graph, cell, precision_bits):
        calls.append(cell)
        width = 4 if cell.lower >= 0 else 1
        zero = Interval.point(0, precision_bits)
        return CellCertificate(
            cell, zero, zero, Interval.from_bounds(0, width, precision_bits)
        )

    monkeypatch.setattr(certificate_module, "certify_cell", fake_certify_cell)
    result = certificate_module._adaptive_cells(
        object(), Fraction(1), P, _plan("f" * 64, max_depth=1, max_cells=4)
    )
    assert result is None
    assert calls[:2] == [
        DyadicCell(Fraction(-1), Fraction(0)),
        DyadicCell(Fraction(0), Fraction(1)),
    ]
    assert calls[2:4] == [
        DyadicCell(Fraction(0), Fraction(1, 2), 1),
        DyadicCell(Fraction(1, 2), Fraction(1), 1),
    ]


def test_linear_endpoint_error_zero():
    _, _, endpoint, curvature, error = _certificate_parts(1)
    assert error.positive_residual.lower == error.positive_residual.upper == 0
    assert error.negative_residual.lower == error.negative_residual.upper == 0
    assert witness_interval(endpoint, curvature, error).contains(1.0)


def test_quadratic_endpoint_remainders_exact():
    _, _, endpoint, _, error = _certificate_parts(2)
    assert error.positive_residual.contains(1.0)
    assert error.negative_residual.contains(1.0)
    assert endpoint.slope.contains(0.0)


def test_cubic_central_secant_remainder():
    _, _, endpoint, curvature, error = _certificate_parts(3)
    interval = witness_interval(endpoint, curvature, error)
    assert interval.contains(0.0)
    assert interval.width() == 0


def test_positive_negative_signed_curvature_cancellation():
    _, _, _, curvature, error = _certificate_parts(3)
    assert error.positive_residual.contains(1.0)
    assert error.negative_residual.contains(-1.0)
    assert (error.positive_residual - error.negative_residual).contains(2.0)
    assert curvature.secant.contains(2.0)


def test_direct_and_curvature_intersection_nonempty():
    _, _, _, curvature, error = _certificate_parts(2)
    assert error.positive_residual.intersect(curvature.positive)
    assert error.negative_residual.intersect(curvature.negative)


def test_disjoint_enclosures_raise_implementation_invalid():
    zero = Interval.point(0, P)
    one = Interval.point(1, P)
    endpoint = EndpointCertificate(Fraction(1), zero, zero, zero, zero)
    curvature = CurvatureCertificate(one, one, zero, one)
    with pytest.raises(RuntimeError, match="CERTIFICATE_IMPLEMENTATION_INVALID"):
        compute_epsilon_psi(endpoint, curvature)


def test_m2_fallback_bounds_endpoint_error():
    _, cells, _, _, error = _certificate_parts(3)
    m2 = compute_m2(cells).upper
    assert error.epsilon_psi <= m2 / 2  # h=1, each one-sided remainder <= M2/2


def test_epsilon_component_accounting():
    _, _, _, _, error = _certificate_parts(2)
    assert error.epsilon_psi == max(error.positive_residual.magnitude(),
                                    error.negative_residual.magnitude())


def test_384_official_512_audit_nested():
    graph = _polynomial_graph(3)
    low = certify_cell(RelationalGraph(graph.nodes, graph.output_id, 384),
                       DyadicCell(Fraction(-1), Fraction(1)), 384)
    high = certify_cell(RelationalGraph(graph.nodes, graph.output_id, 512),
                        DyadicCell(Fraction(-1), Fraction(1)), 512)
    assert low.second.lower <= high.second.lower <= high.second.upper <= low.second.upper


def test_resource_cap_returns_inconclusive_not_success():
    graph = _polynomial_graph(3)
    payload = {
        "precision_bits": P, "output_id": graph.output_id,
        "nodes": [
            {"node_id": node.node_id, "op": node.op, "parents": list(node.parents),
             "params": node.params, "provenance": node.provenance,
             "depends_on_t": node.depends_on_t}
            for node in graph.nodes.values()
        ],
    }
    row = JointWitnessRowSpec("green-v400-row-v1", "0"*64, "synthetic", "1"*64,
                              "2"*64, "3"*64, "4"*64, "5"*64,
                              ("PAT_J", "PAT_B", "TAR_J", "TAR_B"), payload)
    plan = _plan(row.row_hash, max_depth=0, max_cells=2)
    result = certify_joint_witness(row, plan)
    assert result.status == "RESOURCE_INCONCLUSIVE"
    assert result.witness_interval is None


def test_certificate_plan_round_trip_and_multi_radius_execution():
    graph = _polynomial_graph(2)
    payload = {
        "precision_bits": P, "output_id": graph.output_id,
        "nodes": [
            {"node_id": node.node_id, "op": node.op, "parents": list(node.parents),
             "params": node.params, "provenance": node.provenance,
             "depends_on_t": node.depends_on_t}
            for node in graph.nodes.values()
        ],
    }
    row = JointWitnessRowSpec("green-v400-row-v1", "a"*64, "synthetic", "1"*64,
                              "2"*64, "3"*64, "4"*64, "5"*64,
                              ("PAT_J", "PAT_B", "TAR_J", "TAR_B"), payload)
    plan = _plan(row.row_hash, radii=(Dyadic(1, 0), Dyadic(1, -1), Dyadic(1, -2)))
    assert CertificatePlan.from_dict(plan.to_dict()) == plan
    result = certify_joint_witness(row, plan)
    assert result.status == "CERTIFIED"
    assert result.audit_nested
    assert len(result.radii) == 3
    assert result.witness_interval is not None
    assert result.witness_interval.contains(0.0)


def test_legacy_plan_object_is_rejected_instead_of_silent_default():
    graph = _polynomial_graph(2)
    payload = {
        "precision_bits": P, "output_id": graph.output_id,
        "nodes": [
            {"node_id": node.node_id, "op": node.op, "parents": list(node.parents),
             "params": node.params, "provenance": node.provenance,
             "depends_on_t": node.depends_on_t}
            for node in graph.nodes.values()
        ],
    }
    row = JointWitnessRowSpec("green-v400-row-v1", "b"*64, "synthetic", "1"*64,
                              "2"*64, "3"*64, "4"*64, "5"*64,
                              ("PAT_J", "PAT_B", "TAR_J", "TAR_B"), payload)
    with pytest.raises(TypeError, match="validated CertificatePlan"):
        certify_joint_witness(row, object())
