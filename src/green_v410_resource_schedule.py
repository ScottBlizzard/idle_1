"""Deterministic fixed-budget schedules for GREEN v4.1 resource calibration.

This module contains no model or task data.  It turns one closed synthetic
domain into the exact number of graph dispatches charged by the successor
protocol: ``2*L+1`` at 384 bits and ``L+3`` at 512 bits.  The audit replays the
final official partition and never creates an audit-precision split queue.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import gmpy2

from green_bridge_v400_certificate import DyadicCell
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import Jet2
from green_bridge_v400_mpfr_tensor_executor import jet_exact_payload
from green_v410_protocol import sha256_canonical
from green_v410_resource_calibration import CANDIDATES


@dataclass(frozen=True)
class _EvaluatedCell:
    cell: DyadicCell
    jet: Jet2


def _fraction(value) -> Fraction:
    rational = gmpy2.mpq(value)
    return Fraction(int(rational.numerator), int(rational.denominator))


def _cell_payload(row: _EvaluatedCell) -> dict:
    return {
        "lower": [row.cell.lower.numerator, row.cell.lower.denominator],
        "upper": [row.cell.upper.numerator, row.cell.upper.denominator],
        "depth": row.cell.depth,
        "jet": jet_exact_payload(row.jet),
    }


def _point_payload(point: Fraction, jet: Jet2) -> dict:
    return {
        "point": [point.numerator, point.denominator],
        "jet": jet_exact_payload(jet),
    }


def _validate_jet(jet: Jet2, precision_bits: int) -> None:
    if not isinstance(jet, Jet2) or jet.precision_bits != precision_bits:
        raise RuntimeError("resource evaluator returned wrong precision or type")


def _evaluate(evaluator, cell: DyadicCell, precision_bits: int) -> _EvaluatedCell:
    jet = evaluator.evaluate_interval(cell.interval(precision_bits))
    _validate_jet(jet, precision_bits)
    return _EvaluatedCell(cell, jet)


def _priority(row: _EvaluatedCell) -> Fraction:
    # This is the exact calibration form of curvature-weighted interval width.
    # It is intentionally computed from rational MPFR endpoints, never floats.
    domain_width = row.cell.upper - row.cell.lower
    curvature_width = _fraction(row.jet.second.upper) - _fraction(row.jet.second.lower)
    return domain_width * curvature_width


def _assert_partition(rows: Iterable[_EvaluatedCell], lower: Fraction,
                      upper: Fraction, leaf_budget: int) -> tuple[_EvaluatedCell, ...]:
    ordered = tuple(sorted(rows, key=lambda row: (row.cell.lower, row.cell.upper)))
    if len(ordered) != leaf_budget or ordered[0].cell.lower != lower:
        raise RuntimeError("resource partition has wrong leaf count or lower bound")
    if ordered[-1].cell.upper != upper or any(
        left.cell.upper != right.cell.lower for left, right in zip(ordered, ordered[1:])
    ):
        raise RuntimeError("resource partition is not an exact contiguous cover")
    return ordered


def run_official_schedule(evaluator, domain: tuple[Fraction, Fraction],
                          leaf_budget: int, precision_bits: int = 384) -> dict:
    """Run one fixed-budget official calibration schedule.

    Two initial half-domain cells plus two children for each of ``L-2``
    refinements and three point evaluations gives exactly ``2*L+1`` graph
    dispatches.  This mirrors the production ``[-h,0],[0,h]`` accounting on
    every translated/scaled synthetic fixture domain.
    """
    if leaf_budget not in CANDIDATES or precision_bits != 384:
        raise ValueError("official resource schedule identity mismatch")
    lower, upper = (Fraction(value) for value in domain)
    if lower >= upper:
        raise ValueError("resource calibration domain must be nonempty")
    initial = DyadicCell(lower, upper).bisect()
    leaves = [_evaluate(evaluator, cell, precision_bits) for cell in initial]
    dispatches = 2
    while len(leaves) < leaf_budget:
        parent = min(
            leaves,
            key=lambda row: (-_priority(row), row.cell.lower, row.cell.depth),
        )
        leaves.remove(parent)
        children = tuple(
            _evaluate(evaluator, cell, precision_bits) for cell in parent.cell.bisect()
        )
        leaves.extend(children)
        dispatches += 2
    ordered = _assert_partition(leaves, lower, upper, leaf_budget)
    points = (lower, (lower + upper) / 2, upper)
    point_rows = []
    for point in points:
        jet = evaluator.evaluate_interval(Interval.point(
            gmpy2.mpq(point.numerator, point.denominator), precision_bits
        ))
        _validate_jet(jet, precision_bits)
        point_rows.append(_point_payload(point, jet))
        dispatches += 1
    if dispatches != 2 * leaf_budget + 1:
        raise RuntimeError("official resource dispatch accounting drift")
    payload = {
        "schema_version": "green-v410-resource-official-schedule-v1",
        "precision_bits": precision_bits,
        "leaf_budget": leaf_budget,
        "domain": [
            [lower.numerator, lower.denominator],
            [upper.numerator, upper.denominator],
        ],
        "split_policy": "exact_curvature_weighted_width_then_lower_then_depth",
        "dispatch_count": dispatches,
        "max_depth": max(row.cell.depth for row in ordered),
        "final_cells": [_cell_payload(row) for row in ordered],
        "boundary_points": point_rows,
    }
    payload["schedule_sha256"] = sha256_canonical(payload)
    return payload


def _cell_from_payload(payload: dict) -> DyadicCell:
    if set(payload) != {"lower", "upper", "depth", "jet"}:
        raise ValueError("official resource cell payload mismatch")
    return DyadicCell(
        Fraction(*payload["lower"]), Fraction(*payload["upper"]), int(payload["depth"]),
    )


def _interval_nested(high: dict, low: dict) -> bool:
    return (
        Fraction(*low["lower"]) <= Fraction(*high["lower"])
        and Fraction(*high["upper"]) <= Fraction(*low["upper"])
    )


def _jet_nested(high: dict, low: dict) -> bool:
    return all(_interval_nested(high[name], low[name]) for name in ("value", "first", "second"))


def run_frozen_partition_audit(evaluator, official: dict,
                               precision_bits: int = 512) -> dict:
    """Replay the exact official partition at audit precision and check nesting."""
    if (
        official.get("schema_version") != "green-v410-resource-official-schedule-v1"
        or official.get("precision_bits") != 384
        or official.get("schedule_sha256")
        != sha256_canonical({key: value for key, value in official.items()
                             if key != "schedule_sha256"})
        or precision_bits != 512
    ):
        raise ValueError("audit resource schedule identity mismatch")
    cells = [_cell_from_payload(row) for row in official["final_cells"]]
    audit_cells = [_evaluate(evaluator, cell, precision_bits) for cell in cells]
    lower, upper = (Fraction(*value) for value in official["domain"])
    _assert_partition(audit_cells, lower, upper, official["leaf_budget"])
    points = (lower, (lower + upper) / 2, upper)
    audit_points = []
    for point in points:
        jet = evaluator.evaluate_interval(Interval.point(
            gmpy2.mpq(point.numerator, point.denominator), precision_bits
        ))
        _validate_jet(jet, precision_bits)
        audit_points.append(_point_payload(point, jet))
    nested_cells = [
        _jet_nested(jet_exact_payload(high.jet), low["jet"])
        for high, low in zip(audit_cells, official["final_cells"])
    ]
    nested_points = [
        _jet_nested(high["jet"], low["jet"])
        for high, low in zip(audit_points, official["boundary_points"])
    ]
    dispatches = len(audit_cells) + len(audit_points)
    if dispatches != official["leaf_budget"] + 3:
        raise RuntimeError("audit resource dispatch accounting drift")
    payload = {
        "schema_version": "green-v410-resource-frozen-partition-audit-v1",
        "precision_bits": precision_bits,
        "leaf_budget": official["leaf_budget"],
        "official_schedule_sha256": official["schedule_sha256"],
        "dispatch_count": dispatches,
        "audit_cells": [_cell_payload(row) for row in audit_cells],
        "audit_boundary_points": audit_points,
        "cell_nesting": nested_cells,
        "boundary_nesting": nested_points,
        "all_nested": all(nested_cells) and all(nested_points),
        "independent_audit_split_queue": False,
    }
    payload["schedule_sha256"] = sha256_canonical(payload)
    return payload
