"""Fixed-budget multi-output certificates for GREEN v4.1 real rows.

The resource gate freezes a uniform final leaf budget.  This module applies
that budget to the five-output v4.1 TensorProgram without inspecting endpoint
material or using interval width as a stopping rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import gmpy2

from green_bridge_v400_certificate import (
    CellCertificate,
    DyadicCell,
    EndpointCertificate,
    compute_epsilon_psi,
    integrate_signed_curvature,
    witness_interval,
)
from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import EmptyIntersection, Interval
from green_bridge_v400_mpfr_tensor_executor import (
    ResidentStaticRowCache,
    execute_tensor_program_mpfr,
    jet_exact_payload,
    preload_tensor_program_arrays,
)
from green_v410_protocol import RADIUS_PANEL, sha256_canonical


OUTPUT_KEYS = ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "PSI")
RADIUS_FRACTIONS = (Fraction(1), Fraction(1, 2), Fraction(1, 4))


def _fraction(value: Any) -> Fraction:
    rational = gmpy2.mpq(value)
    return Fraction(int(rational.numerator), int(rational.denominator))


def _pair(value: Any) -> list[int]:
    fraction = _fraction(value)
    return [fraction.numerator, fraction.denominator]


def _interval_payload(interval: Interval) -> dict[str, list[int]]:
    return {"lower": _pair(interval.lower), "upper": _pair(interval.upper)}


def _nested(inner: Interval, outer: Interval) -> bool:
    return outer.lower <= inner.lower <= inner.upper <= outer.upper


def _result_jets(result: Mapping[str, Any]) -> dict[str, Any]:
    jets = {
        "PAT_J": result["PAT_J"],
        "PAT_B": result["PAT_B"],
        "TAR_J": result["TAR_J"],
        "TAR_B": result["TAR_B"],
        "PSI": result["output"],
    }
    if any(jet.precision_bits != next(iter(jets.values())).precision_bits for jet in jets.values()):
        raise RuntimeError("multi-output evaluator returned mixed precision")
    return jets


class FixedBudgetProgramEvaluator:
    """Hash-closed five-output TensorProgram evaluator."""

    contains_scientific_outcome = False

    def __init__(self, program: Any, reader: Any, backend: str | Path | Any,
                 *, successful_node_callback=None):
        self.program = program
        self.reader = reader
        self.successful_node_callback = successful_node_callback
        self.backend = (
            CompiledMPFRBackend(Path(backend))
            if isinstance(backend, (str, Path)) else backend
        )
        self.preloaded = preload_tensor_program_arrays(program, reader)
        self._caches: dict[int, ResidentStaticRowCache] = {}

    def evaluate(self, domain: Interval) -> dict[str, Any]:
        cache = self._caches.get(domain.precision_bits)
        if cache is None:
            cache = ResidentStaticRowCache.build_unpacked(
                self.program, self.backend, domain.precision_bits
            )
            self._caches[domain.precision_bits] = cache
        return _result_jets(execute_tensor_program_mpfr(
            self.program,
            self.reader,
            domain,
            self.backend,
            sparse_axis0_execution=True,
            resident_buffer_execution=True,
            resident_static_row_cache=cache,
            preloaded_tensors=self.preloaded,
            successful_node_callback=self.successful_node_callback,
        ))

    def close(self) -> None:
        for cache in self._caches.values():
            cache.close()
        self._caches.clear()


@dataclass(frozen=True)
class EvaluatedCell:
    cell: DyadicCell
    jets: Mapping[str, Any]


def _priority(row: EvaluatedCell) -> Fraction:
    second = row.jets["PSI"].second
    return (row.cell.upper - row.cell.lower) * (
        _fraction(second.upper) - _fraction(second.lower)
    )


def _evaluate_cell(
    evaluator: FixedBudgetProgramEvaluator, cell: DyadicCell, precision_bits: int,
) -> EvaluatedCell:
    jets = evaluator.evaluate(cell.interval(precision_bits))
    if set(jets) != set(OUTPUT_KEYS):
        raise RuntimeError("five-output certificate closure mismatch")
    return EvaluatedCell(cell, jets)


def _partition(
    evaluator: FixedBudgetProgramEvaluator,
    h: Fraction,
    leaf_budget: int,
    precision_bits: int,
    frozen_cells: tuple[DyadicCell, ...] | None = None,
) -> tuple[tuple[EvaluatedCell, ...], tuple[tuple[EvaluatedCell, ...], ...]]:
    if leaf_budget < 2:
        raise ValueError("leaf budget must be at least two")
    if frozen_cells is None:
        leaves = [
            _evaluate_cell(evaluator, cell, precision_bits)
            for cell in (DyadicCell(-h, Fraction(0)), DyadicCell(Fraction(0), h))
        ]
        histories = [tuple(sorted(leaves, key=lambda row: row.cell.lower))]
        while len(leaves) < leaf_budget:
            parent = min(
                leaves,
                key=lambda row: (-_priority(row), row.cell.lower, row.cell.depth),
            )
            leaves.remove(parent)
            leaves.extend(
                _evaluate_cell(evaluator, child, precision_bits)
                for child in parent.cell.bisect()
            )
            histories.append(tuple(sorted(leaves, key=lambda row: row.cell.lower)))
    else:
        leaves = [
            _evaluate_cell(evaluator, cell, precision_bits) for cell in frozen_cells
        ]
        histories = [tuple(sorted(leaves, key=lambda row: row.cell.lower))]
    ordered = tuple(sorted(leaves, key=lambda row: (row.cell.lower, row.cell.upper)))
    if len(ordered) != leaf_budget or ordered[0].cell.lower != -h or ordered[-1].cell.upper != h:
        raise RuntimeError("fixed-budget partition boundary mismatch")
    if any(left.cell.upper != right.cell.lower for left, right in zip(ordered, ordered[1:])):
        raise RuntimeError("fixed-budget partition is not contiguous")
    if any(row.cell.lower < 0 < row.cell.upper for row in ordered):
        raise RuntimeError("fixed-budget partition crosses zero")
    return ordered, tuple(histories)


def _points(
    evaluator: FixedBudgetProgramEvaluator, h: Fraction, precision_bits: int,
) -> dict[Fraction, dict[str, Any]]:
    return {
        point: evaluator.evaluate(Interval.point(
            gmpy2.mpq(point.numerator, point.denominator), precision_bits
        ))
        for point in (-h, Fraction(0), h)
    }


def _witnesses(
    rows: tuple[EvaluatedCell, ...],
    points: Mapping[Fraction, Mapping[str, Any]],
    h: Fraction,
) -> dict[str, Interval]:
    result: dict[str, Interval] = {}
    for name in OUTPUT_KEYS:
        cells = [CellCertificate(
            row.cell,
            row.jets[name].value,
            row.jets[name].first,
            row.jets[name].second,
        ) for row in rows]
        negative = points[-h][name]
        center = points[Fraction(0)][name]
        positive = points[h][name]
        endpoint = EndpointCertificate(
            h, negative.value, center.value, positive.value, center.first
        )
        curvature = integrate_signed_curvature(cells, h)
        error = compute_epsilon_psi(endpoint, curvature)
        result[name] = witness_interval(endpoint, curvature, error)
    return result


def _radius_payload(
    h: Fraction,
    official: tuple[EvaluatedCell, ...],
    audit: tuple[EvaluatedCell, ...],
    official_points: Mapping[Fraction, Mapping[str, Any]],
    audit_points: Mapping[Fraction, Mapping[str, Any]],
    official_witnesses: Mapping[str, Interval],
    audit_witnesses: Mapping[str, Interval],
    official_prefix_witnesses: tuple[Mapping[str, Interval], ...],
) -> dict[str, Any]:
    return {
        "radius": [h.numerator, h.denominator],
        "official_cells": [{
            "lower": [row.cell.lower.numerator, row.cell.lower.denominator],
            "upper": [row.cell.upper.numerator, row.cell.upper.denominator],
            "depth": row.cell.depth,
            "jets": {name: jet_exact_payload(row.jets[name]) for name in OUTPUT_KEYS},
        } for row in official],
        "audit_cells": [{
            "lower": [row.cell.lower.numerator, row.cell.lower.denominator],
            "upper": [row.cell.upper.numerator, row.cell.upper.denominator],
            "depth": row.cell.depth,
            "jets": {name: jet_exact_payload(row.jets[name]) for name in OUTPUT_KEYS},
        } for row in audit],
        "official_points": {
            str(point): {name: jet_exact_payload(jet) for name, jet in jets.items()}
            for point, jets in official_points.items()
        },
        "audit_points": {
            str(point): {name: jet_exact_payload(jet) for name, jet in jets.items()}
            for point, jets in audit_points.items()
        },
        "official_witnesses": {
            name: _interval_payload(value) for name, value in official_witnesses.items()
        },
        "official_prefix_witnesses": [{
            name: _interval_payload(value) for name, value in row.items()
        } for row in official_prefix_witnesses],
        "audit_witnesses": {
            name: _interval_payload(value) for name, value in audit_witnesses.items()
        },
    }


def certify_five_outputs(
    evaluator: FixedBudgetProgramEvaluator,
    *,
    leaf_budget: int,
    official_precision_bits: int = 384,
    audit_precision_bits: int = 512,
) -> dict[str, Any]:
    """Certify all five outputs at every frozen radius and precision."""

    official_intersections: dict[str, Interval] = {}
    audit_intersections: dict[str, Interval] = {}
    radius_reports = []
    for h in RADIUS_FRACTIONS:
        official, official_histories = _partition(
            evaluator, h, leaf_budget, official_precision_bits
        )
        audit, _ = _partition(
            evaluator, h, leaf_budget, audit_precision_bits,
            tuple(row.cell for row in official),
        )
        official_points = _points(evaluator, h, official_precision_bits)
        audit_points = _points(evaluator, h, audit_precision_bits)
        for low, high in zip(official, audit):
            for name in OUTPUT_KEYS:
                if not all(_nested(getattr(high.jets[name], field), getattr(low.jets[name], field))
                           for field in ("value", "first", "second")):
                    raise RuntimeError("CERTIFICATE_PRECISION_NESTING_INVALID")
        for point in (-h, Fraction(0), h):
            for name in OUTPUT_KEYS:
                if not all(_nested(getattr(audit_points[point][name], field),
                                   getattr(official_points[point][name], field))
                           for field in ("value", "first", "second")):
                    raise RuntimeError("CERTIFICATE_ENDPOINT_NESTING_INVALID")
        prefix_witnesses = tuple(
            _witnesses(history, official_points, h)
            for history in official_histories
        )
        official_witnesses: dict[str, Interval] = {}
        for prefix in prefix_witnesses:
            for name in OUTPUT_KEYS:
                try:
                    official_witnesses[name] = (
                        prefix[name] if name not in official_witnesses
                        else official_witnesses[name].intersect(prefix[name])
                    )
                except EmptyIntersection as exc:
                    raise RuntimeError("INVALID_EMPTY_SOUND_INTERSECTION") from exc
        audit_witnesses = _witnesses(audit, audit_points, h)
        for name in OUTPUT_KEYS:
            if not _nested(audit_witnesses[name], official_witnesses[name]):
                raise RuntimeError("CERTIFICATE_WITNESS_NESTING_INVALID")
            try:
                official_intersections[name] = (
                    official_witnesses[name]
                    if name not in official_intersections
                    else official_intersections[name].intersect(official_witnesses[name])
                )
                audit_intersections[name] = (
                    audit_witnesses[name]
                    if name not in audit_intersections
                    else audit_intersections[name].intersect(audit_witnesses[name])
                )
            except EmptyIntersection as exc:
                raise RuntimeError("INVALID_EMPTY_SOUND_INTERSECTION") from exc
            if not _nested(audit_intersections[name], official_intersections[name]):
                raise RuntimeError("CERTIFICATE_CROSS_RADIUS_NESTING_INVALID")
        radius_reports.append(_radius_payload(
            h, official, audit, official_points, audit_points,
            official_witnesses, audit_witnesses, prefix_witnesses,
        ))
    payload = {
        "schema_version": "green-v410-fixed-budget-five-output-certificate-v1",
        "leaf_budget": leaf_budget,
        "radii": list(RADIUS_PANEL),
        "official_precision_bits": official_precision_bits,
        "audit_precision_bits": audit_precision_bits,
        "official_intervals": {
            name: _interval_payload(value) for name, value in official_intersections.items()
        },
        "audit_intervals": {
            name: _interval_payload(value) for name, value in audit_intersections.items()
        },
        "precision_nested": all(
            _nested(audit_intersections[name], official_intersections[name])
            for name in OUTPUT_KEYS
        ),
        "radius_reports": radius_reports,
    }
    return payload | {"certificate_sha256": sha256_canonical(payload)}
