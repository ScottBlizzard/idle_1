"""Outcome-blind bounded adaptive-queue execution on the closed synthetic program."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import resource
import sys
import time

import gmpy2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_certificate import certify_adaptive_cells
from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_mpfr_tensor_executor import (
    ResidentStaticRowCache, execute_tensor_program_mpfr,
)
from green_bridge_v400_resident_plan import load_resident_plan_arrays
from green_bridge_v400_schemas import CertificatePlan, Dyadic
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


def _fraction(value) -> Fraction:
    exact = gmpy2.mpq(value)
    return Fraction(int(exact.numerator), int(exact.denominator))


class ResidentProgramEvaluator:
    contains_scientific_outcome = False

    def __init__(self, program, reader, backend, resident_plan, resident_arrays):
        self.program = program
        self.reader = reader
        self.backend = backend
        self.resident_plan = resident_plan
        self.resident_arrays = resident_arrays
        self.caches = {}
        self.records = []

    def evaluate_interval(self, domain):
        precision = domain.precision_bits
        cache = self.caches.setdefault(
            precision, ResidentStaticRowCache.build(
                self.program, self.resident_plan, self.backend, precision
            )
        )
        started = time.perf_counter()
        result = execute_tensor_program_mpfr(
            self.program, self.reader, domain, self.backend,
            resident_plan=self.resident_plan, resident_arrays=self.resident_arrays,
            resident_static_row_cache=cache, sparse_axis0_execution=True,
            resident_buffer_execution=True, return_runtime_metrics=True,
        )
        elapsed = time.perf_counter() - started
        metrics = result["runtime_metrics"]
        self.records.append({
            "ordinal": len(self.records),
            "precision_bits": precision,
            "domain": {
                "lower": [_fraction(domain.lower).numerator,
                          _fraction(domain.lower).denominator],
                "upper": [_fraction(domain.upper).numerator,
                          _fraction(domain.upper).denominator],
            },
            "elapsed_seconds": elapsed,
            "static_cache_miss_count": sum(
                metrics["static_row_cache_misses_by_kernel"].values()
            ),
            "native_static_cache_hits": metrics["resident_native_static_cache_hits"],
            "native_static_cache_misses": metrics["resident_native_static_cache_misses"],
            "resident_buffer_imported_jet_count": metrics[
                "resident_buffer_imported_jet_count"
            ],
            "resident_buffer_exported_jet_count": metrics[
                "resident_buffer_exported_jet_count"
            ],
        })
        print(json.dumps({
            "status": "ADAPTIVE_CELL_COMPLETE", "ordinal": len(self.records) - 1,
            "precision_bits": precision, "elapsed_seconds": elapsed,
        }, sort_keys=True), flush=True)
        return result["output"]

    def close(self):
        for cache in self.caches.values():
            cache.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--radius-numerator", type=int, default=1)
    parser.add_argument("--radius-exponent-2", type=int, default=-14)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-cells", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("adaptive benchmark output must be a new file on /mnt/sdb")
    if (args.radius_numerator <= 0 or not -1024 <= args.radius_exponent_2 <= 0
            or args.max_depth < 0 or args.max_cells < 2):
        raise ValueError("invalid adaptive benchmark policy")
    radius = Fraction(args.radius_numerator, 1 << (-args.radius_exponent_2))
    program = TensorProgram.from_dict(json.loads(
        Path(args.program).read_text(encoding="utf-8")
    ))
    reader = TensorStoreReader(Path(args.tensor_manifest))
    resident_plan, resident_arrays = load_resident_plan_arrays(
        Path(args.plan), program, reader
    )
    backend = CompiledMPFRBackend(Path(args.library))
    certificate_plan = CertificatePlan(
        "green-v400-certificate-plan-v1", program.semantic_hash(),
        (Dyadic(args.radius_numerator, args.radius_exponent_2),),
        "[-h,0],[0,h]", "curvature-weighted width priority dyadic bisection",
        "0x1p-80", "0x1p-40", args.max_depth, args.max_cells,
        384, 512, (), False,
    )
    evaluator = ResidentProgramEvaluator(
        program, reader, backend, resident_plan, resident_arrays
    )
    peak_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    partition = certify_adaptive_cells(evaluator, radius, 384, certificate_plan)
    total_seconds = time.perf_counter() - started
    peak_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    records = evaluator.records
    evaluator.close()
    expected_initial = (
        {"lower": [-radius.numerator, radius.denominator], "upper": [0, 1]},
        {"lower": [0, 1], "upper": [radius.numerator, radius.denominator]},
    )
    initial_order_correct = (
        len(records) >= 2
        and records[0]["domain"] == expected_initial[0]
        and records[1]["domain"] == expected_initial[1]
    )
    evaluation_count_upper_bound = 2 * args.max_cells - 2
    bounded = 2 <= len(records) <= evaluation_count_upper_bound
    resource_inconclusive = partition is None
    if partition is None:
        partition_payload = None
    else:
        partition_payload = [
            {
                "lower": [cell.cell.lower.numerator, cell.cell.lower.denominator],
                "upper": [cell.cell.upper.numerator, cell.cell.upper.denominator],
                "depth": cell.cell.depth,
            }
            for cell in partition
        ]
    passed = initial_order_correct and bounded and all(
        record["precision_bits"] == 384 for record in records
    )
    report = {
        "schema_version": "green-v400-resident-adaptive-queue-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS_ADAPTIVE_QUEUE_OBSERVATION_ONLY" if passed else "FAIL",
        "contains_scientific_outcome": False,
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
        "fixture": "actual-shape closed GPT-2 synthetic tensor store; values omitted",
        "program_semantic_hash": program.semantic_hash(),
        "resident_plan_semantic_hash": resident_plan["resident_plan_semantic_hash"],
        "backend_sha256": backend.library_sha256,
        "certificate_plan": certificate_plan.to_dict(),
        "radius": {"numerator": radius.numerator, "denominator": radius.denominator},
        "evaluation_records": records,
        "evaluation_count": len(records),
        "initial_partition_order_correct": initial_order_correct,
        "evaluation_count_upper_bound_from_leaf_cap": evaluation_count_upper_bound,
        "evaluation_count_within_derived_bound": bounded,
        "adaptive_result": (
            "RESOURCE_INCONCLUSIVE" if resource_inconclusive else "PARTITION_ACCEPTED"
        ),
        "accepted_partition_cells": partition_payload,
        "audit_precision_executed": False,
        "audit_precision_skip_reason": (
            "official precision exhausted the frozen depth/cell policy"
            if resource_inconclusive else
            "benchmark scope ends after official adaptive partition"
        ),
        "elapsed_seconds": total_seconds,
        "process_peak_rss_before_kib": peak_before_kib,
        "process_peak_rss_after_kib": peak_after_kib,
        "process_peak_rss_delta_kib": max(0, peak_after_kib - peak_before_kib),
        "claim_scope": (
            "official-precision curvature-weighted heap execution under a deliberately "
            "small frozen depth/cell policy; RESOURCE_INCONCLUSIVE is retained as such, "
            "with no scientific label, threshold decision, or formal runtime bound"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "adaptive_result": report["adaptive_result"],
        "evaluation_count": len(records), "output": str(output),
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
