"""Outcome-blind endpoint/center and initial-curvature resident-buffer benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_certificate import (
    CellCertificate, DyadicCell, EndpointCertificate, compute_epsilon_psi,
    integrate_signed_curvature, witness_interval,
)
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr_tensor_executor import (
    ResidentStaticRowCache, execute_tensor_program_mpfr, jet_exact_payload,
)
from green_bridge_v400_resident_plan import load_resident_plan_arrays
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


def _nested_clear(inner, outer) -> bool:
    return all(
        outer_component.lower <= inner_component.lower
        and inner_component.upper <= outer_component.upper
        for inner_component, outer_component in (
            (inner.value, outer.value), (inner.first, outer.first),
            (inner.second, outer.second),
        )
    )


def _interval_nested(inner, outer) -> bool:
    return outer.lower <= inner.lower <= inner.upper <= outer.upper


def _interval_exact_payload(interval) -> dict:
    import gmpy2

    def rational(value) -> list[int]:
        exact = gmpy2.mpq(value)
        return [int(exact.numerator), int(exact.denominator)]

    return {"lower": rational(interval.lower), "upper": rational(interval.upper)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--radius-numerator", type=int, default=1)
    parser.add_argument("--radius-exponent-2", type=int, default=-14)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("endpoint/center output must be a new file on /mnt/sdb")
    if args.radius_numerator <= 0 or not -1024 <= args.radius_exponent_2 <= 0:
        raise ValueError("invalid positive dyadic radius")
    radius = Fraction(args.radius_numerator, 1 << (-args.radius_exponent_2))
    program = TensorProgram.from_dict(json.loads(
        Path(args.program).read_text(encoding="utf-8")
    ))
    reader = TensorStoreReader(Path(args.tensor_manifest))
    plan, arrays = load_resident_plan_arrays(Path(args.plan), program, reader)
    backend = CompiledMPFRBackend(Path(args.library))
    domain_specs = (
        ("negative_endpoint", -radius, -radius),
        ("center", Fraction(0), Fraction(0)),
        ("positive_endpoint", radius, radius),
        ("negative_curvature_cell", -radius, Fraction(0)),
        ("positive_curvature_cell", Fraction(0), radius),
    )
    outputs_by_precision = {}
    certificates_by_precision = {}
    precision_rows = []
    passed = True
    peak_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for precision in (384, 512):
        cache = ResidentStaticRowCache.build(program, plan, backend, precision)
        stage_rows, stage_outputs = [], {}
        for name, lower, upper in domain_specs:
            domain = Interval.from_bounds(lower, upper, precision)
            started = time.perf_counter()
            result = execute_tensor_program_mpfr(
                program, reader, domain, backend, resident_plan=plan,
                resident_arrays=arrays, resident_static_row_cache=cache,
                sparse_axis0_execution=True, resident_buffer_execution=True,
                return_runtime_metrics=True,
            )
            elapsed = time.perf_counter() - started
            stage_outputs[name] = result["output"]
            metrics = result["runtime_metrics"]
            stage_rows.append({
                "stage": name,
                "elapsed_seconds": elapsed,
                "static_cache_initial_entry_count": metrics[
                    "static_row_cache_initial_entry_count"
                ],
                "static_cache_miss_count": sum(
                    metrics["static_row_cache_misses_by_kernel"].values()
                ),
                "native_static_cache_hits": metrics[
                    "resident_native_static_cache_hits"
                ],
                "native_static_cache_misses": metrics[
                    "resident_native_static_cache_misses"
                ],
                "resident_buffer_imported_jet_count": metrics[
                    "resident_buffer_imported_jet_count"
                ],
                "resident_buffer_exported_jet_count": metrics[
                    "resident_buffer_exported_jet_count"
                ],
            })
            print(json.dumps({
                "status": "STAGE_COMPLETE", "precision_bits": precision,
                "stage": name, "elapsed_seconds": elapsed,
            }, sort_keys=True), flush=True)
        serialization_payload = {
            name: jet_exact_payload(stage_outputs[name]) for name, _, _ in domain_specs
        }
        endpoint = EndpointCertificate(
            radius, stage_outputs["negative_endpoint"].value,
            stage_outputs["center"].value,
            stage_outputs["positive_endpoint"].value,
            stage_outputs["center"].first,
        )
        cells = [
            CellCertificate(
                DyadicCell(-radius, Fraction(0)),
                stage_outputs["negative_curvature_cell"].value,
                stage_outputs["negative_curvature_cell"].first,
                stage_outputs["negative_curvature_cell"].second,
            ),
            CellCertificate(
                DyadicCell(Fraction(0), radius),
                stage_outputs["positive_curvature_cell"].value,
                stage_outputs["positive_curvature_cell"].first,
                stage_outputs["positive_curvature_cell"].second,
            ),
        ]
        curvature = integrate_signed_curvature(cells, radius)
        endpoint_error = compute_epsilon_psi(endpoint, curvature)
        witness = witness_interval(endpoint, curvature, endpoint_error)
        certificates_by_precision[precision] = (
            endpoint, curvature, endpoint_error, witness
        )
        certificate_payload = {
            "endpoint": {
                "negative": _interval_exact_payload(endpoint.negative),
                "center": _interval_exact_payload(endpoint.center),
                "positive": _interval_exact_payload(endpoint.positive),
                "slope": _interval_exact_payload(endpoint.slope),
            },
            "curvature": {
                "positive": _interval_exact_payload(curvature.positive),
                "negative": _interval_exact_payload(curvature.negative),
                "secant": _interval_exact_payload(curvature.secant),
                "m2": _interval_exact_payload(curvature.m2),
            },
            "endpoint_error": {
                "positive_residual": _interval_exact_payload(
                    endpoint_error.positive_residual
                ),
                "negative_residual": _interval_exact_payload(
                    endpoint_error.negative_residual
                ),
            },
            "witness": _interval_exact_payload(witness),
        }
        started = time.perf_counter()
        serialized = json.dumps(
            serialization_payload, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        serialization_seconds = time.perf_counter() - started
        outputs_by_precision[precision] = stage_outputs
        precision_rows.append({
            "precision_bits": precision,
            "stages": stage_rows,
            "stage_elapsed_sum_seconds": sum(row["elapsed_seconds"] for row in stage_rows),
            "stage_elapsed_median_seconds": statistics.median(
                row["elapsed_seconds"] for row in stage_rows
            ),
            "exact_payload_serialization_seconds": serialization_seconds,
            "exact_payload_serialized_bytes": len(serialized),
            "exact_payload_sha256": hashlib.sha256(serialized).hexdigest(),
            "signed_certificate_payload_sha256": sha256_canonical(certificate_payload),
            "signed_certificate_payload_bytes": len(json.dumps(
                certificate_payload, sort_keys=True, separators=(",", ":")
            ).encode("ascii")),
            "static_cache_entry_count": cache.entry_count,
            "native_static_cache_entry_count": cache.native_entry_count,
        })
        cache.close()
    nesting = {
        name: _nested_clear(outputs_by_precision[512][name], outputs_by_precision[384][name])
        for name, _, _ in domain_specs
    }
    passed = passed and all(nesting.values())
    endpoint_384, curvature_384, error_384, witness_384 = certificates_by_precision[384]
    endpoint_512, curvature_512, error_512, witness_512 = certificates_by_precision[512]
    certificate_nesting = {
        "endpoint_negative": _interval_nested(endpoint_512.negative, endpoint_384.negative),
        "endpoint_center": _interval_nested(endpoint_512.center, endpoint_384.center),
        "endpoint_positive": _interval_nested(endpoint_512.positive, endpoint_384.positive),
        "endpoint_slope": _interval_nested(endpoint_512.slope, endpoint_384.slope),
        "curvature_positive": _interval_nested(
            curvature_512.positive, curvature_384.positive
        ),
        "curvature_negative": _interval_nested(
            curvature_512.negative, curvature_384.negative
        ),
        "curvature_secant": _interval_nested(curvature_512.secant, curvature_384.secant),
        "curvature_m2": _interval_nested(curvature_512.m2, curvature_384.m2),
        "positive_residual": _interval_nested(
            error_512.positive_residual, error_384.positive_residual
        ),
        "negative_residual": _interval_nested(
            error_512.negative_residual, error_384.negative_residual
        ),
        "witness": _interval_nested(witness_512, witness_384),
    }
    passed = passed and all(certificate_nesting.values())
    peak_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "schema_version": "green-v400-resident-endpoint-center-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS_SIGNED_CURVATURE_OBSERVATION_ONLY" if passed else "FAIL",
        "contains_scientific_outcome": False,
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
        "fixture": "actual-shape closed GPT-2 synthetic tensor store; values omitted",
        "program_semantic_hash": program.semantic_hash(),
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "backend_sha256": backend.library_sha256,
        "radius": {
            "numerator": radius.numerator, "denominator": radius.denominator,
        },
        "fixed_stage_order": [name for name, _, _ in domain_specs],
        "precision_nesting_boolean_only": nesting,
        "signed_certificate_nesting_boolean_only": certificate_nesting,
        "rows": precision_rows,
        "process_peak_rss_before_kib": peak_before_kib,
        "process_peak_rss_after_kib": peak_after_kib,
        "process_peak_rss_delta_kib": max(0, peak_after_kib - peak_before_kib),
        "claim_scope": (
            "endpoint/center and two mandatory initial curvature-cell execution, "
            "signed two-cell curvature integration, exact-payload serialization cost, "
            "and same-process peak RSS; no adaptive subdivision, scientific label, "
            "threshold decision, or formal bound"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "output": str(output),
        "peak_rss_after_kib": peak_after_kib,
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
