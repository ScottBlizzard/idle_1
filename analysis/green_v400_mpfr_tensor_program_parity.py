"""Outcome-blind full four-branch MPFR TensorProgram compiled parity audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from test_green_bridge_v400_gpt2_program import _fixture
from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr_tensor_executor import (
    ResidentStaticRowCache, execute_tensor_program_mpfr, jet_exact_payload,
    tensor_program_required_axis0_rows,
)
from green_bridge_v400_resident_plan import build_resident_plan, load_resident_plan_arrays
from green_bridge_v400_schemas import sha256_canonical


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    library, output_root = Path(args.library).resolve(), Path(args.output_root).resolve()
    if "/mnt/sdb/" not in output_root.as_posix() or output_root.exists():
        raise RuntimeError("parity output root must be a new directory on /mnt/sdb")
    output_root.mkdir(parents=True)
    reader, dims, program = _fixture(output_root)
    resident_plan = build_resident_plan(
        output_root / "resident_plan", "tiny_gpt2_resident", program, reader
    )
    resident_plan, resident_arrays = load_resident_plan_arrays(
        output_root / "resident_plan" / "tiny_gpt2_resident.json", program, reader
    )
    backend = CompiledMPFRBackend(library)
    live_rows = tensor_program_required_axis0_rows(program)
    rows, passed = [], True
    for precision in (384, 512):
        domain = Interval.from_bounds(-2.0**-14, 2.0**-14, precision)
        resident_static_cache = ResidentStaticRowCache.build(
            program, resident_plan, backend, precision
        )
        started = time.perf_counter()
        reference = execute_tensor_program_mpfr(
            program, reader, domain, return_dispatch_trace=True
        )
        reference_seconds = time.perf_counter() - started
        started = time.perf_counter()
        compiled = execute_tensor_program_mpfr(program, reader, domain, backend)
        compiled_seconds = time.perf_counter() - started
        started = time.perf_counter()
        resident_fused = execute_tensor_program_mpfr(
            program, reader, domain, backend, resident_plan=resident_plan,
            resident_arrays=resident_arrays, sparse_axis0_execution=True,
            resident_static_row_cache=resident_static_cache,
            return_runtime_metrics=True, return_dispatch_trace=True,
        )
        resident_fused_seconds = time.perf_counter() - started
        started = time.perf_counter()
        resident_warm = execute_tensor_program_mpfr(
            program, reader, domain, backend, resident_plan=resident_plan,
            resident_arrays=resident_arrays, sparse_axis0_execution=True,
            resident_static_row_cache=resident_static_cache,
            return_runtime_metrics=True, return_dispatch_trace=True,
        )
        resident_warm_seconds = time.perf_counter() - started
        resident_buffer_cache = ResidentStaticRowCache.build(
            program, resident_plan, backend, precision
        )
        started = time.perf_counter()
        resident_buffer = execute_tensor_program_mpfr(
            program, reader, domain, backend, resident_plan=resident_plan,
            resident_arrays=resident_arrays, sparse_axis0_execution=True,
            resident_static_row_cache=resident_buffer_cache,
            resident_buffer_execution=True, return_runtime_metrics=True,
            return_dispatch_trace=True,
        )
        resident_buffer_seconds = time.perf_counter() - started
        roots = {}
        for name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "output"):
            reference_payload = jet_exact_payload(reference[name])
            compiled_payload = jet_exact_payload(compiled[name])
            resident_fused_payload = jet_exact_payload(resident_fused[name])
            resident_warm_payload = jet_exact_payload(resident_warm[name])
            resident_buffer_payload = jet_exact_payload(resident_buffer[name])
            identical = reference_payload == compiled_payload
            resident_fused_identical = reference_payload == resident_fused_payload
            resident_warm_identical = reference_payload == resident_warm_payload
            resident_buffer_identical = reference_payload == resident_buffer_payload
            passed = (passed and identical and resident_fused_identical
                      and resident_warm_identical and resident_buffer_identical)
            roots[name] = {
                "bit_identical": identical,
                "resident_fused_contrast_bit_identical": resident_fused_identical,
                "reference_exact_payload_sha256": sha256_canonical(reference_payload),
                "compiled_exact_payload_sha256": sha256_canonical(compiled_payload),
                "resident_fused_exact_payload_sha256": sha256_canonical(
                    resident_fused_payload
                ),
                "resident_warm_static_cache_bit_identical": resident_warm_identical,
                "resident_buffer_bit_identical": resident_buffer_identical,
                "resident_buffer_exact_payload_sha256": sha256_canonical(
                    resident_buffer_payload
                ),
            }
        resident_trace_identical = (
            resident_fused["dispatch_trace"] == reference["dispatch_trace"]
        )
        passed = passed and resident_trace_identical
        resident_warm_trace_identical = (
            resident_warm["dispatch_trace"] == reference["dispatch_trace"]
        )
        passed = passed and resident_warm_trace_identical
        resident_buffer_trace_identical = (
            resident_buffer["dispatch_trace"] == reference["dispatch_trace"]
        )
        passed = passed and resident_buffer_trace_identical
        rows.append({
            "precision_bits": precision,
            "domain": {"lower": {"numerator": -1, "exponent_2": -14},
                       "upper": {"numerator": 1, "exponent_2": -14}},
            "reference_seconds": reference_seconds,
            "compiled_correctness_ffi_seconds": compiled_seconds,
            "resident_fused_contrast_ffi_seconds": resident_fused_seconds,
            "resident_warm_static_cache_ffi_seconds": resident_warm_seconds,
            "resident_buffer_ffi_seconds": resident_buffer_seconds,
            "resident_cold_over_warm_ratio": resident_fused_seconds / resident_warm_seconds,
            "roots": roots,
            "successful_dispatch_trace_sha256": reference["dispatch_trace"]["trace_sha256"],
            "program_dispatch_signature_sha256": reference["dispatch_trace"][
                "program_dispatch_signature_sha256"
            ],
            "successful_dispatch_event_count": len(reference["dispatch_trace"]["events"]),
            "resident_successful_dispatch_trace_sha256": resident_fused[
                "dispatch_trace"
            ]["trace_sha256"],
            "resident_successful_dispatch_event_count": len(
                resident_fused["dispatch_trace"]["events"]
            ),
            "resident_dispatch_trace_bit_identical": resident_trace_identical,
            "resident_runtime_metrics": resident_fused["runtime_metrics"],
            "resident_warm_dispatch_trace_bit_identical": resident_warm_trace_identical,
            "resident_warm_runtime_metrics": resident_warm["runtime_metrics"],
            "resident_buffer_dispatch_trace_bit_identical": (
                resident_buffer_trace_identical
            ),
            "resident_buffer_runtime_metrics": resident_buffer["runtime_metrics"],
        })
    report = {
        "schema_version": "green-v400-full-tensor-program-mpfr-parity-v13-resident-buffer",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if passed else "FAIL",
        "fixture": "deterministic tiny Transformer; no noun, prompt, donor, or scientific outcome",
        "contains_scientific_outcome": False,
        "program_semantic_hash": program.semantic_hash(),
        "program_node_count": len(program.nodes),
        "dimensions": dims.to_dict(),
        "dependency_mask_closure_sha256": program.resource_formula[
            "dependency_mask_closure_sha256"
        ],
        "backend_version": backend.version,
        "backend_sha256": sha256_file(library),
        "resident_plan_semantic_hash": resident_plan["resident_plan_semantic_hash"],
        "resident_plan_claim_status": resident_plan["claim_status"],
        "resident_plan_native_execution_ready": resident_plan["native_execution_ready"],
        "resident_packed_tensor_inputs_consumed": True,
        "resident_pairwise_affine_batch_abi_consumed": True,
        "resident_gelu_batch_abi_consumed": True,
        "resident_layer_norm_buffer_abi_consumed": True,
        "resident_residual_buffer_abi_consumed": True,
        "resident_attention_buffer_abi_consumed": True,
        "resident_fused_contrast_buffer_abi_consumed": True,
        "resident_attention_all_heads_abi_default_enabled": False,
        "resident_attention_all_heads_abi_decision": (
            "correctness-qualified but disabled: actual-shape benchmark showed no stable "
            "speedup and the tiny end-to-end resident path regressed"
        ),
        "resident_sparse_axis0_execution_consumed": True,
        "resident_static_row_cache_consumed": True,
        "resident_cross_cell_static_row_cache_consumed": True,
        "canonical_zero_endpoint_serialization_consumed": True,
        "python_zero_decode_short_circuit_consumed": True,
        "resident_materialized_axis0_row_count": sum(map(len, live_rows.values())),
        "dense_axis0_row_slot_count": sum(
            node.output_spec.shape[0] for node in program.nodes if node.output_spec.shape
        ),
        "rows": rows,
        "claim_scope": (
            "end-to-end correctness plus packed-input, sparse-row, fused-contrast, and "
            "identity-bound cross-cell static-row parity; "
            "the remaining JSON/FFI dispatcher is not a performance backend"
        ),
        "resident_buffer_executor": True,
        "cap_decision_authorized": False,
    }
    (output_root / "parity_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output_root": str(output_root),
                      "program_node_count": len(program.nodes)}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
