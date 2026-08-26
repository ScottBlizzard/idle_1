"""Outcome-blind resident C++ benchmark for one four-branch GPT-2 dynamic cell."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_resident_resources import (
    PRIMITIVE_TAXONOMY, gpt2_joint_witness_cell_jet2,
)
from green_bridge_v400_gpt2_program import (
    GPT2TailDimensions, validate_gpt2_joint_witness_program,
)
from green_bridge_v400_tensor_program import (
    TensorProgram, tensor_program_dispatch_signature, tensor_program_native_tags,
    tensor_program_native_trace,
)
from green_bridge_v400_tensor_store import TensorStoreReader


DIMENSIONS = {
    "d_model": 768, "d_mlp": 3072, "sequence_length": 12,
    "n_heads": 12, "d_head": 64, "selected_gates": 10,
}
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    library = Path(args.library).resolve()
    program_path = Path(args.program).resolve()
    tensor_store_path = Path(args.tensor_store).resolve()
    output = Path(args.output).resolve()
    if args.repetitions < 3:
        raise RuntimeError("at least three fixed repetitions are required")
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("benchmark output must be a new file on /mnt/sdb")
    program = TensorProgram.from_dict(json.loads(program_path.read_text(encoding="utf-8")))
    reader = TensorStoreReader(tensor_store_path)
    dims_payload = program.resource_formula.get("dimensions", {})
    dims = GPT2TailDimensions(
        int(dims_payload["sequence_length"]), int(dims_payload["d_model"]),
        int(dims_payload["d_mlp"]), int(dims_payload["n_heads"]),
        int(dims_payload["d_head"]), tuple(dims_payload["selected_gates"]),
        int(dims_payload["final_position"]), int(dims_payload["contrast_width"]),
    )
    validate_gpt2_joint_witness_program(program, reader, dims)
    signature = tensor_program_dispatch_signature(program.nodes)
    expected_native_trace = tensor_program_native_trace(program.nodes)
    expected_native_tags = tensor_program_native_tags(program.nodes)
    program_dims = program.resource_formula.get("dimensions", {})
    for key, value in DIMENSIONS.items():
        expected = len(program_dims.get("selected_gates", [])) if key == "selected_gates" else program_dims.get(key)
        if key == "sequence_length" and isinstance(expected, int) and value >= expected:
            continue
        if expected != value:
            raise RuntimeError(f"resident dimensions disagree with TensorProgram at {key}")
    backend = CompiledMPFRBackend(library)
    expected_primitive_count = gpt2_joint_witness_cell_jet2(**DIMENSIONS)
    rows, summaries = [], {}
    for precision in (384, 512):
        precision_rows = []
        for repetition in range(args.repetitions):
            result = backend.benchmark_gpt2_joint_witness_cell(
                precision, **DIMENSIONS,
            )
            row = {"repetition": repetition, **result}
            if ({"event_count": result["dispatch_event_count"],
                 "fnv1a_u64": result["dispatch_trace_fnv1a_u64"]}
                    != expected_native_trace):
                raise RuntimeError("native runtime dispatch trace disagrees with TensorProgram")
            if result["dispatch_tags"] != expected_native_tags:
                raise RuntimeError("native runtime event vector disagrees with TensorProgram")
            if result["mpfr_primitive_count"] != expected_primitive_count:
                raise RuntimeError("native primitive count disagrees with the exact resource formula")
            rows.append(row); precision_rows.append(row)
        checksums = {row["checksum"] for row in precision_rows}
        if len(checksums) != 1:
            raise RuntimeError("resident benchmark checksum is not deterministic")
        primitive_counts = {row["mpfr_primitive_count"] for row in precision_rows}
        if len(primitive_counts) != 1:
            raise RuntimeError("resident benchmark primitive count is not deterministic")
        maximum = max(row["elapsed_seconds"] for row in precision_rows)
        summaries[str(precision)] = {
            "elapsed_seconds": [row["elapsed_seconds"] for row in precision_rows],
            "maximum_seconds": maximum,
            "guardbanded_observed_max_1p25x_seconds": 1.25 * maximum,
            "mpfr_primitive_count_per_cell": next(iter(primitive_counts)),
            "checksum": next(iter(checksums)),
        }
    paired_upper = sum(row["guardbanded_observed_max_1p25x_seconds"] for row in summaries.values())
    paired_primitives = sum(row["mpfr_primitive_count_per_cell"] for row in summaries.values())
    report = {
        "schema_version": "green-v400-resident-joint-witness-cell-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "native deterministic exact-dyadic synthetic jets; public shapes only",
        "dimensions": DIMENSIONS,
        "shape_binding": {
            "tensor_program_dimensions": program_dims,
            "benchmark_relation": "same architecture and gate count; sequence_length is a conservative upper bound",
        },
        "tensor_program_path": str(program_path),
        "tensor_program_sha256": sha256_file(program_path),
        "tensor_store_manifest_path": str(tensor_store_path),
        "tensor_store_manifest_sha256": sha256_file(tensor_store_path),
        "tensor_program_semantic_hash": program.semantic_hash(),
        "tensor_program_scalarization_merkle_root": program.scalarization_merkle_root,
        "program_dispatch_signature": signature,
        "program_dispatch_signature_sha256": sha256_canonical(signature),
        "native_runtime_dispatch_trace": expected_native_trace,
        "native_runtime_dispatch_tags": expected_native_tags,
        "primitive_taxonomy": PRIMITIVE_TAXONOMY,
        "exact_primitive_formula_result_per_precision_cell": expected_primitive_count,
        "backend_version": backend.version,
        "backend_sha256": sha256_file(library),
        "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "logical_cpu_count": os.cpu_count()},
        "repetitions": args.repetitions,
        "rows": rows,
        "summaries": summaries,
        "paired_384_plus_512_one_cell_upper_seconds": paired_upper,
        "paired_384_plus_512_one_cell_mpfr_primitives": paired_primitives,
        "mandatory_two_initial_cells_upper_seconds": 2 * paired_upper,
        "mandatory_two_initial_cells_mpfr_primitives": 2 * paired_primitives,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "coverage": "resident four-branch dynamic call vector with native affine/LN/GELU/attention/residual/contrast",
        "known_exclusions": [
            "actual TensorProgram JSON dispatcher timing", "real tensor-store decode and model weights",
            "exact final-contrast static fusion startup",
            "adaptive priority queue", "endpoint and multi-radius certificate orchestration",
        ],
        "claim_status": "PASS_PROGRAM_KERNEL_ORDER_CHECKED_SYNTHETIC_DYNAMIC_CELL_ONLY",
        "full_tail_equivalent": False,
        "cap_decision_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["claim_status"], "output": str(output),
                      "summaries": summaries,
                      "mandatory_two_initial_cells_upper_seconds": 2 * paired_upper,
                      "peak_rss_kib": report["peak_rss_kib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
