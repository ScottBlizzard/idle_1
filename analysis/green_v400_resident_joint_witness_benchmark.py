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


DIMENSIONS = {
    "d_model": 768, "d_mlp": 3072, "sequence_length": 12,
    "n_heads": 12, "d_head": 64, "selected_gates": 10,
}
PROGRAM_CALL_VECTOR = {
    "affine_scatter.v1": 4, "static_view.v1": 4, "pairwise_affine.v1": 30,
    "layer_norm.v1": 16, "gelu_new.v1": 8, "causal_attention.v1": 4,
    "residual_add.v1": 10, "final_contrast.v1": 4,
    "branch_linear_combination.v1": 1,
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    library, output = Path(args.library).resolve(), Path(args.output).resolve()
    if args.repetitions < 3:
        raise RuntimeError("at least three fixed repetitions are required")
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("benchmark output must be a new file on /mnt/sdb")
    backend = CompiledMPFRBackend(library)
    rows, summaries = [], {}
    for precision in (384, 512):
        precision_rows = []
        for repetition in range(args.repetitions):
            result = backend.benchmark_gpt2_joint_witness_cell(
                precision, **DIMENSIONS,
            )
            row = {"repetition": repetition, **result}
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
            "timing_upper_1p25x_seconds": 1.25 * maximum,
            "mpfr_primitive_count_per_cell": next(iter(primitive_counts)),
            "checksum": next(iter(checksums)),
        }
    paired_upper = sum(row["timing_upper_1p25x_seconds"] for row in summaries.values())
    paired_primitives = sum(row["mpfr_primitive_count_per_cell"] for row in summaries.values())
    report = {
        "schema_version": "green-v400-resident-joint-witness-cell-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "native deterministic exact-dyadic synthetic jets; public shapes only",
        "dimensions": DIMENSIONS,
        "program_call_vector": PROGRAM_CALL_VECTOR,
        "program_call_vector_sha256": sha256_canonical(PROGRAM_CALL_VECTOR),
        "backend_version": backend.version,
        "backend_sha256": sha256_file(library),
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
            "actual TensorProgram JSON dispatcher", "real tensor-store decode and model weights",
            "exact final-contrast static fusion startup",
            "adaptive priority queue", "endpoint and multi-radius certificate orchestration",
        ],
        "claim_status": "PASS_RESIDENT_SYNTHETIC_DYNAMIC_CELL_ONLY",
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
