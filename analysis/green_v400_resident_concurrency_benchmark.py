"""Outcome-blind pinned-process scaling audit for resident MPFR joint-witness cells."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

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


DIMENSIONS = dict(d_model=768, d_mlp=3072, sequence_length=12,
                  n_heads=12, d_head=64, selected_gates=10)
_START_BARRIER = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_cpu_representatives(available: set[int]) -> list[tuple[int, int, int]]:
    representatives = {}
    for cpu in sorted(available):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        socket = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        representatives.setdefault((socket, core), cpu)
    return [(cpu, socket, core) for (socket, core), cpu in sorted(representatives.items())]


def run_cell(task: tuple[str, int, int]) -> dict:
    library, precision, physical_cpu = task
    os.sched_setaffinity(0, {physical_cpu})
    _START_BARRIER.wait()
    backend = CompiledMPFRBackend(Path(library))
    result = backend.benchmark_gpt2_joint_witness_cell(precision, **DIMENSIONS)
    return {**result, "physical_cpu": physical_cpu, "pid": os.getpid(),
            "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-store", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    library_path = Path(args.library).resolve()
    library, program_path = str(library_path), Path(args.program).resolve()
    tensor_store_path = Path(args.tensor_store).resolve()
    output = Path(args.output).resolve()
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
    expected_primitive_count = gpt2_joint_witness_cell_jet2(**DIMENSIONS)
    program_dims = program.resource_formula.get("dimensions", {})
    for key, value in DIMENSIONS.items():
        expected = len(program_dims.get("selected_gates", [])) if key == "selected_gates" else program_dims.get(key)
        if key == "sequence_length" and isinstance(expected, int) and value >= expected:
            continue
        if expected != value:
            raise RuntimeError(f"resident dimensions disagree with TensorProgram at {key}")
    physical_topology = physical_cpu_representatives(set(os.sched_getaffinity(0)))
    physical = [row[0] for row in physical_topology]
    worker_counts = (1, 8, 32, 64)
    if len(physical) < max(worker_counts):
        raise RuntimeError("fewer than 64 physical CPUs are available")
    rows = []
    global _START_BARRIER
    for precision in (384, 512):
        for workers in worker_counts:
            tasks = [(library, precision, physical[index]) for index in range(workers)]
            started = time.perf_counter()
            context = multiprocessing.get_context("fork")
            _START_BARRIER = context.Barrier(workers)
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                results = list(pool.map(run_cell, tasks))
            wall = time.perf_counter() - started
            checksums = {row["checksum"] for row in results}
            counts = {row["mpfr_primitive_count"] for row in results}
            if len(checksums) != 1 or len(counts) != 1:
                raise RuntimeError("concurrent resident execution is not deterministic")
            if counts != {expected_primitive_count}:
                raise RuntimeError("concurrent primitive count disagrees with the exact formula")
            traces = {(row["dispatch_event_count"], row["dispatch_trace_fnv1a_u64"])
                      for row in results}
            if traces != {(expected_native_trace["event_count"], expected_native_trace["fnv1a_u64"])}:
                raise RuntimeError("concurrent native dispatch trace disagrees with TensorProgram")
            if any(row["dispatch_tags"] != expected_native_tags for row in results):
                raise RuntimeError("concurrent native event vector disagrees with TensorProgram")
            if len({row["pid"] for row in results}) != workers:
                raise RuntimeError("concurrent tasks did not occupy unique worker processes")
            rows.append({
                "precision_bits": precision, "workers": workers,
                "wall_seconds": wall,
                "cells_per_wall_second": workers / wall,
                "maximum_worker_kernel_seconds": max(row["elapsed_seconds"] for row in results),
                "minimum_worker_kernel_seconds": min(row["elapsed_seconds"] for row in results),
                "mpfr_primitive_count_per_cell": next(iter(counts)),
                "aggregate_mpfr_primitives": workers * next(iter(counts)),
                "aggregate_mpfr_primitives_per_wall_second": workers * next(iter(counts)) / wall,
                "checksum": next(iter(checksums)),
                "sum_worker_peak_rss_kib": sum(row["worker_peak_rss_kib"] for row in results),
                "max_worker_peak_rss_kib": max(row["worker_peak_rss_kib"] for row in results),
                "physical_cpus": [row["physical_cpu"] for row in results],
                "worker_pids": [row["pid"] for row in results],
                "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            })
    report = {
        "schema_version": "green-v400-resident-concurrency-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "fixed native synthetic cell; no row tensors or outcomes",
        "dimensions": DIMENSIONS,
        "shape_binding": {
            "tensor_program_dimensions": program_dims,
            "benchmark_relation": "same architecture and gate count; sequence_length is a conservative upper bound",
        },
        "tensor_program_path": str(program_path),
        "tensor_program_file_sha256": sha256_file(program_path),
        "tensor_store_manifest_path": str(tensor_store_path),
        "tensor_store_manifest_sha256": sha256_file(tensor_store_path),
        "tensor_program_semantic_hash": program.semantic_hash(),
        "program_dispatch_signature": signature,
        "program_dispatch_signature_sha256": sha256_canonical(signature),
        "native_runtime_dispatch_trace": expected_native_trace,
        "native_runtime_dispatch_tags": expected_native_tags,
        "primitive_taxonomy": PRIMITIVE_TAXONOMY,
        "exact_primitive_formula_result_per_precision_cell": expected_primitive_count,
        "worker_counts": list(worker_counts),
        "physical_cpu_topology": [
            {"cpu": cpu, "socket": socket, "core": core}
            for cpu, socket, core in physical_topology
        ],
        "affinity_policy": "one synchronized unique process per sysfs-unique (socket, core); no SMT sibling",
        "backend_sha256": sha256_file(library_path),
        "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        "rows": rows,
        "status": "PASS_PROGRAM_KERNEL_ORDER_CHECKED_SYNTHETIC_CONCURRENCY_ONLY",
        "cap_decision_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output),
                      "summary": [{"precision_bits": row["precision_bits"],
                                   "workers": row["workers"],
                                   "wall_seconds": row["wall_seconds"],
                                   "cells_per_wall_second": row["cells_per_wall_second"]}
                                  for row in rows]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
