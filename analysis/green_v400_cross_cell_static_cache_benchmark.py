"""Outcome-blind actual-shape benchmark for identity-bound cross-cell static rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr_tensor_executor import (
    ResidentStaticRowCache, execute_tensor_program_mpfr, jet_exact_payload,
)
from green_bridge_v400_resident_plan import load_resident_plan_arrays
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--precision", required=True, type=int, choices=(384, 512))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("cache benchmark output must be a new file on /mnt/sdb")
    program = TensorProgram.from_dict(json.loads(
        Path(args.program).read_text(encoding="utf-8")
    ))
    reader = TensorStoreReader(Path(args.tensor_manifest))
    plan, arrays = load_resident_plan_arrays(Path(args.plan), program, reader)
    backend = CompiledMPFRBackend(Path(args.library))
    rows = []
    passed = True
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(output.suffix + ".cold-checkpoint")
    progress = output.with_suffix(output.suffix + ".node-progress.jsonl")
    if checkpoint.exists() or progress.exists():
        raise RuntimeError("stale benchmark checkpoint exists")

    def record_node(row: dict) -> None:
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    for precision in (args.precision,):
        domain = Interval.from_bounds(-2.0**-14, 2.0**-14, precision)
        cache = ResidentStaticRowCache.build(program, plan, backend, precision)
        started = time.perf_counter()
        cold = execute_tensor_program_mpfr(
            program, reader, domain, backend, resident_plan=plan,
            resident_arrays=arrays, resident_static_row_cache=cache,
            sparse_axis0_execution=True, return_runtime_metrics=True,
            successful_node_callback=record_node,
        )
        cold_seconds = time.perf_counter() - started
        cold_entries = cache.entry_count
        checkpoint.write_text(json.dumps({
            "schema_version": "green-v400-cross-cell-static-cache-cold-checkpoint-v1",
            "contains_scientific_outcome": False,
            "formal_wall_time_upper_bound": False,
            "cap_decision_authorized": False,
            "precision_bits": precision,
            "cold_seconds": cold_seconds,
            "resident_static_cache_entry_count": cold_entries,
            "cold_runtime_metrics": cold["runtime_metrics"],
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "COLD_COMPLETE", "precision_bits": precision,
                          "cold_seconds": cold_seconds}), flush=True)
        started = time.perf_counter()
        warm = execute_tensor_program_mpfr(
            program, reader, domain, backend, resident_plan=plan,
            resident_arrays=arrays, resident_static_row_cache=cache,
            sparse_axis0_execution=True, return_runtime_metrics=True,
        )
        warm_seconds = time.perf_counter() - started
        outputs_identical = all(
            jet_exact_payload(cold[name]) == jet_exact_payload(warm[name])
            for name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "output")
        )
        warm_metrics = warm["runtime_metrics"]
        row_pass = (
            outputs_identical and cold_entries > 0
            and cache.entry_count == cold_entries
            and warm_metrics["static_row_cache_initial_entry_count"] == cold_entries
            and warm_metrics["static_row_cache_misses_by_kernel"] == {}
            and warm_metrics["tensor_store_fallback_reads"] == 0
        )
        passed = passed and row_pass
        rows.append({
            "precision_bits": precision,
            "cold_seconds": cold_seconds,
            "warm_seconds": warm_seconds,
            "cold_over_warm_ratio": cold_seconds / warm_seconds,
            "output_bit_identity_boolean_only": outputs_identical,
            "resident_static_cache_entry_count": cold_entries,
            "cold_runtime_metrics": cold["runtime_metrics"],
            "warm_runtime_metrics": warm_metrics,
            "row_pass": row_pass,
        })
    report = {
        "schema_version": "green-v400-cross-cell-static-cache-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS_CROSS_CELL_STATIC_CACHE_OBSERVATION_ONLY" if passed else "FAIL",
        "contains_scientific_outcome": False,
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
        "program_semantic_hash": program.semantic_hash(),
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "backend_sha256": backend.library_sha256,
        "domain": {"lower": {"numerator": -1, "exponent_2": -14},
                   "upper": {"numerator": 1, "exponent_2": -14}},
        "fixture": "actual-shape closed GPT-2 synthetic tensor store; output values omitted",
        "rows": rows,
        "claim_scope": (
            "cross-cell reuse of exact static rows under program/plan/backend/precision identity; "
            "Python dispatcher remains and timings are observations, not formal bounds"
        ),
        "resident_dispatcher_complete": False,
    }
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    checkpoint.unlink()
    progress.unlink()
    print(json.dumps({
        "status": report["status"], "output": str(output),
        "timings": [{"precision_bits": row["precision_bits"],
                     "cold_seconds": row["cold_seconds"],
                     "warm_seconds": row["warm_seconds"]} for row in rows],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
