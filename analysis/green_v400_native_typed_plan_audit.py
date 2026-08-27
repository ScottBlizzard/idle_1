"""Outcome-blind actual-file audit of retained typed native plan tables."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_v400_native_payload_parser_audit import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
)


EXPECTED_KERNEL_TAGS = [
    1, 1, 2, 4, 4, 3, 3, 5, 5, 2, 3, 7, 4, 3, 3, 3, 6, 3, 7, 4, 3,
    5, 3, 7, 4, 8, 4, 3, 3, 3, 6, 3, 7, 4, 3, 5, 3, 7, 4, 8, 1, 1, 2,
    4, 4, 3, 3, 5, 5, 2, 3, 7, 4, 3, 3, 3, 6, 3, 7, 4, 3, 5, 3, 7,
    4, 8, 4, 3, 3, 3, 6, 3, 7, 4, 3, 5, 3, 7, 4, 8, 9,
]
EXPECTED_LIVENESS_COUNTS = [
    7, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 0, 7, 1, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 7, 1, 1,
    1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 7, 1, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    root = Path("/mnt/sdb").resolve()
    try:
        relative = output.relative_to(root)
    except ValueError as error:
        raise RuntimeError("native typed-plan audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("native typed-plan audit output must be new below /mnt/sdb")
    backend = CompiledMPFRBackend(Path(args.library))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    envelope = backend.open_native_plan_envelope(
        Path(args.descriptor), Path(args.blob), descriptor_sha256=DESCRIPTOR_SHA,
        program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
        blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
        blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
    )
    trace = backend.native_plan_typed_trace(envelope)
    elapsed = time.perf_counter() - started
    info = envelope.info
    handle = envelope.handle
    envelope.close()
    stale_status = backend.library.green_v400_native_plan_typed_info_v1(
        handle, None, None, None, None, None, None
    )
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    expected_info = {
        "descriptor_nbytes": 241_234, "blob_nbytes": BLOB_NBYTES,
        "record_count": 32, "node_count": 81, "binding_count": 150,
        "fusion_weight_count": 768, "payload_tables_validated": True,
        "typed_plan_materialized": True, "liveness_row_count": 196,
        "branch_root_count": 4,
    }
    expected_trace = {
        "kernel_tags": EXPECTED_KERNEL_TAGS,
        "liveness_counts": EXPECTED_LIVENESS_COUNTS,
        "branch_root_indices": [25, 39, 65, 79], "output_root_index": 80,
    }
    passed = info == expected_info and trace == expected_trace and stale_status == 2
    report = {
        "schema_version": "green-v400-native-typed-plan-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "status": "PASS_NATIVE_TYPED_PLAN_MATERIALIZATION_PREPARE_ONLY" if passed else "FAIL",
        "native_execution_ready": False,
        "native_typed_plan_materialization_ready": passed,
        "per_precision_native_context_ready": False,
        "full_native_dispatch_ready": False,
        "backend_sha256": backend.library_sha256,
        "backend_version": backend.version,
        "descriptor_sha256": DESCRIPTOR_SHA,
        "program_execution_sha256": PROGRAM_SHA,
        "dispatch_sha256": DISPATCH_SHA,
        "blob_sha256": BLOB_SHA,
        "fusion_sha256": FUSION_SHA,
        "native_info": info,
        "typed_trace": trace,
        "stale_handle_typed_info_status": stale_status,
        "open_materialize_trace_seconds": elapsed,
        "process_peak_rss_before_kib": before,
        "process_peak_rss_after_kib": after,
        "process_peak_rss_delta_kib": max(0, after - before),
        "claim_scope": (
            "the generation handle retains typed dimensions, records, nodes, kernel attributes, "
            "parent indices, output specs, liveness rows, ordered bindings, exact dyadic fusion, "
            "four branch roots, and output root; no MPFR precision context or node execution occurs"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
