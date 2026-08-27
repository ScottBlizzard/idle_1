"""Outcome-blind actual-file audit of the bounded native payload-table parser."""
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


DESCRIPTOR_SHA = "bc673467ac237e59e542634d38d02b8eaa12053cbb0abfc39e4dcaa6659ba3ee"
PROGRAM_SHA = "38f40999524d465b8ee58fcc8d2d1822caf9af6c36897a72bd404a8fff34fe62"
DISPATCH_SHA = "eb4c907ab4a86f3aac2fda445deed67099f2831c41e9712463688cccf1b6f008"
BLOB_SHA = "34bcd45371c08720c23f66d8f723dfc0249779e9e47eee5499c04d6064dc3560"
FUSION_SHA = "bd734f457bd3baee252af47f1c048dbd606ec15bf6a1b6533751c7bb943319c1"
BLOB_NBYTES = 28_517_632


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
        raise RuntimeError("native parser audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("native parser audit output must be new below /mnt/sdb")
    backend = CompiledMPFRBackend(Path(args.library))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    envelope = backend.open_native_plan_envelope(
        Path(args.descriptor), Path(args.blob), descriptor_sha256=DESCRIPTOR_SHA,
        program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
        blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
        blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
    )
    elapsed = time.perf_counter() - started
    info = envelope.info
    handle = envelope.handle
    envelope.close()
    stale_info_status = backend.library.green_v400_native_plan_envelope_info_v1(
        handle, None, None, None, None, None, None
    )
    stale_validation_status = (
        backend.library.green_v400_native_plan_payload_validated_v1(handle)
    )
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    passed = info == {
        "descriptor_nbytes": 241_234, "blob_nbytes": BLOB_NBYTES,
        "record_count": 32, "node_count": 81, "binding_count": 150,
        "fusion_weight_count": 768, "payload_tables_validated": True,
    } and stale_info_status == 2 and stale_validation_status == 0
    report = {
        "schema_version": "green-v400-native-payload-parser-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "status": "PASS_NATIVE_PAYLOAD_TABLE_PARSER_PREPARE_ONLY" if passed else "FAIL",
        "native_execution_ready": False,
        "native_payload_table_validation_ready": passed,
        "native_typed_plan_materialization_ready": False,
        "full_native_dispatch_ready": False,
        "backend_sha256": backend.library_sha256,
        "backend_version": backend.version,
        "descriptor_sha256": DESCRIPTOR_SHA,
        "program_execution_sha256": PROGRAM_SHA,
        "dispatch_sha256": DISPATCH_SHA,
        "blob_sha256": BLOB_SHA,
        "fusion_sha256": FUSION_SHA,
        "native_info": info,
        "stale_handle_info_status": stale_info_status,
        "stale_handle_validation_status": stale_validation_status,
        "open_hash_parse_validate_mmap_seconds": elapsed,
        "process_peak_rss_before_kib": before,
        "process_peak_rss_after_kib": after,
        "process_peak_rss_delta_kib": max(0, after - before),
        "claim_scope": (
            "bounded native canonical-binary decode and semantic validation of the actual "
            "descriptor records, nodes, roots, liveness, bindings, fusion closure, dimensions, "
            "and runtime policy; parsed values are not yet retained as a typed native plan and "
            "no node execution occurs"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
