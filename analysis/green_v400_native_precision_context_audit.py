"""Outcome-blind audit of 384/512-bit native plan contexts and static buffers."""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        relative = output.relative_to(Path("/mnt/sdb").resolve())
    except ValueError as error:
        raise RuntimeError("native context audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("native context audit output must be new below /mnt/sdb")
    backend = CompiledMPFRBackend(Path(args.library))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    envelope = backend.open_native_plan_envelope(
        Path(args.descriptor), Path(args.blob), descriptor_sha256=DESCRIPTOR_SHA,
        program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
        blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
        blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
    )
    contexts = [backend.open_native_precision_context(envelope, precision)
                for precision in (384, 512)]
    elapsed = time.perf_counter() - started
    infos = [context.info for context in contexts]
    envelope_handle = envelope.handle
    envelope.close()
    retained_statuses = [backend.library.green_v400_native_precision_context_info_v1(
        context.handle, None, None, None, None, None
    ) for context in contexts]
    context_handles = [context.handle for context in contexts]
    for context in contexts:
        context.close()
    stale_statuses = [backend.library.green_v400_native_precision_context_info_v1(
        handle, None, None, None, None, None
    ) for handle in context_handles]
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    expected = [
        {"precision_bits": precision, "static_buffer_count": 5,
         "static_jet_count": 22_272, "node_count": 81, "binding_count": 150,
         "plan_retained": True}
        for precision in (384, 512)
    ]
    passed = (infos == expected and retained_statuses == [0, 0]
              and stale_statuses == [2, 2]
              and backend.library.green_v400_native_plan_envelope_info_v1(
                  envelope_handle, None, None, None, None, None, None) == 2)
    report = {
        "schema_version": "green-v400-native-precision-context-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "status": "PASS_NATIVE_PRECISION_CONTEXT_PREPARE_ONLY" if passed else "FAIL",
        "native_execution_ready": False,
        "per_precision_native_context_ready": passed,
        "full_native_dispatch_ready": False,
        "backend_sha256": backend.library_sha256,
        "backend_version": backend.version,
        "descriptor_sha256": DESCRIPTOR_SHA,
        "program_execution_sha256": PROGRAM_SHA,
        "dispatch_sha256": DISPATCH_SHA,
        "blob_sha256": BLOB_SHA,
        "fusion_sha256": FUSION_SHA,
        "context_info": infos,
        "context_survives_envelope_close_statuses": retained_statuses,
        "stale_context_statuses": stale_statuses,
        "open_plan_and_two_contexts_seconds": elapsed,
        "process_peak_rss_before_kib": before,
        "process_peak_rss_after_kib": after,
        "process_peak_rss_delta_kib": max(0, after - before),
        "claim_scope": (
            "384-bit and 512-bit generation contexts retain the typed plan/blob and own exact "
            "MPFR constant Jet buffers for zero.d_model and four PAT/TAR residual tensors; "
            "no node dispatch or scientific outcome is opened"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
