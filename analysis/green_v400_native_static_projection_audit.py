"""Outcome-blind audit of context-owned historical K/V projection buffers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
        raise RuntimeError("native projection audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("native projection audit output must be new below /mnt/sdb")
    backend = CompiledMPFRBackend(Path(args.library))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    envelope = backend.open_native_plan_envelope(
        Path(args.descriptor), Path(args.blob), descriptor_sha256=DESCRIPTOR_SHA,
        program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
        blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
        blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
    )
    contexts = []
    observations = []
    for precision in (384, 512):
        started = time.perf_counter()
        context = backend.open_native_precision_context(envelope, precision)
        elapsed = time.perf_counter() - started
        contexts.append(context)
        info = backend.native_precision_context_projection_info(context)
        selected_indices = [0, 1, info["projection_buffer_count"] - 2,
                            info["projection_buffer_count"] - 1]
        observations.append({
            "precision_bits": precision,
            "projection_info": info,
            "context_build_seconds": elapsed,
            "selected_projection_indices": selected_indices,
            "selected_projection_payload_sha256": [
                canonical_hash(backend.export_native_precision_context_projection(
                    context, index
                )) for index in selected_indices
            ],
        })
    envelope.close()
    retained = [backend.library.green_v400_native_precision_context_projection_info_v1(
        context.handle, None, None, None, None
    ) for context in contexts]
    for context in contexts:
        context.close()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    expected_info = {
        "projection_buffer_count": 48, "projection_jet_count": 36_864,
        "historical_row_count": 6, "branch_count": 4,
    }
    passed = (all(item["projection_info"] == expected_info for item in observations)
              and retained == [0, 0]
              and all(len(value) == 64 for item in observations
                      for value in item["selected_projection_payload_sha256"]))
    report = {
        "schema_version": "green-v400-native-static-projection-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "status": "PASS_NATIVE_STATIC_KV_PROJECTIONS_PREPARE_ONLY" if passed else "FAIL",
        "native_execution_ready": False,
        "context_owned_static_kv_ready": passed,
        "full_native_dispatch_ready": False,
        "backend_sha256": backend.library_sha256,
        "backend_version": backend.version,
        "descriptor_sha256": DESCRIPTOR_SHA,
        "program_execution_sha256": PROGRAM_SHA,
        "dispatch_sha256": DISPATCH_SHA,
        "blob_sha256": BLOB_SHA,
        "fusion_sha256": FUSION_SHA,
        "precision_observations": observations,
        "context_survives_envelope_close_statuses": retained,
        "process_peak_rss_before_kib": before,
        "process_peak_rss_after_kib": after,
        "process_peak_rss_delta_kib": max(0, after - before),
        "claim_scope": (
            "each precision context owns exact historical K/V buffers for four PAT/TAR "
            "branches and six pre-final rows, built from the closed f32 residual, LayerNorm, "
            "and K/V affine records; selected buffers are serialization-hashed only"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
