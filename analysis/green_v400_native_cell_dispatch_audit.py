"""Outcome-blind actual-shape audit of the single-call 81-node native dispatcher."""
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
from green_bridge_v400_interval import Interval
from green_v400_native_payload_parser_audit import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
)
from green_v400_native_typed_plan_audit import EXPECTED_KERNEL_TAGS


ROOT_NAMES = ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "output")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def endpoints(backend, payload):
    return {
        component: {
            endpoint: backend.exact_fraction(payload[component][endpoint])
            for endpoint in ("lower", "upper")
        }
        for component in ("value", "first", "second")
    }


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
        raise RuntimeError("native dispatch audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("native dispatch audit output must be new below /mnt/sdb")
    backend = CompiledMPFRBackend(Path(args.library))
    envelope = backend.open_native_plan_envelope(
        Path(args.descriptor), Path(args.blob), descriptor_sha256=DESCRIPTOR_SHA,
        program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
        blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
        blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
    )
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    observations = []
    retained = {}
    for precision in (384, 512):
        context_started = time.perf_counter()
        context = backend.open_native_precision_context(envelope, precision)
        context_seconds = time.perf_counter() - context_started
        domain = Interval.from_bounds(-(2.0 ** -14), 2.0 ** -14, precision)
        dispatch_started = time.perf_counter()
        result = backend.dispatch_native_precision_context_cell(context, domain)
        dispatch_seconds = time.perf_counter() - dispatch_started
        if result.get("event_count") != 81 or result.get("kernel_tags") != EXPECTED_KERNEL_TAGS:
            raise RuntimeError("native dispatch trace disagrees with the frozen TensorProgram")
        retained[precision] = {
            name: endpoints(backend, result[name]) for name in ROOT_NAMES
        }
        observations.append({
            "precision_bits": precision,
            "context_build_seconds": context_seconds,
            "cell_dispatch_seconds": dispatch_seconds,
            "event_count": result["event_count"],
            "kernel_tags_sha256": canonical_hash(result["kernel_tags"]),
            "root_payload_sha256": {
                name: canonical_hash(result[name]) for name in ROOT_NAMES
            },
        })
        context.close()
    envelope.close()
    nesting = {}
    for name in ROOT_NAMES:
        nesting[name] = {}
        for component in ("value", "first", "second"):
            official = retained[384][name][component]
            audit = retained[512][name][component]
            nesting[name][component] = (
                official["lower"] <= audit["lower"]
                <= audit["upper"] <= official["upper"]
            )
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    passed = all(value for root in nesting.values() for value in root.values())
    report = {
        "schema_version": "green-v400-native-cell-dispatch-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "status": "PASS_NATIVE_81_NODE_CELL_DISPATCH" if passed else "FAIL",
        "native_execution_ready": passed,
        "full_native_cell_dispatch_ready": passed,
        "backend_sha256": backend.library_sha256,
        "backend_version": backend.version,
        "descriptor_sha256": DESCRIPTOR_SHA,
        "program_execution_sha256": PROGRAM_SHA,
        "dispatch_sha256": DISPATCH_SHA,
        "blob_sha256": BLOB_SHA,
        "fusion_sha256": FUSION_SHA,
        "cell_domain": {"center": "0", "radius": "2^-14"},
        "precision_observations": observations,
        "cross_precision_nesting": nesting,
        "all_root_components_nested": passed,
        "process_peak_rss_before_kib": before,
        "process_peak_rss_after_kib": after,
        "process_peak_rss_delta_kib": max(0, after - before),
        "claim_scope": (
            "one C++ entry point executes the frozen 81-node dynamic final-row graph using "
            "context-owned historical K/V buffers and returns five exact roots; only hashes, "
            "successful event identity, timing, memory, and 512-inside-384 nesting are retained"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
