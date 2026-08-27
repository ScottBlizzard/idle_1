"""Deterministic audit of close winning after context lookup but before execution lock."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_resources import ProcessTreeResourceRecorder
from green_bridge_v400_schemas import sha256_canonical
from green_v400_native_adaptive_policy_audit import _git, _sha256
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
        raise RuntimeError("close-first output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("close-first output must be new below /mnt/sdb")

    library = Path(args.library).resolve()
    descriptor = Path(args.descriptor).resolve()
    blob = Path(args.blob).resolve()
    with ProcessTreeResourceRecorder(sample_interval_seconds=0.01) as resources:
        backend = CompiledMPFRBackend(library)
        enable = backend.library.green_v400_native_audit_after_find_hook_enable_v1
        reached = backend.library.green_v400_native_audit_after_find_hook_reached_v1
        release = backend.library.green_v400_native_audit_after_find_hook_release_v1
        for function in (enable, reached, release):
            function.argtypes = []
            function.restype = ctypes.c_int
        envelope = backend.open_native_plan_envelope(
            descriptor, blob, descriptor_sha256=DESCRIPTOR_SHA,
            program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
            blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
            blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
        )
        context = backend.open_native_precision_context(envelope, 384)
        handle = context.handle
        backend.reset_native_dispatch_concurrency_metrics()
        enable_status = enable()

        def raw_dispatch():
            output_buffer = ctypes.create_string_buffer(1024)
            return backend.library.green_v400_native_precision_context_dispatch_cell_v1(
                handle, b"-1", -14, b"0", 0, output_buffer, len(output_buffer),
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(raw_dispatch)
            deadline = time.monotonic() + 5.0
            while reached() != 1:
                if time.monotonic() >= deadline:
                    raise RuntimeError("after-find audit hook was not reached")
                time.sleep(0.001)
            close_status = (
                backend.library.green_v400_native_precision_context_close_v1(handle)
            )
            context.handle = 0
            close_completed_before_hook_release = not future.done()
            release_status = release()
            dispatch_status = future.result(timeout=5.0)
        metrics = backend.native_dispatch_concurrency_info()
        stale_info_status = (
            backend.library.green_v400_native_precision_context_info_v1(
                handle, None, None, None, None, None,
            )
        )
        envelope.close()

    passed = (
        enable_status == 0
        and close_status == 0
        and close_completed_before_hook_release
        and release_status == 0
        and dispatch_status == 2
        and stale_info_status == 2
        and metrics == {
            "dispatch_entry_count": 0,
            "active_dispatch_count": 0,
            "peak_active_dispatch_count": 0,
        }
    )
    report = {
        "schema_version": "green-v400-native-close-first-hook-audit-v1",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "audit_only_backend_build": True,
        "status": "PASS_CLOSE_FIRST_PRELOCK_WAITER" if passed else "FAIL",
        "hook_enable_status": enable_status,
        "close_status": close_status,
        "close_completed_before_hook_release": close_completed_before_hook_release,
        "hook_release_status": release_status,
        "prelock_waiter_dispatch_status": dispatch_status,
        "post_close_info_status": stale_info_status,
        "native_dispatch_concurrency": metrics,
        "process_tree_resource_record": resources.record.to_dict(),
        "provenance": {
            "repository_commit": _git("rev-parse", "HEAD"),
            "repository_clean": not bool(_git(
                "status", "--porcelain=v1", "--untracked-files=all"
            )),
            "audit_backend_sha256": _sha256(library),
            "descriptor_sha256": _sha256(descriptor),
            "blob_sha256": _sha256(blob),
            "program_execution_sha256": PROGRAM_SHA,
            "dispatch_sha256": DISPATCH_SHA,
            "fusion_sha256": FUSION_SHA,
        },
        "claim_scope": (
            "deterministic audit-only interleaving: dispatch has copied the context "
            "shared_ptr but has not acquired its execution mutex; close removes and "
            "inactivates the context first; the old waiter then returns stale status 2"
        ),
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "wall_seconds": resources.record.wall_seconds,
        "peak_sampled_tree_rss_kib": resources.record.peak_sampled_tree_rss_kib,
        "report_semantic_hash": report["report_semantic_hash"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
