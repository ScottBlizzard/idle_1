"""Outcome-blind native context contention and close-race audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from fractions import Fraction
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_resources import ProcessTreeResourceRecorder
from green_bridge_v400_schemas import sha256_canonical
from green_v400_native_adaptive_policy_audit import _git, _sha256
from green_v400_native_payload_parser_audit import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
)
from green_v400_native_typed_plan_audit import EXPECTED_KERNEL_TAGS


def _open(backend, descriptor, blob):
    return backend.open_native_plan_envelope(
        descriptor, blob, descriptor_sha256=DESCRIPTOR_SHA,
        program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
        blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
        blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
    )


def _trace_valid(payload: dict) -> bool:
    return (
        payload.get("event_count") == 81
        and tuple(payload.get("kernel_tags", ())) == tuple(EXPECTED_KERNEL_TAGS)
    )


def _post_close_statuses(backend, handle: int) -> dict:
    output = ctypes.create_string_buffer(1024)
    return {
        "info": backend.library.green_v400_native_precision_context_info_v1(
            handle, None, None, None, None, None,
        ),
        "projection_info": (
            backend.library.green_v400_native_precision_context_projection_info_v1(
                handle, None, None, None, None,
            )
        ),
        "projection_export": (
            backend.library.green_v400_native_precision_context_projection_export_json_v1(
                handle, 0, output, len(output),
            )
        ),
        "dispatch": backend.library.green_v400_native_precision_context_dispatch_cell_v1(
            handle, b"-1", -14, b"0", 0, output, len(output),
        ),
        "second_close": (
            backend.library.green_v400_native_precision_context_close_v1(handle)
        ),
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
        raise RuntimeError("race audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("race audit output must be new below /mnt/sdb")

    library = Path(args.library).resolve()
    descriptor = Path(args.descriptor).resolve()
    blob = Path(args.blob).resolve()
    domain = Interval.from_bounds(
        Fraction(-1, 2**14), Fraction(0, 1), 384,
    )
    with ProcessTreeResourceRecorder(sample_interval_seconds=0.01) as resources:
        backend = CompiledMPFRBackend(library)
        build_options = backend.mpfr_build_options()
        envelope = _open(backend, descriptor, blob)

        contention_context = backend.open_native_precision_context(envelope, 384)
        backend.reset_native_dispatch_concurrency_metrics()
        start_barrier = threading.Barrier(3)

        def contend():
            start_barrier.wait()
            return backend.dispatch_native_precision_context_cell(
                contention_context, domain,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            contention_futures = tuple(executor.submit(contend) for _ in range(2))
            start_barrier.wait()
            contention_payloads = tuple(future.result() for future in contention_futures)
        contention_global = backend.native_dispatch_concurrency_info()
        contention_local = backend.native_precision_context_dispatch_info(
            contention_context
        )
        contention_equal = contention_payloads[0] == contention_payloads[1]
        contention_traces_valid = all(map(_trace_valid, contention_payloads))
        contention_context.close()
        del contention_payloads

        close_context = backend.open_native_precision_context(envelope, 384)
        close_handle = close_context.handle
        backend.reset_native_dispatch_concurrency_metrics()
        with ThreadPoolExecutor(max_workers=1) as executor:
            dispatch_future = executor.submit(
                backend.dispatch_native_precision_context_cell,
                close_context, domain,
            )
            poll_deadline = time.monotonic() + 5.0
            while backend.native_dispatch_concurrency_info()[
                "active_dispatch_count"
            ] != 1:
                if time.monotonic() >= poll_deadline:
                    raise RuntimeError("dispatch did not enter before close-race deadline")
                time.sleep(0.001)
            close_started = time.monotonic()
            close_status = (
                backend.library.green_v400_native_precision_context_close_v1(
                    close_handle
                )
            )
            close_elapsed = time.monotonic() - close_started
            close_context.handle = 0
            close_payload = dispatch_future.result()
        close_global = backend.native_dispatch_concurrency_info()
        close_trace_valid = _trace_valid(close_payload)
        del close_payload
        post_close_statuses = _post_close_statuses(backend, close_handle)

        double_context = backend.open_native_precision_context(envelope, 384)
        double_handle = double_context.handle
        double_barrier = threading.Barrier(3)

        def double_close():
            double_barrier.wait()
            return backend.library.green_v400_native_precision_context_close_v1(
                double_handle
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            double_futures = tuple(executor.submit(double_close) for _ in range(2))
            double_barrier.wait()
            double_close_statuses = sorted(
                future.result() for future in double_futures
            )
        double_context.handle = 0
        envelope.close()

    passed = (
        build_options["tls_enabled"]
        and contention_global == {
            "dispatch_entry_count": 2,
            "active_dispatch_count": 0,
            "peak_active_dispatch_count": 1,
        }
        and contention_local == contention_global
        and contention_equal
        and contention_traces_valid
        and close_status == 0
        and close_elapsed > 0.01
        and close_global == {
            "dispatch_entry_count": 1,
            "active_dispatch_count": 0,
            "peak_active_dispatch_count": 1,
        }
        and close_trace_valid
        and set(post_close_statuses.values()) == {2}
        and double_close_statuses == [0, 2]
    )
    report = {
        "schema_version": "green-v400-native-context-race-audit-v1",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "status": "PASS_CONTEXT_LIFECYCLE_RACES" if passed else "FAIL",
        "mpfr_build_options": build_options,
        "same_context_contention": {
            "global": contention_global,
            "context": contention_local,
            "decoded_payloads_exactly_equal": contention_equal,
            "dispatch_traces_valid": contention_traces_valid,
        },
        "dispatch_close_race": {
            "close_status": close_status,
            "close_wait_seconds": close_elapsed,
            "global": close_global,
            "dispatch_trace_valid": close_trace_valid,
            "post_close_statuses": post_close_statuses,
        },
        "concurrent_double_close_statuses": double_close_statuses,
        "process_tree_resource_record": resources.record.to_dict(),
        "provenance": {
            "repository_commit": _git("rev-parse", "HEAD"),
            "repository_clean": not bool(_git(
                "status", "--porcelain=v1", "--untracked-files=all"
            )),
            "backend_sha256": _sha256(library),
            "descriptor_sha256": _sha256(descriptor),
            "blob_sha256": _sha256(blob),
            "program_execution_sha256": PROGRAM_SHA,
            "dispatch_sha256": DISPATCH_SHA,
            "fusion_sha256": FUSION_SHA,
        },
        "retained_numeric_scope": (
            "exact-equality and dispatch-trace booleans only; native response "
            "Jet2 payloads discarded before report construction"
        ),
        "claim_scope": (
            "same-context native dispatch serialization, close waits for an "
            "already-active dispatch, stale handle rejection, and concurrent "
            "double-close linearization"
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
        "close_wait_seconds": close_elapsed,
        "report_semantic_hash": report["report_semantic_hash"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
