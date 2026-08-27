"""Outcome-blind 384/512 native overlap, repeat, and nesting audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import sys
import threading

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


ROOT_NAMES = ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "output")
COMPONENT_NAMES = ("value", "first", "second")


def _exact_fraction(payload: dict) -> Fraction:
    raw = payload["significand_hex"]
    negative = raw.startswith("-")
    significand = int(raw[1:] if negative else raw, 16)
    if negative:
        significand = -significand
    exponent = int(payload["exponent_2"])
    if exponent >= 0:
        return Fraction(significand * (1 << exponent), 1)
    return Fraction(significand, 1 << -exponent)


def _nested(high: dict, low: dict) -> list[bool]:
    checks = []
    for root in ROOT_NAMES:
        for component in COMPONENT_NAMES:
            high_interval = high[root][component]
            low_interval = low[root][component]
            checks.append(
                _exact_fraction(low_interval["lower"])
                <= _exact_fraction(high_interval["lower"])
                <= _exact_fraction(high_interval["upper"])
                <= _exact_fraction(low_interval["upper"])
            )
    return checks


def _trace_valid(payload: dict) -> bool:
    return (
        payload.get("event_count") == 81
        and tuple(payload.get("kernel_tags", ())) == tuple(EXPECTED_KERNEL_TAGS)
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
        raise RuntimeError("mixed-precision output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("mixed-precision output must be new below /mnt/sdb")

    library = Path(args.library).resolve()
    descriptor = Path(args.descriptor).resolve()
    blob = Path(args.blob).resolve()
    with ProcessTreeResourceRecorder(sample_interval_seconds=0.01) as resources:
        backend = CompiledMPFRBackend(library)
        build_options = backend.mpfr_build_options()
        envelope = backend.open_native_plan_envelope(
            descriptor, blob, descriptor_sha256=DESCRIPTOR_SHA,
            program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
            blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
            blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
        )
        contexts = {
            precision: backend.open_native_precision_context(envelope, precision)
            for precision in (384, 512)
        }
        domains = {
            precision: Interval.from_bounds(
                Fraction(-1, 2**14), Fraction(0, 1), precision,
            )
            for precision in (384, 512)
        }
        backend.reset_native_dispatch_concurrency_metrics()
        barrier = threading.Barrier(3)

        def dispatch(precision):
            barrier.wait()
            return backend.dispatch_native_precision_context_cell(
                contexts[precision], domains[precision],
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                precision: executor.submit(dispatch, precision)
                for precision in (384, 512)
            }
            barrier.wait()
            concurrent = {
                precision: futures[precision].result()
                for precision in (384, 512)
            }
        concurrency = backend.native_dispatch_concurrency_info()
        per_context = {
            precision: backend.native_precision_context_dispatch_info(context)
            for precision, context in contexts.items()
        }
        sequential = {
            precision: backend.dispatch_native_precision_context_cell(
                contexts[precision], domains[precision],
            )
            for precision in (384, 512)
        }
        repeat_equal = {
            precision: concurrent[precision] == sequential[precision]
            for precision in (384, 512)
        }
        traces_valid = {
            precision: _trace_valid(concurrent[precision])
                       and _trace_valid(sequential[precision])
            for precision in (384, 512)
        }
        nesting_checks = _nested(concurrent[512], concurrent[384])
        for context in contexts.values():
            context.close()
        envelope.close()
        del concurrent, sequential

    passed = (
        build_options["tls_enabled"]
        and concurrency == {
            "dispatch_entry_count": 2,
            "active_dispatch_count": 0,
            "peak_active_dispatch_count": 2,
        }
        and all(info == {
            "dispatch_entry_count": 1,
            "active_dispatch_count": 0,
            "peak_active_dispatch_count": 1,
        } for info in per_context.values())
        and repeat_equal == {384: True, 512: True}
        and traces_valid == {384: True, 512: True}
        and nesting_checks == [True] * 15
    )
    report = {
        "schema_version": "green-v400-native-mixed-precision-overlap-audit-v1",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "status": "PASS_MIXED_PRECISION_OVERLAP" if passed else "FAIL",
        "production_phase_order_unchanged": "ALL_384_THEN_REPLAY_SAME_PARTITION_512",
        "mixed_precision_overlap_is_production_authorized": False,
        "mpfr_build_options": build_options,
        "native_dispatch_concurrency": concurrency,
        "per_context_dispatch_concurrency_before_sequential_repeats": {
            str(key): value for key, value in per_context.items()
        },
        "concurrent_equals_sequential_after_canonical_decode": {
            str(key): value for key, value in repeat_equal.items()
        },
        "all_concurrent_and_sequential_dispatch_traces_valid": {
            str(key): value for key, value in traces_valid.items()
        },
        "precision_nesting_component_count": len(nesting_checks),
        "precision_nesting_pass_count": sum(nesting_checks),
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
            "exact-equality, precision-nesting, and trace booleans only; all "
            "native response Jet2 payloads discarded before report construction"
        ),
        "claim_scope": (
            "thread-safety stress only: one 384-bit and one 512-bit native "
            "dispatch physically overlap on independent contexts, repeat exactly "
            "under sequential execution, and preserve 512-inside-384 nesting"
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
