"""One outcome-blind cold-process sample of the native 81-node evaluator.

The process opens one precision context, evaluates one exact rational domain,
stores hashes of the five returned roots, and exits.  It deliberately never
serializes a Jet2 payload or reads a scientific threshold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
try:
    import resource
except ImportError:  # pragma: no cover - exercised by import-only tests on Windows.
    resource = None
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import canonical_json, sha256_canonical
from green_v400_native_cold_identity import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
    EXPECTED_KERNEL_TAGS,
)


ROOT_NAMES = ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "output")
DOMAIN_CLASSES = (
    "center", "negative_endpoint", "positive_endpoint", "negative_half_cell",
    "positive_half_cell", "deep_negative_dyadic", "deep_positive_dyadic",
)


def _below_mnt_sdb(path: str | Path) -> bool:
    pure = PurePosixPath(Path(path).as_posix())
    return len(pure.parts) > 3 and pure.parts[:3] == ("/", "mnt", "sdb")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _process_start_ticks() -> int:
    text = Path("/proc/self/stat").read_text(encoding="ascii", errors="strict")
    closing = text.rfind(")")
    fields = text[closing + 1:].split() if closing >= 0 else []
    if len(fields) <= 19:
        raise RuntimeError("COLD_SAMPLE_PROCESS_IDENTITY_UNAVAILABLE")
    return int(fields[19])


def _validate_domain(domain_class: str, lower: Fraction, upper: Fraction) -> None:
    if domain_class not in DOMAIN_CLASSES or lower > upper:
        raise ValueError("invalid cold-sample domain")
    valid = {
        "center": lower == upper == 0,
        "negative_endpoint": lower == upper < 0,
        "positive_endpoint": lower == upper > 0,
        "negative_half_cell": lower < 0 and upper == 0,
        "positive_half_cell": lower == 0 and upper > 0,
        "deep_negative_dyadic": lower < upper < 0,
        "deep_positive_dyadic": 0 < lower < upper,
    }[domain_class]
    if not valid:
        raise ValueError("cold-sample class/domain mismatch")


def _gpu_hidden_environment() -> dict[str, str]:
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    nvidia = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if cuda != "" or nvidia not in {"", "none", "void"}:
        raise RuntimeError("COLD_SAMPLE_GPU_VISIBILITY_NOT_DISABLED")
    return {
        "CUDA_VISIBLE_DEVICES": cuda,
        "NVIDIA_VISIBLE_DEVICES": nvidia,
        "gpu_used": False,
    }


def main() -> int:
    if os.name != "posix" or resource is None or not Path("/proc/self/stat").is_file():
        raise RuntimeError("cold cell sampling requires Linux /proc and resource limits")

    from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
    from green_bridge_v400_interval import Interval

    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--precision", type=int, choices=(384, 512), required=True)
    parser.add_argument("--lower-numerator", type=int, required=True)
    parser.add_argument("--lower-denominator", type=int, required=True)
    parser.add_argument("--upper-numerator", type=int, required=True)
    parser.add_argument("--upper-denominator", type=int, required=True)
    parser.add_argument("--domain-class", choices=DOMAIN_CLASSES, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if not _below_mnt_sdb(output) or output.exists():
        raise RuntimeError("cold sample output must be a new file below /mnt/sdb")
    if (args.lower_denominator <= 0 or args.upper_denominator <= 0
            or args.ordinal < 0
            or len(args.manifest_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in args.manifest_sha256)):
        raise ValueError("invalid cold-sample identity or rational denominator")
    lower = Fraction(args.lower_numerator, args.lower_denominator)
    upper = Fraction(args.upper_numerator, args.upper_denominator)
    _validate_domain(args.domain_class, lower, upper)
    gpu_environment = _gpu_hidden_environment()

    library = Path(args.library).resolve(strict=True)
    descriptor = Path(args.descriptor).resolve(strict=True)
    blob = Path(args.blob).resolve(strict=True)
    if _sha256_file(descriptor) != DESCRIPTOR_SHA or _sha256_file(blob) != BLOB_SHA:
        raise RuntimeError("COLD_SAMPLE_NATIVE_ARTIFACT_IDENTITY_MISMATCH")

    pid = os.getpid()
    start_ticks = _process_start_ticks()
    peak_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total_started = time.perf_counter()
    backend = CompiledMPFRBackend(library)
    envelope_started = time.perf_counter()
    envelope = backend.open_native_plan_envelope(
        descriptor, blob, descriptor_sha256=DESCRIPTOR_SHA,
        program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
        blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
        blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
    )
    envelope_seconds = time.perf_counter() - envelope_started
    context = None
    try:
        context_started = time.perf_counter()
        context = backend.open_native_precision_context(envelope, args.precision)
        context_seconds = time.perf_counter() - context_started
        domain = Interval.from_bounds(lower, upper, args.precision)
        dispatch_started = time.perf_counter()
        result = backend.dispatch_native_precision_context_cell(context, domain)
        dispatch_seconds = time.perf_counter() - dispatch_started
        if (result.get("event_count") != 81
                or result.get("kernel_tags") != EXPECTED_KERNEL_TAGS):
            raise RuntimeError("COLD_SAMPLE_NATIVE_DISPATCH_TRACE_INVALID")
        root_hashes = {
            name: sha256_canonical(result[name]) for name in ROOT_NAMES
        }
    finally:
        if context is not None:
            context.close()
        envelope.close()
    total_seconds = time.perf_counter() - total_started
    peak_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    report = {
        "schema_version": "green-v400-native-cold-cell-sample-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "report_contains_scientific_outcome": False,
        "supervisor_applied_scientific_threshold": False,
        "status": "PASS_NATIVE_COLD_CELL_SAMPLE",
        "manifest_sha256": args.manifest_sha256,
        "sample": {
            "sample_id": args.sample_id,
            "ordinal": args.ordinal,
            "precision_bits": args.precision,
            "domain_class": args.domain_class,
            "lower": [lower.numerator, lower.denominator],
            "upper": [upper.numerator, upper.denominator],
        },
        "process_identity": {"pid": pid, "start_ticks": start_ticks},
        "gpu_environment": gpu_environment,
        "native_identity": {
            "backend_sha256": backend.library_sha256,
            "backend_version": backend.version,
            "descriptor_sha256": DESCRIPTOR_SHA,
            "program_execution_sha256": PROGRAM_SHA,
            "dispatch_sha256": DISPATCH_SHA,
            "blob_sha256": BLOB_SHA,
            "fusion_sha256": FUSION_SHA,
            "kernel_tags_sha256": sha256_canonical(EXPECTED_KERNEL_TAGS),
        },
        "observations": {
            "envelope_open_seconds": envelope_seconds,
            "context_build_seconds": context_seconds,
            "cell_dispatch_seconds": dispatch_seconds,
            "total_seconds": total_seconds,
            "process_peak_rss_before_kib": peak_before_kib,
            "process_peak_rss_after_kib": peak_after_kib,
            "process_peak_rss_delta_kib": max(0, peak_after_kib - peak_before_kib),
        },
        "root_payload_sha256": root_hashes,
        "numeric_jet_payload_retained": False,
        "physical_native_dispatch_count": 1,
        "claim_scope": (
            "one fresh process, one exact rational control domain, one precision, "
            "and one successful native 81-node dispatch; only hashes and resource "
            "observations are retained"
        ),
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((canonical_json(report) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    print(canonical_json({
        "status": report["status"], "output": str(output),
        "report_semantic_hash": report["report_semantic_hash"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
