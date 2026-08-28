"""Freeze and execute the outcome-blind 60-process native cold matrix."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import canonical_json, sha256_canonical
from green_v400_native_cold_cell_sample import DOMAIN_CLASSES, ROOT_NAMES
from green_v400_native_cold_identity import (
    BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
    EXPECTED_KERNEL_TAGS,
)
from green_v400_machine_concurrency_identity import (
    collect_machine_concurrency_manifest, validate_machine_concurrency_manifest,
    verify_current_machine_concurrency_manifest,
)


PRECISION_ORDER = (384, 512)
SAMPLES_PER_PRECISION = 30
MAX_CONCURRENT_PROCESSES = 16
CATEGORY_COUNTS = {
    "center": 4,
    "negative_endpoint": 4,
    "positive_endpoint": 4,
    "negative_half_cell": 4,
    "positive_half_cell": 4,
    "deep_negative_dyadic": 5,
    "deep_positive_dyadic": 5,
}
GPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
    "gpu_used": False,
}
SOURCE_RELATIVE_PATHS = (
    "analysis/green_v400_machine_concurrency_identity.py",
    "analysis/green_v400_native_cold_identity.py",
    "analysis/green_v400_native_cold_cell_sample.py",
    "analysis/green_v400_native_cold_matrix.py",
    "scripts/run_green_shared_host.py",
    "src/green_bridge_v400_shared_host.py",
)


def _fraction_payload(value: Fraction) -> list[int]:
    value = Fraction(value)
    return [value.numerator, value.denominator]


def _below_mnt_sdb(value: str) -> bool:
    pure = PurePosixPath(value)
    return len(pure.parts) > 3 and pure.parts[:3] == ("/", "mnt", "sdb")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _domain_templates() -> tuple[dict[str, Any], ...]:
    templates: list[dict[str, Any]] = []

    def add(domain_class: str, lower: Fraction, upper: Fraction, label: str) -> None:
        templates.append({
            "domain_class": domain_class,
            "domain_label": label,
            "lower": _fraction_payload(lower),
            "upper": _fraction_payload(upper),
        })

    for repetition in range(4):
        add("center", Fraction(0), Fraction(0), f"zero-r{repetition}")
    for exponent in (0, 5, 10, 16):
        h = Fraction(1, 2**exponent)
        add("negative_endpoint", -h, -h, f"h2^-{exponent}")
    for exponent in (0, 5, 10, 16):
        h = Fraction(1, 2**exponent)
        add("positive_endpoint", h, h, f"h2^-{exponent}")
    for exponent in (0, 5, 10, 16):
        h = Fraction(1, 2**exponent)
        add("negative_half_cell", -h, Fraction(0), f"h2^-{exponent}")
    for exponent in (0, 5, 10, 16):
        h = Fraction(1, 2**exponent)
        add("positive_half_cell", Fraction(0), h, f"h2^-{exponent}")
    for exponent in (0, 4, 8, 12, 16):
        h = Fraction(1, 2**exponent)
        width = h / (2**24)
        add(
            "deep_negative_dyadic", -h, -h + width,
            f"h2^-{exponent}-depth24",
        )
    for exponent in (0, 4, 8, 12, 16):
        h = Fraction(1, 2**exponent)
        width = h / (2**24)
        add(
            "deep_positive_dyadic", h - width, h,
            f"h2^-{exponent}-depth24",
        )
    if len(templates) != SAMPLES_PER_PRECISION:
        raise AssertionError("cold domain template count is not frozen at 30")
    return tuple(templates)


def expected_jobs() -> tuple[dict[str, Any], ...]:
    jobs: list[dict[str, Any]] = []
    ordinal = 0
    for precision in PRECISION_ORDER:
        per_class: Counter[str] = Counter()
        for template in _domain_templates():
            domain_class = template["domain_class"]
            repetition = per_class[domain_class]
            per_class[domain_class] += 1
            sample_id = f"p{precision}-{ordinal:02d}-{domain_class}-r{repetition:02d}"
            relative_attempt = f"samples/p{precision}/{ordinal:02d}_{domain_class}_r{repetition:02d}"
            jobs.append({
                "ordinal": ordinal,
                "sample_id": sample_id,
                "precision_bits": precision,
                **template,
                "attempt_relative_path": relative_attempt,
                "sample_report_relative_path": f"{relative_attempt}/cold_cell_sample.json",
            })
            ordinal += 1
    return tuple(jobs)


def build_manifest(
    *, output_root: str, library_path: str, library_sha256: str,
    descriptor_path: str, blob_path: str, max_workers: int,
    per_sample_wall_seconds: float, address_space_bytes: int,
    observed_tree_memory_bytes: int, sample_interval_seconds: float,
    repository_commit: str, repository_clean: bool,
    source_sha256: dict[str, str], machine_concurrency_manifest: dict,
) -> dict:
    manifest = {
        "schema_version": "green-v400-native-cold-matrix-manifest-v1",
        "report_contains_scientific_outcome": False,
        "supervisor_applied_scientific_threshold": False,
        "output_root": output_root,
        "precision_order": list(PRECISION_ORDER),
        "samples_per_precision": SAMPLES_PER_PRECISION,
        "total_fresh_processes": 2 * SAMPLES_PER_PRECISION,
        "category_counts_per_precision": CATEGORY_COUNTS,
        "cold_definition": (
            "one newly exec'd process, one native precision context, one exact "
            "rational domain, and one physical native dispatch; OS page-cache coldness "
            "is not claimed"
        ),
        "native_inputs": {
            "library_path": library_path,
            "library_sha256": library_sha256,
            "descriptor_path": descriptor_path,
            "descriptor_sha256": DESCRIPTOR_SHA,
            "blob_path": blob_path,
            "blob_sha256": BLOB_SHA,
            "program_execution_sha256": PROGRAM_SHA,
            "dispatch_sha256": DISPATCH_SHA,
            "fusion_sha256": FUSION_SHA,
            "kernel_tags_sha256": sha256_canonical(EXPECTED_KERNEL_TAGS),
        },
        "execution_policy": {
            "max_workers": max_workers,
            "absolute_max_workers": MAX_CONCURRENT_PROCESSES,
            "per_sample_wall_seconds": per_sample_wall_seconds,
            "per_process_address_space_bytes": address_space_bytes,
            "observed_tree_memory_bytes": observed_tree_memory_bytes,
            "sample_interval_seconds": sample_interval_seconds,
            "all_384_complete_before_512_launch": True,
            "memoization_enabled": False,
        },
        "gpu_environment": GPU_ENVIRONMENT,
        "machine_concurrency_manifest": machine_concurrency_manifest,
        "jobs": list(expected_jobs()),
        "provenance": {
            "repository_commit": repository_commit,
            "repository_clean_before_manifest": repository_clean,
            "source_sha256": dict(sorted(source_sha256.items())),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    expected_fields = {
        "schema_version", "report_contains_scientific_outcome",
        "supervisor_applied_scientific_threshold", "output_root",
        "precision_order", "samples_per_precision", "total_fresh_processes",
        "category_counts_per_precision", "cold_definition", "native_inputs",
        "execution_policy", "gpu_environment", "jobs", "provenance",
        "machine_concurrency_manifest",
    }
    if set(manifest) != expected_fields:
        raise ValueError("cold matrix manifest fields mismatch")
    if (manifest["schema_version"] != "green-v400-native-cold-matrix-manifest-v1"
            or manifest["report_contains_scientific_outcome"] is not False
            or manifest["supervisor_applied_scientific_threshold"] is not False
            or not _below_mnt_sdb(manifest["output_root"])):
        raise ValueError("cold matrix manifest identity/scope invalid")
    if (manifest["precision_order"] != [384, 512]
            or manifest["samples_per_precision"] != 30
            or manifest["total_fresh_processes"] != 60
            or manifest["category_counts_per_precision"] != CATEGORY_COUNTS
            or manifest["gpu_environment"] != GPU_ENVIRONMENT
            or manifest["jobs"] != list(expected_jobs())):
        raise ValueError("cold matrix frozen sample design mismatch")
    policy = manifest["execution_policy"]
    if set(policy) != {
        "max_workers", "absolute_max_workers", "per_sample_wall_seconds",
        "per_process_address_space_bytes", "observed_tree_memory_bytes",
        "sample_interval_seconds", "all_384_complete_before_512_launch",
        "memoization_enabled",
    }:
        raise ValueError("cold matrix execution-policy fields mismatch")
    if (type(policy["max_workers"]) is not int
            or not 1 <= policy["max_workers"] <= MAX_CONCURRENT_PROCESSES
            or policy["absolute_max_workers"] != MAX_CONCURRENT_PROCESSES
            or type(policy["per_sample_wall_seconds"]) not in {int, float}
            or not math.isfinite(policy["per_sample_wall_seconds"])
            or policy["per_sample_wall_seconds"] <= 0
            or type(policy["per_process_address_space_bytes"]) is not int
            or policy["per_process_address_space_bytes"] <= 0
            or type(policy["observed_tree_memory_bytes"]) is not int
            or policy["observed_tree_memory_bytes"] <= 0
            or type(policy["sample_interval_seconds"]) not in {int, float}
            or not math.isfinite(policy["sample_interval_seconds"])
            or not 0.01 <= policy["sample_interval_seconds"] <= 60
            or policy["all_384_complete_before_512_launch"] is not True
            or policy["memoization_enabled"] is not False):
        raise ValueError("cold matrix execution policy invalid")
    native = manifest["native_inputs"]
    required_hashes = {
        "library_sha256", "descriptor_sha256", "blob_sha256",
        "program_execution_sha256", "dispatch_sha256", "fusion_sha256",
        "kernel_tags_sha256",
    }
    if set(native) != required_hashes | {"library_path", "descriptor_path", "blob_path"}:
        raise ValueError("cold matrix native-input fields mismatch")
    if (not all(_below_mnt_sdb(native[name]) for name in (
            "library_path", "descriptor_path", "blob_path"))
            or native["descriptor_sha256"] != DESCRIPTOR_SHA
            or native["blob_sha256"] != BLOB_SHA
            or native["program_execution_sha256"] != PROGRAM_SHA
            or native["dispatch_sha256"] != DISPATCH_SHA
            or native["fusion_sha256"] != FUSION_SHA
            or native["kernel_tags_sha256"] != sha256_canonical(EXPECTED_KERNEL_TAGS)
            or any(not _is_sha256(native[name]) for name in required_hashes)):
        raise ValueError("cold matrix native identities invalid")
    provenance = manifest["provenance"]
    if (set(provenance) != {
            "repository_commit", "repository_clean_before_manifest", "source_sha256"
            }
            or type(provenance["repository_clean_before_manifest"]) is not bool
            or len(provenance["repository_commit"]) not in {40, 64}
            or any(character not in "0123456789abcdef"
                   for character in provenance["repository_commit"])
            or not isinstance(provenance["source_sha256"], dict)
            or set(provenance["source_sha256"]) != set(SOURCE_RELATIVE_PATHS)
            or any(not _is_sha256(value)
                   for value in provenance["source_sha256"].values())):
        raise ValueError("cold matrix provenance invalid")
    validate_machine_concurrency_manifest(
        manifest["machine_concurrency_manifest"],
        expected_max_workers=policy["max_workers"],
        expected_absolute_max_workers=MAX_CONCURRENT_PROCESSES,
        expected_wall_seconds=policy["per_sample_wall_seconds"],
        expected_address_space_bytes=policy["per_process_address_space_bytes"],
        expected_observed_tree_memory_bytes=policy["observed_tree_memory_bytes"],
        expected_sample_interval_seconds=policy["sample_interval_seconds"],
        expected_gpu_environment=GPU_ENVIRONMENT,
        expected_backend_kind="compiled-mpfr-native",
        expected_backend_path=native["library_path"],
        expected_backend_sha256=native["library_sha256"],
        expected_backend_opened=True,
    )


def manifest_semantic_hash(manifest: dict) -> str:
    validate_manifest(manifest)
    return sha256_canonical(manifest)


def _freeze_manifest(output_root: Path, manifest: dict) -> tuple[Path, str]:
    validate_manifest(manifest)
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_hash = manifest_semantic_hash(manifest)
    path = output_root / "cold_matrix_manifest.json"
    payload = {**manifest, "manifest_semantic_hash": manifest_hash}
    with path.open("xb") as stream:
        stream.write((canonical_json(payload) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "posix":
        descriptor = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return path, manifest_hash


def _load_frozen_manifest(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = payload.pop("manifest_semantic_hash", None)
    validate_manifest(payload)
    observed_hash = sha256_canonical(payload)
    if stored_hash != observed_hash:
        raise RuntimeError("COLD_MATRIX_MANIFEST_HASH_MISMATCH")
    return payload, observed_hash


def _verify_frozen_resources(manifest: dict) -> None:
    native = manifest["native_inputs"]
    for path_key, hash_key in (
        ("library_path", "library_sha256"),
        ("descriptor_path", "descriptor_sha256"),
        ("blob_path", "blob_sha256"),
    ):
        if _sha256_file(Path(native[path_key]).resolve(strict=True)) != native[hash_key]:
            raise RuntimeError(f"COLD_MATRIX_FROZEN_RESOURCE_CHANGED:{path_key}")
    for relative_path, expected_hash in manifest["provenance"]["source_sha256"].items():
        if _sha256_file(ROOT / relative_path) != expected_hash:
            raise RuntimeError(f"COLD_MATRIX_FROZEN_SOURCE_CHANGED:{relative_path}")


def _job_command(manifest: dict, manifest_hash: str, job: dict) -> tuple[str, ...]:
    output_root = Path(manifest["output_root"])
    attempt = output_root / job["attempt_relative_path"]
    sample_output = output_root / job["sample_report_relative_path"]
    native = manifest["native_inputs"]
    policy = manifest["execution_policy"]
    lower, upper = job["lower"], job["upper"]
    return (
        sys.executable, str(ROOT / "scripts" / "run_green_shared_host.py"),
        "--storage-root", "/mnt/sdb",
        "--attempt-directory", str(attempt),
        "--cwd", str(ROOT),
        "--wall-seconds", str(policy["per_sample_wall_seconds"]),
        "--address-space-gib", str(policy["per_process_address_space_bytes"] / (1 << 30)),
        "--observed-tree-gib", str(policy["observed_tree_memory_bytes"] / (1 << 30)),
        "--sample-seconds", str(policy["sample_interval_seconds"]),
        "--", sys.executable,
        str(ROOT / "analysis" / "green_v400_native_cold_cell_sample.py"),
        "--library", native["library_path"],
        "--descriptor", native["descriptor_path"],
        "--blob", native["blob_path"],
        "--precision", str(job["precision_bits"]),
        "--lower-numerator", str(lower[0]), "--lower-denominator", str(lower[1]),
        "--upper-numerator", str(upper[0]), "--upper-denominator", str(upper[1]),
        "--domain-class", job["domain_class"],
        "--sample-id", job["sample_id"], "--ordinal", str(job["ordinal"]),
        "--manifest-sha256", manifest_hash,
        "--output", str(sample_output),
    )


def _validate_sample_report(report: dict, manifest: dict, manifest_hash: str,
                            job: dict) -> tuple[int, int]:
    expected_fields = {
        "schema_version", "created_at_utc", "report_contains_scientific_outcome",
        "supervisor_applied_scientific_threshold", "status", "manifest_sha256",
        "sample", "process_identity", "gpu_environment", "native_identity",
        "observations", "root_payload_sha256", "numeric_jet_payload_retained",
        "physical_native_dispatch_count", "claim_scope", "report_semantic_hash",
    }
    if set(report) != expected_fields:
        raise RuntimeError("COLD_SAMPLE_REPORT_FIELDS_INVALID")
    semantic_hash = report["report_semantic_hash"]
    if semantic_hash != sha256_canonical({
            key: value for key, value in report.items() if key != "report_semantic_hash"
            }):
        raise RuntimeError("COLD_SAMPLE_REPORT_HASH_INVALID")
    expected_sample = {
        "sample_id": job["sample_id"], "ordinal": job["ordinal"],
        "precision_bits": job["precision_bits"],
        "domain_class": job["domain_class"],
        "lower": job["lower"], "upper": job["upper"],
    }
    native = manifest["native_inputs"]
    native_identity = report.get("native_identity")
    expected_native_identity_fields = {
        "backend_sha256", "backend_version", "descriptor_sha256",
        "program_execution_sha256", "dispatch_sha256", "blob_sha256",
        "fusion_sha256", "kernel_tags_sha256",
    }
    observations = report.get("observations")
    expected_observation_fields = {
        "envelope_open_seconds", "context_build_seconds", "cell_dispatch_seconds",
        "total_seconds", "process_peak_rss_before_kib",
        "process_peak_rss_after_kib", "process_peak_rss_delta_kib",
    }
    root_hashes = report.get("root_payload_sha256")
    if (report["schema_version"] != "green-v400-native-cold-cell-sample-v1"
            or report["report_contains_scientific_outcome"] is not False
            or report["supervisor_applied_scientific_threshold"] is not False
            or report["status"] != "PASS_NATIVE_COLD_CELL_SAMPLE"
            or report["manifest_sha256"] != manifest_hash
            or report["sample"] != expected_sample
            or report["gpu_environment"] != GPU_ENVIRONMENT
            or report["numeric_jet_payload_retained"] is not False
            or report["physical_native_dispatch_count"] != 1
            or not isinstance(root_hashes, dict)
            or set(root_hashes) != set(ROOT_NAMES)
            or any(not _is_sha256(value) for value in root_hashes.values())
            or not isinstance(native_identity, dict)
            or set(native_identity) != expected_native_identity_fields
            or native_identity["backend_sha256"] != native["library_sha256"]
            or not isinstance(native_identity["backend_version"], str)
            or not native_identity["backend_version"]
            or native_identity["descriptor_sha256"] != DESCRIPTOR_SHA
            or native_identity["program_execution_sha256"] != PROGRAM_SHA
            or native_identity["dispatch_sha256"] != DISPATCH_SHA
            or native_identity["blob_sha256"] != BLOB_SHA
            or native_identity["fusion_sha256"] != FUSION_SHA
            or native_identity["kernel_tags_sha256"]
               != sha256_canonical(EXPECTED_KERNEL_TAGS)
            or not isinstance(observations, dict)
            or set(observations) != expected_observation_fields
            or any(type(observations[name]) not in {int, float}
                   or not math.isfinite(observations[name])
                   or observations[name] < 0 for name in expected_observation_fields)):
        raise RuntimeError("COLD_SAMPLE_REPORT_SCOPE_OR_IDENTITY_INVALID")
    identity = report["process_identity"]
    if (set(identity) != {"pid", "start_ticks"}
            or type(identity["pid"]) is not int or identity["pid"] <= 0
            or type(identity["start_ticks"]) is not int or identity["start_ticks"] <= 0):
        raise RuntimeError("COLD_SAMPLE_PROCESS_IDENTITY_INVALID")
    return identity["pid"], identity["start_ticks"]


def _run_job(manifest_path: Path, manifest_hash: str, job: dict) -> dict:
    manifest, observed_hash = _load_frozen_manifest(manifest_path)
    if observed_hash != manifest_hash:
        raise RuntimeError("COLD_MATRIX_MANIFEST_CHANGED_BEFORE_DISPATCH")
    _verify_frozen_resources(manifest)
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    command = _job_command(manifest, manifest_hash, job)
    completed = subprocess.run(
        command, cwd=ROOT, env=environment,
        text=True, encoding="utf-8", capture_output=True,
        timeout=float(manifest["execution_policy"]["per_sample_wall_seconds"]) + 45,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"COLD_MATRIX_JOB_FAILED:{job['sample_id']}:rc={completed.returncode}:"
            f"stderr={completed.stderr[-1000:]}"
        )
    sample_path = Path(manifest["output_root"]) / job["sample_report_relative_path"]
    report = json.loads(sample_path.read_text(encoding="utf-8"))
    identity = _validate_sample_report(report, manifest, manifest_hash, job)
    resource_path = sample_path.parent / "shared_host_resource_report.json"
    resource_report = json.loads(resource_path.read_text(encoding="utf-8"))
    resource_hash = resource_report.get("report_semantic_hash")
    expected_child_command = list(command[command.index("--") + 1:])
    if (resource_report.get("status") != "COMPLETED"
            or resource_report.get("command_sha256")
               != sha256_canonical(expected_child_command)
            or resource_hash != sha256_canonical({
                key: value for key, value in resource_report.items()
                if key != "report_semantic_hash"
            })
            or not resource_report.get("observations", {}).get(
                "cleanup", {}).get("cleanup_verified", False)):
        raise RuntimeError("COLD_MATRIX_SHARED_HOST_ENVELOPE_INVALID")
    return {
        "ordinal": job["ordinal"], "sample_id": job["sample_id"],
        "precision_bits": job["precision_bits"],
        "domain_class": job["domain_class"], "process_identity": list(identity),
        "sample_report_semantic_hash": report["report_semantic_hash"],
        "shared_host_report_semantic_hash": resource_report["report_semantic_hash"],
        "context_build_seconds": report["observations"]["context_build_seconds"],
        "cell_dispatch_seconds": report["observations"]["cell_dispatch_seconds"],
        "total_seconds": report["observations"]["total_seconds"],
        "peak_rss_after_kib": report["observations"]["process_peak_rss_after_kib"],
    }


def _run_precision_phase(manifest_path: Path, manifest_hash: str,
                         manifest: dict, precision: int) -> list[dict]:
    jobs = [job for job in manifest["jobs"] if job["precision_bits"] == precision]
    results: list[dict] = []
    with ThreadPoolExecutor(
        max_workers=manifest["execution_policy"]["max_workers"],
        thread_name_prefix=f"green-cold-{precision}",
    ) as executor:
        futures = {
            executor.submit(_run_job, manifest_path, manifest_hash, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["ordinal"])
    if len(results) != SAMPLES_PER_PRECISION:
        raise RuntimeError("COLD_MATRIX_PRECISION_PHASE_INCOMPLETE")
    return results


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8",
    ).strip()


def _source_hashes() -> dict[str, str]:
    paths = tuple(ROOT / relative_path for relative_path in SOURCE_RELATIVE_PATHS)
    return {path.relative_to(ROOT).as_posix(): _sha256_file(path) for path in paths}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--per-sample-wall-seconds", type=float, default=300.0)
    parser.add_argument("--address-space-gib", type=float, default=2.0)
    parser.add_argument("--observed-tree-gib", type=float, default=1.0)
    parser.add_argument("--sample-seconds", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    if not _below_mnt_sdb(output_root.as_posix()) or output_root.exists():
        raise RuntimeError("cold matrix output root must be new below /mnt/sdb")
    library = Path(args.library).resolve(strict=True)
    descriptor = Path(args.descriptor).resolve(strict=True)
    blob = Path(args.blob).resolve(strict=True)
    if not all(_below_mnt_sdb(path.as_posix()) for path in (library, descriptor, blob)):
        raise RuntimeError("cold matrix native inputs must resolve below /mnt/sdb")
    if _sha256_file(descriptor) != DESCRIPTOR_SHA or _sha256_file(blob) != BLOB_SHA:
        raise RuntimeError("COLD_MATRIX_NATIVE_INPUT_HASH_MISMATCH")

    manifest = build_manifest(
        output_root=output_root.as_posix(),
        library_path=library.as_posix(), library_sha256=_sha256_file(library),
        descriptor_path=descriptor.as_posix(), blob_path=blob.as_posix(),
        max_workers=args.max_workers,
        per_sample_wall_seconds=args.per_sample_wall_seconds,
        address_space_bytes=int(args.address_space_gib * (1 << 30)),
        observed_tree_memory_bytes=int(args.observed_tree_gib * (1 << 30)),
        sample_interval_seconds=args.sample_seconds,
        repository_commit=_git("rev-parse", "HEAD"),
        repository_clean=not bool(_git("status", "--porcelain=v1", "--untracked-files=all")),
        source_sha256=_source_hashes(),
        machine_concurrency_manifest=collect_machine_concurrency_manifest(
            max_workers=args.max_workers,
            absolute_max_workers=MAX_CONCURRENT_PROCESSES,
            wall_seconds_per_process=args.per_sample_wall_seconds,
            per_process_address_space_bytes=int(args.address_space_gib * (1 << 30)),
            observed_tree_memory_bytes=int(args.observed_tree_gib * (1 << 30)),
            sample_interval_seconds=args.sample_seconds,
            gpu_environment=GPU_ENVIRONMENT,
            backend_kind="compiled-mpfr-native", backend_path=library,
            backend_opened_by_workload=True,
        ),
    )
    manifest_path, manifest_hash = _freeze_manifest(output_root, manifest)
    if args.dry_run:
        print(canonical_json({
            "status": "DRY_RUN_MANIFEST_FROZEN", "manifest": str(manifest_path),
            "manifest_semantic_hash": manifest_hash, "job_count": len(manifest["jobs"]),
        }))
        return 0

    verify_current_machine_concurrency_manifest(
        manifest["machine_concurrency_manifest"]
    )

    all_results: list[dict] = []
    for precision in PRECISION_ORDER:
        frozen, observed_hash = _load_frozen_manifest(manifest_path)
        if observed_hash != manifest_hash:
            raise RuntimeError("COLD_MATRIX_MANIFEST_CHANGED_BETWEEN_PHASES")
        all_results.extend(_run_precision_phase(
            manifest_path, manifest_hash, frozen, precision,
        ))
    identities = [tuple(row["process_identity"]) for row in all_results]
    if len(all_results) != 60 or len(set(identities)) != 60:
        raise RuntimeError("COLD_MATRIX_FRESH_PROCESS_INVARIANT_FAILED")

    summaries = {}
    for precision in PRECISION_ORDER:
        rows = [row for row in all_results if row["precision_bits"] == precision]
        dispatch = [row["cell_dispatch_seconds"] for row in rows]
        summaries[str(precision)] = {
            "sample_count": len(rows),
            "median_cell_dispatch_seconds": statistics.median(dispatch),
            "maximum_cell_dispatch_seconds": max(dispatch),
            "maximum_total_seconds": max(row["total_seconds"] for row in rows),
            "maximum_peak_rss_after_kib": max(row["peak_rss_after_kib"] for row in rows),
            "category_counts": dict(sorted(Counter(
                row["domain_class"] for row in rows
            ).items())),
        }
    report = {
        "schema_version": "green-v400-native-cold-matrix-report-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "report_contains_scientific_outcome": False,
        "supervisor_applied_scientific_threshold": False,
        "status": "PASS_NATIVE_COLD_MATRIX",
        "manifest_semantic_hash": manifest_hash,
        "all_384_complete_before_512_launch": True,
        "fresh_process_identity_count": len(set(identities)),
        "gpu_environment": GPU_ENVIRONMENT,
        "summaries": summaries,
        "samples": all_results,
        "claim_scope": (
            "thirty cold-process observations per precision across seven frozen exact-"
            "domain classes; timing and memory remain observations, not formal bounds"
        ),
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    report_path = output_root / "cold_matrix_report.json"
    with report_path.open("xb") as stream:
        stream.write((canonical_json(report) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    print(canonical_json({
        "status": report["status"], "output": str(report_path),
        "report_semantic_hash": report["report_semantic_hash"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
