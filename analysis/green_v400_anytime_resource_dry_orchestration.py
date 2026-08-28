"""Outcome-blind 17-radius dry orchestration for GREEN v4 resource policy.

This program rehearses admission, process supervision, pass charging, failure
handling, phase ordering, and canonical commit order.  Its worker never opens a
native backend, evaluates a Jet, or applies a scientific threshold.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import canonical_json, sha256_canonical
from green_v400_machine_concurrency_identity import (
    collect_machine_concurrency_manifest, validate_machine_concurrency_manifest,
    verify_current_machine_concurrency_manifest,
)


MODES = ("binding-semantic-dry", "worst-case-L14-resource-stress")
PRECISION_ORDER = (384, 512)
RADIUS_EXPONENTS = tuple(range(17))
MAX_LEAVES = 14
MAX_WORKERS = 16
SOURCE_RELATIVE_PATHS = (
    "analysis/GREEN_V400_ANYTIME_CERTIFICATE_RESOURCE_POLICY_V1_20260827.md",
    "analysis/green_v400_anytime_resource_dry_orchestration.py",
    "analysis/green_v400_machine_concurrency_identity.py",
    "scripts/run_green_shared_host.py",
    "src/green_bridge_v400_shared_host.py",
)
GPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
    "gpu_used": False,
}


def _fraction_payload(value: Fraction) -> list[int]:
    exact = Fraction(value)
    return [exact.numerator, exact.denominator]


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _below_mnt_sdb(value: str | Path) -> bool:
    pure = PurePosixPath(Path(value).as_posix())
    return len(pure.parts) > 3 and pure.parts[:3] == ("/", "mnt", "sdb")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((canonical_json(payload) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _frozen_partition(
    radius: Fraction, final_leaf_count: int,
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """Return synthetic evaluated cells and final exact dyadic leaves.

    The structural widest/lower/depth schedule is fixed only for orchestration.
    It is explicitly not represented as the scientific curvature priority.
    """
    if not 2 <= final_leaf_count <= MAX_LEAVES:
        raise ValueError("final leaf count must lie in [2, 14]")
    leaves = [
        {"lower": -radius, "upper": Fraction(0), "depth": 0},
        {"lower": Fraction(0), "upper": radius, "depth": 0},
    ]
    evaluations = [dict(row, evaluation_role="initial_half_cell") for row in leaves]
    while len(leaves) < final_leaf_count:
        parent_index = min(
            range(len(leaves)),
            key=lambda index: (
                -(leaves[index]["upper"] - leaves[index]["lower"]),
                leaves[index]["lower"], leaves[index]["depth"],
            ),
        )
        parent = leaves.pop(parent_index)
        midpoint = (parent["lower"] + parent["upper"]) / 2
        children = [
            {"lower": parent["lower"], "upper": midpoint,
             "depth": parent["depth"] + 1},
            {"lower": midpoint, "upper": parent["upper"],
             "depth": parent["depth"] + 1},
        ]
        evaluations.extend(
            dict(row, evaluation_role="selected_parent_child") for row in children
        )
        leaves.extend(children)
        leaves.sort(key=lambda row: (row["lower"], row["upper"], row["depth"]))
    return tuple(evaluations), tuple(leaves)


def _domain_job(
    *, ordinal: int, precision: int, radius_index: int, radius: Fraction,
    radius_local_pass: int, pass_kind: str, lower: Fraction, upper: Fraction,
    depth: int | None,
) -> dict:
    pass_id = f"p{precision}-r{radius_index:02d}-q{radius_local_pass:02d}-{pass_kind}"
    attempt = f"attempts/p{precision}/{ordinal:04d}_{pass_id}"
    return {
        "ordinal": ordinal,
        "pass_id": pass_id,
        "precision_bits": precision,
        "phase": "official-384" if precision == 384 else "audit-512",
        "radius_index": radius_index,
        "radius_exponent": radius_index,
        "radius": _fraction_payload(radius),
        "radius_local_pass": radius_local_pass,
        "pass_kind": pass_kind,
        "lower": _fraction_payload(lower),
        "upper": _fraction_payload(upper),
        "depth": depth,
        "attempt_relative_path": attempt,
        "receipt_relative_path": f"{attempt}/dry_pass_receipt.json",
    }


def expected_jobs(leaves_per_radius: Iterable[int]) -> tuple[dict, ...]:
    counts = tuple(leaves_per_radius)
    if len(counts) != 17 or any(
            type(count) is not int or not 2 <= count <= MAX_LEAVES
            for count in counts):
        raise ValueError("exactly 17 leaf counts in [2, 14] are required")
    jobs: list[dict] = []
    ordinal = 0
    partitions: list[tuple[dict, ...]] = []
    for radius_index, leaf_count in enumerate(counts):
        radius = Fraction(1, 2**radius_index)
        evaluated, leaves = _frozen_partition(radius, leaf_count)
        partitions.append(leaves)
        local = 0
        for cell in evaluated:
            jobs.append(_domain_job(
                ordinal=ordinal, precision=384, radius_index=radius_index,
                radius=radius, radius_local_pass=local,
                pass_kind=cell["evaluation_role"], lower=cell["lower"],
                upper=cell["upper"], depth=cell["depth"],
            ))
            ordinal += 1
            local += 1
        for pass_kind, point in (
            ("negative_endpoint", -radius), ("center", Fraction(0)),
            ("positive_endpoint", radius),
        ):
            jobs.append(_domain_job(
                ordinal=ordinal, precision=384, radius_index=radius_index,
                radius=radius, radius_local_pass=local, pass_kind=pass_kind,
                lower=point, upper=point, depth=None,
            ))
            ordinal += 1
            local += 1
    for radius_index, leaves in enumerate(partitions):
        radius = Fraction(1, 2**radius_index)
        local = 0
        for cell in leaves:
            jobs.append(_domain_job(
                ordinal=ordinal, precision=512, radius_index=radius_index,
                radius=radius, radius_local_pass=local,
                pass_kind="frozen_partition_leaf", lower=cell["lower"],
                upper=cell["upper"], depth=cell["depth"],
            ))
            ordinal += 1
            local += 1
        for pass_kind, point in (
            ("negative_endpoint", -radius), ("center", Fraction(0)),
            ("positive_endpoint", radius),
        ):
            jobs.append(_domain_job(
                ordinal=ordinal, precision=512, radius_index=radius_index,
                radius=radius, radius_local_pass=local, pass_kind=pass_kind,
                lower=point, upper=point, depth=None,
            ))
            ordinal += 1
            local += 1
    return tuple(jobs)


def build_manifest(
    *, mode: str, output_root: str, leaves_per_radius: Iterable[int],
    max_workers: int, wall_seconds: float, address_space_bytes: int,
    observed_tree_memory_bytes: int, sample_interval_seconds: float,
    fixture_seconds: float, fixture_memory_bytes: int,
    injected_failure_ordinal: int | None, repository_commit: str,
    repository_clean: bool, source_sha256: dict[str, str],
    backend_path: str, backend_sha256: str,
    machine_concurrency_manifest: dict,
) -> dict:
    counts = tuple(leaves_per_radius)
    jobs = expected_jobs(counts)
    count_384 = sum(job["precision_bits"] == 384 for job in jobs)
    count_512 = sum(job["precision_bits"] == 512 for job in jobs)
    manifest = {
        "schema_version": "green-v400-anytime-resource-dry-manifest-v1",
        "mode": mode,
        "report_contains_scientific_outcome": False,
        "scientific_certificate_authorized": False,
        "supervisor_applied_scientific_threshold": False,
        "nonbinding": mode == "worst-case-L14-resource-stress",
        "binding_resource_failure_semantics": mode == "binding-semantic-dry",
        "output_root": output_root,
        "radius_exponents": list(RADIUS_EXPONENTS),
        "radii": [_fraction_payload(Fraction(1, 2**value))
                  for value in RADIUS_EXPONENTS],
        "precision_order": list(PRECISION_ORDER),
        "leaves_per_radius": list(counts),
        "maximum_leaves_per_radius": MAX_LEAVES,
        "center_reuse": False,
        "memoization": False,
        "partition_semantics": (
            "synthetic exact-dyadic partitions frozen before execution; they rehearse "
            "resource orchestration and do not claim scientific priority equivalence"
        ),
        "phase_policy": {
            "all_384_before_any_512": True,
            "same_frozen_partition_replayed_at_512": True,
            "charge_pass_before_execution": True,
            "failed_or_timed_out_pass_not_refunded": True,
            "binding_384_resource_failure_globally_short_circuits_512": True,
            "canonical_commit_order": "ascending-manifest-ordinal",
        },
        "expected_pass_counts": {
            "384": count_384, "512": count_512, "total": len(jobs),
            "formula": "N384=sum(2L_r+1); N512=sum(L_r+3)",
        },
        "execution_policy": {
            "max_workers": max_workers,
            "absolute_max_workers": MAX_WORKERS,
            "wall_seconds_per_pass": wall_seconds,
            "per_process_address_space_bytes": address_space_bytes,
            "observed_tree_memory_bytes": observed_tree_memory_bytes,
            "sample_interval_seconds": sample_interval_seconds,
            "fixture_seconds": fixture_seconds,
            "fixture_memory_bytes": fixture_memory_bytes,
            "injected_failure_ordinal": injected_failure_ordinal,
            "external_supervisor": "scripts/run_green_shared_host.py",
        },
        "backend_artifact": {
            "path": backend_path,
            "sha256": backend_sha256,
            "opened_by_dry_worker": False,
        },
        "gpu_environment": GPU_ENVIRONMENT,
        "machine_concurrency_manifest": machine_concurrency_manifest,
        "jobs": list(jobs),
        "provenance": {
            "repository_commit": repository_commit,
            "repository_clean_before_manifest": repository_clean,
            "source_sha256": dict(sorted(source_sha256.items())),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    required = {
        "schema_version", "mode", "report_contains_scientific_outcome",
        "scientific_certificate_authorized", "supervisor_applied_scientific_threshold",
        "nonbinding", "binding_resource_failure_semantics", "output_root",
        "radius_exponents", "radii", "precision_order", "leaves_per_radius",
        "maximum_leaves_per_radius", "center_reuse", "memoization",
        "partition_semantics", "phase_policy", "expected_pass_counts",
        "execution_policy", "gpu_environment", "jobs", "provenance",
        "backend_artifact", "machine_concurrency_manifest",
    }
    if set(manifest) != required:
        raise ValueError("resource dry manifest fields mismatch")
    mode = manifest["mode"]
    counts = manifest["leaves_per_radius"]
    if (manifest["schema_version"] != "green-v400-anytime-resource-dry-manifest-v1"
            or mode not in MODES
            or manifest["report_contains_scientific_outcome"] is not False
            or manifest["scientific_certificate_authorized"] is not False
            or manifest["supervisor_applied_scientific_threshold"] is not False
            or manifest["nonbinding"] is not (mode == "worst-case-L14-resource-stress")
            or manifest["binding_resource_failure_semantics"]
               is not (mode == "binding-semantic-dry")
            or not _below_mnt_sdb(manifest["output_root"])
            or manifest["radius_exponents"] != list(range(17))
            or manifest["radii"] != [
                _fraction_payload(Fraction(1, 2**value)) for value in range(17)
            ]
            or manifest["precision_order"] != [384, 512]
            or manifest["maximum_leaves_per_radius"] != 14
            or manifest["center_reuse"] is not False
            or manifest["memoization"] is not False):
        raise ValueError("resource dry manifest scope/identity invalid")
    if (not isinstance(counts, list) or len(counts) != 17
            or any(type(value) is not int or not 2 <= value <= 14 for value in counts)
            or (mode == "worst-case-L14-resource-stress"
                and counts != [14] * 17)):
        raise ValueError("resource dry leaf design invalid")
    expected = list(expected_jobs(counts))
    if manifest["jobs"] != expected:
        raise ValueError("resource dry frozen jobs mismatch")
    count_384 = sum(job["precision_bits"] == 384 for job in expected)
    count_512 = sum(job["precision_bits"] == 512 for job in expected)
    expected_counts = manifest["expected_pass_counts"]
    if (expected_counts != {
            "384": count_384, "512": count_512, "total": len(expected),
            "formula": "N384=sum(2L_r+1); N512=sum(L_r+3)",
            }
            or (mode == "worst-case-L14-resource-stress"
                and expected_counts != {
                    "384": 493, "512": 289, "total": 782,
                    "formula": "N384=sum(2L_r+1); N512=sum(L_r+3)",
                })):
        raise ValueError("resource dry pass counts invalid")
    phase_policy = manifest["phase_policy"]
    if phase_policy != {
        "all_384_before_any_512": True,
        "same_frozen_partition_replayed_at_512": True,
        "charge_pass_before_execution": True,
        "failed_or_timed_out_pass_not_refunded": True,
        "binding_384_resource_failure_globally_short_circuits_512": True,
        "canonical_commit_order": "ascending-manifest-ordinal",
    }:
        raise ValueError("resource dry phase policy invalid")
    execution = manifest["execution_policy"]
    if set(execution) != {
        "max_workers", "absolute_max_workers", "wall_seconds_per_pass",
        "per_process_address_space_bytes", "observed_tree_memory_bytes",
        "sample_interval_seconds", "fixture_seconds", "fixture_memory_bytes",
        "injected_failure_ordinal", "external_supervisor",
    }:
        raise ValueError("resource dry execution fields mismatch")
    failure = execution["injected_failure_ordinal"]
    if (type(execution["max_workers"]) is not int
            or not 1 <= execution["max_workers"] <= MAX_WORKERS
            or execution["absolute_max_workers"] != MAX_WORKERS
            or type(execution["wall_seconds_per_pass"]) not in {int, float}
            or not math.isfinite(execution["wall_seconds_per_pass"])
            or execution["wall_seconds_per_pass"] <= 0
            or type(execution["per_process_address_space_bytes"]) is not int
            or execution["per_process_address_space_bytes"] <= 0
            or type(execution["observed_tree_memory_bytes"]) is not int
            or execution["observed_tree_memory_bytes"] <= 0
            or type(execution["sample_interval_seconds"]) not in {int, float}
            or not math.isfinite(execution["sample_interval_seconds"])
            or not 0.01 <= execution["sample_interval_seconds"] <= 60
            or type(execution["fixture_seconds"]) not in {int, float}
            or not math.isfinite(execution["fixture_seconds"])
            or execution["fixture_seconds"] < 0
            or type(execution["fixture_memory_bytes"]) is not int
            or execution["fixture_memory_bytes"] < 0
            or execution["fixture_memory_bytes"]
               > execution["observed_tree_memory_bytes"]
            or execution["fixture_memory_bytes"]
               > execution["per_process_address_space_bytes"]
            or (failure is not None and (
                type(failure) is not int or not 0 <= failure < len(expected)))
            or execution["external_supervisor"] != "scripts/run_green_shared_host.py"):
        raise ValueError("resource dry execution policy invalid")
    if manifest["gpu_environment"] != GPU_ENVIRONMENT:
        raise ValueError("resource dry GPU policy invalid")
    backend = manifest["backend_artifact"]
    if (not isinstance(backend, dict)
            or set(backend) != {"path", "sha256", "opened_by_dry_worker"}
            or not isinstance(backend["path"], str)
            or not _below_mnt_sdb(backend["path"])
            or not _is_sha256(backend["sha256"])
            or backend["opened_by_dry_worker"] is not False):
        raise ValueError("resource dry backend identity invalid")
    provenance = manifest["provenance"]
    if (set(provenance) != {
            "repository_commit", "repository_clean_before_manifest", "source_sha256"
            }
            or type(provenance["repository_clean_before_manifest"]) is not bool
            or len(provenance["repository_commit"]) not in {40, 64}
            or any(character not in "0123456789abcdef"
                   for character in provenance["repository_commit"])
            or set(provenance["source_sha256"]) != set(SOURCE_RELATIVE_PATHS)
            or any(not _is_sha256(value)
                   for value in provenance["source_sha256"].values())):
        raise ValueError("resource dry provenance invalid")
    validate_machine_concurrency_manifest(
        manifest["machine_concurrency_manifest"],
        expected_max_workers=execution["max_workers"],
        expected_absolute_max_workers=MAX_WORKERS,
        expected_wall_seconds=execution["wall_seconds_per_pass"],
        expected_address_space_bytes=execution["per_process_address_space_bytes"],
        expected_observed_tree_memory_bytes=execution["observed_tree_memory_bytes"],
        expected_sample_interval_seconds=execution["sample_interval_seconds"],
        expected_gpu_environment=GPU_ENVIRONMENT,
        expected_backend_kind="compiled-mpfr-native",
        expected_backend_path=backend["path"],
        expected_backend_sha256=backend["sha256"],
        expected_backend_opened=False,
    )


def manifest_semantic_hash(manifest: dict) -> str:
    validate_manifest(manifest)
    return sha256_canonical(manifest)


def freeze_manifest(output_root: Path, manifest: dict) -> tuple[Path, str]:
    validate_manifest(manifest)
    output_root.mkdir(parents=True, exist_ok=False)
    semantic_hash = manifest_semantic_hash(manifest)
    path = output_root / "resource_dry_manifest.json"
    _canonical_write_new(path, {**manifest, "manifest_semantic_hash": semantic_hash})
    if os.name == "posix":
        directory_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return path, semantic_hash


def load_manifest(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = payload.pop("manifest_semantic_hash", None)
    validate_manifest(payload)
    semantic_hash = sha256_canonical(payload)
    if stored_hash != semantic_hash:
        raise RuntimeError("RESOURCE_DRY_MANIFEST_HASH_MISMATCH")
    expected_path = Path(payload["output_root"]) / "resource_dry_manifest.json"
    if path.resolve() != expected_path.resolve():
        raise RuntimeError("RESOURCE_DRY_MANIFEST_PATH_MISMATCH")
    return payload, semantic_hash


def _verify_sources(manifest: dict) -> None:
    for relative_path, expected_hash in manifest["provenance"]["source_sha256"].items():
        if _sha256_file(ROOT / relative_path) != expected_hash:
            raise RuntimeError(f"RESOURCE_DRY_FROZEN_SOURCE_CHANGED:{relative_path}")
    backend = manifest["backend_artifact"]
    if _sha256_file(Path(backend["path"]).resolve(strict=True)) != backend["sha256"]:
        raise RuntimeError("RESOURCE_DRY_FROZEN_BACKEND_CHANGED")


def _worker_command(manifest: dict, manifest_hash: str, job: dict) -> tuple[str, ...]:
    output_root = Path(manifest["output_root"])
    attempt = output_root / job["attempt_relative_path"]
    receipt = output_root / job["receipt_relative_path"]
    execution = manifest["execution_policy"]
    lower, upper = job["lower"], job["upper"]
    return (
        sys.executable, str(ROOT / "scripts" / "run_green_shared_host.py"),
        "--storage-root", "/mnt/sdb", "--attempt-directory", str(attempt),
        "--cwd", str(ROOT), "--wall-seconds", str(execution["wall_seconds_per_pass"]),
        "--address-space-gib",
        str(execution["per_process_address_space_bytes"] / (1 << 30)),
        "--observed-tree-gib",
        str(execution["observed_tree_memory_bytes"] / (1 << 30)),
        "--sample-seconds", str(execution["sample_interval_seconds"]),
        "--", sys.executable, str(Path(__file__).resolve()),
        "--worker-receipt", str(receipt), "--manifest", str(
            output_root / "resource_dry_manifest.json"
        ),
        "--manifest-sha256", manifest_hash, "--pass-id", job["pass_id"],
        "--ordinal", str(job["ordinal"]), "--precision", str(job["precision_bits"]),
        "--lower-numerator", str(lower[0]), "--lower-denominator", str(lower[1]),
        "--upper-numerator", str(upper[0]), "--upper-denominator", str(upper[1]),
        "--fixture-seconds", str(execution["fixture_seconds"]),
        "--fixture-memory-bytes", str(execution["fixture_memory_bytes"]),
        "--fixture-fail", str(
            job["ordinal"] == execution["injected_failure_ordinal"]
        ).lower(),
    )


def _validate_receipt(
    report: dict, manifest_hash: str, job: dict,
) -> None:
    expected_fields = {
        "schema_version", "created_at_utc", "status", "manifest_sha256",
        "pass_id", "ordinal", "precision_bits", "lower", "upper",
        "gpu_environment", "native_backend_loaded", "jet_evaluated",
        "scientific_threshold_read", "scientific_outcome_read_or_retained",
        "fixture_observation", "report_semantic_hash",
    }
    semantic_hash = report.get("report_semantic_hash")
    if (set(report) != expected_fields
            or semantic_hash != sha256_canonical({
                key: value for key, value in report.items()
                if key != "report_semantic_hash"
            })
            or report["schema_version"] != "green-v400-resource-dry-pass-v1"
            or report["status"] != "PASS_RESOURCE_DRY_FIXTURE"
            or report["manifest_sha256"] != manifest_hash
            or report["pass_id"] != job["pass_id"]
            or report["ordinal"] != job["ordinal"]
            or report["precision_bits"] != job["precision_bits"]
            or report["lower"] != job["lower"] or report["upper"] != job["upper"]
            or report["gpu_environment"] != GPU_ENVIRONMENT
            or report["native_backend_loaded"] is not False
            or report["jet_evaluated"] is not False
            or report["scientific_threshold_read"] is not False
            or report["scientific_outcome_read_or_retained"] is not False):
        raise RuntimeError("RESOURCE_DRY_RECEIPT_INVALID")


def _run_one(manifest_path: Path, manifest_hash: str, job: dict) -> dict:
    manifest, observed_hash = load_manifest(manifest_path)
    if observed_hash != manifest_hash:
        raise RuntimeError("RESOURCE_DRY_MANIFEST_CHANGED_BEFORE_PASS")
    _verify_sources(manifest)
    command = _worker_command(manifest, manifest_hash, job)
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    execution = manifest["execution_policy"]
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, text=True, encoding="utf-8",
            capture_output=True, check=False,
            timeout=float(execution["wall_seconds_per_pass"]) + 45,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "ordinal": job["ordinal"], "pass_id": job["pass_id"],
            "precision_bits": job["precision_bits"], "status": "RESOURCE_INCONCLUSIVE",
            "reason": "OUTER_SUPERVISOR_TIMEOUT", "charged_passes": 1,
            "detail_sha256": sha256_canonical(str(error)),
        }
    resource_path = (Path(manifest["output_root"])
                     / job["attempt_relative_path"]
                     / "shared_host_resource_report.json")
    if not resource_path.is_file():
        return {
            "ordinal": job["ordinal"], "pass_id": job["pass_id"],
            "precision_bits": job["precision_bits"], "status": "RESOURCE_INCONCLUSIVE",
            "reason": "SUPERVISOR_REPORT_MISSING", "charged_passes": 1,
            "detail_sha256": sha256_canonical(completed.stderr[-1000:]),
        }
    resource_report = json.loads(resource_path.read_text(encoding="utf-8"))
    resource_hash = resource_report.get("report_semantic_hash")
    report_valid = resource_hash == sha256_canonical({
        key: value for key, value in resource_report.items()
        if key != "report_semantic_hash"
    })
    expected_child_command = list(command[command.index("--") + 1:])
    completed_cleanly = (
        completed.returncode == 0 and report_valid
        and resource_report.get("status") == "COMPLETED"
        and resource_report.get("command_sha256")
        == sha256_canonical(expected_child_command)
        and resource_report.get("observations", {}).get("cleanup", {}).get(
            "cleanup_verified", False
        )
    )
    if not completed_cleanly:
        observed_status = resource_report.get("status")
        reason = (observed_status if observed_status in {
            "WORKER_FAILED", "WALL_DEADLINE_REACHED",
            "DESCENDANT_PROCESS_POLICY_REACHED", "OBSERVED_TREE_MEMORY_REACHED",
            "SUPERVISOR_INFRASTRUCTURE_FAILED", "SUPERVISOR_CLEANUP_FAILED",
        } else "SUPERVISOR_INVALID")
        return {
            "ordinal": job["ordinal"], "pass_id": job["pass_id"],
            "precision_bits": job["precision_bits"], "status": "RESOURCE_INCONCLUSIVE",
            "reason": reason,
            "charged_passes": 1,
            "shared_host_report_semantic_hash": resource_hash,
            "detail_sha256": sha256_canonical(completed.stderr[-1000:]),
        }
    receipt_path = Path(manifest["output_root"]) / job["receipt_relative_path"]
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _validate_receipt(receipt, manifest_hash, job)
    except Exception as error:
        return {
            "ordinal": job["ordinal"], "pass_id": job["pass_id"],
            "precision_bits": job["precision_bits"], "status": "RESOURCE_INCONCLUSIVE",
            "reason": "RECEIPT_INVALID", "charged_passes": 1,
            "shared_host_report_semantic_hash": resource_hash,
            "detail_sha256": sha256_canonical(repr(error)),
        }
    return {
        "ordinal": job["ordinal"], "pass_id": job["pass_id"],
        "precision_bits": job["precision_bits"], "status": "PASS_DRY_ORCHESTRATION",
        "reason": None, "charged_passes": 1,
        "receipt_semantic_hash": receipt["report_semantic_hash"],
        "shared_host_report_semantic_hash": resource_hash,
    }


Runner = Callable[[dict], dict]


def _run_phase(
    jobs: list[dict], *, runner: Runner, max_workers: int,
    stop_admission_on_failure: bool,
) -> list[dict]:
    iterator = iter(jobs)
    active: dict[Future, dict] = {}
    completed_rows: list[dict] = []
    admission_stopped = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        def admit() -> None:
            while not admission_stopped and len(active) < max_workers:
                try:
                    job = next(iterator)
                except StopIteration:
                    return
                active[executor.submit(runner, job)] = job

        admit()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    row = {
                        "ordinal": job["ordinal"], "pass_id": job["pass_id"],
                        "precision_bits": job["precision_bits"],
                        "status": "RESOURCE_INCONCLUSIVE",
                        "reason": "ORCHESTRATOR_EXCEPTION", "charged_passes": 1,
                        "detail_sha256": sha256_canonical(repr(error)),
                    }
                completed_rows.append(row)
                if (stop_admission_on_failure
                        and row["status"] != "PASS_DRY_ORCHESTRATION"):
                    admission_stopped = True
            admit()
    completed_rows.sort(key=lambda row: row["ordinal"])
    return completed_rows


def execute_schedule(manifest: dict, runner: Runner) -> tuple[list[dict], list[dict]]:
    jobs_384 = [job for job in manifest["jobs"] if job["precision_bits"] == 384]
    jobs_512 = [job for job in manifest["jobs"] if job["precision_bits"] == 512]
    binding = manifest["mode"] == "binding-semantic-dry"
    results_384 = _run_phase(
        jobs_384, runner=runner,
        max_workers=manifest["execution_policy"]["max_workers"],
        stop_admission_on_failure=binding,
    )
    if binding and any(
            row["status"] != "PASS_DRY_ORCHESTRATION" for row in results_384):
        return results_384, []
    results_512 = _run_phase(
        jobs_512, runner=runner,
        max_workers=manifest["execution_policy"]["max_workers"],
        stop_admission_on_failure=False,
    )
    return results_384, results_512


def execute_manifest(manifest_path: Path) -> Path:
    manifest, manifest_hash = load_manifest(manifest_path)
    _verify_sources(manifest)
    verify_current_machine_concurrency_manifest(
        manifest["machine_concurrency_manifest"]
    )
    runner = lambda job: _run_one(manifest_path, manifest_hash, job)
    results_384, results_512 = execute_schedule(manifest, runner)
    all_results = results_384 + results_512
    binding_short_circuit = (
        manifest["mode"] == "binding-semantic-dry"
        and len(results_512) == 0
        and any(row["status"] != "PASS_DRY_ORCHESTRATION" for row in results_384)
    )
    all_passed = (
        len(all_results) == manifest["expected_pass_counts"]["total"]
        and all(row["status"] == "PASS_DRY_ORCHESTRATION" for row in all_results)
    )
    if manifest["mode"] == "binding-semantic-dry":
        status = (
            "PASS_BINDING_SEMANTIC_DRY_ORCHESTRATION" if all_passed
            else "RESOURCE_INCONCLUSIVE"
        )
    else:
        status = (
            "PASS_NONBINDING_WORST_CASE_L14_RESOURCE_STRESS" if all_passed
            else "FAIL_NONBINDING_WORST_CASE_L14_RESOURCE_STRESS"
        )
    report = {
        "schema_version": "green-v400-anytime-resource-dry-report-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "mode": manifest["mode"],
        "manifest_semantic_hash": manifest_hash,
        "report_contains_scientific_outcome": False,
        "scientific_certificate_authorized": False,
        "supervisor_applied_scientific_threshold": False,
        "nonbinding": manifest["nonbinding"],
        "binding_384_resource_failure_short_circuited_512": binding_short_circuit,
        "charged_pass_count": sum(row["charged_passes"] for row in all_results),
        "completed_pass_count": sum(
            row["status"] == "PASS_DRY_ORCHESTRATION" for row in all_results
        ),
        "launched_384_count": len(results_384),
        "launched_512_count": len(results_512),
        "expected_pass_counts": manifest["expected_pass_counts"],
        "canonical_commit_ordinals": [row["ordinal"] for row in all_results],
        "pass_records": all_results,
        "claim_scope": (
            "outcome-blind command, resource-accounting, external-supervisor, "
            "failure, phase-order, and canonical-commit rehearsal only; no native "
            "Jet evaluation, certificate interval, or scientific decision"
        ),
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    report_path = Path(manifest["output_root"]) / "resource_dry_report.json"
    _canonical_write_new(report_path, report)
    return report_path


def _worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-receipt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--precision", type=int, choices=(384, 512), required=True)
    parser.add_argument("--lower-numerator", type=int, required=True)
    parser.add_argument("--lower-denominator", type=int, required=True)
    parser.add_argument("--upper-numerator", type=int, required=True)
    parser.add_argument("--upper-denominator", type=int, required=True)
    parser.add_argument("--fixture-seconds", type=float, required=True)
    parser.add_argument("--fixture-memory-bytes", type=int, required=True)
    parser.add_argument("--fixture-fail", choices=("true", "false"), required=True)
    args = parser.parse_args(arguments)
    receipt = Path(args.worker_receipt).resolve()
    manifest_path = Path(args.manifest).resolve(strict=True)
    if (not _below_mnt_sdb(receipt) or receipt.exists()
            or not _below_mnt_sdb(manifest_path)):
        raise RuntimeError("resource dry worker paths must be new/below /mnt/sdb")
    manifest, observed_hash = load_manifest(manifest_path)
    if observed_hash != args.manifest_sha256:
        raise RuntimeError("resource dry worker manifest identity mismatch")
    _verify_sources(manifest)
    if not 0 <= args.ordinal < len(manifest["jobs"]):
        raise RuntimeError("resource dry worker ordinal out of range")
    job = manifest["jobs"][args.ordinal]
    lower = Fraction(args.lower_numerator, args.lower_denominator)
    upper = Fraction(args.upper_numerator, args.upper_denominator)
    expected_receipt = (Path(manifest["output_root"])
                        / job["receipt_relative_path"]).resolve()
    execution = manifest["execution_policy"]
    if (job["ordinal"] != args.ordinal or receipt != expected_receipt
            or job["pass_id"] != args.pass_id
            or job["precision_bits"] != args.precision
            or job["lower"] != _fraction_payload(lower)
            or job["upper"] != _fraction_payload(upper)
            or args.fixture_seconds != execution["fixture_seconds"]
            or args.fixture_memory_bytes != execution["fixture_memory_bytes"]
            or (args.fixture_fail == "true")
               is not (args.ordinal == execution["injected_failure_ordinal"])):
        raise RuntimeError("resource dry worker job binding mismatch")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or os.environ.get("NVIDIA_VISIBLE_DEVICES") not in {"", "none", "void"}):
        raise RuntimeError("resource dry worker GPU visibility not disabled")
    if args.fixture_fail == "true":
        return 17
    allocation = bytearray(args.fixture_memory_bytes)
    started = time.perf_counter()
    if args.fixture_seconds:
        time.sleep(args.fixture_seconds)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "green-v400-resource-dry-pass-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS_RESOURCE_DRY_FIXTURE",
        "manifest_sha256": observed_hash,
        "pass_id": args.pass_id,
        "ordinal": args.ordinal,
        "precision_bits": args.precision,
        "lower": _fraction_payload(lower), "upper": _fraction_payload(upper),
        "gpu_environment": GPU_ENVIRONMENT,
        "native_backend_loaded": False,
        "jet_evaluated": False,
        "scientific_threshold_read": False,
        "scientific_outcome_read_or_retained": False,
        "fixture_observation": {
            "elapsed_seconds": elapsed,
            "allocated_bytes": len(allocation),
        },
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    _canonical_write_new(receipt, report)
    return 0


def _parse_leaf_counts(value: str, mode: str) -> tuple[int, ...]:
    pieces = tuple(int(piece) for piece in value.split(",") if piece)
    if len(pieces) == 1:
        pieces = pieces * 17
    if mode == "worst-case-L14-resource-stress" and pieces != (14,) * 17:
        raise ValueError("worst-case stress requires exactly fourteen leaves per radius")
    if len(pieces) != 17 or any(not 2 <= value <= 14 for value in pieces):
        raise ValueError("--leaves must contain one or seventeen integers in [2,14]")
    return pieces


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8",
    ).strip()


def _source_hashes() -> dict[str, str]:
    return {
        relative_path: _sha256_file(ROOT / relative_path)
        for relative_path in SOURCE_RELATIVE_PATHS
    }


def _driver_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output-root")
    parser.add_argument("--backend")
    parser.add_argument("--leaves", default="14")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--wall-seconds", type=float, default=30.0)
    parser.add_argument("--address-space-gib", type=float, default=1.0)
    parser.add_argument("--observed-tree-gib", type=float, default=0.5)
    parser.add_argument("--sample-seconds", type=float, default=0.05)
    parser.add_argument("--fixture-seconds", type=float, default=0.0)
    parser.add_argument("--fixture-memory-mib", type=float, default=0.0)
    parser.add_argument("--inject-failure-ordinal", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-frozen-manifest")
    args = parser.parse_args(arguments)
    if args.execute_frozen_manifest:
        if any(value is not None for value in (
                args.mode, args.output_root, args.backend)) or args.dry_run:
            parser.error("--execute-frozen-manifest is exclusive with planning arguments")
        path = Path(args.execute_frozen_manifest).resolve(strict=True)
        if not _below_mnt_sdb(path):
            raise RuntimeError("frozen manifest must be below /mnt/sdb")
        report_path = execute_manifest(path)
        print(canonical_json({"status": "EXECUTED", "report": str(report_path)}))
        return 0
    if args.mode is None or args.output_root is None or args.backend is None:
        parser.error("--mode, --output-root, and --backend are required when freezing")
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or not _below_mnt_sdb(output_root):
        raise RuntimeError("resource dry output root must be new below /mnt/sdb")
    leaves = _parse_leaf_counts(args.leaves, args.mode)
    backend = Path(args.backend).resolve(strict=True)
    if not _below_mnt_sdb(backend):
        raise RuntimeError("resource dry backend must resolve below /mnt/sdb")
    backend_sha256 = _sha256_file(backend)
    address_space_bytes = int(args.address_space_gib * (1 << 30))
    observed_tree_memory_bytes = int(args.observed_tree_gib * (1 << 30))
    manifest = build_manifest(
        mode=args.mode, output_root=output_root.as_posix(),
        leaves_per_radius=leaves, max_workers=args.max_workers,
        wall_seconds=args.wall_seconds,
        address_space_bytes=address_space_bytes,
        observed_tree_memory_bytes=observed_tree_memory_bytes,
        sample_interval_seconds=args.sample_seconds,
        fixture_seconds=args.fixture_seconds,
        fixture_memory_bytes=int(args.fixture_memory_mib * (1 << 20)),
        injected_failure_ordinal=args.inject_failure_ordinal,
        repository_commit=_git("rev-parse", "HEAD"),
        repository_clean=not bool(_git(
            "status", "--porcelain=v1", "--untracked-files=all"
        )),
        source_sha256=_source_hashes(),
        backend_path=backend.as_posix(), backend_sha256=backend_sha256,
        machine_concurrency_manifest=collect_machine_concurrency_manifest(
            max_workers=args.max_workers, absolute_max_workers=MAX_WORKERS,
            wall_seconds_per_process=args.wall_seconds,
            per_process_address_space_bytes=address_space_bytes,
            observed_tree_memory_bytes=observed_tree_memory_bytes,
            sample_interval_seconds=args.sample_seconds,
            gpu_environment=GPU_ENVIRONMENT,
            backend_kind="compiled-mpfr-native", backend_path=backend,
            backend_opened_by_workload=False,
        ),
    )
    manifest_path, semantic_hash = freeze_manifest(output_root, manifest)
    if args.dry_run:
        print(canonical_json({
            "status": "DRY_RUN_MANIFEST_FROZEN", "manifest": str(manifest_path),
            "manifest_semantic_hash": semantic_hash,
            "expected_pass_counts": manifest["expected_pass_counts"],
        }))
        return 0
    report_path = execute_manifest(manifest_path)
    print(canonical_json({"status": "EXECUTED", "report": str(report_path)}))
    return 0


def main() -> int:
    if "--worker-receipt" in sys.argv[1:]:
        return _worker_main(sys.argv[1:])
    return _driver_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
