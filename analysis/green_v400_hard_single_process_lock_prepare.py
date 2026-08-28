"""Freeze a prepare-only source/evidence closure for the strict resource lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_schemas import canonical_json, sha256_canonical


SOURCE_RELATIVE_PATHS = (
    "analysis/GREEN_V400_ANYTIME_512_FULL_HISTORY_CORRIGENDUM_20260828.md",
    "analysis/GREEN_V400_HARD_SINGLE_PROCESS_RESOURCE_LOCK_CANDIDATE_20260828.md",
    "analysis/green_v400_hard_single_process_actual_shape_audit.py",
    "analysis/green_v400_hard_single_process_lock_prepare.py",
    "scripts/run_green_shared_host.py",
    "src/green_bridge_v400_certificate.py",
    "src/green_bridge_v400_compiled_mpfr.py",
    "src/green_bridge_v400_interval.py",
    "src/green_bridge_v400_interval_jet.py",
    "src/green_bridge_v400_mpfr.py",
    "src/green_bridge_v400_relational_graph.py",
    "src/green_bridge_v400_resources.py",
    "src/green_bridge_v400_schemas.py",
    "src/green_bridge_v400_shared_host.py",
    "src/green_bridge_v400_shared_host_exec.py",
    "src/green_bridge_v400_strict_resource_lock.py",
    "src/green_bridge_v400_transformer_ops.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8",
    ).strip()


def _below_mnt_sdb(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(Path("/mnt/sdb").resolve())
    except ValueError:
        return False
    return bool(relative.parts)


def _load_hashed_json(path: Path, hash_field: str) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.pop(hash_field, None)
    if not isinstance(stored, str) or stored != sha256_canonical(payload):
        raise RuntimeError(f"HASH_MISMATCH:{path}")
    payload[hash_field] = stored
    return payload, stored


def _validate_strict_probe(wrapper: dict, numerics: dict) -> None:
    before = numerics.get("hard_lock_readback_before")
    after = numerics.get("hard_lock_readback_after")
    if (
        wrapper.get("status") != "COMPLETED"
        or wrapper.get("observations", {}).get("exit_code") != 0
        or wrapper.get("observations", {}).get("peak_process_count") != 1
        or wrapper.get("observations", {}).get("peak_descendant_count") != 0
        or wrapper.get("guarantee_scope", {}).get(
            "hard_single_process_creation_limit") is not True
        or wrapper.get("guarantee_scope", {}).get(
            "hard_aggregate_user_space_address_space_upper_bound") is not True
        or wrapper.get("guarantee_scope", {}).get(
            "cgroup_v2_enforcement_claimed") is not False
        or numerics.get("hard_single_process_lock_verified") is not True
        or before != after
        or not isinstance(before, dict)
        or before.get("rlimit_nproc") != [1, 1]
        or before.get("rlimit_as") != [4294967296, 4294967296]
        or before.get("rlimit_core") != [0, 0]
        or before.get("threads") != 1
        or before.get("cap_sys_admin_effective") is not False
        or before.get("cap_sys_resource_effective") is not False
        or numerics.get("dispatch_count_by_precision") != {"384": 6}
        or len(numerics.get("evaluated_domains", ())) != 6
        or numerics.get("status_matches_expected") is not True
        or numerics.get("contains_scientific_outcome") is not False
        or numerics.get("scientific_threshold_applied") is not False
        or numerics.get("gpu_used") is not False
    ):
        raise RuntimeError("STRICT_ACTUAL_SHAPE_PROBE_INVALID")


def build_prepare_closure(
    *, calibration_manifest: Path, wrapper_report: Path,
    numerics_report: Path, library: Path, descriptor: Path, blob: Path,
) -> dict:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("REPOSITORY_NOT_CLEAN")
    wrapper, wrapper_hash = _load_hashed_json(
        wrapper_report, "report_semantic_hash")
    numerics, numerics_hash = _load_hashed_json(
        numerics_report, "report_semantic_hash")
    _validate_strict_probe(wrapper, numerics)
    calibration, calibration_hash = _load_hashed_json(
        calibration_manifest, "manifest_semantic_hash")
    if (
        calibration.get("scientific_threshold_application_authorized") is not False
        or calibration.get("report_contains_scientific_outcome") is not False
    ):
        raise RuntimeError("CALIBRATION_MANIFEST_SCOPE_INVALID")

    source_hashes = {
        relative: _sha256_file(ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }
    return {
        "schema_version": "green-v400-hard-single-process-lock-prepare-v1",
        "production_authorized": False,
        "protocol_freeze_authorized": False,
        "real_certificate_execution_authorized": False,
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "budget_selection_status": "PENDING_CALIBRATION_COMPLETION",
        "selected_final_leaf_budget": None,
        "resource_definition": (
            "trusted_single_process_hard_user_address_space_bound"
        ),
        "memory_enforcement": "RLIMIT_AS_hard_equal_soft",
        "process_creation_enforcement": "RLIMIT_NPROC_hard_equal_soft_one",
        "numeric_thread_environment": {
            key: "1" for key in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "cgroup_v2_enforcement_claimed": False,
        "gpu_memory_in_scope": False,
        "kernel_memory_in_scope": False,
        "worker_concurrency": 1,
        "official_precision_bits": 384,
        "audit_precision_bits": 512,
        "audit_history_policy": (
            "complete_frozen_official_split_history_independent_recurrence"
        ),
        "per_radius_no_cache_count_formula": (
            "N384=2L+1;N512=2L+1;total=4L+2"
        ),
        "provenance": {
            "repository_commit": _git("rev-parse", "HEAD"),
            "source_sha256": dict(sorted(source_hashes.items())),
            "source_closure_sha256": sha256_canonical(source_hashes),
        },
        "native_artifacts": {
            "library_path": str(library),
            "library_sha256": _sha256_file(library),
            "descriptor_path": str(descriptor),
            "descriptor_sha256": _sha256_file(descriptor),
            "blob_path": str(blob),
            "blob_sha256": _sha256_file(blob),
        },
        "calibration_manifest": {
            "path": str(calibration_manifest),
            "semantic_hash": calibration_hash,
        },
        "strict_actual_shape_probe": {
            "wrapper_report_path": str(wrapper_report),
            "wrapper_report_semantic_hash": wrapper_hash,
            "numerics_report_path": str(numerics_report),
            "numerics_report_semantic_hash": numerics_hash,
        },
        "remaining_blocker": (
            "finish frozen calibration, select the outcome-blind budget, and "
            "freeze a reviewed production protocol using this exact resource definition"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--wrapper-report", required=True)
    parser.add_argument("--numerics-report", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        key: Path(value).resolve(strict=(key != "output"))
        for key, value in vars(args).items()
    }
    output = paths.pop("output")
    if output.exists() or not _below_mnt_sdb(output):
        raise RuntimeError("prepare closure output must be new below /mnt/sdb")
    payload = build_prepare_closure(**paths)
    payload["report_semantic_hash"] = sha256_canonical(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((canonical_json(payload) + "\n").encode("utf-8"))
        stream.flush()
    print(json.dumps({
        "output": str(output),
        "repository_commit": payload["provenance"]["repository_commit"],
        "source_closure_sha256": payload["provenance"]["source_closure_sha256"],
        "report_semantic_hash": payload["report_semantic_hash"],
        "production_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
