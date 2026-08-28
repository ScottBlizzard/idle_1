"""Prepare-only driver for outcome-blind actual-shape budget calibration.

The frozen native API currently lacks two required public capabilities.  This
driver therefore freezes the complete four-budget experiment and fails closed,
with a machine-readable blocker report, before launching any budget worker.
It must not be upgraded by assembling private certificate helpers here.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import canonical_json, sha256_canonical
from green_v400_machine_concurrency_identity import (
    collect_machine_concurrency_manifest, validate_machine_concurrency_manifest,
    verify_current_machine_concurrency_manifest,
)
from green_v400_native_cold_identity import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
    EXPECTED_KERNEL_TAGS,
)


CANDIDATE_FINAL_LEAF_BUDGETS = (4, 8, 16, 32)
RADIUS = Fraction(1, 2**14)
RADIUS_EXPONENTS = tuple(range(17))
OFFICIAL_PRECISION = 384
AUDIT_PRECISION = 512
ROW_HASH = hashlib.sha256(
    b"green-v400-closed-synthetic-native-adaptive-policy-audit-v1"
).hexdigest()
GPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
    "gpu_used": False,
}
BLOCKER_SYNTHETIC_BOUNDARY = (
    "COMPILED_NATIVE_EVALUATOR_SYNTHETIC_ONLY_BOUNDARY_MISSING"
)
BLOCKER_AUDIT_REPLAY = (
    "PUBLIC_ANYTIME_FROZEN_PARTITION_512_REPLAY_AND_NESTING_API_MISSING"
)
SOURCE_RELATIVE_PATHS = (
    "analysis/GREEN_V400_ANYTIME_CERTIFICATE_RESOURCE_POLICY_V1_20260827.md",
    "analysis/green_v400_actual_shape_budget_calibration.py",
    "analysis/green_v400_machine_concurrency_identity.py",
    "analysis/green_v400_native_cold_identity.py",
    "scripts/run_green_shared_host.py",
    "src/green_bridge_v400_certificate.py",
    "src/green_bridge_v400_compiled_mpfr.py",
    "src/green_bridge_v400_schemas.py",
    "src/green_bridge_v400_shared_host.py",
)


def _below_mnt_sdb(value: str | Path) -> bool:
    pure = PurePosixPath(Path(value).as_posix())
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


def _canonical_write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((canonical_json(payload) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _module_tree(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def _class_has_true_assignment(tree: ast.Module, class_name: str,
                               field_name: str) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for statement in node.body:
                if (isinstance(statement, ast.Assign)
                        and any(isinstance(target, ast.Name)
                                and target.id == field_name
                                for target in statement.targets)):
                    try:
                        return ast.literal_eval(statement.value) is True
                    except (ValueError, TypeError):
                        return False
    return False


def _public_function_names(tree: ast.Module) -> set[str]:
    return {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def inspect_required_api_capabilities() -> dict:
    compiled_tree = _module_tree("src/green_bridge_v400_compiled_mpfr.py")
    certificate_tree = _module_tree("src/green_bridge_v400_certificate.py")
    public = _public_function_names(certificate_tree)
    official_api = {
        "initialize_monotone_anytime_state", "advance_monotone_anytime_state",
        "serialize_monotone_anytime_state", "restore_monotone_anytime_state",
    }
    replay_names = {
        "replay_monotone_anytime_state_at_audit_precision",
        "audit_monotone_anytime_frozen_partition",
    }
    synthetic_boundary = _class_has_true_assignment(
        compiled_tree, "CompiledSyntheticNativeJointWitnessEvaluator", "synthetic_only",
    )
    public_replay = bool(public & replay_names)
    blockers = []
    if not synthetic_boundary:
        blockers.append(BLOCKER_SYNTHETIC_BOUNDARY)
    if not public_replay:
        blockers.append(BLOCKER_AUDIT_REPLAY)
    return {
        "schema_version": "green-v400-budget-calibration-api-preflight-v1",
        "official_anytime_state_api_present": official_api <= public,
        "compiled_native_explicit_synthetic_only_boundary": synthetic_boundary,
        "public_512_frozen_partition_replay_and_nesting_api": public_replay,
        "private_legacy_same_partition_helper_is_not_authorized": True,
        "blockers": blockers,
        "execution_ready": not blockers and official_api <= public,
    }


def _budget_jobs(output_root: str) -> tuple[dict, ...]:
    jobs = []
    for ordinal, leaf_budget in enumerate(CANDIDATE_FINAL_LEAF_BUDGETS):
        maximum_384 = 17 * (2 * leaf_budget + 1)
        maximum_512 = 17 * (leaf_budget + 3)
        attempt = f"attempts/{ordinal:02d}_L{leaf_budget}"
        jobs.append({
            "ordinal": ordinal,
            "leaf_budget": leaf_budget,
            "radius_exponents": list(RADIUS_EXPONENTS),
            "official_precision_bits": OFFICIAL_PRECISION,
            "audit_precision_bits": AUDIT_PRECISION,
            "maximum_charged_passes_384": maximum_384,
            "maximum_charged_passes_512": maximum_512,
            "maximum_charged_passes_total": maximum_384 + maximum_512,
            "count_formula": (
                "per-radius N384=2L_r+1,N512=L_r+3,total=3L_r+4; "
                "maximum uses L_r=B for all 17 radii"
            ),
            "attempt_relative_path": attempt,
            "selector_safe_resource_relative_path": (
                f"{attempt}/selector_safe/resource_report.json"
            ),
            "selector_inaccessible_numerics_relative_path": (
                f"{attempt}/numerics_audit/numerics_report.json"
            ),
        })
    return tuple(jobs)


def exact_no_cache_counts(achieved_leaves: list[int] | tuple[int, ...]) -> dict:
    leaves = tuple(achieved_leaves)
    if (len(leaves) != 17 or any(type(value) is not int or value < 2
                                 for value in leaves)):
        raise ValueError("exactly 17 achieved leaf counts >=2 are required")
    official = sum(2 * value + 1 for value in leaves)
    audit = sum(value + 3 for value in leaves)
    return {"384": official, "512": audit, "total": official + audit}


def select_largest_resource_safe_budget(
    manifest: dict, selector_safe_records: list[dict],
) -> int | None:
    """Apply the frozen resource-only selector; reject all extra fields."""
    validate_manifest(manifest)
    required = {
        "budget", "charged_pass_counts", "fault_reason", "timing_seconds",
        "rss_bytes", "machine_manifest_hash",
    }
    if len(selector_safe_records) != 4:
        raise ValueError("selector requires four resource-only records")
    by_budget = {}
    for record in selector_safe_records:
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError("selector-safe resource record fields invalid")
        budget = record["budget"]
        counts = record["charged_pass_counts"]
        if (budget not in CANDIDATE_FINAL_LEAF_BUDGETS or budget in by_budget
                or not isinstance(counts, dict)
                or set(counts) != {"384", "512", "total"}
                or any(type(value) is not int or value < 0 for value in counts.values())
                or counts["total"] != counts["384"] + counts["512"]
                or record["fault_reason"] is not None
                   and record["fault_reason"] not in {
                       "WALL_DEADLINE_REACHED", "OBSERVED_TREE_MEMORY_REACHED",
                       "WORKER_FAILED", "SUPERVISOR_INFRASTRUCTURE_FAILED",
                       "SUPERVISOR_CLEANUP_FAILED",
                   }
                or type(record["timing_seconds"]) not in {int, float}
                or record["timing_seconds"] < 0
                or type(record["rss_bytes"]) is not int or record["rss_bytes"] < 0
                or record["machine_manifest_hash"] != manifest[
                    "machine_concurrency_manifest"
                ]["machine_manifest_semantic_hash"]):
            raise ValueError("selector-safe resource record invalid")
        by_budget[budget] = record
    execution = manifest["execution_policy"]
    time_limit = execution["wall_seconds_per_budget"] * Fraction(4, 5)
    memory_limit = execution["observed_tree_memory_bytes"] * Fraction(4, 5)
    safe = [
        budget for budget in CANDIDATE_FINAL_LEAF_BUDGETS
        if by_budget[budget]["fault_reason"] is None
        and by_budget[budget]["timing_seconds"] <= time_limit
        and by_budget[budget]["rss_bytes"] <= memory_limit
    ]
    return max(safe, default=None)


def build_manifest(
    *, output_root: str, library_path: str, library_sha256: str,
    descriptor_path: str, blob_path: str, max_depth: int,
    wall_seconds_per_budget: float, address_space_bytes: int,
    observed_tree_memory_bytes: int, sample_interval_seconds: float,
    repository_commit: str, repository_clean: bool,
    source_sha256: dict[str, str], machine_concurrency_manifest: dict,
    api_preflight: dict,
) -> dict:
    jobs = _budget_jobs(output_root)
    manifest = {
        "schema_version": "green-v400-actual-shape-budget-calibration-manifest-v1",
        "report_contains_scientific_outcome": False,
        "scientific_threshold_application_authorized": False,
        "calibration_is_binding_production_result": False,
        "output_root": output_root,
        "closed_synthetic_identity": {
            "row_hash": ROW_HASH,
            "split": "synthetic",
            "actual_shape": True,
            "continuation_anchor_radius": [RADIUS.numerator, RADIUS.denominator],
            "standalone_radius_exponents": list(RADIUS_EXPONENTS),
        },
        "dual_track_policy": {
            "continuation_to_32_anchor_radius": [RADIUS.numerator, RADIUS.denominator],
            "continuation_prefix_leaf_counts": list(CANDIDATE_FINAL_LEAF_BUDGETS),
            "continuation_same_path_required": True,
            "standalone_fresh_process_per_budget": True,
            "standalone_all_17_radii": True,
        },
        "candidate_final_leaf_budgets": list(CANDIDATE_FINAL_LEAF_BUDGETS),
        "budget_order_frozen_before_observations": True,
        "official_precision_bits": OFFICIAL_PRECISION,
        "audit_precision_bits": AUDIT_PRECISION,
        "adaptive_semantics": {
            "initial_partition": "[-h,0],[0,h]",
            "split_policy": "curvature-weighted width priority dyadic bisection",
            "absolute_width_tolerance": "0x1p-80",
            "relative_width_tolerance": "0x1p-40",
            "max_depth": max_depth,
            "stop_at_tolerance_or_leaf_budget": True,
            "raw_to_monotone_intersection_must_not_expand": True,
            "audit_replays_exact_frozen_official_partition": True,
            "audit_512_must_nest_inside_official_384": True,
        },
        "accounting_policy": {
            "charge_before_native_dispatch": True,
            "failed_timeout_or_dead_process_not_refunded": True,
            "center_reuse": False,
            "memoization": False,
            "fresh_isolated_process_per_budget": True,
            "per_budget_all_17_radii_384_before_any_512": True,
            "any_384_failure_launches_zero_512": True,
            "achieved_leaves_may_be_below_budget_on_tolerance": True,
            "per_radius_no_cache_count_formula": "384=2L+1;512=L+3;total=3L+4",
        },
        "artifact_separation_policy": {
            "selector_safe_resource_artifact": {
                "allowed_fields": [
                    "budget", "charged_pass_counts", "fault_reason",
                    "timing_seconds", "rss_bytes", "machine_manifest_hash",
                ],
                "interval_width_forbidden": True,
                "certificate_status_forbidden": True,
                "state_or_split_hash_payload_forbidden": True,
            },
            "selector_inaccessible_numerics_artifact": {
                "exact_domains_allowed": True,
                "exact_interval_widths_allowed": True,
                "nesting_booleans_allowed": True,
                "split_and_state_hashes_allowed": True,
                "jet_payload_forbidden": True,
                "scientific_threshold_forbidden": True,
            },
        },
        "selector_policy": {
            "inputs": "selector-safe-resource-artifacts-only",
            "rule": "largest-candidate-budget-within-hard-limits-and-4/5-guardband",
            "candidate_order": list(CANDIDATE_FINAL_LEAF_BUDGETS),
            "time_guardband": [4, 5],
            "memory_guardband": [4, 5],
            "width_sign_p13_or_numerics_input_forbidden": True,
        },
        "native_artifacts": {
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
            "max_workers": 1,
            "wall_seconds_per_budget": wall_seconds_per_budget,
            "per_process_address_space_bytes": address_space_bytes,
            "observed_tree_memory_bytes": observed_tree_memory_bytes,
            "sample_interval_seconds": sample_interval_seconds,
            "external_supervisor": "scripts/run_green_shared_host.py",
        },
        "gpu_environment": GPU_ENVIRONMENT,
        "machine_concurrency_manifest": machine_concurrency_manifest,
        "api_preflight": api_preflight,
        "jobs": list(jobs),
        "provenance": {
            "repository_commit": repository_commit,
            "repository_clean_before_manifest": repository_clean,
            "source_sha256": dict(sorted(source_sha256.items())),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_api_preflight(payload: dict) -> None:
    if (not isinstance(payload, dict) or set(payload) != {
            "schema_version", "official_anytime_state_api_present",
            "compiled_native_explicit_synthetic_only_boundary",
            "public_512_frozen_partition_replay_and_nesting_api",
            "private_legacy_same_partition_helper_is_not_authorized",
            "blockers", "execution_ready",
            }):
        raise ValueError("budget calibration API preflight fields mismatch")
    expected_blockers = []
    if payload["compiled_native_explicit_synthetic_only_boundary"] is not True:
        expected_blockers.append(BLOCKER_SYNTHETIC_BOUNDARY)
    if payload["public_512_frozen_partition_replay_and_nesting_api"] is not True:
        expected_blockers.append(BLOCKER_AUDIT_REPLAY)
    ready = (
        payload["official_anytime_state_api_present"] is True
        and payload["compiled_native_explicit_synthetic_only_boundary"] is True
        and payload["public_512_frozen_partition_replay_and_nesting_api"] is True
    )
    if (payload["schema_version"]
            != "green-v400-budget-calibration-api-preflight-v1"
            or type(payload["official_anytime_state_api_present"]) is not bool
            or type(payload["compiled_native_explicit_synthetic_only_boundary"])
               is not bool
            or type(payload["public_512_frozen_partition_replay_and_nesting_api"])
               is not bool
            or payload["private_legacy_same_partition_helper_is_not_authorized"]
               is not True
            or payload["blockers"] != expected_blockers
            or payload["execution_ready"] is not ready):
        raise ValueError("budget calibration API preflight invalid")


def validate_manifest(manifest: dict) -> None:
    required = {
        "schema_version", "report_contains_scientific_outcome",
        "scientific_threshold_application_authorized",
        "calibration_is_binding_production_result", "output_root",
        "closed_synthetic_identity", "dual_track_policy",
        "candidate_final_leaf_budgets",
        "budget_order_frozen_before_observations", "official_precision_bits",
        "audit_precision_bits", "adaptive_semantics", "accounting_policy",
        "artifact_separation_policy", "selector_policy", "native_artifacts",
        "execution_policy",
        "gpu_environment", "machine_concurrency_manifest", "api_preflight",
        "jobs", "provenance",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("budget calibration manifest fields mismatch")
    if (manifest["schema_version"]
            != "green-v400-actual-shape-budget-calibration-manifest-v1"
            or manifest["report_contains_scientific_outcome"] is not False
            or manifest["scientific_threshold_application_authorized"] is not False
            or manifest["calibration_is_binding_production_result"] is not False
            or not _below_mnt_sdb(manifest["output_root"])
            or manifest["candidate_final_leaf_budgets"] != [4, 8, 16, 32]
            or manifest["budget_order_frozen_before_observations"] is not True
            or manifest["official_precision_bits"] != 384
            or manifest["audit_precision_bits"] != 512
            or manifest["gpu_environment"] != GPU_ENVIRONMENT
            or manifest["jobs"] != list(_budget_jobs(manifest["output_root"]))):
        raise ValueError("budget calibration frozen design mismatch")
    identity = manifest["closed_synthetic_identity"]
    if identity != {
        "row_hash": ROW_HASH, "split": "synthetic", "actual_shape": True,
        "continuation_anchor_radius": [1, 16384],
        "standalone_radius_exponents": list(range(17)),
    }:
        raise ValueError("budget calibration synthetic identity invalid")
    if manifest["dual_track_policy"] != {
        "continuation_to_32_anchor_radius": [1, 16384],
        "continuation_prefix_leaf_counts": [4, 8, 16, 32],
        "continuation_same_path_required": True,
        "standalone_fresh_process_per_budget": True,
        "standalone_all_17_radii": True,
    }:
        raise ValueError("budget calibration dual-track policy invalid")
    adaptive = manifest["adaptive_semantics"]
    if (not isinstance(adaptive, dict) or set(adaptive) != {
            "initial_partition", "split_policy", "absolute_width_tolerance",
            "relative_width_tolerance", "max_depth",
            "stop_at_tolerance_or_leaf_budget",
            "raw_to_monotone_intersection_must_not_expand",
            "audit_replays_exact_frozen_official_partition",
            "audit_512_must_nest_inside_official_384",
            }
            or adaptive["initial_partition"] != "[-h,0],[0,h]"
            or adaptive["split_policy"]
               != "curvature-weighted width priority dyadic bisection"
            or adaptive["absolute_width_tolerance"] != "0x1p-80"
            or adaptive["relative_width_tolerance"] != "0x1p-40"
            or type(adaptive["max_depth"]) is not int
            or adaptive["max_depth"] < 5
            or any(adaptive[name] is not True for name in (
                "stop_at_tolerance_or_leaf_budget",
                "raw_to_monotone_intersection_must_not_expand",
                "audit_replays_exact_frozen_official_partition",
                "audit_512_must_nest_inside_official_384",
            ))):
        raise ValueError("budget calibration adaptive semantics invalid")
    if manifest["accounting_policy"] != {
        "charge_before_native_dispatch": True,
        "failed_timeout_or_dead_process_not_refunded": True,
        "center_reuse": False, "memoization": False,
        "fresh_isolated_process_per_budget": True,
        "per_budget_all_17_radii_384_before_any_512": True,
        "any_384_failure_launches_zero_512": True,
        "achieved_leaves_may_be_below_budget_on_tolerance": True,
        "per_radius_no_cache_count_formula": "384=2L+1;512=L+3;total=3L+4",
    }:
        raise ValueError("budget calibration accounting policy invalid")
    if manifest["artifact_separation_policy"] != {
        "selector_safe_resource_artifact": {
            "allowed_fields": [
                "budget", "charged_pass_counts", "fault_reason",
                "timing_seconds", "rss_bytes", "machine_manifest_hash",
            ],
            "interval_width_forbidden": True,
            "certificate_status_forbidden": True,
            "state_or_split_hash_payload_forbidden": True,
        },
        "selector_inaccessible_numerics_artifact": {
            "exact_domains_allowed": True,
            "exact_interval_widths_allowed": True,
            "nesting_booleans_allowed": True,
            "split_and_state_hashes_allowed": True,
            "jet_payload_forbidden": True,
            "scientific_threshold_forbidden": True,
        },
    }:
        raise ValueError("budget calibration artifact separation invalid")
    if manifest["selector_policy"] != {
        "inputs": "selector-safe-resource-artifacts-only",
        "rule": "largest-candidate-budget-within-hard-limits-and-4/5-guardband",
        "candidate_order": [4, 8, 16, 32],
        "time_guardband": [4, 5], "memory_guardband": [4, 5],
        "width_sign_p13_or_numerics_input_forbidden": True,
    }:
        raise ValueError("budget calibration selector policy invalid")
    native = manifest["native_artifacts"]
    hashes = {
        "library_sha256", "descriptor_sha256", "blob_sha256",
        "program_execution_sha256", "dispatch_sha256", "fusion_sha256",
        "kernel_tags_sha256",
    }
    if (not isinstance(native, dict)
            or set(native) != hashes | {"library_path", "descriptor_path", "blob_path"}
            or not all(_below_mnt_sdb(native[name]) for name in (
                "library_path", "descriptor_path", "blob_path"
            ))
            or any(not _is_sha256(native[name]) for name in hashes)
            or native["descriptor_sha256"] != DESCRIPTOR_SHA
            or native["blob_sha256"] != BLOB_SHA
            or native["program_execution_sha256"] != PROGRAM_SHA
            or native["dispatch_sha256"] != DISPATCH_SHA
            or native["fusion_sha256"] != FUSION_SHA
            or native["kernel_tags_sha256"]
               != sha256_canonical(EXPECTED_KERNEL_TAGS)):
        raise ValueError("budget calibration native identity invalid")
    execution = manifest["execution_policy"]
    if (not isinstance(execution, dict) or set(execution) != {
            "max_workers", "wall_seconds_per_budget",
            "per_process_address_space_bytes", "observed_tree_memory_bytes",
            "sample_interval_seconds", "external_supervisor",
            }
            or execution["max_workers"] != 1
            or type(execution["wall_seconds_per_budget"]) not in {int, float}
            or execution["wall_seconds_per_budget"] <= 0
            or type(execution["per_process_address_space_bytes"]) is not int
            or execution["per_process_address_space_bytes"] <= 0
            or type(execution["observed_tree_memory_bytes"]) is not int
            or execution["observed_tree_memory_bytes"] <= 0
            or type(execution["sample_interval_seconds"]) not in {int, float}
            or not 0.01 <= execution["sample_interval_seconds"] <= 60
            or execution["external_supervisor"]
               != "scripts/run_green_shared_host.py"):
        raise ValueError("budget calibration execution policy invalid")
    validate_api_preflight(manifest["api_preflight"])
    provenance = manifest["provenance"]
    if (not isinstance(provenance, dict) or set(provenance) != {
            "repository_commit", "repository_clean_before_manifest", "source_sha256"
            }
            or type(provenance["repository_clean_before_manifest"]) is not bool
            or len(provenance["repository_commit"]) not in {40, 64}
            or any(character not in "0123456789abcdef"
                   for character in provenance["repository_commit"])
            or set(provenance["source_sha256"]) != set(SOURCE_RELATIVE_PATHS)
            or any(not _is_sha256(value)
                   for value in provenance["source_sha256"].values())):
        raise ValueError("budget calibration provenance invalid")
    validate_machine_concurrency_manifest(
        manifest["machine_concurrency_manifest"], expected_max_workers=1,
        expected_absolute_max_workers=1,
        expected_wall_seconds=execution["wall_seconds_per_budget"],
        expected_address_space_bytes=execution["per_process_address_space_bytes"],
        expected_observed_tree_memory_bytes=execution["observed_tree_memory_bytes"],
        expected_sample_interval_seconds=execution["sample_interval_seconds"],
        expected_gpu_environment=GPU_ENVIRONMENT,
        expected_backend_kind="compiled-mpfr-native",
        expected_backend_path=native["library_path"],
        expected_backend_sha256=native["library_sha256"],
        expected_backend_opened=True,
    )


def manifest_semantic_hash(manifest: dict) -> str:
    validate_manifest(manifest)
    return sha256_canonical(manifest)


def freeze_manifest(output_root: Path, manifest: dict) -> tuple[Path, str]:
    validate_manifest(manifest)
    output_root.mkdir(parents=True, exist_ok=False)
    semantic_hash = manifest_semantic_hash(manifest)
    path = output_root / "budget_calibration_manifest.json"
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
        raise RuntimeError("BUDGET_CALIBRATION_MANIFEST_HASH_MISMATCH")
    expected = Path(payload["output_root"]) / "budget_calibration_manifest.json"
    if path.resolve() != expected.resolve():
        raise RuntimeError("BUDGET_CALIBRATION_MANIFEST_PATH_MISMATCH")
    return payload, semantic_hash


def _verify_frozen_resources(manifest: dict) -> None:
    native = manifest["native_artifacts"]
    for path_key, hash_key in (
        ("library_path", "library_sha256"),
        ("descriptor_path", "descriptor_sha256"),
        ("blob_path", "blob_sha256"),
    ):
        if _sha256_file(Path(native[path_key]).resolve(strict=True)) != native[hash_key]:
            raise RuntimeError(f"BUDGET_CALIBRATION_RESOURCE_CHANGED:{path_key}")
    for relative_path, expected_hash in manifest["provenance"]["source_sha256"].items():
        if _sha256_file(ROOT / relative_path) != expected_hash:
            raise RuntimeError(f"BUDGET_CALIBRATION_SOURCE_CHANGED:{relative_path}")
    if inspect_required_api_capabilities() != manifest["api_preflight"]:
        raise RuntimeError("BUDGET_CALIBRATION_API_PREFLIGHT_CHANGED")


def budget_worker_command(manifest: dict, manifest_hash: str, job: dict) -> tuple[str, ...]:
    output_root = Path(manifest["output_root"])
    attempt = output_root / job["attempt_relative_path"]
    resource_report = output_root / job["selector_safe_resource_relative_path"]
    numerics_report = output_root / job[
        "selector_inaccessible_numerics_relative_path"
    ]
    native = manifest["native_artifacts"]
    execution = manifest["execution_policy"]
    return (
        sys.executable, str(ROOT / "scripts" / "run_green_shared_host.py"),
        "--storage-root", "/mnt/sdb", "--attempt-directory", str(attempt),
        "--cwd", str(ROOT),
        "--wall-seconds", str(execution["wall_seconds_per_budget"]),
        "--address-space-gib",
        str(execution["per_process_address_space_bytes"] / (1 << 30)),
        "--observed-tree-gib",
        str(execution["observed_tree_memory_bytes"] / (1 << 30)),
        "--sample-seconds", str(execution["sample_interval_seconds"]),
        "--", sys.executable, str(Path(__file__).resolve()),
        "--budget-worker", "--manifest", str(
            output_root / "budget_calibration_manifest.json"
        ),
        "--manifest-sha256", manifest_hash,
        "--ordinal", str(job["ordinal"]),
        "--leaf-budget", str(job["leaf_budget"]),
        "--library", native["library_path"],
        "--descriptor", native["descriptor_path"],
        "--blob", native["blob_path"],
        "--selector-safe-resource-output", str(resource_report),
        "--selector-inaccessible-numerics-output", str(numerics_report),
    )


def _rational_width(interval_payload: dict) -> list[int]:
    lower = Fraction(*interval_payload["lower"])
    upper = Fraction(*interval_payload["upper"])
    width = upper - lower
    return [width.numerator, width.denominator]


def _exact_ratio_payload(value) -> list[int]:
    numerator, denominator = value.as_integer_ratio()
    return [int(numerator), int(denominator)]


def _numerics_radius_summary(state, audit_report: dict) -> dict:
    def widths(section: str) -> dict:
        return {
            name: {
                "official_width": _rational_width(row["official"]),
                "audit_width": _rational_width(row["audit"]),
                "audit_inside_official": row["audit_inside_official"],
            }
            for name, row in audit_report[section].items()
        }

    return {
        "radius": list(state.radius),
        "achieved_final_leaves": len(state.leaves),
        "official_state_semantic_hash": state.semantic_hash(),
        "audit_report_semantic_hash": audit_report["report_semantic_hash"],
        "frozen_domains": [
            {"lower": list(leaf.lower), "upper": list(leaf.upper),
             "depth": leaf.depth}
            for leaf in state.leaves
        ],
        "raw_curvature_widths": widths("raw_curvature"),
        "monotone_curvature_widths": widths("monotone_curvature"),
        "raw_residual_widths": widths("raw_residual"),
        "monotone_residual_widths": widths("monotone_residual"),
        "raw_witness_widths": {
            "official_width": _rational_width(
                audit_report["raw_witness"]["official"]
            ),
            "audit_width": _rational_width(audit_report["raw_witness"]["audit"]),
            "audit_inside_official": True,
        },
        "monotone_witness_widths": {
            "official_width": _rational_width(
                audit_report["monotone_witness"]["official"]
            ),
            "audit_width": _rational_width(
                audit_report["monotone_witness"]["audit"]
            ),
            "audit_inside_official": True,
        },
        "all_cell_value_first_second_nesting_checks_passed": all(
            component["audit_inside_official"] is True
            for cell in audit_report["cells"]
            for component in cell["components"].values()
        ),
        "all_endpoint_nesting_checks_passed": all(
            row["audit_inside_official"] is True
            for row in audit_report["endpoints"].values()
        ),
    }


class _DurableChargedSyntheticEvaluator:
    """Outcome-blind admission ledger around an already authorized evaluator."""

    contains_scientific_outcome = False
    synthetic_only = True

    def __init__(self, evaluator, ledger_path: Path):
        if (getattr(evaluator, "synthetic_only", None) is not True
                or getattr(evaluator, "contains_scientific_outcome", None) is not False):
            raise RuntimeError("CALIBRATION_UNAUTHORIZED_EVALUATOR")
        self.evaluator = evaluator
        self.certificate_row_hash = evaluator.certificate_row_hash
        self.evaluator_identity_sha256 = evaluator.evaluator_identity_sha256
        self.ledger_path = ledger_path
        self.charged_by_precision = {384: 0, 512: 0}

    def evaluate_interval(self, domain):
        precision = int(domain.precision_bits)
        ordinal = sum(self.charged_by_precision.values())
        entry = {
            "schema_version": "green-v400-budget-calibration-admission-v1",
            "ordinal": ordinal,
            "precision_bits": precision,
            "lower": _exact_ratio_payload(domain.lower),
            "upper": _exact_ratio_payload(domain.upper),
            "contains_scientific_outcome": False,
            "scientific_threshold_applied": False,
        }
        with self.ledger_path.open("ab") as stream:
            stream.write((canonical_json(entry) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        self.charged_by_precision[precision] += 1
        return self.evaluator.evaluate_interval(domain)


def _certificate_plan(leaf_budget: int, max_depth: int):
    from green_bridge_v400_schemas import CertificatePlan, Dyadic

    return CertificatePlan(
        "green-v400-certificate-plan-v1", ROW_HASH,
        tuple(Dyadic(1, -exponent) for exponent in RADIUS_EXPONENTS),
        "[-h,0],[0,h]",
        "curvature-weighted width priority dyadic bisection",
        "0x1p-80", "0x1p-40", max_depth, leaf_budget,
        OFFICIAL_PRECISION, AUDIT_PRECISION, (), False,
    )


def _advance_state_to_leaf_budget(state, evaluator, plan, leaf_budget: int):
    from green_bridge_v400_certificate import advance_monotone_anytime_state

    while len(state.leaves) < leaf_budget:
        try:
            state = advance_monotone_anytime_state(state, evaluator, plan)
        except RuntimeError as error:
            if str(error) == "ANYTIME_PARTITION_TOLERANCES_MET":
                break
            raise
        if state.computation_status != "PROVISIONAL":
            raise RuntimeError(
                f"CALIBRATION_OFFICIAL_PARTITION_TERMINATED:{state.resource_reason}"
            )
    return state


def run_budget_worker(
    manifest_path: Path, expected_manifest_hash: str, ordinal: int,
    leaf_budget: int, library: Path, descriptor: Path, blob: Path,
    resource_output: Path, numerics_output: Path,
) -> int:
    from green_bridge_v400_certificate import (
        audit_monotone_anytime_frozen_partitions,
        initialize_monotone_anytime_state,
    )
    from green_bridge_v400_compiled_mpfr import (
        CompiledMPFRBackend, CompiledSyntheticNativeJointWitnessEvaluator,
    )
    from green_bridge_v400_resources import ProcessTreeResourceRecorder

    manifest, manifest_hash = load_manifest(manifest_path)
    if manifest_hash != expected_manifest_hash:
        raise RuntimeError("BUDGET_CALIBRATION_WORKER_MANIFEST_HASH_MISMATCH")
    _verify_frozen_resources(manifest)
    verify_current_machine_concurrency_manifest(
        manifest["machine_concurrency_manifest"]
    )
    if manifest["api_preflight"]["execution_ready"] is not True:
        raise RuntimeError("BUDGET_CALIBRATION_WORKER_API_NOT_READY")
    job = manifest["jobs"][ordinal]
    if job["ordinal"] != ordinal or job["leaf_budget"] != leaf_budget:
        raise RuntimeError("BUDGET_CALIBRATION_WORKER_JOB_IDENTITY_MISMATCH")
    native = manifest["native_artifacts"]
    expected_paths = tuple(Path(native[name]).resolve() for name in (
        "library_path", "descriptor_path", "blob_path",
    ))
    if tuple(path.resolve() for path in (library, descriptor, blob)) != expected_paths:
        raise RuntimeError("BUDGET_CALIBRATION_WORKER_NATIVE_PATH_MISMATCH")
    if resource_output.resolve() != Path(
            manifest["output_root"],
            job["selector_safe_resource_relative_path"]).resolve():
        raise RuntimeError("BUDGET_CALIBRATION_RESOURCE_OUTPUT_PATH_MISMATCH")
    if numerics_output.resolve() != Path(
            manifest["output_root"],
            job["selector_inaccessible_numerics_relative_path"]).resolve():
        raise RuntimeError("BUDGET_CALIBRATION_NUMERICS_OUTPUT_PATH_MISMATCH")
    if resource_output.exists() or numerics_output.exists():
        raise RuntimeError("BUDGET_CALIBRATION_WORKER_OUTPUT_ALREADY_EXISTS")

    attempt = Path(manifest["output_root"], job["attempt_relative_path"])
    ledger_path = attempt / "admission_ledger.jsonl"
    if ledger_path.exists():
        raise RuntimeError("BUDGET_CALIBRATION_ADMISSION_LEDGER_ALREADY_EXISTS")
    plan = _certificate_plan(leaf_budget, manifest["adaptive_semantics"]["max_depth"])
    authorization = {
        "schema_version": "green-v400-native-synthetic-authorization-v1",
        "execution_scope": "outcome_blind_synthetic_only",
        "certificate_row_hash": ROW_HASH,
        "synthetic_artifact_semantic_hash": manifest_hash,
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
    }
    states = []
    started = time.monotonic()
    with ProcessTreeResourceRecorder(
        sample_interval_seconds=manifest["execution_policy"][
            "sample_interval_seconds"
        ]
    ) as resources:
        backend = CompiledMPFRBackend(library)
        envelope = backend.open_native_plan_envelope(
            descriptor, blob, descriptor_sha256=DESCRIPTOR_SHA,
            program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
            blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
            blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
        )
        contexts = {}
        try:
            contexts = {
                precision: backend.open_native_precision_context(envelope, precision)
                for precision in (OFFICIAL_PRECISION, AUDIT_PRECISION)
            }
            native_evaluator = CompiledSyntheticNativeJointWitnessEvaluator(
                backend, contexts, certificate_row_hash=ROW_HASH,
                expected_kernel_tags=tuple(EXPECTED_KERNEL_TAGS),
                synthetic_authorization=authorization,
            )
            evaluator = _DurableChargedSyntheticEvaluator(
                native_evaluator, ledger_path,
            )
            for radius in plan.radii:
                state = initialize_monotone_anytime_state(
                    evaluator, radius.as_fraction(), OFFICIAL_PRECISION, plan,
                    resource_lock_semantic_hash=manifest_hash,
                )
                states.append(_advance_state_to_leaf_budget(
                    state, evaluator, plan, leaf_budget,
                ))
            if evaluator.charged_by_precision[512] != 0:
                raise RuntimeError("CALIBRATION_512_LAUNCHED_BEFORE_ALL_384_COMPLETE")
            achieved = [len(state.leaves) for state in states]
            expected_counts = exact_no_cache_counts(achieved)
            if evaluator.charged_by_precision[384] != expected_counts["384"]:
                raise RuntimeError("CALIBRATION_OFFICIAL_ACCOUNTING_MISMATCH")
            audit = audit_monotone_anytime_frozen_partitions(
                states, evaluator, plan,
            )
            if (evaluator.charged_by_precision != {
                    384: expected_counts["384"], 512: expected_counts["512"]}
                    or audit["accounting"]["exact_cache_hits"] != 0):
                raise RuntimeError("CALIBRATION_NO_CACHE_ACCOUNTING_MISMATCH")
        finally:
            for context in contexts.values():
                context.close()
            envelope.close()
    elapsed = time.monotonic() - started
    record = resources.record
    counts = exact_no_cache_counts([len(state.leaves) for state in states])
    resource_report = {
        "budget": leaf_budget,
        "charged_pass_counts": counts,
        "fault_reason": None,
        "timing_seconds": elapsed,
        "rss_bytes": int(record.peak_sampled_tree_rss_kib) * 1024,
        "machine_manifest_hash": manifest["machine_concurrency_manifest"][
            "machine_manifest_semantic_hash"
        ],
    }
    radius_summaries = [
        _numerics_radius_summary(state, report)
        for state, report in zip(states, audit["radius_reports"])
    ]
    numerics_payload = {
        "schema_version": "green-v400-budget-calibration-numerics-v1",
        "execution_scope": "outcome_blind_synthetic_only",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "selector_may_read_this_artifact": False,
        "track": "fresh_process_standalone_all_17_radii",
        "budget": leaf_budget,
        "manifest_semantic_hash": manifest_hash,
        "certificate_plan_semantic_hash": sha256_canonical(plan),
        "evaluator_identity_sha256": evaluator.evaluator_identity_sha256,
        "radius_summaries": radius_summaries,
        "cross_radius_prefix_intersection_widths": [
            {
                "radius": row["radius"],
                "official_width": _rational_width(row["official_intersection"]),
                "audit_width": _rational_width(row["audit_intersection"]),
                "audit_inside_official": row["audit_inside_official"],
            }
            for row in audit["cross_radius_prefix_intersections"]
        ],
        "charged_pass_counts": counts,
        "exact_cache_hits": audit["accounting"]["exact_cache_hits"],
        "all_nesting_checks_passed": True,
    }
    numerics_payload["report_semantic_hash"] = sha256_canonical(numerics_payload)
    _canonical_write_new(numerics_output, numerics_payload)
    _canonical_write_new(resource_output, resource_report)
    print(canonical_json({
        "status": "PASS_ACTUAL_SHAPE_BUDGET_CALIBRATION_WORKER",
        "budget": leaf_budget,
        "charged_pass_counts": counts,
        "resource_output": str(resource_output),
        "numerics_output": str(numerics_output),
    }))
    return 0


def write_blocker_report(manifest_path: Path) -> Path:
    manifest, semantic_hash = load_manifest(manifest_path)
    _verify_frozen_resources(manifest)
    verify_current_machine_concurrency_manifest(
        manifest["machine_concurrency_manifest"]
    )
    preflight = manifest["api_preflight"]
    if preflight["execution_ready"]:
        raise RuntimeError(
            "CALIBRATION_API_NOW_READY_BUT_WORKER_NOT_AUTHORIZED_IN_PREPARE_ONLY_DRIVER"
        )
    report = {
        "schema_version": "green-v400-budget-calibration-blocker-report-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "BLOCKED_BEFORE_ANY_BUDGET_PROCESS",
        "manifest_semantic_hash": semantic_hash,
        "report_contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "budget_processes_launched": 0,
        "native_dispatches_admitted": 0,
        "native_dispatches_completed": 0,
        "charged_native_dispatches": 0,
        "blockers": preflight["blockers"],
        "private_legacy_same_partition_helper_used": False,
        "required_next_api_scope": [
            "explicit hash-bound synthetic-only native evaluator authorization",
            "public 512-bit replay of exact leaves from MonotoneAnytimeCertificateState",
            "public raw/monotone/component/witness 512-inside-384 nesting result",
        ],
        "claim_scope": (
            "prepare-only fail-closed API capability report; no budget worker, native "
            "dispatch, Jet payload, interval result, or scientific threshold was opened"
        ),
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    path = Path(manifest["output_root"]) / "budget_calibration_blocker_report.json"
    _canonical_write_new(path, report)
    return path


def _admission_counts(ledger_path: Path) -> dict:
    counts = {384: 0, 512: 0}
    if not ledger_path.is_file():
        return counts
    for expected_ordinal, line in enumerate(
            ledger_path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if (row.get("schema_version")
                != "green-v400-budget-calibration-admission-v1"
                or row.get("ordinal") != expected_ordinal
                or row.get("precision_bits") not in counts
                or row.get("contains_scientific_outcome") is not False
                or row.get("scientific_threshold_applied") is not False):
            raise RuntimeError("BUDGET_CALIBRATION_ADMISSION_LEDGER_INVALID")
        counts[row["precision_bits"]] += 1
    return counts


def _materialize_failed_resource_record(manifest: dict, job: dict) -> dict:
    output_root = Path(manifest["output_root"])
    attempt = output_root / job["attempt_relative_path"]
    wrapper_path = attempt / "shared_host_resource_report.json"
    if not wrapper_path.is_file():
        raise RuntimeError("BUDGET_CALIBRATION_WRAPPER_REPORT_MISSING")
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    status = wrapper["status"]
    allowed = {
        "WALL_DEADLINE_REACHED", "OBSERVED_TREE_MEMORY_REACHED",
        "WORKER_FAILED", "SUPERVISOR_INFRASTRUCTURE_FAILED",
        "SUPERVISOR_CLEANUP_FAILED",
    }
    if status not in allowed:
        raise RuntimeError(f"BUDGET_CALIBRATION_UNEXPECTED_WRAPPER_STATUS:{status}")
    charged = _admission_counts(attempt / "admission_ledger.jsonl")
    record = {
        "budget": job["leaf_budget"],
        "charged_pass_counts": {
            "384": charged[384], "512": charged[512],
            "total": charged[384] + charged[512],
        },
        "fault_reason": status,
        "timing_seconds": wrapper["observations"]["elapsed_seconds"],
        "rss_bytes": wrapper["observations"]["peak_tree_rss_bytes"],
        "machine_manifest_hash": manifest["machine_concurrency_manifest"][
            "machine_manifest_semantic_hash"
        ],
    }
    path = output_root / job["selector_safe_resource_relative_path"]
    if not path.exists():
        _canonical_write_new(path, record)
    return record


def execute_frozen_manifest(manifest_path: Path) -> Path:
    manifest, semantic_hash = load_manifest(manifest_path)
    _verify_frozen_resources(manifest)
    verify_current_machine_concurrency_manifest(
        manifest["machine_concurrency_manifest"]
    )
    if manifest["api_preflight"]["execution_ready"] is not True:
        return write_blocker_report(manifest_path)
    records = []
    for job in manifest["jobs"]:
        output_root = Path(manifest["output_root"])
        resource_path = output_root / job[
            "selector_safe_resource_relative_path"
        ]
        if resource_path.is_file():
            records.append(json.loads(resource_path.read_text(encoding="utf-8")))
            continue
        attempt = output_root / job["attempt_relative_path"]
        if attempt.exists():
            records.append(_materialize_failed_resource_record(manifest, job))
            continue
        completed = subprocess.run(
            budget_worker_command(manifest, semantic_hash, job),
            cwd=ROOT, env=os.environ | {
                "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none",
            }, check=False,
        )
        if completed.returncode == 0 and resource_path.is_file():
            records.append(json.loads(resource_path.read_text(encoding="utf-8")))
        else:
            failed = _materialize_failed_resource_record(manifest, job)
            records.append(failed)
            if failed["fault_reason"] not in {
                    "WALL_DEADLINE_REACHED", "OBSERVED_TREE_MEMORY_REACHED"}:
                raise RuntimeError(
                    "BUDGET_CALIBRATION_NONRESOURCE_FAILURE_FAIL_CLOSED:"
                    + failed["fault_reason"]
                )
    selected = select_largest_resource_safe_budget(manifest, records)
    payload = {
        "schema_version": "green-v400-budget-calibration-selection-candidate-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "status": (
            "PROVISIONAL_RESOURCE_CANDIDATE_SELECTED"
            if selected is not None else "NO_CANDIDATE_WITHIN_GUARDBAND"
        ),
        "report_contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "manifest_semantic_hash": semantic_hash,
        "selector_inputs": records,
        "selected_final_leaf_budget": selected,
        "selection_used_numerics_or_interval_widths": False,
        "production_authorized": False,
        "formal_hard_memory_lock_available_on_this_host": False,
        "formal_limitation": (
            "school host has no delegated cgroup-v2 memory.max/swap.max control; "
            "selection is an outcome-blind resource candidate, not a production lock"
        ),
    }
    payload["report_semantic_hash"] = sha256_canonical(payload)
    path = Path(manifest["output_root"]) / "resource_selection_candidate.json"
    _canonical_write_new(path, payload)
    return path


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
    parser.add_argument("--library")
    parser.add_argument("--descriptor")
    parser.add_argument("--blob")
    parser.add_argument("--output-root")
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--wall-seconds-per-budget", type=float, default=7200.0)
    parser.add_argument("--address-space-gib", type=float, default=4.0)
    parser.add_argument("--observed-tree-gib", type=float, default=3.0)
    parser.add_argument("--sample-seconds", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-frozen-manifest")
    args = parser.parse_args(arguments)
    if args.execute_frozen_manifest:
        if any(value is not None for value in (
                args.library, args.descriptor, args.blob, args.output_root)) or args.dry_run:
            parser.error("--execute-frozen-manifest is exclusive with planning inputs")
        path = Path(args.execute_frozen_manifest).resolve(strict=True)
        if not _below_mnt_sdb(path):
            raise RuntimeError("budget calibration manifest must be below /mnt/sdb")
        report = execute_frozen_manifest(path)
        print(canonical_json({"status": "EXECUTION_FINISHED", "report": str(report)}))
        return 0
    if any(value is None for value in (
            args.library, args.descriptor, args.blob, args.output_root)):
        parser.error("--library, --descriptor, --blob, and --output-root are required")
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or not _below_mnt_sdb(output_root):
        raise RuntimeError("budget calibration output root must be new below /mnt/sdb")
    library = Path(args.library).resolve(strict=True)
    descriptor = Path(args.descriptor).resolve(strict=True)
    blob = Path(args.blob).resolve(strict=True)
    if not all(_below_mnt_sdb(path) for path in (library, descriptor, blob)):
        raise RuntimeError("budget calibration native artifacts must be below /mnt/sdb")
    library_sha = _sha256_file(library)
    if _sha256_file(descriptor) != DESCRIPTOR_SHA or _sha256_file(blob) != BLOB_SHA:
        raise RuntimeError("BUDGET_CALIBRATION_NATIVE_ARTIFACT_IDENTITY_MISMATCH")
    address_bytes = int(args.address_space_gib * (1 << 30))
    observed_bytes = int(args.observed_tree_gib * (1 << 30))
    machine = collect_machine_concurrency_manifest(
        max_workers=1, absolute_max_workers=1,
        wall_seconds_per_process=args.wall_seconds_per_budget,
        per_process_address_space_bytes=address_bytes,
        observed_tree_memory_bytes=observed_bytes,
        sample_interval_seconds=args.sample_seconds,
        gpu_environment=GPU_ENVIRONMENT,
        backend_kind="compiled-mpfr-native", backend_path=library,
        backend_opened_by_workload=True,
    )
    manifest = build_manifest(
        output_root=output_root.as_posix(), library_path=library.as_posix(),
        library_sha256=library_sha, descriptor_path=descriptor.as_posix(),
        blob_path=blob.as_posix(), max_depth=args.max_depth,
        wall_seconds_per_budget=args.wall_seconds_per_budget,
        address_space_bytes=address_bytes,
        observed_tree_memory_bytes=observed_bytes,
        sample_interval_seconds=args.sample_seconds,
        repository_commit=_git("rev-parse", "HEAD"),
        repository_clean=not bool(_git(
            "status", "--porcelain=v1", "--untracked-files=all"
        )),
        source_sha256=_source_hashes(), machine_concurrency_manifest=machine,
        api_preflight=inspect_required_api_capabilities(),
    )
    manifest_path, manifest_hash = freeze_manifest(output_root, manifest)
    if args.dry_run:
        print(canonical_json({
            "status": "DRY_RUN_MANIFEST_FROZEN",
            "manifest": str(manifest_path),
            "manifest_semantic_hash": manifest_hash,
            "api_preflight": manifest["api_preflight"],
        }))
        return 0
    if manifest["api_preflight"]["execution_ready"] is not True:
        report = write_blocker_report(manifest_path)
        print(canonical_json({"status": "BLOCKED", "report": str(report)}))
        return 4
    report = execute_frozen_manifest(manifest_path)
    print(canonical_json({"status": "EXECUTION_FINISHED", "report": str(report)}))
    return 0


def main() -> int:
    if "--budget-worker" in sys.argv[1:]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--budget-worker", action="store_true")
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--manifest-sha256", required=True)
        parser.add_argument("--ordinal", type=int, required=True)
        parser.add_argument("--leaf-budget", type=int, required=True)
        parser.add_argument("--library", required=True)
        parser.add_argument("--descriptor", required=True)
        parser.add_argument("--blob", required=True)
        parser.add_argument("--selector-safe-resource-output", required=True)
        parser.add_argument("--selector-inaccessible-numerics-output", required=True)
        args = parser.parse_args()
        return run_budget_worker(
            Path(args.manifest).resolve(strict=True), args.manifest_sha256,
            args.ordinal, args.leaf_budget,
            Path(args.library).resolve(strict=True),
            Path(args.descriptor).resolve(strict=True),
            Path(args.blob).resolve(strict=True),
            Path(args.selector_safe_resource_output).resolve(),
            Path(args.selector_inaccessible_numerics_output).resolve(),
        )
    return _driver_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
