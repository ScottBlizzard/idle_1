"""Audit a sealed GREEN prepare bundle without opening scientific outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_bundle(
    *,
    manifest_path: Path,
    registry_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if manifest.get("contains_scientific_outcome") is not False:
        errors.append("manifest is not outcome-free")
    if registry.get("contains_scientific_outcome") is not False:
        errors.append("direction registry is not outcome-free")
    if plan.get("execution_enabled") is not False:
        errors.append("sealed prepare plan unexpectedly enables execution")
    if plan.get("real_outcomes_authorized") is not False:
        errors.append("sealed prepare plan unexpectedly authorizes outcomes")
    allowed_prepare_gates = {
        "PLAN_COMPILED_AWAITING_SCIENTIFIC_AUTHORIZATION",
        "PLAN_COMPILED_BLOCKED_BY_BASELINES",
    }
    if plan.get("plan_gate") not in allowed_prepare_gates:
        errors.append("plan is not at a recognized prepare-only barrier")
    if plan.get("plan_gate") == "PLAN_COMPILED_BLOCKED_BY_BASELINES" and plan.get(
        "baseline_readiness", {}
    ).get("ready_for_untouched_execution") is not False:
        errors.append("baseline-blocked plan lacks a failing baseline audit")

    manifest_sha256 = canonical_sha256(manifest)
    if registry.get("manifest_sha256") != manifest_sha256:
        errors.append("direction registry manifest binding mismatch")
    registry_without_hash = dict(registry)
    claimed_registry_hash = registry_without_hash.pop("registry_sha256", None)
    if canonical_sha256(registry_without_hash) != claimed_registry_hash:
        errors.append("direction registry self hash mismatch")
    if plan.get("direction_registry_sha256") != claimed_registry_hash:
        errors.append("plan direction registry binding mismatch")
    plan_without_hash = dict(plan)
    claimed_plan_hash = plan_without_hash.pop("plan_sha256", None)
    if canonical_sha256(plan_without_hash) != claimed_plan_hash:
        errors.append("sealed execution plan self hash mismatch")

    payloads: dict[str, Any] = {}
    for panel in ("green", "endpoint"):
        record = registry.get("panels", {}).get(panel, {})
        payload_path = registry_path.parent / "directions" / str(
            record.get("payload_filename", "")
        )
        if not payload_path.is_file():
            errors.append(f"{panel} direction payload is missing")
            continue
        observed = file_sha256(payload_path)
        if observed != record.get("payload_file_sha256"):
            errors.append(f"{panel} direction payload hash mismatch")
        payloads[panel] = {
            "path": str(payload_path),
            "file_sha256": observed,
            "size_bytes": payload_path.stat().st_size,
            "shape": record.get("shape"),
            "dtype": record.get("dtype"),
            "seed_sha256": record.get("seed_sha256"),
            "seed_value_serialized": record.get("seed_value_serialized"),
            "prediction_process_access": record.get("prediction_process_access"),
        }
    if payloads.get("endpoint", {}).get("prediction_process_access") is not False:
        errors.append("endpoint payload is exposed to prediction processes")
    if registry.get("endpoint_payload_hidden_from_prediction_process") is not True:
        errors.append("endpoint payload firewall is not asserted")

    precision = plan.get("response_evaluation_precision", {})
    if precision.get("response_evaluation_dtype") != "float64":
        errors.append("plan does not bind float64 response evaluation")
    if precision.get("model_manifest_tensor_hash_scheme") != (
        "sha256-contiguous-numpy-native-bytes-v1"
    ):
        errors.append("plan does not bind the frozen model hash scheme")
    if plan.get("gpu_policy", {}).get("physical_gpu_indices") != [4, 5, 6, 7]:
        errors.append("plan GPU policy differs from physical GPUs 4 through 7")
    if plan.get("storage_policy", {}).get("required_prefix") != (
        "/mnt/sdb/ccj/iclr_1_runs/"
    ):
        errors.append("plan storage policy does not require /mnt/sdb")

    return {
        "schema_version": "green-v400-prepare-artifact-audit-v1",
        "contains_scientific_outcome": False,
        "protocol_id": plan.get("protocol_id"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "direction_registry_path": str(registry_path),
        "direction_registry_file_sha256": file_sha256(registry_path),
        "direction_registry_sha256": claimed_registry_hash,
        "sealed_plan_path": str(plan_path),
        "sealed_plan_file_sha256": file_sha256(plan_path),
        "plan_sha256": claimed_plan_hash,
        "plan_gate": plan.get("plan_gate"),
        "execution_enabled": False,
        "queue_counts": plan.get("queue_counts"),
        "baseline_blockers": plan.get("baseline_readiness", {}).get(
            "not_ready_required", []
        ),
        "direction_payloads": payloads,
        "errors": errors,
        "verdict": "PASS_PREPARE_BUNDLE_AUDIT" if not errors else "FAIL_PREPARE_BUNDLE_AUDIT",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direction-registry", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_bundle(
        manifest_path=args.manifest,
        registry_path=args.direction_registry,
        plan_path=args.plan,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
