"""Compile the sealed, outcome-free GREEN v4 execution queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.green_v400_baseline_readiness import audit_baseline_readiness
from analysis.green_v400_silent_failure_prepare import _atomic_write_json, sha256_value


PREDICTION_ROLES = ("development", "confirmation")
GPU_POLICY = (4, 5, 6, 7)


def _check_hash(payload: Any, expected: Any, label: str) -> None:
    if sha256_value(payload) != expected:
        raise ValueError(f"{label} hash mismatch")


def _job(kind: str, role: str, site: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "kind": kind,
        "role": role,
        "site_row_id": site["row_id"],
        "prompt_row_id": site["prompt_row_id"],
        "layer": site["layer"],
        "hook": site["hook"],
    }
    return {
        "job_id": sha256_value(identity),
        **identity,
        "contains_scientific_outcome": False,
    }


def compile_execution_plan(
    challenge: dict[str, Any],
    universe: dict[str, Any],
    manifest: dict[str, Any],
    readiness: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    if challenge.get("real_outcomes_authorized") is not False:
        raise ValueError("challenge must remain prepare-only")
    if universe.get("contains_scientific_outcome") is not False:
        raise ValueError("universe contains scientific outcomes")
    if manifest.get("contains_scientific_outcome") is not False:
        raise ValueError("manifest contains scientific outcomes")
    protocol_id = challenge.get("protocol_id")
    if universe.get("protocol_id") != protocol_id or manifest.get("protocol_id") != protocol_id:
        raise ValueError("protocol identifiers do not match")

    rows = universe.get("rows", [])
    _check_hash(rows, universe.get("rows_sha256"), "universe rows")
    if manifest.get("universe_rows_sha256") != universe.get("rows_sha256"):
        raise ValueError("manifest is not bound to universe rows")
    prediction_sites = manifest.get("prediction_sites", [])
    _check_hash(
        prediction_sites,
        manifest.get("prediction_sites_sha256"),
        "prediction sites",
    )
    endpoint_calibration = manifest.get("endpoint_calibration", {})
    _check_hash(
        endpoint_calibration.get("sites", []),
        endpoint_calibration.get("sites_sha256"),
        "endpoint calibration sites",
    )
    reserve = manifest.get("unused_reserve", {})
    _check_hash(
        reserve.get("sites", []), reserve.get("sites_sha256"), "reserve sites"
    )

    role_by_prompt = {row["row_id"]: row["role"] for row in rows}
    if len(role_by_prompt) != len(rows):
        raise ValueError("universe prompt identifiers are not unique")
    queues: dict[str, list[dict[str, Any]]] = {
        "development_prediction": [],
        "development_endpoint": [],
        "confirmation_prediction": [],
        "confirmation_endpoint": [],
        "endpoint_calibration": [],
    }
    for site in prediction_sites:
        role = role_by_prompt.get(site["prompt_row_id"])
        if role not in PREDICTION_ROLES:
            raise ValueError("prediction site references a non-prediction prompt")
        queues[f"{role}_prediction"].append(_job("prediction", role, site))
        endpoint_job = _job("endpoint", role, site)
        endpoint_job["requires_prediction_commitment"] = True
        queues[f"{role}_endpoint"].append(endpoint_job)
    for site in endpoint_calibration.get("sites", []):
        if role_by_prompt.get(site["prompt_row_id"]) != "endpoint_calibration":
            raise ValueError("endpoint calibration site has the wrong prompt role")
        queues["endpoint_calibration"].append(
            _job("endpoint_calibration", "endpoint_calibration", site)
        )
    if reserve.get("execution_forbidden") is not True:
        raise ValueError("reserve execution must be forbidden")

    all_job_ids = [job["job_id"] for queue in queues.values() for job in queue]
    if len(all_job_ids) != len(set(all_job_ids)):
        raise ValueError("execution job identifiers are not unique")

    baseline_audit = audit_baseline_readiness(readiness, repository_root)
    baseline_ready = baseline_audit["ready_for_untouched_execution"]
    plan = {
        "schema_version": "green-v400-sealed-execution-plan-v1",
        "protocol_id": protocol_id,
        "real_outcomes_authorized": False,
        "execution_enabled": False,
        "contains_scientific_outcome": False,
        "untouched_rows_evaluated": 0,
        "universe_rows_sha256": universe["rows_sha256"],
        "prediction_sites_sha256": manifest["prediction_sites_sha256"],
        "gpu_policy": {
            "physical_gpu_indices": list(GPU_POLICY),
            "physical_gpu_indices_0_through_3_forbidden": True,
        },
        "storage_policy": {
            "required_prefix": "/mnt/sdb/ccj/iclr_1_runs/",
            "root_disk_output_forbidden": True,
        },
        "worker_routes": {
            "prediction": "separate_prediction_process",
            "endpoint": "separate_endpoint_process",
            "shared_model_instance_forbidden": True,
        },
        "phase_locks": [
            "endpoint calibration cannot enter prediction processes",
            "every row prediction packet must be committed before its endpoint job",
            "confirmation queues remain sealed until the frozen development decision opens them",
            "reserve queues have no executable jobs",
        ],
        "baseline_readiness": baseline_audit,
        "queue_counts": {name: len(queue) for name, queue in queues.items()},
        "queues_sha256": {name: sha256_value(queue) for name, queue in queues.items()},
        "queues": queues,
        "plan_gate": (
            "PLAN_COMPILED_AWAITING_SCIENTIFIC_AUTHORIZATION"
            if baseline_ready
            else "PLAN_COMPILED_BLOCKED_BY_BASELINES"
        ),
    }
    plan["plan_sha256"] = sha256_value(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.challenge, args.universe, args.manifest, args.readiness)
    ]
    plan = compile_execution_plan(*payloads, repository_root=args.repository_root)
    _atomic_write_json(args.output, plan)
    print(json.dumps({key: plan[key] for key in (
        "plan_gate", "plan_sha256", "queue_counts", "execution_enabled"
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
