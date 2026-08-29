"""Compile the sealed, outcome-free GREEN v4 execution queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.green_v400_baseline_readiness import audit_baseline_readiness
from analysis.green_v400_silent_failure_prepare import _atomic_write_json, sha256_value


PREDICTION_ROLES = ("development", "confirmation")
GPU_POLICY = (4, 5, 6, 7)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value


def _direction_binding_map(
    registry: dict[str, Any], panel: str
) -> dict[str, str]:
    rows = registry.get("panels", {}).get(panel, {}).get("row_bindings", [])
    result = {}
    for row in rows:
        row_id = row.get("row_id")
        if row_id in result:
            raise ValueError(f"duplicate {panel} direction binding row")
        result[row_id] = _validate_digest(
            row.get("binding_sha256"), f"{panel} direction binding"
        )
        binding = row.get("binding", {})
        if (
            binding.get("row_id") != row_id
            or binding.get("panel_kind") != panel
            or binding.get("protocol_id") != registry.get("protocol_id")
        ):
            raise ValueError(f"{panel} direction binding identity mismatch")
    return result


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


def _grant_cohort_job(
    role: str,
    layer: int,
    hook: str,
    sites: list[dict[str, Any]],
    *,
    protocol_id: str,
    capture_spec_sha256: str,
    seed_domain: str,
    root_seed: int,
) -> dict[str, Any]:
    site_row_ids = sorted(site["row_id"] for site in sites)
    identity = {
        "kind": "grant_cohort_prediction",
        "role": role,
        "layer": layer,
        "hook": hook,
        "cohort_site_row_ids_sha256": sha256_value(site_row_ids),
        "cohort_size": len(site_row_ids),
        "grant_capture_spec_sha256": capture_spec_sha256,
    }
    job_id = sha256_value(identity)
    analysis_seed = int(
        sha256_value(
            [seed_domain, root_seed, protocol_id, role, layer, identity["cohort_site_row_ids_sha256"]]
        )[:16],
        16,
    )
    return {
        "job_id": job_id,
        **identity,
        "contains_scientific_outcome": False,
        "analysis_seed": analysis_seed,
        "official_semantics": "grant_style_downstream_contextual_divergence_extension_not_per_row_classifier",
        "must_commit_before_phase_endpoints": True,
    }


def compile_execution_plan(
    challenge: dict[str, Any],
    universe: dict[str, Any],
    manifest: dict[str, Any],
    readiness: dict[str, Any],
    direction_registry: dict[str, Any],
    decision_spec: dict[str, Any],
    model_manifest: dict[str, Any],
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
    if readiness.get("protocol_id") != protocol_id:
        raise ValueError("baseline readiness protocol identifier does not match")
    if decision_spec.get("contains_scientific_outcome") is not False:
        raise ValueError("decision spec contains scientific outcomes")
    if protocol_id not in decision_spec.get("applies_to_protocols", []):
        raise ValueError("shared decision spec does not apply to protocol")
    decision_sha256 = sha256_value(decision_spec)
    if challenge.get("shared_decision_rule", {}).get("canonical_sha256") != decision_sha256:
        raise ValueError("challenge is not bound to the shared decision spec")
    grant_capture_path = repository_root / "configs/green_v400_grant_capture_spec.json"
    grant_capture_spec = json.loads(grant_capture_path.read_text(encoding="utf-8"))
    if (
        grant_capture_spec.get("status") != "FROZEN_BEFORE_OUTCOMES"
        or grant_capture_spec.get("contains_scientific_outcome") is not False
        or protocol_id not in grant_capture_spec.get("applies_to_protocols", [])
        or grant_capture_spec.get("measurement_hook")
        != "blocks.10.hook_resid_post"
        or grant_capture_spec.get("candidate_layers") != list(range(9))
        or grant_capture_spec.get("measurement_must_be_strictly_downstream")
        is not True
        or grant_capture_spec.get("vectors_per_site_row") != 1
        or grant_capture_spec.get("measurement_position") != "final_prompt_position"
        or grant_capture_spec.get(
            "measurement_position_must_be_strictly_after_candidate_position"
        )
        is not True
        or grant_capture_spec.get("firewall", {}).get("prediction_route_only")
        is not True
        or grant_capture_spec.get("firewall", {}).get(
            "green_direction_payload_access"
        )
        is not False
        or grant_capture_spec.get("firewall", {}).get(
            "heldout_direction_payload_access"
        )
        is not False
        or grant_capture_spec.get("firewall", {}).get("heldout_outcome_access")
        is not False
    ):
        raise ValueError("Grant capture specification is not the frozen prediction-only extension")
    grant_capture_sha256 = sha256_value(grant_capture_spec)
    grant_seed_domain = grant_capture_spec.get("sampling", {}).get("seed_domain")
    grant_root_seed = grant_capture_spec.get("sampling", {}).get("root_seed")
    if grant_seed_domain != "GREEN_V400_GRANT_SPLIT_V1" or grant_root_seed != 40029017:
        raise ValueError("Grant sampling seed domain is not frozen")
    _validate_digest(model_manifest.get("full_model_hash"), "full model hash")
    if model_manifest.get("model_revision") != "607a30d783dfa663caf39e06633721c8d4cfcd7e":
        raise ValueError("model revision is not the frozen GPT-2 revision")
    challenge_sha256 = sha256_value(challenge)
    if manifest.get("config_sha256") != challenge_sha256:
        raise ValueError("manifest is not bound to the challenge configuration")
    precision = challenge.get("response_evaluation_precision", {})
    if (
        precision.get("checkpoint_storage_dtype") != "float32"
        or precision.get("response_evaluation_dtype") != "float64"
        or precision.get("model_manifest_tensor_hash_scheme")
        != "sha256-contiguous-numpy-native-bytes-v1"
        or precision.get("checkpoint_values_must_roundtrip_to_float32_bit_exactly")
        is not True
        or precision.get("float32_response_outcomes_forbidden") is not True
    ):
        raise ValueError("response evaluation precision contract is not binding")
    precision_audit_path = repository_root / str(precision.get("historical_audit", ""))
    if not precision_audit_path.is_file():
        raise ValueError("response precision historical audit is missing")
    precision_audit_file_hash = _file_sha256(precision_audit_path)
    if precision_audit_file_hash != precision.get("historical_audit_file_sha256"):
        raise ValueError("response precision historical audit file hash mismatch")
    precision_audit = json.loads(precision_audit_path.read_text(encoding="utf-8"))
    if (
        precision_audit.get("verdict")
        != "PASS_REQUIRE_FLOAT64_SAME_CHECKPOINT_RESPONSE_EVALUATION"
        or precision_audit.get("float64_response_evaluation_required") is not True
    ):
        raise ValueError("response precision historical audit did not pass")

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
    if reserve.get("execution_forbidden") is not True:
        raise ValueError("reserve execution must be forbidden")

    if direction_registry.get("protocol_id") != protocol_id:
        raise ValueError("direction registry protocol identifier does not match")
    if direction_registry.get("contains_scientific_outcome") is not False:
        raise ValueError("direction registry contains scientific outcomes")
    if direction_registry.get("manifest_sha256") != sha256_value(manifest):
        raise ValueError("direction registry is not bound to the manifest")
    registry_without_self_hash = dict(direction_registry)
    registry_hash = registry_without_self_hash.pop("registry_sha256", None)
    if sha256_value(registry_without_self_hash) != registry_hash:
        raise ValueError("direction registry self hash mismatch")

    green_bindings = _direction_binding_map(direction_registry, "green")
    endpoint_bindings = _direction_binding_map(direction_registry, "endpoint")
    prediction_site_ids = {site["row_id"] for site in prediction_sites}
    replay_site_ids = {site["row_id"] for site in endpoint_calibration.get("sites", [])}
    if set(green_bindings) != prediction_site_ids:
        raise ValueError("GREEN direction registry does not exactly cover prediction sites")
    if set(endpoint_bindings) != prediction_site_ids | replay_site_ids:
        raise ValueError(
            "endpoint direction registry does not exactly cover prediction and numerical-replay sites"
        )
    for panel in ("green", "endpoint"):
        _validate_digest(
            direction_registry["panels"][panel].get("payload_file_sha256"),
            f"{panel} direction payload file hash",
        )

    role_by_prompt = {row["row_id"]: row["role"] for row in rows}
    if len(role_by_prompt) != len(rows):
        raise ValueError("universe prompt identifiers are not unique")
    queues: dict[str, list[dict[str, Any]]] = {
        "development_prediction": [],
        "development_grant_cohort_prediction": [],
        "development_endpoint": [],
        "confirmation_prediction": [],
        "confirmation_grant_cohort_prediction": [],
        "confirmation_endpoint": [],
        "endpoint_numerical_replay": [],
    }
    for site in prediction_sites:
        role = role_by_prompt.get(site["prompt_row_id"])
        if role not in PREDICTION_ROLES:
            raise ValueError("prediction site references a non-prediction prompt")
        prediction_job = _job("prediction", role, site)
        prediction_job["green_direction_binding_sha256"] = green_bindings[site["row_id"]]
        queues[f"{role}_prediction"].append(prediction_job)
        endpoint_job = _job("endpoint", role, site)
        endpoint_job["requires_prediction_commitment"] = True
        endpoint_job["endpoint_direction_binding_sha256"] = endpoint_bindings[site["row_id"]]
        queues[f"{role}_endpoint"].append(endpoint_job)
    site_definition = manifest.get("site_definition", {})
    for role in PREDICTION_ROLES:
        for layer in site_definition.get("layers", []):
            cohort_sites = [
                site
                for site in prediction_sites
                if site["layer"] == layer
                and role_by_prompt.get(site["prompt_row_id"]) == role
            ]
            if not cohort_sites:
                raise ValueError(f"Grant cohort is empty for {role}/layer{layer}")
            queues[f"{role}_grant_cohort_prediction"].append(
                _grant_cohort_job(
                    role,
                    layer,
                    site_definition["hook"],
                    cohort_sites,
                    protocol_id=protocol_id,
                    capture_spec_sha256=grant_capture_sha256,
                    seed_domain=grant_seed_domain,
                    root_seed=grant_root_seed,
                )
            )
    for site in endpoint_calibration.get("sites", []):
        if role_by_prompt.get(site["prompt_row_id"]) != "endpoint_calibration":
            raise ValueError("endpoint calibration site has the wrong prompt role")
        replay_job = _job(
            "endpoint_numerical_replay", "endpoint_calibration", site
        )
        replay_job["endpoint_direction_binding_sha256"] = endpoint_bindings[
            site["row_id"]
        ]
        replay_job["legacy_prompt_role"] = "endpoint_calibration"
        queues["endpoint_numerical_replay"].append(replay_job)
    all_job_ids = [job["job_id"] for queue in queues.values() for job in queue]
    if len(all_job_ids) != len(set(all_job_ids)):
        raise ValueError("execution job identifiers are not unique")

    baseline_audit = audit_baseline_readiness(readiness, repository_root)
    baseline_ready = baseline_audit["ready_for_untouched_execution"]
    task_specific_sources = (
        ["src/green_v400_greater_than_response_adapter.py", "src/green_v400_greater_than_endpoint.py"]
        if "GT_REPLICATION" in protocol_id
        else ["src/green_v400_ioi_response_adapter.py", "src/green_v400_ioi_nmh_endpoint.py"]
    )
    source_paths = [
        "analysis/green_v400_baseline_readiness.py",
        "analysis/green_v400_execution_plan_prepare.py",
        "analysis/green_v400_formal_worker.py",
        "analysis/green_v400_formal_batch_worker.py",
        "analysis/green_v400_development_activation.py",
        "analysis/green_v400_phase_ledger.py",
        "analysis/green_v400_shared_decision_analyzer.py",
        "src/green_v400_direction_binding.py",
        "src/green_v400_endpoint_calibration.py",
        "src/green_v400_endpoint_firewall.py",
        "src/green_v400_endpoint_worker.py",
        "src/green_v400_execution_receipts.py",
        "src/green_v400_formal_prediction_runner.py",
        "src/green_v400_formal_replay_runner.py",
        "src/green_v400_formal_endpoint_runner.py",
        "src/green_v400_formal_grant_runner.py",
        "src/green_v400_four_branch_baseline.py",
        "src/green_v400_grant_divergence.py",
        "src/green_v400_grant_prediction_worker.py",
        "src/green_v400_matched_bypass_adapter.py",
        "src/green_v400_prediction_worker.py",
        "src/green_v400_response_baselines.py",
        "src/green_v400_response_precision.py",
        *task_specific_sources,
    ]
    source_hashes = {
        path: _file_sha256(repository_root / path) for path in source_paths
    }
    plan = {
        "schema_version": "green-v400-sealed-execution-plan-v1",
        "protocol_id": protocol_id,
        "real_outcomes_authorized": False,
        "execution_enabled": False,
        "contains_scientific_outcome": False,
        "untouched_rows_evaluated": 0,
        "challenge_sha256": challenge_sha256,
        "universe_sha256": sha256_value(universe),
        "manifest_sha256": sha256_value(manifest),
        "readiness_registry_sha256": sha256_value(readiness),
        "direction_registry_sha256": registry_hash,
        "decision_spec_sha256": decision_sha256,
        "grant_capture_spec_path": "configs/green_v400_grant_capture_spec.json",
        "grant_capture_spec_sha256": grant_capture_sha256,
        "decision_analyzer_sha256": source_hashes[
            "analysis/green_v400_shared_decision_analyzer.py"
        ],
        "model_manifest_sha256": sha256_value(model_manifest),
        "full_model_hash": model_manifest["full_model_hash"],
        "model_revision": model_manifest["model_revision"],
        "response_evaluation_precision": {
            "checkpoint_storage_dtype": "float32",
            "response_evaluation_dtype": "float64",
            "model_manifest_tensor_hash_scheme": (
                "sha256-contiguous-numpy-native-bytes-v1"
            ),
            "checkpoint_values_must_roundtrip_to_float32_bit_exactly": True,
            "historical_audit_file_sha256": precision_audit_file_hash,
        },
        "source_file_sha256": source_hashes,
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
            "prediction": "persistent_prediction_process_per_gpu_shard",
            "grant": "persistent_grant_process_per_gpu_shard_without_directions",
            "replay": "separate_replay_process",
            "endpoint": "separate_endpoint_process",
            "cross_route_shared_model_instance_forbidden": True,
            "prediction_route_persistent_model_allowed": True,
            "grant_route_separate_from_direction_bearing_prediction_route": True,
        },
        "prediction_execution": {
            "integrated_gradients_steps": 65,
            "ms_hvp_segments": 8,
            "response_batch_chunk_size": 16,
            "shard_count": 4,
            "physical_gpu_by_shard": {"0": 4, "1": 5, "2": 6, "3": 7},
            "jobs_serial_within_shard": True,
            "model_persistent_within_route_plan_phase_shard": True,
            "automatic_numerical_parameter_change_on_failure_forbidden": True,
        },
        "phase_locks": [
            "endpoint numerical replay cannot enter prediction processes or define a scientific null",
            "all planned numerical replay pairs for a layer must pass before any endpoint job at that layer",
            "every row prediction packet must be committed before its endpoint job",
            "every phase-layer Grant cohort packet must be committed before any endpoint job in that phase",
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
    parser.add_argument("--direction-registry", type=Path, required=True)
    parser.add_argument("--decision-spec", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            args.challenge,
            args.universe,
            args.manifest,
            args.readiness,
            args.direction_registry,
            args.decision_spec,
            args.model_manifest,
        )
    ]
    plan = compile_execution_plan(*payloads, repository_root=args.repository_root)
    _atomic_write_json(args.output, plan)
    print(json.dumps({key: plan[key] for key in (
        "plan_gate", "plan_sha256", "queue_counts", "execution_enabled"
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
