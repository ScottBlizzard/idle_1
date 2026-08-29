import json
from pathlib import Path

from analysis.green_v400_prepare_artifact_audit import (
    audit_bundle,
    canonical_sha256,
    file_sha256,
)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def bundle(tmp_path: Path):
    directions = tmp_path / "directions"
    directions.mkdir()
    (directions / "green.npy").write_bytes(b"green")
    (directions / "endpoint.npy").write_bytes(b"endpoint")
    manifest = {"contains_scientific_outcome": False, "rows": []}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    registry = {
        "contains_scientific_outcome": False,
        "manifest_sha256": canonical_sha256(manifest),
        "endpoint_payload_hidden_from_prediction_process": True,
        "panels": {
            "green": {
                "payload_filename": "green.npy",
                "payload_file_sha256": file_sha256(directions / "green.npy"),
                "prediction_process_access": True,
            },
            "endpoint": {
                "payload_filename": "endpoint.npy",
                "payload_file_sha256": file_sha256(directions / "endpoint.npy"),
                "prediction_process_access": False,
            },
        },
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    registry_path = tmp_path / "registry.json"
    write_json(registry_path, registry)
    plan = {
        "protocol_id": "P",
        "execution_enabled": False,
        "real_outcomes_authorized": False,
        "plan_gate": "PLAN_COMPILED_AWAITING_SCIENTIFIC_AUTHORIZATION",
        "direction_registry_sha256": registry["registry_sha256"],
        "response_evaluation_precision": {
            "response_evaluation_dtype": "float64",
            "model_manifest_tensor_hash_scheme": (
                "sha256-contiguous-numpy-native-bytes-v1"
            ),
        },
        "gpu_policy": {"physical_gpu_indices": [4, 5, 6, 7]},
        "storage_policy": {"required_prefix": "/mnt/sdb/ccj/iclr_1_runs/"},
        "queue_counts": {},
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    return manifest_path, registry_path, plan_path


def test_prepare_bundle_audit_passes_and_payload_tamper_fails(tmp_path):
    manifest, registry, plan = bundle(tmp_path)
    report = audit_bundle(
        manifest_path=manifest,
        registry_path=registry,
        plan_path=plan,
    )
    assert report["verdict"] == "PASS_PREPARE_BUNDLE_AUDIT"
    (tmp_path / "directions" / "endpoint.npy").write_bytes(b"changed")
    report = audit_bundle(
        manifest_path=manifest,
        registry_path=registry,
        plan_path=plan,
    )
    assert "endpoint direction payload hash mismatch" in report["errors"]


def test_prepare_bundle_audit_accepts_an_honest_baseline_blocker(tmp_path):
    manifest, registry, plan_path = bundle(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["plan_gate"] = "PLAN_COMPILED_BLOCKED_BY_BASELINES"
    plan["baseline_readiness"] = {
        "ready_for_untouched_execution": False,
        "not_ready_required": ["grant_divergence"],
    }
    plan.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(plan)
    write_json(plan_path, plan)
    report = audit_bundle(
        manifest_path=manifest,
        registry_path=registry,
        plan_path=plan_path,
    )
    assert report["verdict"] == "PASS_PREPARE_BUNDLE_AUDIT"
    assert report["baseline_blockers"] == ["grant_divergence"]
