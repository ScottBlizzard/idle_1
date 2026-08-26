"""Independent, read-only audit for a completed GREEN v4 formal prepare."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_spec import BINDING_PARENT_COMMIT, BRANCH, PROTOCOL_ID


REQUIRED = (
    "protocol_lock.json", "repository_manifest.json", "rounding_environment.json",
    "model_manifest.json", "sealed_exclusion_audit.json", "row_universe_manifest.json",
    "donor_feasibility.jsonl", "graph_manifest.jsonl", "certificate_plan.jsonl",
    "boundary_design_lock.json", "primitive_op_coverage.json", "theorem_test_report.json",
    "formal_prepare_summary.json",
    "engineering_corrections.jsonl",
)
SYNTHETIC_REQUIRED = (
    "fixture_manifest.json", "fixture_certificates.jsonl",
    "offgrid_curvature_counterexample.json", "tiny_transformer_graph.json",
    "tiny_transformer_certificate.json", "precision_nesting_report.json",
    "synthetic_test_summary.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(output_root: Path) -> dict:
    before = {name: sha256_file(output_root / name) for name in REQUIRED if (output_root / name).is_file()}
    missing = sorted(set(REQUIRED) - set(before))
    if missing:
        raise RuntimeError(f"AUDIT_MISSING_ARTIFACTS: {missing}")
    protocol = read_json(output_root / "protocol_lock.json")
    if protocol["protocol_id"] != PROTOCOL_ID or protocol["binding_parent_commit"] != BINDING_PARENT_COMMIT:
        raise RuntimeError("AUDIT_PROTOCOL_IDENTITY")
    if protocol["branch"] != BRANCH or not protocol["formal_prepare_only"]:
        raise RuntimeError("AUDIT_BRANCH_SCOPE")
    if protocol["development_authorized"] or protocol["confirmation_authorized"]:
        raise RuntimeError("AUDIT_AUTHORIZATION_OPEN")
    repository = read_json(output_root / "repository_manifest.json")
    if not repository["clean"] or repository["binding_parent_commit"] != BINDING_PARENT_COMMIT:
        raise RuntimeError("AUDIT_REPOSITORY_STATE")
    if any(row["file_class"] == "inherited_read_only" and not row["immutable_hash_match"] for row in repository["files"]):
        raise RuntimeError("AUDIT_IMMUTABLE_MISMATCH")
    corrections = read_jsonl(output_root / "engineering_corrections.jsonl")
    expected_corrections = {
        "path_plumbing", "environment_plumbing", "model_cache_plumbing",
        "pre_model_attempt_recovery",
    }
    if len(corrections) != 4 or {row["category"] for row in corrections} != expected_corrections:
        raise RuntimeError("AUDIT_ENGINEERING_CORRECTIONS")
    if any(row["scientific_semantics_changed"] or row["storage_device_changed"] for row in corrections):
        raise RuntimeError("AUDIT_ENGINEERING_CORRECTION_SCOPE")
    environment = read_json(output_root / "rounding_environment.json")
    if environment["official_precision_bits"] != 384 or environment["audit_precision_bits"] != 512:
        raise RuntimeError("AUDIT_PRECISION_POLICY")
    if environment["gpu_used_for_certificate"] is not False:
        raise RuntimeError("AUDIT_CERTIFICATE_GPU")
    model = read_json(output_root / "model_manifest.json")
    if not model["evaluation_mode"] or not model["singleton_hf_transformerlens_parity"]["passed"]:
        raise RuntimeError("AUDIT_MODEL_SEMANTICS_PARITY")
    exclusion = read_json(output_root / "sealed_exclusion_audit.json")
    if not exclusion["passed"] or any(exclusion["intersection_counts"].values()):
        raise RuntimeError("AUDIT_SEALED_EXCLUSION")
    if exclusion["confirmation_content_opened"] is not False:
        raise RuntimeError("AUDIT_CONFIRMATION_ACCESS")
    universe = read_json(output_root / "row_universe_manifest.json")
    if universe["contains_response_fields"] is not False:
        raise RuntimeError("AUDIT_ROW_OUTCOME_FIELD")
    feasibility = read_jsonl(output_root / "donor_feasibility.jsonl")
    if not feasibility or any(row["contains_response_outcome"] or not row["sealed_exclusion_pass"] for row in feasibility):
        raise RuntimeError("AUDIT_DONOR_SCOPE")
    graphs = read_jsonl(output_root / "graph_manifest.jsonl")
    if len(graphs) != len(feasibility) or any(row["contains_endpoint_or_derivative_values"] for row in graphs):
        raise RuntimeError("AUDIT_GRAPH_SCOPE")
    if any(not row["supported_operation_coverage"] for row in graphs):
        raise RuntimeError("AUDIT_GRAPH_COVERAGE")
    plans = read_jsonl(output_root / "certificate_plan.jsonl")
    if len(plans) != len(feasibility) or any(row["execution_authorized"] for row in plans):
        raise RuntimeError("AUDIT_REAL_CERTIFICATE_AUTHORIZATION")
    boundary = read_json(output_root / "boundary_design_lock.json")
    if boundary["contains_observed_v4_outcome"] or boundary["q_selection_authorized"] or boundary["outcome_replay_authorized"]:
        raise RuntimeError("AUDIT_BOUNDARY_OUTCOME_SCOPE")
    coverage = read_json(output_root / "primitive_op_coverage.json")
    if coverage["coverage_status"] != "PASS" or coverage["unsupported_operations"]:
        raise RuntimeError("AUDIT_PRIMITIVE_COVERAGE")
    tests = read_json(output_root / "theorem_test_report.json")
    if (tests["passed"], tests["failed"], tests["skipped"], tests["xfailed"]) != (70, 0, 0, 0):
        raise RuntimeError("AUDIT_THEOREM_BARRIER")
    synthetic_root = output_root / "synthetic"
    if any(not (synthetic_root / name).is_file() for name in SYNTHETIC_REQUIRED):
        raise RuntimeError("AUDIT_SYNTHETIC_ARTIFACTS")
    synthetic_hashes = {name: sha256_file(synthetic_root / name) for name in SYNTHETIC_REQUIRED}
    if tests["fixture_artifact_hashes"] != synthetic_hashes:
        raise RuntimeError("AUDIT_SYNTHETIC_HASHES")
    summary = read_json(output_root / "formal_prepare_summary.json")
    if summary["status"] != "PREPARE_PASS_STATIC_THEOREM_ONLY":
        raise RuntimeError("AUDIT_SUMMARY_STATUS")
    if summary["scientific_response_counts"] != {}:
        raise RuntimeError("AUDIT_SCIENTIFIC_RESPONSE_COUNTS")
    if summary["terminal_text"] != "STOP_AFTER_FORMAL_PREPARE_RETURN_TO_GPTPRO":
        raise RuntimeError("AUDIT_TERMINAL_TEXT")
    for name, expected in summary["upstream_artifact_hashes"].items():
        if name == "formal_prepare_summary.json" or sha256_file(output_root / name) != expected:
            raise RuntimeError(f"AUDIT_HASH_CLOSURE: {name}")
    launch = (ROOT / "scripts/launch_green_bridge_v400_formal_prepare.sh").read_text(encoding="utf-8")
    if "src/green_bridge_v400_prepare.py" not in launch or "src/exp_green_bridge_v300.py" in launch:
        raise RuntimeError("AUDIT_LAUNCH_SCOPE")
    if "development" in launch.lower() or "confirmation" in launch.lower():
        raise RuntimeError("AUDIT_LAUNCH_FORBIDDEN_PHASE")
    after = {name: sha256_file(output_root / name) for name in REQUIRED}
    if before != after:
        raise RuntimeError("AUDIT_NOT_READ_ONLY")
    return {
        "schema_version": "green-v400-formal-prepare-independent-audit-v1",
        "status": "PASS",
        "artifacts_verified": len(REQUIRED),
        "donor_rows_verified": len(feasibility),
        "theorem_tests_verified": 70,
        "read_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    allowed = Path("/mnt/sdb").resolve()
    if root != allowed and allowed not in root.parents:
        raise RuntimeError("AUDIT_STORAGE_ESCAPE")
    print(json.dumps(audit(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
