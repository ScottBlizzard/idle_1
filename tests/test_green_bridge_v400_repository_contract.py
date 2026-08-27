from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import green_bridge_v400_prepare as prepare
import green_bridge_v400_schemas as schemas
import green_bridge_v400_spec as spec


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def test_binding_parent_commit_exact():
    assert spec.BINDING_PARENT_COMMIT == "48182844a43d391439704f27aa26d513d33adaa0"
    assert subprocess.run(["git", "merge-base", "--is-ancestor", spec.BINDING_PARENT_COMMIT, "HEAD"], cwd=ROOT).returncode == 0


def test_actual_v300_paths_present():
    required = ("src/green_bridge_v300_prepare.py", "src/exp_green_bridge_v300.py", "analysis/green_v300_postcorrigendum_diagnostic.py")
    assert all((ROOT / path).is_file() for path in required)


def test_invented_v300_paths_absent():
    forbidden = ("src/green_bridge_v300_formal_prepare.py", "src/green_bridge_v300_experiment.py", "scripts/green_v300_postcorrigendum_diagnostic.py")
    assert all(not (ROOT / path).exists() for path in forbidden)


def test_v300_files_read_only_hashes():
    changed = _git("diff", "--name-only", spec.BINDING_PARENT_COMMIT, "--", *prepare.INHERITED_LOCK_FILES)
    assert changed == ""


def test_v3_confirmation_ids_excluded():
    payload = json.loads(spec.SEALED_NOUN_HASHES_PATH.read_text(encoding="utf-8"))
    forbidden = set(payload["hashes"])
    candidates = {line.strip() for line in spec.CANDIDATE_NOUNS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert len(candidates) >= 40
    candidate_hashes = {
        hashlib.sha256(f"{payload['salt']}|{noun}".encode()).hexdigest()
        for noun in candidates
    }
    assert not candidate_hashes & forbidden


def test_model_tokenizer_config_hashes():
    assert spec.MODEL_ID == "openai-community/gpt2"
    assert len(spec.MODEL_REVISION) == 40
    assert spec.TRANSFORMER_LENS_VERSION == "3.6.0"
    assert spec.TRANSFORMER_LENS_RELEASE_TAG == "v3.6.0"
    assert len(spec.TRANSFORMER_LENS_COMMIT) == 40
    payload = {"model": spec.MODEL_ID, "revision": spec.MODEL_REVISION}
    assert len(hashlib.sha256(schemas.canonical_json(payload).encode()).hexdigest()) == 64


def test_transformerlens_semantics_flags_frozen():
    flags = spec.TRANSFORMER_SEMANTICS_FLAGS
    assert flags["normalization_type"] == "LN" and flags["attention_implementation"] == "eager"
    assert flags["evaluation_mode"]
    assert not any(flags[key] for key in ("fold_ln", "center_writing_weights", "center_unembed", "refactor_factored_attn_matrices"))


def test_control_ast_affine_and_hashed():
    assert spec.CONTROL_AST["operation"] == "affine_control"
    assert spec.CONTROL_AST["form"] == "A0_plus_t_times_D"
    assert spec.CONTROL_AST["dynamic_hook_selection"] is False
    assert spec.CONTROL_AST["branches"]["J"] == {
        "selected_gate_posts": "live", "residual_bypass_kept": True,
    }
    assert spec.CONTROL_AST["branches"]["B"] == {
        "selected_gate_posts": "frozen_to_anchor", "residual_bypass_kept": True,
    }
    assert spec.CONTROL_AST["internal_residual_subtraction_is_official_curve"] is False
    assert schemas.sha256_canonical(spec.CONTROL_AST) == schemas.sha256_canonical(dict(reversed(list(spec.CONTROL_AST.items()))))


def test_contrast_and_branch_order_hashed():
    assert spec.BRANCH_ORDER == ("PAT_J", "PAT_B", "TAR_J", "TAR_B")
    assert spec.BRANCH_CONTRAST == (1, -1, -1, 1)
    assert len(schemas.sha256_canonical({"branches": spec.BRANCH_ORDER, "contrast": spec.BRANCH_CONTRAST})) == 64


def test_outputs_resolve_under_mnt_sdb():
    payload = json.loads(spec.CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["output_root"] == "/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare"
    launch = (ROOT / "scripts/launch_green_bridge_v400_formal_prepare.sh").read_text(encoding="utf-8")
    assert "/mnt/sdb" in launch and "D:" not in launch


def test_no_development_entrypoint_import_or_launch():
    source = inspect.getsource(prepare)
    launch = (ROOT / "scripts/launch_green_bridge_v400_formal_prepare.sh").read_text(encoding="utf-8")
    assert "import exp_green_bridge_v300" not in source
    assert "src/exp_green_bridge_v300.py" not in launch
    assert not spec.DEVELOPMENT_AUTHORIZED


def test_no_confirmation_reader_import_or_open():
    source = inspect.getsource(prepare)
    assert "CONFIRMATION_NOUNS" not in source
    assert "DEVELOPMENT_NOUNS" not in source
    assert "green_bridge_v300_spec" not in source
    assert not spec.CONFIRMATION_AUTHORIZED


def test_prepare_artifacts_canonical_json():
    left = {"z": [3, 2, 1], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "z": [3, 2, 1]}
    assert schemas.canonical_json(left) == schemas.canonical_json(right)
    assert schemas.canonical_json(left) == json.dumps(left, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_prepare_artifact_hash_chain():
    first = {"protocol": spec.PROTOCOL_ID, "parent": spec.BINDING_PARENT_COMMIT}
    second = first | {"upstream": schemas.sha256_canonical(first)}
    assert second["upstream"] == schemas.sha256_canonical(first)
    assert schemas.sha256_canonical(second) != second["upstream"]


def test_synthetic_artifacts_are_executed_not_self_reported(tmp_path):
    from green_bridge_v400_relational_graph import extract_joint_witness_graph
    hashes = prepare._write_synthetic_artifacts(tmp_path)
    graph_record = json.loads(
        (tmp_path / "synthetic" / "tiny_transformer_graph.json").read_text(encoding="utf-8")
    )
    graph_hash = graph_record.pop("graph_hash")
    for key in ("fixture_schema_version", "exact_ieee_constants", "tokens", "heads", "d_model"):
        graph_record.pop(key)
    row = schemas.JointWitnessRowSpec(
        "green-v400-row-v1", "0"*64, "synthetic", "1"*64, "2"*64,
        "3"*64, "4"*64, "5"*64,
        ("PAT_J", "PAT_B", "TAR_J", "TAR_B"), graph_record,
    )
    graph = extract_joint_witness_graph(row)
    certificate = json.loads(
        (tmp_path / "synthetic" / "tiny_transformer_certificate.json").read_text(encoding="utf-8")
    )
    assert graph.semantic_hash() == graph_hash == certificate["graph_hash"]
    assert len(graph.nodes) >= 35
    assert certificate["precision_nested"] is True
    assert certificate["proof_source"] == "executed serialized relational graph"
    assert len(hashes) == 7


def test_template_graph_manifest_cannot_pass_internal_gate():
    row_hash = "a" * 64
    plan = schemas.CertificatePlan(
        "green-v400-certificate-plan-v1", row_hash, (schemas.Dyadic(1, 0),),
        "[-h,0],[0,h]", "curvature-weighted width priority dyadic bisection",
        "0x1p-80", "0x1p-40", 24, 262144, 384, 512, (), False,
    ).to_dict()
    coverage = {"coverage_status": "PASS", "unsupported_operations": []}
    with pytest.raises(RuntimeError, match="GRAPH_NOT_REPLAYABLE"):
        prepare._validate_static_replayability(
            [{"row_hash": row_hash}],
            [{"row_hash": row_hash, "supported_operation_coverage": True}],
            [plan], coverage,
        )


def _candidate_resource_lock(**changes):
    payload = {
        "schema_version": "green-v400-certificate-resource-lock-v1",
        "row_hash": "0" * 64,
        "certificate_plan_semantic_hash": "9" * 64,
        "radii_order_sha256": "a" * 64,
        "radii_count": 17,
        "phase_order": "ALL_384_THEN_REPLAY_SAME_PARTITION_512",
        "official_precision": 384,
        "audit_precision": 512,
        "max_depth": 24,
        "max_final_leaves_per_radius": 14,
        "center_reuse": False,
        "endpoint_passes_per_radius_precision": 3,
        "charge_on_admission": True,
        "failed_dispatch_refund": False,
        "fte_formula_version": "green-v400-fte-pass-v1",
        "primitive_taxonomy_version": "green-v400-directed-primitives-v1",
        "primitive_charge_per_dispatch": 352_275_450,
        "token_weight_384": 90,
        "token_weight_512": 100,
        "token_budget": 75_600,
        "orchestration_reserve_seconds": 10_800,
        "wall_deadline_seconds": 86_400,
        "memory_max_bytes": 68_719_476_736,
        "partial_success_allowed": False,
        "scientific_threshold_reads_before_interval_complete": False,
        "worker_concurrency": 1,
        "memory_enforcement": "cgroup_v2_memory.max",
        "swap_enforcement": "cgroup_v2_memory.swap.max=0",
        "deadline_enforcement": "external_monotonic_supervisor_v1",
        "supervisor_process_scope": "outside_worker_cgroup_pidfd_timerfd",
        "deadline_scope": "pre_exec_validation_through_atomic_publish",
        "publication_policy": "TWO_PHASE_SUPERVISOR_COMMIT",
        "resource_reasons": schemas.RESOURCE_REASONS,
        "reachable_primary_reasons": (
            "MAX_FINAL_LEAVES_PER_RADIUS_REACHED",
            "WALL_DEADLINE_REACHED",
            "MEMORY_MAX_REACHED",
        ),
        "repository_commit": "b" * 40,
        "python_source_manifest_sha256": "c" * 64,
        "supervisor_executable_sha256": "d" * 64,
        "resource_corrigendum_sha256": "e" * 64,
        "backend_sha256": "1" * 64,
        "descriptor_sha256": "2" * 64,
        "blob_sha256": "3" * 64,
        "program_execution_sha256": "4" * 64,
        "dispatch_sha256": "5" * 64,
        "fusion_sha256": "6" * 64,
        "rounding_environment_sha256": "7" * 64,
        "hardware_manifest_sha256": "8" * 64,
        "production_authorized": False,
    }
    return schemas.CertificateResourceLock(**(payload | changes))


def test_candidate_resource_lock_is_hash_closed_and_cost_consistent():
    lock = _candidate_resource_lock()
    assert lock.worst_case_passes_384 == 493
    assert lock.worst_case_passes_512 == 289
    assert lock.worst_case_total_passes == 782
    assert lock.worst_case_charged_primitives == 275_479_401_900
    assert schemas.CertificateResourceLock.from_dict(lock.to_dict()) == lock
    assert len(schemas.sha256_canonical(lock)) == 64
    assert len(lock.semantic_hash()) == 64


@pytest.mark.parametrize("changes", [
    {"phase_order": "RADIUS_MAJOR"},
    {"charge_on_admission": False},
    {"scientific_threshold_reads_before_interval_complete": True},
    {"memory_enforcement": "proc_sampler"},
    {"swap_enforcement": "swap_allowed"},
    {"token_budget": 73_269},
    {"resource_reasons": schemas.RESOURCE_REASONS[:-1]},
    {"backend_sha256": "A" * 64},
    {"production_authorized": True},
    {"worker_concurrency": True},
])
def test_candidate_resource_lock_rejects_unfrozen_or_unsafe_variants(changes):
    with pytest.raises(ValueError):
        _candidate_resource_lock(**changes)
