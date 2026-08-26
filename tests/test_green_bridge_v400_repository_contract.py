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
    assert len(spec.TRANSFORMER_LENS_COMMIT) == 40
    payload = {"model": spec.MODEL_ID, "revision": spec.MODEL_REVISION}
    assert len(hashlib.sha256(schemas.canonical_json(payload).encode()).hexdigest()) == 64


def test_transformerlens_semantics_flags_frozen():
    flags = spec.TRANSFORMER_SEMANTICS_FLAGS
    assert flags["normalization_type"] == "LN" and flags["attention_implementation"] == "eager"
    assert flags["evaluation_mode"]
    assert not any(flags[key] for key in ("fold_ln", "center_writing_weights", "center_unembed", "refactor_factored_attn_matrices"))


def test_control_ast_affine_and_hashed():
    assert spec.CONTROL_AST == {"operation": "affine_control", "form": "A0_plus_t_times_D", "dynamic_hook_selection": False}
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
        "[-h,0],[0,h]", "left-to-right dyadic bisection",
        "0x1p-80", "0x1p-40", 24, 262144, 384, 512, (), False,
    ).to_dict()
    coverage = {"coverage_status": "PASS", "unsupported_operations": []}
    with pytest.raises(RuntimeError, match="GRAPH_NOT_REPLAYABLE"):
        prepare._validate_static_replayability(
            [{"row_hash": row_hash}],
            [{"row_hash": row_hash, "supported_operation_coverage": True}],
            [plan], coverage,
        )
