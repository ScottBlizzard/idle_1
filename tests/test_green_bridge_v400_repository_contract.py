from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys


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
    from green_bridge_spec import DONOR_NOUNS, EVALUATION_NOUNS
    from green_bridge_v300_spec import CONFIRMATION_NOUNS, DEVELOPMENT_NOUNS
    forbidden = set(DONOR_NOUNS) | set(EVALUATION_NOUNS) | set(CONFIRMATION_NOUNS) | set(DEVELOPMENT_NOUNS)
    candidates = {line.strip() for line in spec.CANDIDATE_NOUNS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert len(candidates) >= 40
    assert not candidates & forbidden


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
    assert payload["output_root"] == "/mnt/sdb/outputs/green_bridge_v400_formal_prepare"
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
    assert "confirmation_v300" not in source
    assert "confirmation_artifact" not in source
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
