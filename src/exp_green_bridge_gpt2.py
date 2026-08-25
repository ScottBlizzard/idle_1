"""Frozen, resumable server runner for the GPT-2 matched-bypass bridge.

The command line exposes hardware scheduling only.  Scientific constants are
imported from ``green_bridge_spec`` and cannot be overridden.  Confirmation
records are inaccessible until the development decision and its source hashes
have been atomically frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Sequence

import numpy as np

from analyze_green_bridge import BASELINES, development_decision, freeze_confirmation, confirmation_decision, spearman
from green_bridge_dataset import (
    ConfirmationLock,
    PairRecord,
    build_evaluation_records,
    build_legacy_donor_records,
    plan_payload,
    split_records,
    validate_evaluation_plan,
)
from green_bridge_spec import (
    ALL_GATE_FRAME_DIM,
    AMENDMENT_ID,
    COMMON_FRAME_DIM,
    DIMENSIONS,
    FIRST_ORDER_COEFFICIENT_SEED,
    FIRST_ORDER_COEFFICIENT_SHA256,
    FIRST_ORDER_RESIDUAL_DIRECTIONS,
    FROZEN_SPEC,
    GATE_RADIUS,
    GATE04_AMENDMENT_ID,
    GATE04_HOLDOUT_PAIR_SLICE,
    GATE04_LEGACY_PAIR_SLICE,
    HALF_RADIUS_MULTIPLIER,
    HF_ATTN_IMPLEMENTATION,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_ROOT,
    PARENT_PROTOCOL_ID,
    PREDECESSOR_RUN,
    PROBE_FRAME_DIM,
    PROTOCOL_ID,
    PROTOCOL_RUN_ID,
    PROJECT_ROOT,
    RESIDUAL_RADIUS_MULTIPLIER,
    SELECTED_GATES,
    SHIFT_GRADIENT_NORMALIZED_MAX,
    STRUCTURAL_ATOM_RESIDUAL_MAX,
    STRUCTURAL_FRAME_ORTHOGONAL_MAX,
    STRUCTURAL_GRADIENT_AUTOGRAD_MAX_ABS,
    STRUCTURAL_GRADIENT_AUTOGRAD_RELATIVE,
    STRUCTURAL_GRADIENT_RESIDUAL_MAX,
    THRESHOLDS,
    TAIL_DERIVATIVE_REFERENCE_NORM_FLOOR,
    TAIL_EQUIVALENCE_OUTPUT_DIM,
    TAIL_FIXED_BATCH_SIZE,
    TRANSFORMER_LENS_COMMIT,
    canonical_json,
    frozen_spec_hash,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from green_bridge_structural_frame import (
    canonical_all_gate_frame,
    canonical_common_frame,
    canonical_gate_frame,
    first_order_coefficient_directions,
    frame_containment_metrics,
    frame_sha256,
    layernorm_gate_atom,
    residual_radius,
    target_physical_vector,
)
from green_bridge_whitebox_audit import (
    gradient_envelope_residual,
    layernorm_gate_gradient_autograd,
    layernorm_gate_gradient_formula,
    shift_null_metric,
    whitebox_A_coordinates,
)
from green_bridge_numerics import (
    active_contraction_bound,
    active_envelope_contraction_bound,
    cell_error_bound,
    certified_null_bound,
    richardson_numerical_bounds,
    sum_item_error_bounds,
)
from green_bridge_tail import GreenBridgeTail, TailAnchor, capture_tail_anchor, gather_year_logits
from green_bridge_path_target import (
    TargetAnchor,
    evaluate_joint_target,
    finite_path_effect,
    target_jvp,
)
from matched_bypass_gate import (
    GateJet,
    cosine,
    extrapolate_gate_jet,
    identify_gate,
    direct_bypass_in_common_frame,
    operator_action,
    reconstruct_cotangent,
    symmetric_relative_change,
)


SOURCE_FILES = (
    "src/green_bridge_spec.py",
    "src/green_bridge_dataset.py",
    "src/matched_bypass_gate.py",
    "src/green_bridge_structural_frame.py",
    "src/green_bridge_whitebox_audit.py",
    "src/green_bridge_numerics.py",
    "src/green_bridge_tail.py",
    "src/green_bridge_path_target.py",
    "src/exp_green_bridge_gpt2.py",
    "src/green_bridge_multigpu_worker.py",
    "src/analyze_green_bridge.py",
    "src/test_green_bridge_contract.py",
    "src/launch_green_bridge.sh",
    "src/launch_green_bridge_v131.sh",
    "src/launch_green_bridge_v132.sh",
    "src/launch_green_bridge_v133.sh",
    "src/launch_green_bridge_v135.sh",
)
PROTOCOL_FILES = (
    "analysis/GPTPRO_GREEN_BRIDGE_20260805.md",
    "analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md",
    "analysis/GREEN_SERVER_GATE04_20260805.md",
    "analysis/GREEN_SERVER_GATE08_20260805.md",
    "analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md",
    "analysis/GREEN_SERVER_GATE08_V12_20260805.md",
    "analysis/GPTPRO_GREEN_GATE08_V12_DECISION_20260805.md",
    "analysis/GREEN_V13_HASH_CORRIGENDUM_REQUEST_20260806.md",
    "analysis/GREEN_SERVER_V13_PREPARE_STOP_20260825.md",
    "analysis/GPTPRO_GREEN_V13_MANUAL_TAIL_DECISION_20260825.md",
    "analysis/archive/green_v13_stop_20260825/archive_manifest.json",
    "analysis/archive/green_v13_stop_20260825/green_bridge_v13_prepare.log",
    "analysis/GREEN_SERVER_V131_PREPARE_STOP_20260825.md",
    "analysis/GREEN_V131_BATCH_SHAPE_DIAGNOSTIC_20260825.json",
    "analysis/CODEX_GREEN_V132_BATCH_SHAPE_DECISION_20260825.md",
    "analysis/GREEN_SERVER_V132_DEVELOPMENT_STOP_20260825.md",
    "analysis/CODEX_GREEN_V133_ANCHOR_RECENTER_DECISION_20260825.md",
    "analysis/GREEN_SERVER_V133_PREPARE_STOP_20260825.md",
    "analysis/GREEN_V134_ANCHOR_RELATIVE_DIAGNOSTIC_20260825.json",
    "analysis/CODEX_GREEN_V134_EXACT_BATCH1_MULTIGPU_DECISION_20260825.md",
    "analysis/CODEX_GREEN_V135_GATEJET_RESPONSE_PAIRING_DECISION_20260825.md",
    "requirements-green-bridge.lock",
)
EXPECTED_PACKAGES = {
    "torch": "2.7.1",
    "transformer-lens": "3.6.0",
    "transformers": "5.13.0",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "pandas": "2.2.3",
    "pyarrow": "19.0.1",
    "threadpoolctl": "3.6.0",
}
TL_SOURCE_SHA256 = {
    "HookedTransformer.py": "f80ee1ec42039a287a2b9366c75f98eec23ff33c6e941ffeee03f0374eb20af3",
    "HookedRootModule.py": "e7144971a973ec2d63bf7400db6443caba5d03f22f310f6789d52fa4a56ad245",
    "components/mlps/mlp.py": "615cb178d3ce65d8784af18dec86fbfe2b3957ddc02d3b99bdd2d45aa6759b32",
    "utilities/addmm.py": "f9e72f6a3d6c508814fa8e69918c20e1cb72cbc9ae7bcb1a1abb2476e246bc38",
    "components/unembed.py": "efde00edc62c521c4da216266ea698876b46df6aa8bc3b0632cfe929dbfdce6f",
}
FORWARD_COUNTS = {
    "mixed_per_tensor_item": 2082,
    "first_order_per_tensor_item": 2082,
    "factorial_per_tensor_item": 16,
    "first_order_residual_directions": 250,
    "tensor_items_total": 384,
    "energy_items_total": 384,
    "tensor_item_unique_calls": 4_180,
    "tensor_tail_total": 1_605_120,
    "energy_tail_total": 4_608,
    "tail_evaluations_total": 1_609_824,
    "jvp_invocations_total": 1_152,
    "full_model_evaluations_total": 2_496,
    "raw_invocations_total": 1_613_472,
    "effective_units_total": 1_614_624,
    "conservative_units_total": 1_627_104,
    "development_effective_units": 538_336,
    "confirmation_effective_units": 1_076_288,
}
REVIEW_COMMIT = "67bd92c72057db48642280dd28ef2fa9b03c0cac"
GATE04_ORDERED_PROMPT_HASH = "619d21c10d4f30e6ce2597c3ba4df1de72cf0cb4f6cce322d82c2d3ec62803ce"
ACTIVE_MANUAL_TAIL_BATCH_SIZE = TAIL_FIXED_BATCH_SIZE


class GreenStop(RuntimeError):
    def __init__(self, gate: str, detail: str):
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


V13_TERMINAL_HASHES = {
    "outputs/green_bridge/result.json": "6f61c77b262eee821970dc19ff98f3baaf78e0aa9a65135bed343ed54ac7445d",
    "outputs/green_bridge/run_ledger.json": "a4c21ea2bea3e42de13bd7789a17db849290556147250ba6f284b3aefa51172c",
    "outputs/green_bridge/hook_audit.json": "49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99",
    "outputs/green_bridge/structural_frame_preflight.json": "e0f65f22d29fb8db891094c407c25234f7f8f9f19738d4edaf3fb2ed5a19a05a",
    "outputs/green_bridge/first_order_coefficients.npy": "d9305194f8d026ddde1a1d9084dd74409eae21e25b0b7600ca51f8887ff7b926",
    "outputs/green_bridge/splits.json": "0490113fbfe66bcab1fba924896f832fac4668f2566402aa0107ed4fa43ed0ca",
    "outputs/green_bridge/development_splits.json": "7fb05a1bf83d0083c622630694df09485dbaf18f4caaf6f5614200e0d8d2baf0",
    "outputs/green_bridge/model_fingerprint.json": "fb9bd5a686d1bb09fa31c4cc308ff51f26c1d64075feb57d5a330db8fcaa6cb0",
    "outputs/green_bridge/gate04_legacy_panel.json": "646d2ebcf1229645c83ebadea7f39d782e12152a8248dbd122f8c11e58c83df1",
    "analysis/archive/green_v13_stop_20260825/green_bridge_v13_prepare.log": "28c2788da0477b5c95c4498d70a9a4183f2188419dbca6e5f4725b60d5dc8e52",
}

V131_TERMINAL_HASHES = {
    "outputs/green_bridge_v131/result.json": "e911860ea406e6b38d7dc475dffd500dde68044185c11e0bc7be605f899ebbbf",
    "outputs/green_bridge_v131/run_ledger.json": "a65eb0aed2611996ecd7074512450efe1e598f392127c398d28d53bfe02bb47b",
    "outputs/green_bridge_v131/manual_tail_root_cause_reproduction_v131.json": "742d9bdac6d423e73836772960037ff1699bf953cdfaa5eec1dd18e9bfbf1d80",
    "outputs/green_bridge_v131/manual_tail_stage_trace_v131.json": "a5d599f8bc1e31c4678b3e2cc1955df35ffa3cbd7ec3b2b537ef5fc175b0936f",
    "outputs/green_bridge_v131/manual_tail_equivalence_v131.json": "a3151c719a855990d49539f42f9a00bb25e505c24803d0e197df3eeac75209c6",
    "outputs/green_bridge_v131/manual_tail_derivative_v131.json": "9072473e263ad7cc06bfe80bafb0fccea1e9016c78f0611ce32c1ef69835eee2",
    "outputs/green_bridge_v131/path_target_equivalence_v131.json": "dd5cc30ed9506b602339f2688f0d1eed5a34ba397fe1dd526a4427ed74564545",
    "outputs/green_bridge_v131/scientific_invariance_v131.json": "adf1d39fd1b8d29e9d1d124f56ecbd1f75bf34d8f3ec95dc7f56cbaf193d464e",
    "outputs/green_bridge_v131/structural_frame_preflight.json": "e0f65f22d29fb8db891094c407c25234f7f8f9f19738d4edaf3fb2ed5a19a05a",
}

V132_TERMINAL_HASHES = {
    "outputs/green_bridge_v132/prepare_result.json": "da51fece46dc9d04c3764727f9b89820c1bdd8d3dfc5d0dc9f977b3c8ca1e088",
    "outputs/green_bridge_v132/manifest.json": "090255e5822f39d937c1cd1b6657c7994d171b306874be7175a6e4baebe6bfb9",
    "outputs/green_bridge_v132/run_ledger.json": "e0eea0bdd595498bfccb8376e5ff45a6df07c37ade1c5ba0746dddebdc785df0",
    "outputs/green_bridge_v132/manual_tail_batch_equivalence_v132.json": "24355829c76b3c8c82b9ff5ff8bb6c7d9ffce0b07d75c3ade67ab2e65b9350c5",
    "outputs/green_bridge_v132/scientific_invariance_v132.json": "1a476b737c673aaaa8910245f735f500bfe50f5d0784e26364eee7921953c05d",
    "outputs/green_bridge_v132/development_structural_input_hashes.json": "c67346f629c94058d3081d2d364e257166e303acac7a93a42a77a86e4dc7bf98",
    "outputs/green_bridge_v132/development_throughput_preflight.json": "b6b617212c8681f4f97d0ce8964199b1f4f56c0858b13475bbc9f68c4d00368e",
    "outputs/green_bridge_v132/dev_energy_targets.parquet": "3faefcbd96503da9cb83b181336d9f2a914434affc561200e57a227329f00c44",
    "outputs/green_bridge_v132/dev_tensor_scores.parquet": "9a81230ab4979c4460f9bd6d8ff59a48d41b813bd878b59d65725cece4c936d6",
    "outputs/green_bridge_v132/dev_cells.json": "1294a76d6d79c81f240c20c4257aa6b0fe76457d46b30cfc5d5699e27759ae1f",
    "outputs/green_bridge_v132/endpoint_ledger.jsonl": "78defc4809b0e8dc260fc885c9cd92dd2e15055ba6f9cff5be3149b0b2dc8788",
}

V133_TERMINAL_HASHES = {
    "outputs/green_bridge_v133/result.json": "e1084e999ff3c94c7d7cec343f22b6d7462f142440955edcde561b860d36a1d8",
    "outputs/green_bridge_v133/run_ledger.json": "72d4c9ae7f3af99bb25c1a28c3576ad29772a1585dbc8c304a9b482dfd11a51d",
    "outputs/green_bridge_v133/scientific_invariance_v133.json": "cd4395f49f5073c09a2d20579006ee3c96046c7a855f5cc7353e414491c66297",
    "outputs/green_bridge_v133/manual_tail_equivalence_v133.json": "ecdffe77adf8df5831b4d93c48dd960442afaeeddd51b9b38d5d3f1bbe7eac73",
    "outputs/green_bridge_v133/manual_tail_derivative_v133.json": "1e52604f351abd8c16804025cfffddcf6bc5f9192ea33afbf3b853f2fa2dfe08",
    "outputs/green_bridge_v133/path_target_equivalence_v133.json": "0d93c9992f974baa6fc9bf36f54c99be8a81775ef499903996d26e3973bbf76b",
}


def _scientific_payload(spec: dict | None = None) -> dict:
    """Return only the frozen scientific contract, excluding technical identity."""
    payload = json.loads(canonical_json(FROZEN_SPEC if spec is None else spec))
    for key in ("schema_version", "protocol_id", "amendment_id"):
        payload.pop(key, None)
    amendment = payload.get("structural_envelope_amendment")
    if isinstance(amendment, dict):
        amendment.pop("id", None)
    return payload


def activate_hardware_batch_plan(output_root: Path) -> dict:
    """Load the immutable prepare-selected hardware plan for this process."""
    global ACTIVE_MANUAL_TAIL_BATCH_SIZE
    path = output_root / "hardware_batch_plan.json"
    if not path.is_file():
        raise GreenStop("17_MANIFEST_FREEZE", "hardware batch plan missing")
    plan = json.loads(path.read_text(encoding="utf-8"))
    selected = plan.get("manual_tail_batch_size")
    if selected != TAIL_FIXED_BATCH_SIZE:
        raise GreenStop("17_MANIFEST_FREEZE", f"invalid manual batch {selected}")
    ACTIVE_MANUAL_TAIL_BATCH_SIZE = int(selected)
    return plan


def verify_v13_terminal_archive() -> dict:
    """Verify the immutable v1.3 STOP and its copied external evidence."""
    for relative, expected in V13_TERMINAL_HASHES.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise GreenStop("00A_PREDECESSOR_ARCHIVE", f"missing {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise GreenStop(
                "00A_PREDECESSOR_ARCHIVE",
                f"hash mismatch {relative}: {actual} != {expected}",
            )

    old_root = PROJECT_ROOT / "outputs" / "green_bridge"
    result = json.loads((old_root / "result.json").read_text(encoding="utf-8"))
    ledger = json.loads((old_root / "run_ledger.json").read_text(encoding="utf-8"))
    required = {
        "result_verdict": result.get("verdict") == "STOP",
        "result_gate": result.get("first_failed_gate") == "06_MANUAL_TAIL",
        "attempt": ledger.get("attempt_index") == 1,
        "retry": ledger.get("retry_allowed") is False,
        "development": ledger.get("development_started") is False,
        "confirmation": ledger.get("confirmation_started") is False,
    }
    if not all(required.values()):
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", str(required))
    for name in (
        "frozen_analysis.json",
        "dev_tensor_scores.parquet",
        "dev_energy_targets.parquet",
        "dev_cells.json",
        "dev_result.json",
        "confirm_tensor_scores.parquet",
        "confirm_energy_targets.parquet",
        "confirm_cells.json",
    ):
        if (old_root / name).exists():
            raise GreenStop("00A_PREDECESSOR_ARCHIVE", f"unexpected {name}")

    archive_root = PROJECT_ROOT / "analysis" / "archive" / "green_v13_stop_20260825"
    manifest_path = archive_root / "archive_manifest.json"
    frozen_path = archive_root / "frozen_scientific_spec_v13.json"
    if not manifest_path.is_file() or not frozen_path.is_file():
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", "archive metadata is incomplete")
    archive = json.loads(manifest_path.read_text(encoding="utf-8"))
    if archive.get("old_files_modified") is not False:
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", "archive modification declaration missing")
    parent = json.loads(frozen_path.read_text(encoding="utf-8"))
    parent_payload = parent.get("scientific_payload")
    parent_hash = parent.get("scientific_sha256")
    if not isinstance(parent_payload, dict) or sha256_text(canonical_json(parent_payload)) != parent_hash:
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", "invalid frozen scientific payload")
    return {
        "predecessor": PREDECESSOR_RUN,
        "terminal_hashes": dict(V13_TERMINAL_HASHES),
        "archive_manifest_sha256": sha256_file(manifest_path),
        "frozen_scientific_sha256": parent_hash,
        "frozen_scientific_path_sha256": sha256_file(frozen_path),
    }


def verify_v131_terminal_archive() -> dict:
    """Verify the immutable v1.3.1 technical STOP."""
    grandparent = verify_v13_terminal_archive()
    for relative, expected in V131_TERMINAL_HASHES.items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise GreenStop("00A_PREDECESSOR_ARCHIVE", f"v1.3.1 mismatch: {relative}")
    root = PROJECT_ROOT / "outputs" / "green_bridge_v131"
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    ledger = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
    required = {
        "verdict": result.get("verdict") == "STOP",
        "gate": result.get("first_failed_gate") == "06E_BATCH_SHAPE_EQUIVALENCE",
        "attempt": ledger.get("attempt_index") == 1,
        "retry": ledger.get("retry_allowed") is False,
        "development": ledger.get("development_started") is False,
        "confirmation": ledger.get("confirmation_started") is False,
    }
    if not all(required.values()):
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", str(required))
    diagnostic = PROJECT_ROOT / "analysis" / "GREEN_V131_BATCH_SHAPE_DIAGNOSTIC_20260825.json"
    if not diagnostic.is_file() or sha256_file(diagnostic) != "666a20604fa4b123732bd68a15681fa7a16cafeef8edc2b61544fd911567d07d":
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", "batch diagnostic mismatch")
    return {
        "predecessor": {
            "schema_version": "green-bridge-v1.3",
            "protocol_run_id": "green-bridge-v1.3-one-shot",
            "execution_commit": "ed4b3b4c55ba2c7acfda1291b4814957ce90c845",
            "first_failed_gate": "06_MANUAL_TAIL",
        },
        "terminal_hashes": dict(V131_TERMINAL_HASHES),
        "diagnostic_sha256": sha256_file(diagnostic),
        "grandparent": grandparent,
    }


def verify_v132_terminal_archive() -> dict:
    """Verify the immutable v1.3.2 development failure before v1.3.3."""
    grandparent = verify_v131_terminal_archive()
    for relative, expected in V132_TERMINAL_HASHES.items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise GreenStop("00A_PREDECESSOR_ARCHIVE", f"v1.3.2 mismatch: {relative}")
    root = PROJECT_ROOT / "outputs" / "green_bridge_v132"
    prepare_result = json.loads(
        (root / "prepare_result.json").read_text(encoding="utf-8")
    )
    ledger = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
    dev_cells = json.loads((root / "dev_cells.json").read_text(encoding="utf-8"))
    cells = dev_cells.get("cells", [])
    required = {
        "prepare_pass": prepare_result.get("verdict") == "PREPARE_PASS",
        "attempt": ledger.get("attempt_index") == 1,
        "retry": ledger.get("retry_allowed") is False,
        "development": ledger.get("development_started") is True,
        "confirmation": ledger.get("confirmation_started") is False,
        "cell_count": len(cells) == 16,
        "zero_survivors": all(cell.get("survived") is False for cell in cells),
        "zero_tensor_rows": all(cell.get("n_tensor") == 0 for cell in cells),
    }
    if not all(required.values()):
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", str(required))
    return {
        "predecessor": PREDECESSOR_RUN,
        "terminal_hashes": dict(V132_TERMINAL_HASHES),
        "observed_terminal_condition": "zero surviving development cells",
        "grandparent": grandparent,
    }


def verify_v133_terminal_archive() -> dict:
    """Verify the immutable v1.3.3 prepare STOP before v1.3.4."""
    grandparent = verify_v132_terminal_archive()
    for relative, expected in V133_TERMINAL_HASHES.items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise GreenStop("00A_PREDECESSOR_ARCHIVE", f"v1.3.3 mismatch: {relative}")
    root = PROJECT_ROOT / "outputs" / "green_bridge_v133"
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    ledger = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
    required = {
        "verdict": result.get("verdict") == "STOP",
        "gate": result.get("first_failed_gate") == "06E_FIXED_BATCH_GRAPH",
        "attempt": ledger.get("attempt_index") == 1,
        "retry": ledger.get("retry_allowed") is False,
        "development": ledger.get("development_started") is False,
        "confirmation": ledger.get("confirmation_started") is False,
    }
    if not all(required.values()):
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", str(required))
    diagnostic = PROJECT_ROOT / "analysis" / "GREEN_V134_ANCHOR_RELATIVE_DIAGNOSTIC_20260825.json"
    expected_diagnostic = "7a35dd8d3ad21973850e20466a8d39cb3f0eea4ab73a725617c6c30ea2da0ab5"
    if not diagnostic.is_file() or sha256_file(diagnostic) != expected_diagnostic:
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", "v1.3.4 diagnostic mismatch")
    return {
        "predecessor": PREDECESSOR_RUN,
        "terminal_hashes": dict(V133_TERMINAL_HASHES),
        "diagnostic_sha256": expected_diagnostic,
        "grandparent": grandparent,
    }


def verify_v134_terminal_archive() -> dict:
    """Verify the immutable v1.3.4 engineering STOP before v1.3.5."""
    grandparent = verify_v133_terminal_archive()
    archive = PROJECT_ROOT / "analysis" / "GREEN_V134_TERMINAL_ARCHIVE_20260825"
    result_path = archive / "result.json"
    ledger_path = archive / "run_ledger.json"
    if not result_path.is_file() or sha256_file(result_path) != PREDECESSOR_RUN["result_sha256"]:
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", "v1.3.4 result mismatch")
    if not ledger_path.is_file() or sha256_file(ledger_path) != "923def2e80e680b8004ad6d2235587cffbc8f4287326e502ec87a2184ac3b97d":
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", "v1.3.4 ledger mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    logs = sorted((archive / "development").glob("worker_*/worker.log"))
    required = {
        "verdict": result.get("verdict") == "STOP",
        "gate": result.get("first_failed_gate") == "11_MULTIGPU_WORKER",
        "attempt": ledger.get("attempt_index") == 1,
        "retry": ledger.get("retry_allowed") is False,
        "development": ledger.get("development_started") is True,
        "confirmation": ledger.get("confirmation_started") is False,
        "eight_worker_logs": len(logs) == 8,
        "uniform_interface_failure": all(
            "GateIdentification' object has no attribute 'G'" in path.read_text(encoding="utf-8")
            for path in logs
        ),
    }
    if not all(required.values()):
        raise GreenStop("00A_PREDECESSOR_ARCHIVE", str(required))
    return {
        "predecessor": PREDECESSOR_RUN,
        "result_sha256": sha256_file(result_path),
        "ledger_sha256": sha256_file(ledger_path),
        "observed_terminal_condition": "GateIdentification/GateJet interface mismatch",
        "grandparent": grandparent,
    }


def torch_module():
    import torch
    return torch


def terminal_stop(output_root: Path, gate: str, detail: str) -> None:
    payload = {
        "schema_version": "green-bridge-terminal-v1.3.5",
        "verdict": "STOP",
        "first_failed_gate": gate,
        "detail": detail,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(output_root / "result.json", payload)
    raise GreenStop(gate, detail)


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def assert_clean_repository() -> dict:
    branch = git_text("branch", "--show-current")
    commit = git_text("rev-parse", "HEAD")
    status = git_text("status", "--porcelain=v1", "--untracked-files=all")
    if branch != "main":
        raise GreenStop("00_REPOSITORY_CLEAN", f"branch={branch!r}, expected 'main'")
    if status != "":
        raise GreenStop("00_REPOSITORY_CLEAN", status)
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", REVIEW_COMMIT, commit],
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        raise GreenStop("00_REPOSITORY_CLEAN", "review commit is not an ancestor") from exc
    return {
        "branch": branch,
        "commit": commit,
        "status_porcelain": status,
        "clean": True,
        "review_commit_is_ancestor": True,
    }


def assert_empty_prepare_root(output_root: Path) -> None:
    if output_root.exists():
        raise GreenStop("00_OUTPUT_ROOT_NOT_EMPTY", str(output_root))


def write_run_ledger(output_root: Path, repository: dict) -> None:
    write_json_atomic(output_root / "run_ledger.json", {
        "protocol_run_id": PROTOCOL_RUN_ID,
        "attempt_index": 1,
        "retry_allowed": False,
        "prepare_restart_allowed": False,
        "development_restart_allowed": False,
        "confirmation_restart_allowed": False,
        "execution_commit": repository["commit"],
        "prepare_started": True,
        "development_started": False,
        "confirmation_started": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def claim_phase(output_root: Path, phase: str) -> None:
    """Irreversibly claim a continuation phase before any model response."""
    if phase not in {"development", "confirmation"}:
        raise ValueError(f"invalid continuation phase: {phase}")
    path = output_root / "run_ledger.json"
    if not path.is_file():
        raise GreenStop("17_MANIFEST_FREEZE", "run_ledger.json missing")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("attempt_index") != 1 or ledger.get("retry_allowed") is not False:
        raise GreenStop("17_MANIFEST_FREEZE", "one-shot ledger contract changed")
    key = phase + "_started"
    if ledger.get(key) is not False:
        raise GreenStop("17_MANIFEST_FREEZE", f"{phase} has already been claimed")
    if phase == "confirmation" and ledger.get("development_started") is not True:
        raise GreenStop("17_MANIFEST_FREEZE", "development was not claimed")
    ledger[key] = True
    ledger[phase + "_started_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    write_json_atomic(path, ledger)


def first_order_directions() -> np.ndarray:
    return first_order_coefficient_directions()


def configure_runtime(device: str, physical_gpu: int = 4) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu) or device != "cuda:0":
        raise GreenStop(
            "01_ENVIRONMENT",
            f"v1.3.5 requires physical GPU {physical_gpu} exposed as cuda:0",
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise GreenStop(
            "01_ENVIRONMENT",
            "CUBLAS_WORKSPACE_CONFIG must equal :4096:8",
        )
    thread_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
    }
    if any(value != "1" for value in thread_environment.values()):
        raise GreenStop("01_ENVIRONMENT", f"thread environment={thread_environment}")
    torch = torch_module()
    versions = {}
    for package, expected in EXPECTED_PACKAGES.items():
        actual = importlib.metadata.version(package)
        versions[package] = actual
        if package == "torch":
            matches = actual == expected or actual == expected + "+cu126"
        else:
            matches = actual == expected
        if not matches:
            raise GreenStop("01_ENVIRONMENT", f"{package}={actual}, expected {expected}")
    if platform.python_version() != "3.11.13":
        raise GreenStop("01_ENVIRONMENT", f"Python={platform.python_version()}, expected 3.11.13")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise GreenStop("01_ENVIRONMENT", "a CUDA device is required")
    if torch.cuda.device_count() != 1 or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise GreenStop(
            "01_ENVIRONMENT",
            f"expected one visible RTX 4090, got {torch.cuda.device_count()} / {torch.cuda.get_device_name(0)}",
        )
    if torch.version.cuda != "12.6" or not torch.__version__.startswith("2.7.1"):
        raise GreenStop("01_ENVIRONMENT", f"torch={torch.__version__}, CUDA={torch.version.cuda}")
    import transformer_lens
    tl_root = Path(transformer_lens.__file__).resolve().parent
    actual_source = {name: sha256_file(tl_root / name) for name in TL_SOURCE_SHA256}
    if actual_source != TL_SOURCE_SHA256:
        raise GreenStop("01_ENVIRONMENT", "TransformerLens source does not match frozen commit 4a4dc26")
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    np.random.seed(20260805)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise GreenStop("01_ENVIRONMENT", "deterministic algorithms are disabled")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(torch.device(device)),
        "transformer_lens_source_sha256": actual_source,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "thread_environment": thread_environment,
    }


def load_models(device: str, tokenizer=None):
    torch = torch_module()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval().to(device)
    hf_model.config.use_cache = False
    if getattr(hf_model.config, "_attn_implementation", None) != "eager":
        raise GreenStop(
            "03_MODEL_CONFIG",
            "Hugging Face attention implementation is not eager",
        )
    model = HookedTransformer.from_pretrained_no_processing(
        "gpt2", hf_model=hf_model, tokenizer=tokenizer, device=device, dtype=torch.float32,
        default_prepend_bos=False,
    ).eval()
    cfg = model.cfg
    observed = {
        "n_layers": int(cfg.n_layers), "d_model": int(cfg.d_model),
        "n_heads": int(cfg.n_heads), "d_mlp": int(cfg.d_mlp),
        "normalization_type": str(cfg.normalization_type),
        "act_fn": str(cfg.act_fn), "eps": float(cfg.eps),
        "hf_attention_implementation": getattr(
            hf_model.config, "_attn_implementation", None
        ),
        "hf_use_cache": bool(hf_model.config.use_cache),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    required = asdict(DIMENSIONS)
    for name in ("n_layers", "d_model", "n_heads", "d_mlp"):
        if observed[name] != required[name]:
            raise GreenStop("03_MODEL_CONFIG", f"{name}={observed[name]}")
    if observed["normalization_type"] != "LN" or abs(observed["eps"] - 1e-5) > 1e-12:
        raise GreenStop("03_MODEL_CONFIG", f"normalization mismatch: {observed}")
    if "gelu" not in observed["act_fn"].lower():
        raise GreenStop("03_MODEL_CONFIG", f"activation mismatch: {observed['act_fn']}")
    return tokenizer, hf_model, model, observed


def tokenize_one(tokenizer, prompt: str, device: str):
    torch = torch_module()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return torch.tensor([ids], dtype=torch.long, device=device)


def validate_tokenizer(tokenizer, records: Sequence[PairRecord]) -> tuple[list[int], dict]:
    suffix_ids = []
    for suffix in range(100):
        text = f"{suffix:02d}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != text:
            raise GreenStop("02_TOKENIZATION", f"suffix {text} is not an exact single token")
        suffix_ids.append(ids[0])
    if len(set(suffix_ids)) != 100:
        raise GreenStop("02_TOKENIZATION", "suffix token IDs are not unique")

    lengths = set()
    for row in records:
        for suffix in (row.y, row.y_prime):
            year = tokenizer.encode(f" {row.century:02d}{suffix:02d}", add_special_tokens=False)
            century = tokenizer.encode(f" {row.century:02d}", add_special_tokens=False)
            if len(year) != 2 or year[1] != suffix_ids[suffix] or len(century) != 1:
                raise GreenStop("02_TOKENIZATION", f"year token contract failed for {row.century}{suffix:02d}")
        clean = tokenizer.encode(row.clean_prompt, add_special_tokens=False)
        corrupt = tokenizer.encode(row.corrupt_prompt, add_special_tokens=False)
        if len(clean) != len(corrupt) or clean[-1] != tokenizer.encode(
            f" {row.century:02d}", add_special_tokens=False
        )[0]:
            raise GreenStop("02_TOKENIZATION", f"prompt contract failed for {row.pair_digest}")
        lengths.add(len(clean))
    return suffix_ids, {"suffix_ids": suffix_ids, "sequence_lengths": sorted(lengths)}


def token_pair_allowed(tokenizer, first: str, second: str) -> bool:
    ids_first = tokenizer.encode(first, add_special_tokens=False)
    ids_second = tokenizer.encode(second, add_special_tokens=False)
    if len(ids_first) != len(ids_second):
        return False
    for prompt, prompt_ids in ((first, ids_first), (second, ids_second)):
        match = re.search(r"from the year (\d{2})(\d{2}) to the year (\d{2})$", prompt)
        if match is None or match.group(1) != match.group(3):
            return False
        century, suffix = match.group(1), match.group(2)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        year_ids = tokenizer.encode(" " + century + suffix, add_special_tokens=False)
        century_ids = tokenizer.encode(" " + century, add_special_tokens=False)
        if len(suffix_ids) != 1 or len(year_ids) != 2 or year_ids[1] != suffix_ids[0]:
            return False
        if len(century_ids) != 1 or prompt_ids[-1] != century_ids[0]:
            return False
    return True


def write_initial_manifest(
    output_root: Path,
    repository: dict,
    environment: dict,
    model_cfg: dict,
    evaluation_payload: dict,
    donor_v2_payload: dict,
    legacy_donor_plan_sha256: str,
    gate04_panel: dict,
) -> dict:
    directions = first_order_directions()
    direction_path = output_root / "first_order_directions.npy"
    np.save(direction_path, directions, allow_pickle=False)
    payload = {
        "schema_version": "green-bridge-manifest-v1.2",
        "theory_base_commit": "126556f",
        "repository": {
            "url": "https://github.com/ScottBlizzard/idle_1",
            "branch": repository["branch"],
            "review_commit": REVIEW_COMMIT,
            "execution_commit": repository["commit"],
            "review_commit_is_ancestor": True,
            "status_porcelain": "",
            "repository_dirty_at_launch": False,
        },
        "run": {
            "protocol_run_id": "green-bridge-v1.2-one-shot",
            "attempt_index": 1,
            "retry_allowed": False,
            "prepare_restart_allowed": False,
            "development_restart_allowed": False,
            "confirmation_restart_allowed": False,
        },
        "frozen_spec": FROZEN_SPEC,
        "frozen_spec_sha256": frozen_spec_hash(),
        "source_sha256": source_hashes(),
        "requirements_sha256": sha256_file(PROJECT_ROOT / "requirements-green-bridge.lock"),
        "protocol_sha256": {
            name: sha256_file(PROJECT_ROOT / name) for name in PROTOCOL_FILES
        },
        "evaluation_plan_sha256": evaluation_payload["records_sha256"],
        "legacy_donor_plan_sha256": legacy_donor_plan_sha256,
        "basis_v2_full_plan_sha256": donor_v2_payload["records_sha256"],
        "basis_fit_ordered_keys_sha256": donor_v2_payload["basis_fit_ordered_keys_sha256"],
        "basis_holdout_ordered_keys_sha256": donor_v2_payload["basis_holdout_ordered_keys_sha256"],
        "radius_v2_ordered_keys_sha256": donor_v2_payload["radius_v2_ordered_keys_sha256"],
        "basis_v2_all_prompt_keys_sha256": donor_v2_payload["basis_v2_all_prompt_keys_sha256"],
        "first_order_directions_sha256": sha256_file(direction_path),
        "environment": environment,
        "model_config": model_cfg,
        "forward_counts": FORWARD_COUNTS,
        "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
        "amendment": {
            "id": GATE08_AMENDMENT_ID,
            "decision_document": "analysis/GPTPRO_GREEN_GATE08_DECISION_20260805.md",
            "prior_gate04_amendment": GATE04_AMENDMENT_ID,
            "prior_execution_commit": "5083774e03b99c9958312c6686cf3ead40c3c115",
            "prior_first_failed_gate": "08_BASIS_SPECTRUM",
            "prior_sigma4_over_sigma5": 1.04,
            "prior_sigma4_over_sigma1": 0.5501,
            "prior_development_responses_observed": False,
            "prior_confirmation_responses_observed": False,
            "confirmation_remained_locked": True,
            "scientific_design_change": {
                "residual_rank": "4 -> 5",
                "basis_object": "projector-covariant",
                "new_donor_population": True,
            },
            "unchanged": {
                "theorem_identity": True,
                "actual_gate_coordinates": True,
                "intervention_sites": True,
                "matched_control": True,
                "independent_target": True,
                "residual_bypass_subtraction": True,
                "evaluation_population": True,
                "radii_multiplier": True,
                "baselines": True,
                "development_rules": True,
                "confirmation_rules": True,
            },
        },
        "prior_artifacts": {
            "protocol_v1_gate04_stop_preserved": True,
            "protocol_v1_1_gate08_stop_preserved": True,
            "gate08_stop": {
                "result_sha256": "7d52411b487f7e85f0dc539c760541d16bf5c9b756da75490edd8b9ad5ad7f90",
                "manifest_sha256": "baff192581726f4cae8f23418df5600ccb0fff549b0c81edff8c2c1f95d914df",
                "hook_audit_sha256": "49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99",
                "log_sha256": "845cb7746be048dacbcb6c841e45d29e3d51d7e7632074e08b63c92dea5d8fb8",
                "repository_dirty_at_launch": True,
                "dirty_reason": "untracked_offline_transport_bundle",
            },
        },
        "dimensions": {
            "residual_rank": 5,
            "selected_gates": 10,
            "output_dimension": 100,
            "tensor_shape": [100, 5, 10],
            "kronecker_design_rank": 50,
        },
        "structural_object": {
            "equivalence": "(U,A,P,D) ~ (UQ,AQ,PQ,DQ), Q in O(5)",
            "physical_projector": "Pi = U U^T",
            "gate_coordinates_rotated": False,
            "matched_bypass_identity": "H_path - H_control = C A",
            "inverse_changed": False,
        },
        "donor_v2": {
            "salt": "idle1-gt-bridge-basis-v2-20260805",
            "nouns": list(BASIS_V2_DONOR_NOUNS),
            "centuries": [11, 13, 15, 17],
            "roles": {
                "basis_fit": {"pairs": 512, "pairs_per_cell": 4, "orientation": {"up": 2, "down": 2}},
                "basis_holdout": {"pairs": 256, "pairs_per_cell": 2, "orientation": {"up": 1, "down": 1}},
                "radius_v2": {"pairs": 512, "pairs_per_cell": 4, "orientation": {"up": 2, "down": 2}},
            },
            "prompt_level_disjointness": True,
            "unique_prompts": 2560,
            "failed_quota_replacement_allowed": False,
            "old_donor_responses_reused": False,
        },
        "basis": {
            "fit_matrix_shape": [512, 768],
            "holdout_matrix_shape": [256, 768],
            "rank": 5,
            "centered": False,
            "dtype": "float64",
            "device": "CPU",
            "lapack_driver": "gesvd",
            "scipy_function": "scipy.linalg.svd",
            "full_matrices": False,
            "overwrite_a": False,
            "check_finite": True,
            "sign_rule": "largest_absolute_coordinate_positive_first_index_tie",
            "threadpoolctl_version": "3.6.0",
            "blas_threads": 1,
            "bootstrap_replicates": BASIS_V2_BOOTSTRAP_REPLICATES,
            "bootstrap_quantile": BASIS_V2_BOOTSTRAP_QUANTILE,
            "thresholds": {
                "fit_sigma5_over_sigma6": THRESHOLDS.basis_fit_gap_min,
                "fit_sigma5_over_sigma1": THRESHOLDS.basis_rank_floor,
                "holdout_sigma5_over_sigma6": THRESHOLDS.basis_holdout_gap_min,
                "holdout_sigma5_over_sigma1": THRESHOLDS.basis_rank_floor,
                "fit_holdout_angle_degrees": THRESHOLDS.basis_angle_max_degrees,
                "holdout_efficiency": THRESHOLDS.basis_holdout_efficiency_min,
                "leave_one_noun_angle_degrees": THRESHOLDS.basis_angle_max_degrees,
                "bootstrap_q95_angle_degrees": THRESHOLDS.basis_bootstrap_q95_max_degrees,
            },
            "bootstrap": {
                "replicates": BASIS_V2_BOOTSTRAP_REPLICATES,
                "unit": "noun",
                "rng": "numpy.PCG64",
                "seed_material": "idle1-gt-bridge-basis-v2-20260805:noun-bootstrap",
                "quantile": BASIS_V2_BOOTSTRAP_QUANTILE,
                "quantile_method": "higher",
            },
        },
        "radii": {
            "residual_scale": "median(norm(U^T d)/sqrt(5))",
            "multiplier": 0.20,
            "gate_scale_unchanged": True,
            "donor_role": "radius_v2",
            "leave_one_noun_relative_change_max": 0.20,
            "search_allowed": False,
            "inflation_allowed": False,
        },
        "gate04_replay": {
            "audit_version": "hf-tl-fidelity-v2",
            "legacy_panel_replayed": True,
            "ordered_prompt_keys_sha256": GATE04_ORDERED_PROMPT_HASH,
            "backend": "eager",
            "batch_size": 1,
            "thresholds_changed": False,
            "error_enters_epsilon_y": False,
        },
        "confirmation": {
            "locked_at_prepare": True,
            "all_v1_1_rules_unchanged": True,
            "retries": 0,
        },
        "protocol_files": list(PROTOCOL_FILES),
        "gate04": {
            "audit_version": "hf-tl-fidelity-v2",
            "hf_attention_implementation": HF_ATTN_IMPLEMENTATION,
            "transformer_lens_processing": "none",
            "dtype": "float32",
            "batch_size": 1,
            "use_cache": False,
            "prompt_selection": {
                "population": "donor",
                "ordering": "ascending pair_digest",
                "excluded_legacy_pair_ranks": {
                    "start_inclusive": 0, "stop_exclusive": 16,
                },
                "audited_holdout_pair_ranks": {
                    "start_inclusive": 16, "stop_exclusive": 32,
                },
                "prompts_per_pair": ["clean", "corrupt"],
                "n_pairs": 16,
                "n_prompts": 32,
                **gate04_panel,
            },
            "parameter_mapping": {
                "converter": "transformer_lens.pretrained.weight_conversions.gpt2.convert_gpt2_weights",
                "mapped_tensors_must_be_bitwise_equal": True,
                "allowed_extra_parameter": "unembed.b_U",
                "allowed_extra_parameter_must_be_zero": True,
            },
            "thresholds": gate04_thresholds(),
            "downstream_error": {
                "enters_epsilon_y": False,
                "reporting_only_after_gate_pass": True,
            },
        },
        "same_transformerlens_audits": {
            "no_op_max_abs": THRESHOLDS.no_op_max_abs,
            "tail_max_abs": THRESHOLDS.tail_max_abs,
            "tail_derivative_relative": THRESHOLDS.tail_derivative_relative,
            "center_rms": THRESHOLDS.center_rms,
            "center_max_abs": THRESHOLDS.center_max_abs,
            "hook_untouched_max": THRESHOLDS.hook_untouched_max,
        },
        "numerical_error_contract": {
            "version": "frozen-richardson-propagation-v1",
            "epsilon_y_source": "same-TransformerLens duplicate audit only",
            "eta_G": "3*epsilon_y/h2",
            "eta_C": "64*epsilon_y/(3*h2^2)",
            "eta_J": "3*epsilon_y/h1",
            "eta_H": "17*epsilon_y/(3*h1*h2)",
            "active_tensor_snr_uses_epsilon_P_F": True,
            "certified_null_bound_enters_theta_error": True,
        },
        "preserved": {
            "matched_bypass_theorem": True,
            "selected_gates": True,
            "resid_mid_site": True,
            "matched_control": True,
            "independent_target": True,
            "residual_bypass_subtraction": True,
            "basis_design": False,
            "radii": True,
            "finite_population": True,
            "baselines": True,
            "development_rules": True,
            "confirmation_lock": True,
            "confirmation_rules": True,
        },
        "compute": FORWARD_COUNTS,
        "terminal_rule": {
            "rank5_gate_failure": "terminate_oral_line",
            "rank6_fallback": False,
            "donor_replacement": False,
            "threshold_amendment": False,
            "second_basis_run": False,
        },
        "confirmation_open": False,
    }
    write_json_atomic(output_root / "manifest.json", payload)
    return payload


def selected_position(anchor: TailAnchor, field: str):
    torch = torch_module()
    value = getattr(anchor, field)
    rows = torch.arange(value.shape[0], device=value.device)
    return value[rows, anchor.final_positions]


def capture_item_systems(model, tokenizer, suffix_ids, record: PairRecord, device: str, anchor_cache=None):
    torch = torch_module()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    clean_tokens = tokenize_one(tokenizer, record.clean_prompt, device)
    corrupt_tokens = tokenize_one(tokenizer, record.corrupt_prompt, device)
    cached = {} if anchor_cache is None else anchor_cache.get(record.pair_digest, {})
    target = cached.get("tar") or capture_tail_anchor(
        model, clean_tokens, suffix_tensor, system="tar"
    )
    corrupt = cached.get("cor") or capture_tail_anchor(
        model, corrupt_tokens, suffix_tensor, system="cor"
    )
    patched = cached.get("pat") or capture_tail_anchor(
        model, corrupt_tokens, suffix_tensor, system="pat", block8_patch=target.mlp8_out
    )
    return {"tar": target, "cor": corrupt, "pat": patched}, clean_tokens, corrupt_tokens


def _anchor_to_plain(anchor: TailAnchor) -> dict:
    fields = ("resid_mid", "pre", "post", "resid_post", "year_logits", "final_positions", "mlp8_out")
    return {name: (None if getattr(anchor, name) is None else getattr(anchor, name).detach().cpu()) for name in fields} | {
        "system": anchor.system
    }


def load_anchor_cache(path: Path, device: str) -> dict:
    torch = torch_module()
    if not path.is_file():
        return {}
    plain = torch.load(path, map_location=device, weights_only=True)
    return {
        digest: {system: TailAnchor(**values) for system, values in systems.items()}
        for digest, systems in plain.items()
    }


def merge_anchor_caches(*caches: dict) -> dict:
    merged = {}
    for cache in caches:
        for digest, systems in cache.items():
            merged.setdefault(digest, {}).update(systems)
    return merged


def _repeat_anchor(anchor: TailAnchor, count: int) -> TailAnchor:
    torch = torch_module()
    index = torch.zeros(count, dtype=torch.long, device=anchor.resid_mid.device)
    def repeat(value):
        return None if value is None else value.index_select(0, index)
    return TailAnchor(
        resid_mid=repeat(anchor.resid_mid), pre=repeat(anchor.pre), post=repeat(anchor.post),
        resid_post=repeat(anchor.resid_post), year_logits=repeat(anchor.year_logits),
        final_positions=repeat(anchor.final_positions), system=anchor.system,
        mlp8_out=repeat(anchor.mlp8_out),
    )


def margin_vector(clean_suffix: int, device: str):
    torch = torch_module()
    value = torch.empty(100, dtype=torch.float64, device=device)
    value[: clean_suffix + 1] = -1.0 / (clean_suffix + 1)
    value[clean_suffix + 1 :] = 1.0 / (99 - clean_suffix)
    return value


def margin(logits, contrast):
    return (logits.double() * contrast).sum(dim=-1)


def gate04_thresholds() -> dict:
    return {
        "raw_year_logits": {
            "max_abs": THRESHOLDS.hf_tl_raw_year_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_raw_year_pooled_rms,
        },
        "centered_year_logits": {
            "max_abs": THRESHOLDS.hf_tl_centered_year_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_centered_year_pooled_rms,
        },
        "task_margin": {
            "max_abs": THRESHOLDS.hf_tl_margin_max_abs,
            "rms": THRESHOLDS.hf_tl_margin_rms,
        },
        "resid_mid": {
            "max_abs": THRESHOLDS.hf_tl_resid_mid_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_resid_mid_pooled_rms,
        },
        "selected_pre": {
            "max_abs": THRESHOLDS.hf_tl_selected_pre_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_selected_pre_pooled_rms,
        },
        "selected_post": {
            "max_abs": THRESHOLDS.hf_tl_selected_post_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_selected_post_pooled_rms,
        },
    }


def gate04_record_panels(records):
    ranked = sorted(records, key=lambda row: row.pair_digest)
    legacy = ranked[slice(*GATE04_LEGACY_PAIR_SLICE)]
    holdout = ranked[slice(*GATE04_HOLDOUT_PAIR_SLICE)]
    legacy_ids = {row.pair_digest for row in legacy}
    holdout_ids = {row.pair_digest for row in holdout}
    if len(legacy) != 16 or len(holdout) != 16:
        raise GreenStop("04_HF_TL_FIDELITY", "Gate-04 panel has wrong size")
    if len(legacy_ids) != 16 or len(holdout_ids) != 16 or legacy_ids & holdout_ids:
        raise GreenStop("04_HF_TL_FIDELITY", "Gate-04 panels overlap or contain duplicates")
    return legacy, holdout


def gate04_panel_metadata(legacy, holdout) -> dict:
    ordered_keys = [
        [row.pair_digest, system]
        for row in holdout
        for system in ("clean", "corrupt")
    ]
    return {
        "legacy_pair_digests": [row.pair_digest for row in legacy],
        "holdout_pair_digests": [row.pair_digest for row in holdout],
        "ordered_prompt_keys": ordered_keys,
        "ordered_prompt_keys_sha256": sha256_text(canonical_json(ordered_keys)),
    }


def pooled_error_metrics(errors) -> dict:
    arrays = [np.asarray(error, dtype=np.float64).reshape(-1) for error in errors]
    if not arrays or any(array.size == 0 for array in arrays):
        raise ValueError("pooled error metrics require nonempty arrays")
    total_count = sum(array.size for array in arrays)
    sum_squares = sum(float(array @ array) for array in arrays)
    return {
        "max_abs": max(float(np.max(np.abs(array))) for array in arrays),
        "pooled_rms": float(math.sqrt(sum_squares / total_count)),
    }


def centered_year_error(hf_year, tl_year) -> np.ndarray:
    hf = np.asarray(hf_year, dtype=np.float64)
    tl = np.asarray(tl_year, dtype=np.float64)
    return (hf - hf.mean()) - (tl - tl.mean())


def task_margin_error(hf_year, tl_year, clean_suffix: int) -> float:
    contrast = np.empty(100, dtype=np.float64)
    contrast[: clean_suffix + 1] = -1.0 / (clean_suffix + 1)
    contrast[clean_suffix + 1 :] = 1.0 / (99 - clean_suffix)
    return float(contrast @ (
        np.asarray(hf_year, dtype=np.float64) - np.asarray(tl_year, dtype=np.float64)
    ))


def all_gate04_submetrics_pass(metrics: dict, thresholds: dict) -> bool:
    for name, limits in thresholds.items():
        observed = metrics[name]
        for statistic, limit in limits.items():
            if observed[statistic] > limit:
                return False
    return True


def capture_hf_gate04(hf_model, tokens):
    torch = torch_module()
    captured = {}

    def resid_mid_pre_hook(_module, args):
        captured["resid_mid"] = args[0].detach()

    def pre_hook(_module, _args, output):
        captured["pre"] = output.detach()

    def post_pre_hook(_module, args):
        captured["post"] = args[0].detach()

    handles = [
        hf_model.transformer.h[10].ln_2.register_forward_pre_hook(
            resid_mid_pre_hook
        ),
        hf_model.transformer.h[10].mlp.c_fc.register_forward_hook(pre_hook),
        hf_model.transformer.h[10].mlp.c_proj.register_forward_pre_hook(
            post_pre_hook
        ),
    ]
    try:
        with torch.inference_mode():
            logits = hf_model(
                input_ids=tokens,
                use_cache=False,
                return_dict=True,
            ).logits
    finally:
        for handle in handles:
            handle.remove()
    required = {"resid_mid", "pre", "post"}
    if set(captured) != required:
        raise GreenStop(
            "04_HF_TL_FIDELITY",
            f"incomplete Hugging Face capture: {sorted(captured)}",
        )
    return logits, captured


def _exact_tensor_equal(actual, expected) -> bool:
    try:
        torch = torch_module()
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            return bool(torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    except ImportError:
        pass
    return bool(np.array_equal(np.asarray(actual), np.asarray(expected)))


def _nonzero_count(value) -> int:
    try:
        torch = torch_module()
        if isinstance(value, torch.Tensor):
            return int(torch.count_nonzero(value.detach()).item())
    except ImportError:
        pass
    return int(np.count_nonzero(np.asarray(value)))


def weight_mapping_report(expected_state, actual_state, named_parameters=None) -> dict:
    named_parameters = actual_state if named_parameters is None else named_parameters
    missing, shapes, dtypes, values = [], [], [], []
    for key, expected in expected_state.items():
        if key not in actual_state:
            missing.append(key)
            continue
        actual = actual_state[key]
        if tuple(actual.shape) != tuple(expected.shape):
            shapes.append(key)
            continue
        if actual.dtype != expected.dtype:
            dtypes.append(key)
            continue
        if not _exact_tensor_equal(actual, expected):
            values.append(key)
    bias = actual_state.get("unembed.b_U")
    bias_nonzero = 0 if bias is None else _nonzero_count(bias)
    unexpected_nonzero = sorted(
        key for key, value in named_parameters.items()
        if key not in expected_state
        and key != "unembed.b_U"
        and _nonzero_count(value) != 0
    )
    mismatch_count = (
        len(missing) + len(shapes) + len(dtypes) + len(values)
        + len(unexpected_nonzero) + int(bias_nonzero != 0)
    )
    return {
        "mapped_tensor_count": len(expected_state),
        "missing_keys": missing,
        "shape_mismatches": shapes,
        "dtype_mismatches": dtypes,
        "value_mismatches": values,
        "unexpected_nonzero_keys": unexpected_nonzero,
        "unembed_b_U_present": bias is not None,
        "unembed_b_U_nonzero_count": bias_nonzero,
        "mismatch_count": mismatch_count,
        "passed": mismatch_count == 0,
    }


def audit_converted_weights(hf_model, model) -> dict:
    from transformer_lens.pretrained.weight_conversions.gpt2 import convert_gpt2_weights
    expected_tl_state = convert_gpt2_weights(hf_model, model.cfg)
    result = weight_mapping_report(
        expected_tl_state,
        model.state_dict(),
        dict(model.named_parameters()),
    )
    if not result["passed"]:
        raise GreenStop("04_HF_TL_WEIGHT_MAP", canonical_json(result))
    return result


def hf_tl_audit(tokenizer, hf_model, model, legacy, holdout, suffix_ids, device: str) -> dict:
    torch = torch_module()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    panel = gate04_panel_metadata(legacy, holdout)
    weight_mapping = audit_converted_weights(hf_model, model)
    families = {
        name: [] for name in (
            "raw_year_logits", "centered_year_logits", "resid_mid",
            "selected_pre", "selected_post",
        )
    }
    margin_errors, per_prompt, references = [], [], {}
    for row in holdout:
        for system, prompt in (("clean", row.clean_prompt), ("corrupt", row.corrupt_prompt)):
            tokens = tokenize_one(tokenizer, prompt, device)
            if tokens.shape[0] != 1:
                raise GreenStop("04_HF_TL_FIDELITY", "Gate-04 batch size is not one")
            hf_logits, hf_capture = capture_hf_gate04(hf_model, tokens)
            position = tokens.shape[1] - 1
            positions = torch.tensor([position], device=device)
            hf_year = gather_year_logits(model, hf_logits, positions, suffix_tensor)[0]
            with torch.inference_mode():
                anchor = capture_tail_anchor(model, tokens, suffix_tensor, system="hf-tl")
            tl_year = anchor.year_logits[0]
            raw = hf_year.detach().double().cpu().numpy() - tl_year.detach().double().cpu().numpy()
            centered = centered_year_error(
                hf_year.detach().double().cpu().numpy(),
                tl_year.detach().double().cpu().numpy(),
            )
            resid = (
                hf_capture["resid_mid"][0, position].detach().double().cpu().numpy()
                - anchor.resid_mid[0, position].detach().double().cpu().numpy()
            )
            selected = list(SELECTED_GATES)
            pre = (
                hf_capture["pre"][0, position, selected].detach().double().cpu().numpy()
                - anchor.pre[0, position, selected].detach().double().cpu().numpy()
            )
            post = (
                hf_capture["post"][0, position, selected].detach().double().cpu().numpy()
                - anchor.post[0, position, selected].detach().double().cpu().numpy()
            )
            margin_error = task_margin_error(
                hf_year.detach().double().cpu().numpy(),
                tl_year.detach().double().cpu().numpy(),
                row.y,
            )
            prompt_errors = {
                "raw_year_logits": raw,
                "centered_year_logits": centered,
                "resid_mid": resid,
                "selected_pre": pre,
                "selected_post": post,
            }
            for name, error in prompt_errors.items():
                families[name].append(error)
            margin_errors.append(margin_error)
            per_prompt.append({
                "pair_digest": row.pair_digest,
                "system": system,
                "clean_suffix": row.y,
                **{
                    name + "_max_abs": float(np.max(np.abs(error)))
                    for name, error in prompt_errors.items()
                },
                "task_margin_error": margin_error,
                "task_margin_abs_error": abs(margin_error),
            })
            references[row.pair_digest + "|" + system] = {
                "year_logits": tl_year.detach().double().cpu().numpy().tolist(),
                "mlp8_out": anchor.mlp8_out[0].detach().float().cpu().numpy().tolist(),
            }
    metrics = {name: pooled_error_metrics(errors) for name, errors in families.items()}
    margin_array = np.asarray(margin_errors, dtype=np.float64)
    metrics["task_margin"] = {
        "max_abs": float(np.max(np.abs(margin_array))),
        "rms": float(np.sqrt(np.mean(margin_array**2))),
    }
    thresholds = gate04_thresholds()
    passed = all_gate04_submetrics_pass(metrics, thresholds)
    result = {
        "audit_version": "hf-tl-fidelity-v2",
        "hf_attention_implementation": getattr(
            hf_model.config, "_attn_implementation", None
        ),
        "batch_size": 1,
        **panel,
        "n_pairs": len(holdout),
        "n_prompts": len(per_prompt),
        "weight_mapping": weight_mapping,
        "metrics": metrics,
        "thresholds": thresholds,
        "per_prompt": per_prompt,
        "hf_tl_error_enters_epsilon_y": False,
        "passed": passed,
        "tl_references": references,
    }
    if not passed:
        raise GreenStop("04_HF_TL_FIDELITY", canonical_json(metrics))
    return result


def no_op_audit(model, tokenizer, holdout, suffix_ids, device: str, references: dict) -> dict:
    torch = torch_module()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    errors = []
    with torch.inference_mode():
        for row in holdout:
            for system, prompt in (("clean", row.clean_prompt), ("corrupt", row.corrupt_prompt)):
                tokens = tokenize_one(tokenizer, prompt, device)
                cached = references[row.pair_digest + "|" + system]
                reference = torch.as_tensor(
                    cached["year_logits"], dtype=torch.float64, device=device
                )
                block8 = torch.as_tensor(cached["mlp8_out"], dtype=torch.float32, device=device)[None]
                replay = capture_tail_anchor(
                    model, tokens, suffix_tensor, system=system, block8_patch=block8
                )
                errors.append(float((reference - replay.year_logits[0].double()).abs().max()))
    result = {"n": len(errors), "max_abs": max(errors), "errors": errors}
    if result["max_abs"] > THRESHOLDS.no_op_max_abs:
        raise GreenStop("05_HOOK_NOOP", f"max error {result['max_abs']:.3e}")
    return result


def donor_anchors(model, tokenizer, suffix_ids, records: Sequence[PairRecord], device: str) -> dict:
    """Collect only CPU float64 statistics; donor activations never enter scores."""
    chords, clean_pre, corrupt_pre, rms_anchor, metadata = [], [], [], [], []
    torch = torch_module()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    audit_digests = {
        row.pair_digest for row in sorted(
            records, key=lambda row: sha256_text("tail-audit|" + row.pair_digest)
        )[:32]
    }
    audit_anchors = {}
    with torch.inference_mode():
        for index, row in enumerate(records):
            clean = capture_tail_anchor(
                model, tokenize_one(tokenizer, row.clean_prompt, device), suffix_tensor, system="donor-clean"
            )
            corrupt = capture_tail_anchor(
                model, tokenize_one(tokenizer, row.corrupt_prompt, device), suffix_tensor, system="donor-corrupt"
            )
            if row.pair_digest in audit_digests:
                audit_anchors[row.pair_digest] = _anchor_to_plain(clean)
            clean_resid = selected_position(clean, "resid_mid").double().cpu().numpy()[0]
            corrupt_resid = selected_position(corrupt, "resid_mid").double().cpu().numpy()[0]
            clean_gate = selected_position(clean, "pre").double().cpu().numpy()[0, list(SELECTED_GATES)]
            corrupt_gate = selected_position(corrupt, "pre").double().cpu().numpy()[0, list(SELECTED_GATES)]
            chords.append(clean_resid - corrupt_resid)
            clean_pre.append(clean_gate)
            corrupt_pre.append(corrupt_gate)
            rms_anchor.append([
                float(np.sqrt(np.mean(clean_resid**2))),
                float(np.sqrt(np.mean(corrupt_resid**2))),
            ])
            metadata.append({
                "noun": row.noun,
                "century": row.century,
                "distance_bin": row.distance_bin,
                "role": row.role,
                "pair_digest": row.pair_digest,
            })
            if (index + 1) % 64 == 0:
                print(f"donors {index + 1}/{len(records)}", flush=True)
    return {
        "chords": np.stack(chords), "clean_pre": np.stack(clean_pre),
        "corrupt_pre": np.stack(corrupt_pre), "rms_anchor": np.array(rms_anchor),
        "metadata": metadata,
        "audit_anchors": audit_anchors,
    }


def build_basis_and_radii(donor: dict, output_root: Path) -> tuple[np.ndarray, dict]:
    roles = np.array([row["role"] for row in donor["metadata"]])
    nouns = np.array([row["noun"] for row in donor["metadata"]])
    fit_mask = roles == "basis_fit"
    holdout_mask = roles == "basis_holdout"
    radius_mask = roles == "radius_v2"
    if (int(fit_mask.sum()), int(holdout_mask.sum()), int(radius_mask.sum())) != (512, 256, 512):
        raise GreenStop("07_DONOR_V2_PLAN", "donor role counts changed after capture")
    try:
        fitted = fit_rank5_basis(
            donor["chords"][fit_mask],
            donor["chords"][holdout_mask],
            nouns[fit_mask],
            BASIS_V2_DONOR_NOUNS,
        )
        radii = construct_rank5_radii(
            donor["chords"][radius_mask],
            donor["clean_pre"][radius_mask],
            donor["corrupt_pre"][radius_mask],
            donor["rms_anchor"][radius_mask],
            nouns[radius_mask],
            fitted["U"],
            BASIS_V2_DONOR_NOUNS,
        )
    except BasisAuditError as exc:
        raise GreenStop(exc.gate, exc.detail) from exc
    pair_digests = np.array([row["pair_digest"] for row in donor["metadata"]])
    fit_matrix = donor["chords"][fit_mask].astype(np.float64, copy=False)
    holdout_matrix = donor["chords"][holdout_mask].astype(np.float64, copy=False)
    radius_matrix = donor["chords"][radius_mask].astype(np.float64, copy=False)
    matrix_hashes = {
        "fit": matrix_sha256(fit_matrix),
        "holdout": matrix_sha256(holdout_matrix),
        "radius": matrix_sha256(radius_matrix),
        "shapes": {
            "fit": list(fit_matrix.shape),
            "holdout": list(holdout_matrix.shape),
            "radius": list(radius_matrix.shape),
        },
        "dtype": "float64",
        "finite": bool(
            np.isfinite(fit_matrix).all()
            and np.isfinite(holdout_matrix).all()
            and np.isfinite(radius_matrix).all()
        ),
    }
    write_json_atomic(output_root / "donor_v2_matrix_hashes.json", matrix_hashes)
    np.savez(
        output_root / "donor_basis.npz",
        U=fitted["U"],
        projector=fitted["projector"],
        singular_fit=fitted["singular_fit"],
        singular_holdout=fitted["singular_holdout"],
        U_holdout=fitted["U_holdout"],
        leave_one_names=np.array(BASIS_V2_DONOR_NOUNS),
        leave_one_bases=fitted["leave_one_bases"],
        leave_one_angles=np.array([
            fitted["leave_one_angles"][noun] for noun in BASIS_V2_DONOR_NOUNS
        ]),
        fit_pair_digests=pair_digests[fit_mask],
        holdout_pair_digests=pair_digests[holdout_mask],
    )
    np.savez(
        output_root / "basis_bootstrap.npz",
        sampled_noun_indices=fitted["sampled_noun_indices"],
        angles_degrees=fitted["bootstrap_angles"],
        rank_floor_failures=np.array(fitted["bootstrap_floor_failures"], dtype=np.int64),
    )
    basis_audit = fitted["audit"] | {"matrix_hashes": matrix_hashes}
    write_json_atomic(output_root / "basis_audit.json", basis_audit)
    write_json_atomic(output_root / "radii.json", radii)
    return fitted["U"], radii


def target_basis_stability(model, tokenizer, suffix_ids, output_root: Path, radii: dict, records, device: str) -> dict:
    torch = torch_module()
    completed = output_root / "target_basis_stability.json"
    cache_path = output_root / ".development_anchor_cache.pt"
    if completed.is_file() and cache_path.is_file():
        return json.loads(completed.read_text(encoding="utf-8"))
    archive = np.load(output_root / "donor_basis.npz", allow_pickle=False)
    full_U = archive["U"]
    names = archive["leave_one_names"].tolist()
    bases = [full_U, *list(archive["leave_one_bases"])]
    selected_records = []
    for cid in sorted({row.cell_id for row in records}):
        energy = [row for row in records if row.cell_id == cid and row.role == "energy"]
        selected_records.append(min(energy, key=lambda row: sha256_text("basis-target|" + row.pair_digest)))
    cached = []
    serialized = {}
    for row in selected_records:
        anchors, _, _ = capture_item_systems(model, tokenizer, suffix_ids, row, device)
        cached.append((row, anchors))
        serialized[row.pair_digest] = {
            system: _anchor_to_plain(anchor) for system, anchor in anchors.items()
        }
    torch.save(serialized, cache_path)
    vectors = []
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        for basis_np in bases:
            basis = torch.as_tensor(basis_np, dtype=torch.float32, device=device)
            values = []
            for row, anchors in cached:
                chord = (selected_position(anchors["tar"], "resid_mid") - selected_position(anchors["cor"], "resid_mid"))[0]
                q = (basis.T @ chord).double()
                direction = radii["h1"] * q / torch.clamp(torch.linalg.vector_norm(q), min=1e-30)
                effects = {}
                for system in ("tar", "pat"):
                    anchor = anchors[system]
                    target_anchor = TargetAnchor(anchor.resid_mid, anchor.pre, anchor.post, anchor.final_positions, system)
                    effects[system] = float(finite_path_effect(
                        model, target_anchor, basis, suffix_tensor, direction.float()[None], [row.y], rho=1.0
                    )[0].item())
                values.append(abs(effects["pat"] - effects["tar"]))
            vectors.append(np.array(values, dtype=np.float64))
    full = vectors[0]
    results = {}
    for name, value in zip(names, vectors[1:]):
        symmetric = np.abs(full - value) / np.maximum((np.abs(full) + np.abs(value)) / 2, 0.05)
        row = {"spearman": spearman(full, value), "median_change": float(np.median(symmetric))}
        results[name] = row
        if row["spearman"] < 0.90 or row["median_change"] > 0.20:
            raise GreenStop("08_TARGET_BASIS_STABILITY", f"leave-{name}: {row}")
    payload = {"n_cells": len(selected_records), "leave_one": results}
    write_json_atomic(completed, payload)
    return payload


def full_hook_endpoint_physical(
    model,
    tokens,
    suffix_ids,
    anchor: TailAnchor,
    residual_delta,
    z,
    *,
    mode: str,
    gate_slot: int | None = None,
    block8_patch=None,
    subtract_residual_bypass: bool = False,
    return_trace: bool = False,
):
    """Independent full-model physical-vector reference endpoint."""
    torch = torch_module()
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=tokens.device)
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    positions = torch.full_like(rows, tokens.shape[1] - 1)
    gate_ids = torch.as_tensor(SELECTED_GATES, dtype=torch.long, device=tokens.device)
    counts = {"x": 0, "z": 0, "post": 0, "subtract": 0, "patch": 0}
    trace = {}

    def remember(name, value):
        if return_trace:
            trace[name] = value.detach()

    def assert_untouched(before, after, zero_declared, label):
        difference = (after - before).clone()
        zero_declared(difference)
        maximum = float(difference.abs().max().item())
        if maximum > THRESHOLDS.hook_untouched_max:
            raise GreenStop("05_HOOK_UNTOUCHED", f"{label} changed an undeclared entry by {maximum}")

    def patch_hook(value, hook):
        counts["patch"] += 1
        result = value.clone()
        result[rows, positions] = block8_patch
        assert_untouched(value, result, lambda d: d.__setitem__((rows, positions), 0), "patch")
        return result

    def x_hook(value, hook):
        counts["x"] += 1
        result = value.clone()
        result[rows, positions] += residual_delta.to(result.dtype)
        assert_untouched(value, result, lambda d: d.__setitem__((rows, positions), 0), "x")
        remember("resid_mid_after_x", result)
        return result

    def z_hook(value, hook):
        counts["z"] += 1
        result = value.clone()
        if mode == "joint":
            result[rows[:, None], positions[:, None], gate_ids[None]] += z.to(result.dtype)
        elif mode == "path":
            result[rows, positions, SELECTED_GATES[gate_slot]] += z.reshape(-1).to(result.dtype)
        if mode == "joint":
            assert_untouched(
                value, result,
                lambda d: d.__setitem__((rows[:, None], positions[:, None], gate_ids[None]), 0), "z-joint"
            )
        elif mode == "path":
            gate = SELECTED_GATES[gate_slot]
            assert_untouched(value, result, lambda d: d.__setitem__((rows, positions, gate), 0), "z-path")
        else:
            assert_untouched(value, result, lambda d: None, "z-control")
        remember("pre_after_z", result)
        return result

    def post_hook(value, hook):
        counts["post"] += 1
        result = value.clone()
        result[rows, positions, :] = anchor.post[rows, anchor.final_positions, :]
        if mode in {"path", "joint"}:
            if mode == "path":
                gate = SELECTED_GATES[gate_slot]
                result[rows, positions, gate] = value[rows, positions, gate]
            else:
                result[rows[:, None], positions[:, None], gate_ids[None]] = value[
                    rows[:, None], positions[:, None], gate_ids[None]
                ]
        else:
            gate = SELECTED_GATES[gate_slot]
            controlled = anchor.pre[rows, anchor.final_positions, gate] + z.reshape(-1)
            result[rows, positions, gate] = model.blocks[10].mlp.act_fn(controlled)
        assert_untouched(value, result, lambda d: d.__setitem__((rows, positions), 0), "post")
        remember("post_after_anchor", result)
        return result

    def subtract_hook(value, hook):
        counts["subtract"] += 1
        remember("resid_post_before_subtraction", value)
        result = value.clone()
        result[rows, positions] -= residual_delta.to(result.dtype)
        assert_untouched(value, result, lambda d: d.__setitem__((rows, positions), 0), "subtract")
        remember("resid_post_after_subtraction", result)
        return result

    def capture_resid_post(value, hook):
        remember("resid_post_before_subtraction", value)
        remember("resid_post_after_subtraction", value)
        return value

    def capture_mlp_out(value, hook):
        remember("mlp_out", value)
        return value

    def capture_block11(value, hook):
        remember("block11_resid_post", value)
        return value

    def capture_unembed(value, hook):
        remember("unembed_pre_softcap_full", value)
        return value

    hooks = []
    if block8_patch is not None:
        hooks.append(("blocks.8.hook_mlp_out", patch_hook))
    hooks.extend([
        ("blocks.10.hook_resid_mid", x_hook),
        ("blocks.10.mlp.hook_pre", z_hook),
        ("blocks.10.mlp.hook_post", post_hook),
        ("blocks.10.hook_mlp_out", capture_mlp_out),
        ("blocks.11.hook_resid_post", capture_block11),
        ("unembed.hook_out", capture_unembed),
    ])
    if subtract_residual_bypass:
        hooks.append(("blocks.10.hook_resid_post", subtract_hook))
    else:
        hooks.append(("blocks.10.hook_resid_post", capture_resid_post))

    module_handles = []
    if return_trace:
        module_handles.append(model.blocks[10].ln2.register_forward_hook(
            lambda module, inputs, output: remember("ln2_output", output)
        ))
        module_handles.append(model.ln_final.register_forward_hook(
            lambda module, inputs, output: remember("ln_final_output", output)
        ))
    try:
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    finally:
        for handle in module_handles:
            handle.remove()
        model.reset_hooks()
    required = {"x": 1, "z": 1, "post": 1, "subtract": int(subtract_residual_bypass), "patch": int(block8_patch is not None)}
    if counts != required:
        raise GreenStop("05_HOOK_INVOCATION", f"hook counts {counts}, expected {required}")
    year_logits = gather_year_logits(model, logits, positions, suffix_tensor)
    if return_trace:
        trace["unembed_post_softcap_full"] = logits.detach()
        trace["year_logits"] = year_logits.detach()
        required_trace = {
            "resid_mid_after_x", "ln2_output", "pre_after_z",
            "post_after_anchor", "mlp_out", "resid_post_before_subtraction",
            "resid_post_after_subtraction", "block11_resid_post",
            "ln_final_output", "unembed_pre_softcap_full",
            "unembed_post_softcap_full", "year_logits",
        }
        if set(trace) != required_trace:
            raise GreenStop(
                "06B_MANUAL_TAIL_STAGE_TRACE",
                f"full-hook trace keys {sorted(trace)}, expected {sorted(required_trace)}",
            )
        return year_logits, trace
    return year_logits


def full_hook_endpoint(
    model, tokens, suffix_ids, anchor: TailAnchor, U, x, z, *, mode: str,
    gate_slot: int | None = None, block8_patch=None,
    subtract_residual_bypass: bool = False, return_trace: bool = False,
):
    """Coordinate wrapper around the independent physical-vector reference."""
    residual_delta = x @ U.T
    return full_hook_endpoint_physical(
        model,
        tokens,
        suffix_ids,
        anchor,
        residual_delta,
        z,
        mode=mode,
        gate_slot=gate_slot,
        block8_patch=block8_patch,
        subtract_residual_bypass=subtract_residual_bypass,
        return_trace=return_trace,
    )


def tail_audit(model, tokenizer, suffix_ids, U_np, radii: dict, records, device: str, cached_anchors: dict) -> dict:
    torch = torch_module()
    U = torch.as_tensor(U_np, dtype=torch.float32, device=device)
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    tail = GreenBridgeTail(model, U, suffix_tensor)
    errors, derivative_errors = [], []
    kinds = ["center"] * 8 + ["x"] * 8 + ["z"] * 8 + ["path"] * 4 + ["control"] * 4
    ranked = sorted(records, key=lambda row: sha256_text("tail-audit|" + row.pair_digest))[:32]
    with torch.inference_mode():
        for index, (row, kind) in enumerate(zip(ranked, kinds)):
            tokens = tokenize_one(tokenizer, row.clean_prompt, device)
            plain = cached_anchors[row.pair_digest]
            anchor = TailAnchor(**{
                key: (value.to(device) if hasattr(value, "to") else value)
                for key, value in plain.items()
            })
            x = torch.zeros((1, DIMENSIONS.residual_rank), dtype=torch.float32, device=device)
            z = torch.zeros(1, dtype=torch.float32, device=device)
            gate_slot = index % 10
            mode = "path"
            if kind == "x":
                x[0, index % DIMENSIONS.residual_rank] = radii["h1"]
            elif kind == "z":
                z[0] = radii["h2"][gate_slot]
            elif kind in {"path", "control"}:
                mode = kind
                x[0, index % DIMENSIONS.residual_rank] = radii["h1"]
                z[0] = radii["h2"][gate_slot]
            manual = tail.evaluate(anchor, x, z, mode=mode, gate_slot=gate_slot)
            full = full_hook_endpoint(
                model, tokens, suffix_ids, anchor, U, x, z, mode=mode, gate_slot=gate_slot
            )
            errors.append(float((manual.double() - full.double()).abs().max().item()))
            manual_delta = manual.double() - anchor.year_logits.double()
            full_delta = full.double() - anchor.year_logits.double()
            relative = float(
                torch.linalg.vector_norm(manual_delta - full_delta).item()
                / max(float(torch.linalg.vector_norm(full_delta).item()), 1e-5)
            )
            derivative_errors.append(relative)
    result = {
        "n": 32, "condition_types": kinds, "max_abs": max(errors), "errors": errors,
        "max_derivative_relative": max(derivative_errors),
        "derivative_relative_errors": derivative_errors,
    }
    if result["max_abs"] > THRESHOLDS.tail_max_abs:
        raise GreenStop("06_MANUAL_TAIL", f"max error {result['max_abs']:.3e}")
    if result["max_derivative_relative"] > THRESHOLDS.tail_derivative_relative:
        raise GreenStop("06_MANUAL_TAIL_DERIVATIVE", f"relative error {result['max_derivative_relative']:.3e}")
    return result


def _jet_at_radius(tail: GreenBridgeTail, anchor: TailAnchor, gate_slot: int, hx: float, hz: float, center) -> GateJet:
    """Evaluate the exact rank-five 52-condition design in two batched forwards."""
    torch = torch_module()
    device = anchor.resid_mid.device
    rank = DIMENSIONS.residual_rank
    path_x, path_z = [], []
    # Two z-axis endpoints.
    for sz in (1.0, -1.0):
        path_x.append(np.zeros(rank)); path_z.append(sz * hz)
    # Ten x-axis endpoints.
    for axis in range(rank):
        for sx in (1.0, -1.0):
            value = np.zeros(rank); value[axis] = sx * hx
            path_x.append(value); path_z.append(0.0)
    # Twenty path corners, ordered by axis then (++,+-,-+,--).
    for axis in range(rank):
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            value = np.zeros(rank); value[axis] = sx * hx
            path_x.append(value); path_z.append(sz * hz)
    control_x, control_z = [], []
    for axis in range(rank):
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            value = np.zeros(rank); value[axis] = sx * hx
            control_x.append(value); control_z.append(sz * hz)
    px = torch.as_tensor(np.stack(path_x), dtype=torch.float32, device=device)
    pz = torch.as_tensor(path_z, dtype=torch.float32, device=device)
    cx = torch.as_tensor(np.stack(control_x), dtype=torch.float32, device=device)
    cz = torch.as_tensor(control_z, dtype=torch.float32, device=device)
    path_count = 2 + 6 * rank
    control_count = 4 * rank
    path = tail.evaluate(_repeat_anchor(anchor, path_count), px, pz, mode="path", gate_slot=gate_slot).double()
    control = tail.evaluate(_repeat_anchor(anchor, control_count), cx, cz, mode="control", gate_slot=gate_slot).double()
    G = (path[0] - path[1]) / (2 * hz)
    C = (path[0] - 2 * center + path[1]) / (hz * hz)
    J, HP, HC = [], [], []
    mixed_start = 2 + 2 * rank
    for axis in range(rank):
        J.append((path[2 + 2 * axis] - path[3 + 2 * axis]) / (2 * hx))
        p = path[mixed_start + 4 * axis:mixed_start + 4 + 4 * axis]
        c = control[4 * axis:4 + 4 * axis]
        HP.append((p[0] - p[1] - p[2] + p[3]) / (4 * hx * hz))
        HC.append((c[0] - c[1] - c[2] + c[3]) / (4 * hx * hz))
    return GateJet(
        G.cpu().numpy(), C.cpu().numpy(), torch.stack(J).cpu().numpy(),
        torch.stack(HP).cpu().numpy(), torch.stack(HC).cpu().numpy(),
    )


def whitebox_A(model, anchor: TailAnchor, U, gate_slot: int) -> np.ndarray:
    """Architecture-derived audit only; its return value never enters P."""
    gate = SELECTED_GATES[gate_slot]
    resid = selected_position(anchor, "resid_mid")[0].detach().double().cpu().numpy()
    basis = U.detach().double().cpu().numpy()
    weight = model.blocks[10].ln2.w.detach().double().cpu().numpy()
    w_in = model.blocks[10].mlp.W_in[:, gate].detach().double().cpu().numpy()
    centered = resid - resid.mean()
    variance = float(np.mean(centered**2) + model.cfg.eps)
    scale = math.sqrt(variance)
    dimension = len(resid)
    projector = (
        np.eye(dimension, dtype=np.float64)
        - np.ones((dimension, dimension), dtype=np.float64) / dimension
        - np.outer(centered, centered) / (dimension * variance)
    )
    jacobian_ln = (weight[:, None] * projector) / scale
    return w_in @ jacobian_ln @ basis


def classify_gate(full, half, rich, wb_A, contrast_norm: float, delta_norm: float, epsilon_y: float, hx: float, hz: float) -> tuple[str, dict]:
    numerical = richardson_numerical_bounds(
        rich, half, epsilon_y=epsilon_y, h1=hx, h2=hz
    )
    rich_gate_response_norm = float(np.linalg.norm(rich.G))
    full_gate_response_norm = float(np.linalg.norm(full.G))
    half_gate_response_norm = float(np.linalg.norm(half.G))
    rich_curvature_norm = float(np.linalg.norm(rich.C))
    wb_norm = np.linalg.norm(wb_A)
    full_id = half_id = rich_id = None
    if numerical.inverse_admissible:
        try:
            full_id, half_id, rich_id = (
                identify_gate(full), identify_gate(half), identify_gate(rich)
            )
        except ValueError:
            pass
    wb_error = math.inf if rich_id is None else float(np.linalg.norm(rich_id.A - wb_A))
    wb_ok = (
        rich_id is not None
        and wb_error <= THRESHOLDS.whitebox_a_relative_max * max(wb_norm, 1e-6)
    )
    if rich_id is not None and wb_norm < 1e-6:
        wb_ok = wb_error <= THRESHOLDS.whitebox_a_small_absolute_max
    active = (
        rich_id is not None
        and rich_curvature_norm / 10 >= THRESHOLDS.curvature_rms_min
        and rich_curvature_norm >= THRESHOLDS.curvature_snr_min * numerical.epsilon_C
        and rich_gate_response_norm / 10 >= THRESHOLDS.gate_response_rms_min
        and rich_gate_response_norm >= THRESHOLDS.gate_response_snr_min * numerical.epsilon_G
        and rich_id.factorization_residual <= THRESHOLDS.factorization_residual_max
        and wb_ok
        and cosine(full_id.P, half_id.P) >= THRESHOLDS.tensor_cosine_min
        and symmetric_relative_change(full_id.P, half_id.P) <= THRESHOLDS.tensor_symmetric_change_max
        and np.linalg.norm(rich_id.P - half_id.P) / max(np.linalg.norm(rich_id.P), 1e-8)
            <= THRESHOLDS.richardson_change_max
        and np.linalg.norm(rich_id.P) >= THRESHOLDS.tensor_snr_min * numerical.epsilon_P_F
    )
    null_bound = certified_null_bound(
        contrast_norm, delta_norm, rich_gate_response_norm,
        numerical.epsilon_G, wb_norm,
    )
    full_bound = contrast_norm * delta_norm * full_gate_response_norm * wb_norm
    half_bound = contrast_norm * delta_norm * half_gate_response_norm * wb_norm
    null = (
        rich_gate_response_norm <= 5 * numerical.epsilon_G
        and null_bound <= 0.005
        and abs(full_bound - half_bound) <= 0.005
    )
    label = "active-identified" if active else ("certified-target-null" if null else "invalid")
    audit = {
        "label": label, "curvature_norm": rich_curvature_norm,
        "gate_response_norm": rich_gate_response_norm,
        "factorization_residual": (
            None if rich_id is None else rich_id.factorization_residual
        ),
        "whitebox_error": None if not np.isfinite(wb_error) else float(wb_error),
        "whitebox_norm": float(wb_norm),
        "full_half_cosine": (
            None if rich_id is None else cosine(full_id.P, half_id.P)
        ),
        "full_half_change": (
            None if rich_id is None else symmetric_relative_change(full_id.P, half_id.P)
        ),
        "richardson_change": (
            None if rich_id is None else float(
                np.linalg.norm(rich_id.P - half_id.P)
                / max(np.linalg.norm(rich_id.P), 1e-8)
            )
        ),
        "null_bound": float(null_bound),
        "null_full_half_bound_change": float(abs(full_bound - half_bound)),
        "inverse_admissible": numerical.inverse_admissible,
        "eta_G": numerical.eta_G,
        "eta_C": numerical.eta_C,
        "eta_J": numerical.eta_J,
        "eta_H": numerical.eta_H,
        "epsilon_G": numerical.epsilon_G,
        "epsilon_C": numerical.epsilon_C,
        "epsilon_delta_H": numerical.epsilon_delta_H.tolist(),
        "A_max": [float(value) if np.isfinite(value) else None for value in numerical.A_max],
        "epsilon_A": [float(value) if np.isfinite(value) else None for value in numerical.epsilon_A],
        "epsilon_P": [float(value) if np.isfinite(value) else None for value in numerical.epsilon_P],
        "epsilon_P_F": (
            numerical.epsilon_P_F if np.isfinite(numerical.epsilon_P_F) else None
        ),
    }
    return label, {
        "audit": audit,
        "full": full_id,
        "half": half_id,
        "rich": rich_id,
        "numerical": numerical,
    }


def mixed_system(tail, model, anchor, U, radii, delta, contrast, epsilon_y: float) -> dict:
    torch = torch_module()
    zero_x = torch.zeros(
        (1, DIMENSIONS.residual_rank), dtype=torch.float32,
        device=anchor.resid_mid.device,
    )
    zero_z = torch.zeros(1, dtype=torch.float32, device=anchor.resid_mid.device)
    center = tail.evaluate(anchor, zero_x, zero_z, mode="path", gate_slot=0)[0].double()
    center_error = center - anchor.year_logits[0].double()
    center_rms = float(torch.sqrt(torch.mean(center_error**2)).item())
    center_max = float(center_error.abs().max().item())
    if center_rms > THRESHOLDS.center_rms or center_max > THRESHOLDS.center_max_abs:
        return {"theta": 0.0, "theta_full": 0.0, "theta_half": 0.0, "active_gates": 0,
                "all_valid": False, "bypass_disagreement": None, "gates": [],
                "admissible": False, "center_rms": center_rms, "center_max": center_max,
                "theta_error": None}
    gates, theta = [], 0.0
    theta_full, theta_half = 0.0, 0.0
    direct, gate_error_bounds = [], []
    for gate_slot in range(10):
        full = _jet_at_radius(tail, anchor, gate_slot, radii["h1"], radii["h2"][gate_slot], center)
        half = _jet_at_radius(tail, anchor, gate_slot, radii["h1"] / 2, radii["h2"][gate_slot] / 2, center)
        rich = extrapolate_gate_jet(full, half)
        wb = whitebox_A(model, anchor, U, gate_slot)
        label, values = classify_gate(
            full, half, rich, wb, float(np.linalg.norm(contrast)), float(np.linalg.norm(delta)),
            epsilon_y, radii["h1"], radii["h2"][gate_slot],
        )
        if label == "active-identified":
            for key, accumulator in (("rich", "theta"), ("full", "theta_full"), ("half", "theta_half")):
                contraction = contrast @ (delta @ values[key].P)
                if accumulator == "theta": theta += contraction
                elif accumulator == "theta_full": theta_full += contraction
                else: theta_half += contraction
            direct.append(values["rich"].D)
            gate_error_bounds.append(active_contraction_bound(
                float(np.linalg.norm(contrast)),
                float(np.linalg.norm(delta)),
                values["numerical"].epsilon_P_F,
            ))
        elif label == "certified-target-null":
            gate_error_bounds.append(values["audit"]["null_bound"])
        gates.append(values["audit"])
    active = sum(row["label"] == "active-identified" for row in gates)
    complete = all(row["label"] != "invalid" for row in gates)
    bypass = None
    if direct:
        stack = np.stack(direct)
        mean = stack.mean(axis=0)
        bypass = float(np.sqrt(np.mean((stack - mean) ** 2)) / max(np.sqrt(np.mean(mean**2)), 1e-12))
    return {
        "theta": float(theta), "theta_full": float(theta_full), "theta_half": float(theta_half),
        "theta_error": sum_item_error_bounds(gate_error_bounds) if complete else None,
        "active_gates": active, "all_valid": complete,
        "bypass_disagreement": bypass, "gates": gates,
        "admissible": (
            complete
            and active >= THRESHOLDS.active_gates_min
            and bypass is not None
            and bypass <= THRESHOLDS.bypass_disagreement_max
        ),
        "center_rms": center_rms, "center_max": center_max,
    }


def joint_margins(tail, anchor, xs: np.ndarray, zs: np.ndarray, contrast) -> np.ndarray:
    torch = torch_module()
    device = anchor.resid_mid.device
    device_name = torch.cuda.get_device_name(device)
    batch_limit = 512 if "4090" in device_name else 1024
    outputs = []
    for start in range(0, len(xs), batch_limit):
        stop = min(start + batch_limit, len(xs))
        x = torch.as_tensor(xs[start:stop], dtype=torch.float32, device=device)
        z = torch.as_tensor(zs[start:stop], dtype=torch.float32, device=device)
        logits = tail.evaluate(_repeat_anchor(anchor, stop - start), x, z, mode="joint")
        outputs.append(margin(logits, contrast).detach().cpu().numpy())
    return np.concatenate(outputs)


def first_order_system(tail, anchor, radii, directions: np.ndarray, contrast) -> dict:
    xs, zs, descriptors = [], [], []
    for radius_name, rho in (("full", 1.0), ("half", 0.5)):
        for kind in ("x", "z"):
            count = FIRST_ORDER_RESIDUAL_DIRECTIONS if kind == "x" else 10
            for axis in range(count):
                for sign in (1.0, -1.0):
                    x = np.zeros(DIMENSIONS.residual_rank); z = np.zeros(10)
                    if kind == "x": x = sign * rho * radii["h1"] * directions[axis]
                    else: z[axis] = sign * rho * radii["h2"][axis]
                    xs.append(x); zs.append(z); descriptors.append((radius_name, kind, axis, sign, rho))
    values = joint_margins(tail, anchor, np.stack(xs), np.stack(zs), contrast)
    response = {
        "full": {"x": np.zeros(FIRST_ORDER_RESIDUAL_DIRECTIONS), "z": np.zeros(10)},
        "half": {"x": np.zeros(FIRST_ORDER_RESIDUAL_DIRECTIONS), "z": np.zeros(10)},
    }
    endpoints = {}
    for value, descriptor in zip(values, descriptors):
        radius_name, kind, axis, sign, rho = descriptor
        endpoints[radius_name, kind, axis, sign] = value
    for radius_name, rho in (("full", 1.0), ("half", 0.5)):
        for kind, count in (("x", FIRST_ORDER_RESIDUAL_DIRECTIONS), ("z", 10)):
            for axis in range(count):
                response[radius_name][kind][axis] = (
                    endpoints[radius_name, kind, axis, 1.0]
                    - endpoints[radius_name, kind, axis, -1.0]
                ) / (2 * rho)
    rich = {
        kind: (4 * response["half"][kind] - response["full"][kind]) / 3
        for kind in ("x", "z")
    }
    return {"response": response, "rich": rich}


def factorial_system(tail, anchor, delta, zeta, contrast) -> dict:
    xs, zs, descriptors = [], [], []
    for radius_name, rho in (("full", 1.0), ("half", 0.5)):
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            xs.append(sx * rho * delta); zs.append(sz * rho * zeta)
            descriptors.append((radius_name, rho, sx, sz))
    values = joint_margins(tail, anchor, np.stack(xs), np.stack(zs), contrast)
    endpoints = {(name, sx, sz): value for value, (name, _, sx, sz) in zip(values, descriptors)}
    rows = {}
    for name, rho in (("full", 1.0), ("half", 0.5)):
        pp, pm = endpoints[name, 1, 1], endpoints[name, 1, -1]
        mp, mm = endpoints[name, -1, 1], endpoints[name, -1, -1]
        rows[name] = {
            "single": (pp - mm) / (2 * rho),
            "pie": (pp - pm - mp + mm) / (4 * rho * rho),
            "bx": (pp + pm - mp - mm) / (4 * rho),
            "bz": (pp - pm + mp - mm) / (4 * rho),
        }
    return {key: (4 * rows["half"][key] - rows["full"][key]) / 3 for key in rows["full"]} | {
        "raw": rows
    }


def tensor_item(model, tokenizer, suffix_ids, U_np, radii, record, device, epsilon_y, directions) -> dict:
    torch = torch_module()
    anchors, _, _ = capture_item_systems(model, tokenizer, suffix_ids, record, device)
    U = torch.as_tensor(U_np, dtype=torch.float32, device=device)
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    tail = GreenBridgeTail(model, U, suffix_tensor)
    contrast_t = margin_vector(record.y, device)
    contrast = contrast_t.cpu().numpy()
    chord = (selected_position(anchors["tar"], "resid_mid") - selected_position(anchors["cor"], "resid_mid"))[0]
    q = (U.T @ chord).double().cpu().numpy()
    q_norm = float(np.linalg.norm(q))
    direction_valid = q_norm >= THRESHOLDS.projected_chord_sigma_min * radii["sigma_x"]
    delta = radii["h1"] * q / max(q_norm, 1e-30)

    mixed = {}
    with torch.inference_mode():
        for system in ("tar", "pat"):
            mixed[system] = mixed_system(tail, model, anchors[system], U, radii, delta, contrast, epsilon_y)

    pre_tar = selected_position(anchors["tar"], "pre")[0, list(SELECTED_GATES)].double().cpu().numpy()
    pre_cor = selected_position(anchors["cor"], "pre")[0, list(SELECTED_GATES)].double().cpu().numpy()
    v = (pre_tar - pre_cor) / np.asarray(radii["h2"])
    zeta = np.asarray(radii["h2"]) * v / max(float(np.linalg.norm(v)), 1e-30)
    fo, factorial = {}, {}
    with torch.inference_mode():
        for system in ("tar", "pat"):
            fo[system] = first_order_system(tail, anchors[system], radii, directions, contrast_t)
            factorial[system] = factorial_system(tail, anchors[system], delta, zeta, contrast_t)
    center_tar = float(margin(anchors["tar"].year_logits, contrast_t)[0].item())
    center_pat = float(margin(anchors["pat"].year_logits, contrast_t)[0].item())
    first_order_score = math.sqrt(
        float(np.mean((fo["pat"]["rich"]["x"] - fo["tar"]["rich"]["x"]) ** 2))
        + float(np.mean((fo["pat"]["rich"]["z"] - fo["tar"]["rich"]["z"]) ** 2))
    )
    admissible = direction_valid and mixed["tar"]["admissible"] and mixed["pat"]["admissible"]
    return {
        "pair_digest": record.pair_digest, "cell_id": record.cell_id,
        "split": record.split, "distance_bin": record.distance_bin, "orientation": record.orientation,
        "admissible": admissible, "q_norm": q_norm,
        "theta_tar": mixed["tar"]["theta"], "theta_pat": mixed["pat"]["theta"],
        "theta_full_tar": mixed["tar"]["theta_full"], "theta_full_pat": mixed["pat"]["theta_full"],
        "theta_half_tar": mixed["tar"]["theta_half"], "theta_half_pat": mixed["pat"]["theta_half"],
        "theta_error_tar": mixed["tar"].get("theta_error", float("inf")),
        "theta_error_pat": mixed["pat"].get("theta_error", float("inf")),
        "behavioral": abs(center_pat - center_tar),
        "single": abs(factorial["pat"]["single"] - factorial["tar"]["single"]),
        "first_order": first_order_score,
        "pie": abs(factorial["pat"]["pie"] - factorial["tar"]["pie"]),
        "cancellation_dx": factorial["pat"]["bx"] - factorial["tar"]["bx"],
        "cancellation_dz": factorial["pat"]["bz"] - factorial["tar"]["bz"],
        "mixed_audit": {system: {key: value for key, value in mixed[system].items() if key != "gates"} | {"gates": mixed[system]["gates"]} for system in mixed},
    }


def energy_item(model, tokenizer, suffix_ids, U_np, radii, record, device, anchor_cache=None) -> dict:
    torch = torch_module()
    anchors, _, _ = capture_item_systems(
        model, tokenizer, suffix_ids, record, device, anchor_cache=anchor_cache
    )
    U = torch.as_tensor(U_np, dtype=torch.float32, device=device)
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    chord = (selected_position(anchors["tar"], "resid_mid") - selected_position(anchors["cor"], "resid_mid"))[0]
    q = (U.T @ chord).double().cpu().numpy()
    q_norm = float(np.linalg.norm(q))
    direction_valid = q_norm >= THRESHOLDS.projected_chord_sigma_min * radii["sigma_x"]
    delta_np = radii["h1"] * q / max(q_norm, 1e-30)
    direction = torch.as_tensor(delta_np[None], dtype=torch.float32, device=device)
    systems = {}
    for name in ("tar", "pat", "cor"):
        anchor = anchors[name]
        target_anchor = TargetAnchor(
            resid_mid=anchor.resid_mid, pre=anchor.pre, post=anchor.post,
            final_positions=anchor.final_positions, system=name,
        )
        full = finite_path_effect(
            model, target_anchor, U, suffix_tensor, direction, [record.y], rho=1.0
        )[0]
        half = finite_path_effect(
            model, target_anchor, U, suffix_tensor, direction, [record.y], rho=0.5
        )[0]
        jvp = target_jvp(model, target_anchor, U, suffix_tensor, direction, [record.y])[0]
        full_v, half_v, jvp_v = float(full.item()), float(half.item()), float(jvp.item())
        rich_v = (4 * half_v - full_v) / 3
        absolute = abs(rich_v - jvp_v)
        relative = absolute / max(abs(jvp_v), 0.05)
        locality = abs(full_v - rich_v)
        admissible = (
            absolute <= 0.01 and relative <= 0.05
            and locality <= max(0.02, 0.25 * abs(rich_v))
        )
        systems[name] = {
            "full": full_v, "half": half_v, "jvp": jvp_v,
            "richardson": rich_v,
            "jvp_absolute_error": absolute, "jvp_relative_error": relative,
            "locality_error": locality, "admissible": admissible,
        }
    return {
        "pair_digest": record.pair_digest, "cell_id": record.cell_id,
        "split": record.split, "distance_bin": record.distance_bin, "orientation": record.orientation,
        "q_norm": q_norm, "admissible": direction_valid and all(row["admissible"] for row in systems.values()),
        "systems": systems,
    }


def append_journal(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        _fsync_parent(path)


def read_journal(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_parquet(path: Path, rows: list[dict]) -> None:
    import pandas as pd
    flattened = []
    for row in rows:
        record = {}
        for key, value in row.items():
            record[key] = canonical_json(value) if isinstance(value, (dict, list)) else value
        flattened.append(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(flattened).to_parquet(temporary, index=False, engine="pyarrow")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)


def aggregate_cells(tensor_rows: list[dict], energy_rows: list[dict], *, dev_sd: float | None = None) -> tuple[dict, float]:
    cells = []
    cell_ids = sorted({row["cell_id"] for row in tensor_rows} | {row["cell_id"] for row in energy_rows})
    provisional = []
    for cid in cell_ids:
        tensor = [row for row in tensor_rows if row["cell_id"] == cid and row["admissible"]]
        energy = [row for row in energy_rows if row["cell_id"] == cid and row["admissible"]]
        survived = len(tensor) >= 6 and len(energy) >= 6
        if not survived:
            provisional.append({"cell_id": cid, "survived": False, "n_tensor": len(tensor), "n_energy": len(energy)})
            continue
        mean = lambda values: float(np.mean(list(values)))
        theta_tar = mean(row["theta_tar"] for row in tensor)
        theta_pat = mean(row["theta_pat"] for row in tensor)
        target_tar = mean(row["systems"]["tar"]["full"] for row in energy)
        target_pat = mean(row["systems"]["pat"]["full"] for row in energy)
        target_cor = mean(row["systems"]["cor"]["full"] for row in energy)
        row = {
            "cell_id": cid, "distance_bin": tensor[0]["distance_bin"], "survived": True,
            "n_tensor": len(tensor), "n_energy": len(energy),
            "mixed": abs(theta_pat - theta_tar),
            "mixed_full": abs(mean(r["theta_full_pat"] for r in tensor) - mean(r["theta_full_tar"] for r in tensor)),
            "mixed_half": abs(mean(r["theta_half_pat"] for r in tensor) - mean(r["theta_half_tar"] for r in tensor)),
            "target": abs(target_pat - target_tar),
            "conditioning_gap": abs(target_tar - target_cor),
            "baselines": {name: mean(r[name] for r in tensor) for name in BASELINES},
            "cancellation_dx": mean(r["cancellation_dx"] for r in tensor),
            "cancellation_dz": mean(r["cancellation_dz"] for r in tensor),
        }
        # Conservative cell error from duplicated-logit floor, used only as the
        # preregistered SNR gate; no observed target is fed into the predictor.
        row["error_bound"] = max(
            1e-7,
            cell_error_bound(
                (r["theta_error_tar"] for r in tensor),
                (r["theta_error_pat"] for r in tensor),
            ),
        )
        row["snr"] = row["mixed"] / row["error_bound"]
        provisional.append(row)
    surviving = [row for row in provisional if row["survived"]]
    if dev_sd is None:
        dev_sd = float(np.std([row["conditioning_gap"] for row in surviving], ddof=1)) if len(surviving) > 1 else 0.0
    for row in surviving:
        row["conditioned"] = (
            row["conditioning_gap"] >= THRESHOLDS.conditioning_absolute
            or row["conditioning_gap"] >= THRESHOLDS.conditioning_dev_sd * dev_sd
        )
    cells.extend(provisional)
    return {"cells": cells, "conditioning_dev_sd": dev_sd}, dev_sd


def run_split(model, tokenizer, suffix_ids, U, radii, records, split: str, output_root: Path, device: str, epsilon_y: float, dev_sd: float | None = None) -> dict:
    tensor_records = [row for row in records if row.role == "tensor"]
    energy_records = [row for row in records if row.role == "energy"]
    tensor_journal = output_root / f".{split}_tensor.journal.jsonl"
    energy_journal = output_root / f".{split}_energy.journal.jsonl"
    tensor_rows, energy_rows = read_journal(tensor_journal), read_journal(energy_journal)
    tensor_done = {row["pair_digest"] for row in tensor_rows}
    energy_done = {row["pair_digest"] for row in energy_rows}
    directions = first_order_directions()
    anchor_cache = merge_anchor_caches(
        load_anchor_cache(output_root / ".development_anchor_cache.pt", device)
        if split == "development" else {},
        load_anchor_cache(output_root / f".{split}_noise_anchor_cache.pt", device),
    )
    if split == "development" and not (output_root / "throughput_preflight.json").is_file():
        sample_tensor = sorted(
            tensor_records, key=lambda row: sha256_text("preflight-tensor|" + row.pair_digest)
        )[:math.ceil(0.02 * len(tensor_records))]
        sample_energy = sorted(
            energy_records, key=lambda row: sha256_text("preflight-energy|" + row.pair_digest)
        )[:math.ceil(0.02 * len(energy_records))]
        tensor_start = time.perf_counter()
        for record in sample_tensor:
            if record.pair_digest not in tensor_done:
                row = tensor_item(model, tokenizer, suffix_ids, U, radii, record, device, epsilon_y, directions)
                append_journal(tensor_journal, row); tensor_rows.append(row); tensor_done.add(record.pair_digest)
        tensor_seconds = time.perf_counter() - tensor_start
        energy_start = time.perf_counter()
        for record in sample_energy:
            if record.pair_digest not in energy_done:
                row = energy_item(model, tokenizer, suffix_ids, U, radii, record, device, anchor_cache)
                append_journal(energy_journal, row); energy_rows.append(row); energy_done.add(record.pair_digest)
        energy_seconds = time.perf_counter() - energy_start
        forecast_hours = (
            tensor_seconds / len(sample_tensor) * len(tensor_records)
            + energy_seconds / len(sample_energy) * len(energy_records)
        ) / 3600
        memory_gb = torch_module().cuda.max_memory_allocated() / 2**30
        ceiling, cap = (20.0, 24.0) if "4090" in torch_module().cuda.get_device_name() else (32.0, 40.0)
        write_json_atomic(output_root / "throughput_preflight.json", {
            "fraction": 0.02, "tensor_sample": len(sample_tensor), "energy_sample": len(sample_energy),
            "tensor_seconds": tensor_seconds, "energy_seconds": energy_seconds,
            "forecast_hours": forecast_hours, "peak_allocated_gb": memory_gb,
            "memory_ceiling_gb": ceiling, "hard_hours": cap, "outputs_reused": True,
        })
        if memory_gb > ceiling or forecast_hours > cap:
            raise GreenStop("10_THROUGHPUT_MEMORY", f"forecast={forecast_hours:.2f}h memory={memory_gb:.2f}GB")
    for index, record in enumerate(tensor_records):
        if record.pair_digest in tensor_done:
            continue
        row = tensor_item(model, tokenizer, suffix_ids, U, radii, record, device, epsilon_y, directions)
        append_journal(tensor_journal, row); tensor_rows.append(row)
        print(f"{split} tensor {len(tensor_rows)}/{len(tensor_records)} admissible={row['admissible']}", flush=True)
    for record in energy_records:
        if record.pair_digest in energy_done:
            continue
        row = energy_item(model, tokenizer, suffix_ids, U, radii, record, device, anchor_cache)
        append_journal(energy_journal, row); energy_rows.append(row)
        print(f"{split} energy {len(energy_rows)}/{len(energy_records)} admissible={row['admissible']}", flush=True)
    prefix = "dev" if split == "development" else "confirm"
    write_parquet(output_root / f"{prefix}_tensor_scores.parquet", tensor_rows)
    write_parquet(output_root / f"{prefix}_energy_targets.parquet", energy_rows)
    payload, derived_sd = aggregate_cells(tensor_rows, energy_rows, dev_sd=dev_sd)
    return payload | {"tensor_rows": len(tensor_rows), "energy_rows": len(energy_rows)}, derived_sd


def duplicate_noise_audit(model, tokenizer, suffix_ids, U_np, radii, records, device: str, phase: str, output_root: Path) -> dict:
    torch = torch_module()
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    U = torch.as_tensor(U_np, dtype=torch.float32, device=device)
    tail = GreenBridgeTail(model, U, suffix_tensor)
    n_full = 64 if phase == "development" else 32
    ranked = sorted(records, key=lambda row: sha256_text(f"noise-{phase}|" + row.pair_digest))
    errors, serialized, captured = [], {}, {}
    with torch.inference_mode():
        for row in ranked[:n_full]:
            tokens = tokenize_one(tokenizer, row.clean_prompt, device)
            first = capture_tail_anchor(model, tokens, suffix_tensor, system="noise")
            second = capture_tail_anchor(model, tokens, suffix_tensor, system="noise")
            errors.append(float((first.year_logits.double() - second.year_logits.double()).abs().max().item()))
            serialized[row.pair_digest] = {"tar": _anchor_to_plain(replace(first, system="tar"))}
            captured[row.pair_digest] = first
        for index, row in enumerate(ranked[:32]):
            tokens = tokenize_one(tokenizer, row.clean_prompt, device)
            anchor = captured[row.pair_digest]
            x = torch.zeros((1, DIMENSIONS.residual_rank), dtype=torch.float32, device=device)
            z = torch.zeros(1, dtype=torch.float32, device=device)
            x[0, index % DIMENSIONS.residual_rank] = radii["h1"] * (1 if index % 2 else -1)
            z[0] = radii["h2"][index % 10] * (1 if index % 3 else -1)
            first = tail.evaluate(anchor, x, z, mode="path", gate_slot=index % 10)
            second = tail.evaluate(anchor, x, z, mode="path", gate_slot=index % 10)
            errors.append(float((first.double() - second.double()).abs().max().item()))
    torch.save(serialized, output_root / f".{phase}_noise_anchor_cache.pt")
    maximum = max([0.0, *errors])
    result = {"phase": phase, "n_full": n_full, "n_tail": 32, "max_abs": maximum, "errors": errors}
    return result


def load_split_file(output_root: Path, filename: str = "splits.json") -> list[PairRecord]:
    payload = json.loads((output_root / filename).read_text(encoding="utf-8"))
    return [PairRecord(**row) for row in payload["records"]]


def load_frozen_numeric(output_root: Path, device: str):
    basis = np.load(output_root / "donor_basis.npz", allow_pickle=False)["U"]
    radii = json.loads((output_root / "radii.json").read_text(encoding="utf-8"))
    tokenizer, hf_model, model, cfg = load_models(device)
    suffix_ids = json.loads((output_root / "model_fingerprint.json").read_text(encoding="utf-8"))["tokenizer"]["suffix_ids"]
    return basis, radii, tokenizer, hf_model, model, suffix_ids, cfg


def prepare_v12_inactive(output_root: Path, device: str) -> None:
    repository = assert_clean_repository()
    assert_empty_prepare_root(output_root)
    environment = configure_runtime(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    pair_allowed = lambda first, second: token_pair_allowed(tokenizer, first, second)
    evaluation = build_evaluation_records(pair_allowed)
    validate_evaluation_plan(evaluation)
    legacy_donors = build_legacy_donor_records(pair_allowed)
    donors = inactive_v12_donor_records(pair_allowed)
    inactive_v12_validate_plan(donors)
    legacy_gate04, holdout_gate04 = gate04_record_panels(legacy_donors)
    gate04_panel = gate04_panel_metadata(legacy_gate04, holdout_gate04)
    if gate04_panel["ordered_prompt_keys_sha256"] != GATE04_ORDERED_PROMPT_HASH:
        raise GreenStop(
            "04_HF_TL_FIDELITY",
            "legacy Gate-04 ordered prompt panel hash changed",
        )
    evaluation_payload = plan_payload(evaluation)
    donor_v2_payload = basis_v2_plan_payload(donors)
    legacy_payload = plan_payload(legacy_donors)
    split_payload = evaluation_payload | {
        "basis_v2_ordered_pair_keys": donor_v2_payload["ordered_pair_keys"],
        "basis_v2_ordered_prompt_keys": donor_v2_payload["ordered_prompt_keys"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_run_ledger(output_root, repository)
    write_json_atomic(output_root / "splits.json", split_payload)
    write_json_atomic(output_root / "gate04_legacy_panel.json", gate04_panel)
    write_basis_v2_plan(output_root / "donor_v2_plan.json", donors)
    # Development gets its own physically separate view so its process never
    # parses confirmation prompt strings before the frozen analysis exists.
    write_json_atomic(
        output_root / "development_splits.json",
        plan_payload([row for row in evaluation if row.split == "development"]),
    )
    tokenizer, hf_model, model, cfg = load_models(device, tokenizer=tokenizer)
    suffix_ids, tokenizer_meta = validate_tokenizer(
        tokenizer, evaluation + legacy_donors + donors
    )
    manifest = write_initial_manifest(
        output_root,
        repository,
        environment,
        cfg,
        evaluation_payload,
        donor_v2_payload,
        legacy_payload["records_sha256"],
        gate04_panel,
    )
    fingerprint = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
        "config": cfg, "tokenizer": tokenizer_meta,
        "embedding_sha256": hashlib.sha256(model.W_E.detach().cpu().numpy().tobytes()).hexdigest(),
        "unembedding_sha256": hashlib.sha256(model.W_U.detach().cpu().numpy().tobytes()).hexdigest(),
    }
    write_json_atomic(output_root / "model_fingerprint.json", fingerprint)
    hf_audit = hf_tl_audit(
        tokenizer, hf_model, model, legacy_gate04, holdout_gate04,
        suffix_ids, device,
    )
    noop = no_op_audit(
        model, tokenizer, holdout_gate04, suffix_ids, device,
        hf_audit["tl_references"],
    )
    write_json_atomic(output_root / "hook_audit.json", {"hf_vs_tl": hf_audit, "no_op_patch": noop})
    del hf_model
    torch_module().cuda.empty_cache()
    donor = donor_anchors(model, tokenizer, suffix_ids, donors, device)
    U, radii = build_basis_and_radii(donor, output_root)
    tail_result = tail_audit(
        model, tokenizer, suffix_ids, U, radii, donors, device, donor["audit_anchors"]
    )
    write_json_atomic(output_root / "tail_audit.json", tail_result)
    manifest["artifact_sha256"] = {
        name: sha256_file(output_root / name)
        for name in (
            "model_fingerprint.json", "splits.json", "development_splits.json",
            "gate04_legacy_panel.json", "hook_audit.json", "donor_v2_plan.json",
            "donor_v2_matrix_hashes.json", "donor_basis.npz", "basis_audit.json",
            "basis_bootstrap.npz", "radii.json", "first_order_directions.npy",
            "tail_audit.json", "run_ledger.json",
        )
    }
    manifest["prepare_complete"] = True
    write_json_atomic(output_root / "manifest.json", manifest)


def verify_freeze(output_root: Path, require_confirmation: bool = False) -> dict:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise GreenStop("17_MANIFEST_FREEZE", "manifest.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "green-bridge-manifest-v1.3.5":
        raise GreenStop("17_MANIFEST_FREEZE", "manifest schema changed")
    run = manifest.get("run", {})
    if (
        run.get("protocol_id") != PROTOCOL_ID
        or run.get("parent_protocol_id") != PARENT_PROTOCOL_ID
        or run.get("protocol_run_id") != PROTOCOL_RUN_ID
    ):
        raise GreenStop("17_MANIFEST_FREEZE", "v1.3.5 protocol identity changed")
    if manifest.get("prepare_complete") is not True:
        raise GreenStop("17_MANIFEST_FREEZE", "prepare did not complete")
    stopped = output_root / "result.json"
    if stopped.is_file():
        result = json.loads(stopped.read_text(encoding="utf-8"))
        if result.get("verdict") == "STOP":
            raise GreenStop("17_MANIFEST_FREEZE", "terminal STOP cannot continue")
    ledger_path = output_root / "run_ledger.json"
    if not ledger_path.is_file():
        raise GreenStop("17_MANIFEST_FREEZE", "run ledger missing")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("attempt_index") != 1 or ledger.get("retry_allowed") is not False:
        raise GreenStop("17_MANIFEST_FREEZE", "one-shot ledger contract changed")
    if manifest["frozen_spec_sha256"] != frozen_spec_hash():
        raise GreenStop("17_MANIFEST_FREEZE", "frozen spec hash changed")
    if manifest["source_sha256"] != source_hashes():
        raise GreenStop("17_MANIFEST_FREEZE", "source hashes changed after launch")
    actual_protocol = {
        name: sha256_file(PROJECT_ROOT / name) for name in PROTOCOL_FILES
    }
    if manifest.get("protocol_sha256") != actual_protocol:
        raise GreenStop("17_MANIFEST_FREEZE", "protocol hashes changed after launch")
    if manifest.get("requirements_sha256") != sha256_file(
        PROJECT_ROOT / "requirements-green-bridge.lock"
    ):
        raise GreenStop("17_MANIFEST_FREEZE", "requirements hash changed after launch")
    verify_v134_terminal_archive()
    invariance_path = output_root / "scientific_invariance_v135.json"
    if not invariance_path.is_file():
        raise GreenStop("17_MANIFEST_FREEZE", "scientific invariance record missing")
    invariance = json.loads(invariance_path.read_text(encoding="utf-8"))
    if (
        invariance.get("scientific_payload_equal") is not True
        or invariance.get("parent_scientific_sha256")
        != invariance.get("current_scientific_sha256")
    ):
        raise GreenStop("17_MANIFEST_FREEZE", "scientific invariance changed")
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = output_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise GreenStop("17_MANIFEST_FREEZE", f"artifact hash mismatch: {name}")
    if require_confirmation:
        frozen_path = output_root / "frozen_analysis.json"
        if not frozen_path.is_file() or not manifest.get("confirmation_open", False):
            raise GreenStop("17_MANIFEST_FREEZE", "confirmation lock is closed")
        if sha256_file(frozen_path) != manifest.get("frozen_analysis_sha256"):
            raise GreenStop("17_MANIFEST_FREEZE", "frozen analysis hash mismatch")
    return manifest


def development_phase_v12_inactive(output_root: Path, device: str) -> None:
    manifest = verify_freeze(output_root)
    claim_phase(output_root, "development")
    U, radii, tokenizer, hf_model, model, suffix_ids, _ = load_frozen_numeric(output_root, device)
    del hf_model
    records = load_split_file(output_root, "development_splits.json")
    development = split_records(records, "development")
    target_basis_stability(model, tokenizer, suffix_ids, output_root, radii, development, device)
    noise = duplicate_noise_audit(
        model, tokenizer, suffix_ids, U, radii, development, device, "development", output_root
    )
    epsilon_y = max(1e-7, noise["max_abs"])
    noise["epsilon_y_dev"] = epsilon_y
    write_json_atomic(output_root / "noise_audit_dev.json", noise)
    payload, dev_sd = run_split(
        model, tokenizer, suffix_ids, U, radii, development, "development",
        output_root, device, epsilon_y,
    )
    write_json_atomic(output_root / "dev_cells.json", payload)
    decision = development_decision(payload)
    write_json_atomic(output_root / "dev_result.json", decision)
    if decision["n_surviving_cells"] < THRESHOLDS.development_cells_min:
        raise GreenStop("12_DEVELOPMENT_SURVIVAL", str(decision))
    if decision["n_conditioned_cells"] < THRESHOLDS.development_cells_min:
        raise GreenStop("13_DEVELOPMENT_CONDITIONING", str(decision))
    if decision["n_snr_cells"] < THRESHOLDS.development_snr_cells_min:
        raise GreenStop("14_DEVELOPMENT_SNR", str(decision))
    if decision["verdict"] != "OPEN_CONFIRMATION":
        raise GreenStop("16_DEVELOPMENT_GAIN", str(decision))
    frozen = freeze_confirmation(payload, output_root / "frozen_analysis.json")
    frozen["source_sha256"] = source_hashes()
    frozen["manifest_preconfirmation_sha256"] = sha256_file(output_root / "manifest.json")
    frozen["conditioning_dev_sd"] = dev_sd
    write_json_atomic(output_root / "frozen_analysis.json", frozen)
    manifest["confirmation_open"] = True
    manifest["frozen_analysis_sha256"] = sha256_file(output_root / "frozen_analysis.json")
    manifest["development_artifact_sha256"] = {
        name: sha256_file(output_root / name) for name in (
            "noise_audit_dev.json", "dev_tensor_scores.parquet", "dev_energy_targets.parquet",
            "dev_result.json", "frozen_analysis.json", "target_basis_stability.json",
        )
    }
    write_json_atomic(output_root / "manifest.json", manifest)


def confirmation_phase_v12_inactive(output_root: Path, device: str) -> None:
    manifest = verify_freeze(output_root, require_confirmation=True)
    claim_phase(output_root, "confirmation")
    frozen = json.loads((output_root / "frozen_analysis.json").read_text(encoding="utf-8"))
    if frozen["source_sha256"] != source_hashes():
        raise GreenStop("17_MANIFEST_FREEZE", "frozen source hash mismatch")
    U, radii, tokenizer, hf_model, model, suffix_ids, _ = load_frozen_numeric(output_root, device)
    del hf_model
    records = load_split_file(output_root)
    confirmation = split_records(
        records, "confirmation", confirmation_lock=ConfirmationLock(output_root / "frozen_analysis.json")
    )
    noise = duplicate_noise_audit(
        model, tokenizer, suffix_ids, U, radii, confirmation, device, "confirmation", output_root
    )
    dev_noise = json.loads((output_root / "noise_audit_dev.json").read_text(encoding="utf-8"))["epsilon_y_dev"]
    permitted = max(2 * dev_noise, 2e-6)
    noise["permitted_max_abs"] = permitted
    write_json_atomic(output_root / "noise_audit_confirm.json", noise)
    if noise["max_abs"] > permitted:
        raise GreenStop("18_CONFIRMATION_NOISE", f"{noise['max_abs']} > {permitted}")
    payload, _ = run_split(
        model, tokenizer, suffix_ids, U, radii, confirmation, "confirmation",
        output_root, device, dev_noise, dev_sd=float(frozen["conditioning_dev_sd"]),
    )
    write_json_atomic(output_root / "confirm_cells.json", payload)
    result = confirmation_decision(payload, frozen)
    dev_cells = json.loads((output_root / "dev_cells.json").read_text(encoding="utf-8"))["cells"]
    total_survival = sum(row.get("survived", False) for row in dev_cells + payload["cells"])
    result["total_surviving_cells"] = total_survival
    if total_survival < THRESHOLDS.total_cells_technical_min:
        result["verdict"] = "FAIL_TOTAL_SURVIVAL"
        result["first_failed_gate"] = "19_TOTAL_SURVIVAL"
    elif result["verdict"] != "ORAL_RESULT_PASS":
        if result.get("n_cells", 0) < THRESHOLDS.confirmation_technical_min:
            result["first_failed_gate"] = "19_CONFIRMATION_SURVIVAL"
        elif result.get("n_conditioned", 0) < THRESHOLDS.confirmation_oral_min:
            result["first_failed_gate"] = "19_CONFIRMATION_CONDITIONING"
        else:
            result["first_failed_gate"] = "20_CONFIRMATORY_THRESHOLD"
    else:
        result["first_failed_gate"] = None
    result["schema_version"] = "green-bridge-terminal-v1"
    write_json_atomic(output_root / "result.json", result)
    finalize_hashes(output_root)


# ---------------------------------------------------------------------------
# Active protocol v1.3 implementation.  The v1.2 functions above remain only
# as readable provenance for the archived STOP and are never dispatched.

def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:  # Windows local contract-test fallback
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(path: Path, value) -> None:
    torch = torch_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)


def _atomic_np_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)


def _atomic_np_savez(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)


def _endpoint_ledger_path(output_root: Path) -> Path:
    return output_root / "endpoint_ledger.jsonl"


def _assert_no_uncommitted_endpoint(output_root: Path) -> None:
    rows = read_journal(_endpoint_ledger_path(output_root))
    started = {row["batch_id"] for row in rows if row.get("event") == "endpoint_batch_started"}
    committed = {row["batch_id"] for row in rows if row.get("event") == "endpoint_batch_committed"}
    dangling = sorted(started - committed)
    if dangling:
        raise GreenStop("17_ENDPOINT_LEDGER", f"started-but-uncommitted batches: {dangling}")


def _run_endpoint_batch(output_root: Path, batch_id: str, declaration: dict, evaluate):
    _assert_no_uncommitted_endpoint(output_root)
    ledger = _endpoint_ledger_path(output_root)
    if any(row.get("batch_id") == batch_id for row in read_journal(ledger)):
        raise GreenStop("17_ENDPOINT_LEDGER", f"endpoint batch already exists: {batch_id}")
    started = {
        "event": "endpoint_batch_started",
        "batch_id": batch_id,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    } | declaration
    started["declaration_sha256"] = sha256_text(canonical_json(declaration))
    append_journal(ledger, started)
    result = evaluate()
    artifact = output_root / "endpoint_batches" / f"{batch_id}.json"
    write_json_atomic(artifact, result)
    artifact_hash = sha256_file(artifact)
    append_journal(ledger, {
        "event": "endpoint_batch_committed",
        "batch_id": batch_id,
        "artifact": str(artifact.relative_to(output_root)),
        "artifact_sha256": artifact_hash,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return result


def _anchor_plain_to_device(values: dict, device: str) -> TailAnchor:
    return TailAnchor(**{
        key: (value.to(device) if hasattr(value, "to") else value)
        for key, value in values.items()
    })


def _selected_numpy(anchor: TailAnchor, field: str) -> np.ndarray:
    return selected_position(anchor, field)[0].detach().double().cpu().numpy()


def _capture_structural_inputs(
    model, tokenizer, suffix_ids, records, device: str, output_root: Path, phase: str
) -> dict:
    """Capture and durably freeze all natural anchors before frame construction."""
    plain: dict[str, dict] = {}
    for record in records:
        anchors, _, _ = capture_item_systems(model, tokenizer, suffix_ids, record, device)
        plain[record.pair_digest] = {
            system: _anchor_to_plain(anchor) for system, anchor in anchors.items()
        }
    inputs_path = output_root / f"{phase}_anchor_cache.pt"
    _atomic_torch_save(inputs_path, plain)
    structural: dict[str, np.ndarray] = {
        "ln_scale": model.blocks[10].ln2.w.detach().double().cpu().numpy(),
        "selected_W_in": model.blocks[10].mlp.W_in[:, list(SELECTED_GATES)].detach().double().cpu().numpy(),
    }
    for record in records:
        for system in ("tar", "pat", "cor"):
            anchor = _anchor_plain_to_device(plain[record.pair_digest][system], "cpu")
            structural[f"{record.pair_digest}__{system}__resid_mid"] = _selected_numpy(anchor, "resid_mid")
            structural[f"{record.pair_digest}__{system}__selected_pre"] = _selected_numpy(anchor, "pre")[list(SELECTED_GATES)]
    structural_path = output_root / f"{phase}_structural_inputs.npz"
    _atomic_np_savez(structural_path, structural)
    raw_hashes = {
        "anchor_cache_sha256": sha256_file(inputs_path),
        "structural_inputs_sha256": sha256_file(structural_path),
        "ordered_pair_digests_sha256": sha256_text(
            canonical_json([record.pair_digest for record in records])
        ),
        "ln_scale_sha256": hashlib.sha256(
            model.blocks[10].ln2.w.detach().double().cpu().numpy().tobytes()
        ).hexdigest(),
        "mlp_W_in_sha256": hashlib.sha256(
            model.blocks[10].mlp.W_in.detach().double().cpu().numpy().tobytes()
        ).hexdigest(),
    }
    write_json_atomic(output_root / f"{phase}_structural_input_hashes.json", raw_hashes)
    return plain


def _construct_structural_design(
    model, records, plain: dict, output_root: Path, phase: str
) -> dict:
    """Construct frames only after raw inputs and hashes are durable."""
    gamma = model.blocks[10].ln2.w.detach().double().cpu().numpy()
    W_in = model.blocks[10].mlp.W_in.detach().double().cpu().numpy()
    gate_atoms = [layernorm_gate_atom(gamma, W_in, gate) for gate in SELECTED_GATES]
    eps = float(model.cfg.eps)
    arrays: dict[str, np.ndarray] = {}
    target_arrays: dict[str, np.ndarray] = {}
    audit: dict[str, dict] = {}
    radii: dict[str, dict] = {}
    design: dict[str, dict] = {}
    for record in records:
        digest = record.pair_digest
        anchors = {
            system: _anchor_plain_to_device(values, "cpu")
            for system, values in plain[digest].items()
        }
        residuals = {system: _selected_numpy(anchor, "resid_mid") for system, anchor in anchors.items()}
        common = canonical_common_frame(residuals["tar"], residuals["pat"], residuals["cor"])
        gate_frames = [canonical_gate_frame(common, atom) for atom in gate_atoms]
        all_gate = canonical_all_gate_frame(common, gate_atoms)
        radius = residual_radius(residuals["tar"], residuals["pat"], residuals["cor"])
        vector = target_physical_vector(residuals["tar"], residuals["cor"], float(radius["h_x"]))
        arrays[f"{digest}__common"] = common
        arrays[f"{digest}__all_gate"] = all_gate
        target_arrays[digest] = vector
        item_audit = {
            "common_sha256": frame_sha256(common),
            "all_gate_sha256": frame_sha256(all_gate),
            "common": frame_containment_metrics(
                common,
                np.column_stack((
                    np.ones(768) / np.sqrt(768.0),
                    residuals["tar"] - residuals["tar"].mean(),
                    residuals["pat"] - residuals["pat"].mean(),
                    residuals["cor"] - residuals["cor"].mean(),
                )),
            ),
            "gates": [],
        }
        gate_floor_values = []
        for slot, (gate, atom, frame) in enumerate(zip(SELECTED_GATES, gate_atoms, gate_frames)):
            arrays[f"{digest}__gate_{slot}"] = frame
            raw_atoms = np.column_stack((
                np.ones(768) / np.sqrt(768.0),
                residuals["tar"] - residuals["tar"].mean(),
                residuals["pat"] - residuals["pat"].mean(),
                residuals["cor"] - residuals["cor"].mean(),
                atom,
            ))
            metrics = frame_containment_metrics(frame, raw_atoms)
            gradients = {}
            for system in ("tar", "pat", "cor"):
                formula = layernorm_gate_gradient_formula(
                    residuals[system], gamma, W_in[:, gate], eps=eps
                )
                automatic = layernorm_gate_gradient_autograd(
                    residuals[system], gamma, W_in[:, gate], eps=eps
                )
                difference = formula - automatic
                envelope = gradient_envelope_residual(frame, formula)
                gradients[system] = {
                    "envelope_absolute": envelope["absolute"],
                    "envelope_relative": envelope["relative"],
                    "autograd_max_abs": float(np.max(np.abs(difference))),
                    "autograd_relative": float(
                        np.linalg.norm(difference) / max(np.linalg.norm(automatic), 1e-12)
                    ),
                    "shift_null": shift_null_metric(formula),
                }
                if (
                    envelope["relative"] > STRUCTURAL_GRADIENT_RESIDUAL_MAX
                    or gradients[system]["autograd_max_abs"] > STRUCTURAL_GRADIENT_AUTOGRAD_MAX_ABS
                    or gradients[system]["autograd_relative"] > STRUCTURAL_GRADIENT_AUTOGRAD_RELATIVE
                    or gradients[system]["shift_null"] > SHIFT_GRADIENT_NORMALIZED_MAX
                ):
                    raise GreenStop("07_STRUCTURAL_FRAME", f"{digest} gate={gate} system={system}: {gradients[system]}")
            if (
                metrics["orthogonal_max_abs"] > STRUCTURAL_FRAME_ORTHOGONAL_MAX
                or metrics["atom_residual_relative"] > STRUCTURAL_ATOM_RESIDUAL_MAX
            ):
                raise GreenStop("07_STRUCTURAL_FRAME", f"{digest} gate={gate}: {metrics}")
            natural = [
                abs(float(_selected_numpy(anchors[system], "pre")[gate]))
                for system in ("tar", "pat", "cor")
            ]
            gate_floor = (2.0 ** -10) * max(1.0, float(np.median(natural)))
            gate_floor_values.append(gate_floor)
            item_audit["gates"].append({
                "gate": gate, "frame_sha256": frame_sha256(frame),
                "containment": metrics, "gradients": gradients,
                "gate_radius_floor": gate_floor,
                "gate_radius_floor_pass": GATE_RADIUS >= gate_floor,
            })
        all_metrics = frame_containment_metrics(all_gate, np.column_stack(gate_atoms))
        item_audit["all_gate"] = all_metrics
        if all_metrics["orthogonal_max_abs"] > STRUCTURAL_FRAME_ORTHOGONAL_MAX or all_metrics["atom_residual_relative"] > STRUCTURAL_ATOM_RESIDUAL_MAX:
            raise GreenStop("07_STRUCTURAL_FRAME", f"{digest} all-gate: {all_metrics}")
        radius["gate_radius"] = GATE_RADIUS
        radius["gate_half_radius"] = GATE_RADIUS * HALF_RADIUS_MULTIPLIER
        radius["gate_floor_pass"] = all(GATE_RADIUS >= value for value in gate_floor_values)
        radii[digest] = radius
        audit[digest] = item_audit
        design[digest] = {"common": common, "gate_frames": gate_frames, "all_gate": all_gate, "target": vector, "radius": radius}
    _atomic_np_savez(output_root / f"{phase}_frames.npz", arrays)
    write_json_atomic(output_root / f"{phase}_frame_audit.json", audit)
    write_json_atomic(output_root / f"{phase}_radii.json", radii)
    _atomic_np_savez(output_root / f"{phase}_target_vectors.npz", target_arrays)
    return design


def _jet_at_radius_physical(
    tail: GreenBridgeTail, anchor: TailAnchor, frame: np.ndarray,
    gate_slot: int, hx: float, hz: float, center,
) -> GateJet:
    torch = torch_module()
    device = anchor.resid_mid.device
    path_delta, path_z = [], []
    for sign_z in (1.0, -1.0):
        path_delta.append(np.zeros(768)); path_z.append(sign_z * hz)
    for axis in range(PROBE_FRAME_DIM):
        for sign_x in (1.0, -1.0):
            path_delta.append(sign_x * hx * frame[:, axis]); path_z.append(0.0)
    for axis in range(PROBE_FRAME_DIM):
        for sign_x, sign_z in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            path_delta.append(sign_x * hx * frame[:, axis]); path_z.append(sign_z * hz)
    control_delta, control_z = [], []
    for axis in range(PROBE_FRAME_DIM):
        for sign_x, sign_z in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            control_delta.append(sign_x * hx * frame[:, axis]); control_z.append(sign_z * hz)
    path_count = 2 + 6 * PROBE_FRAME_DIM
    control_count = 4 * PROBE_FRAME_DIM
    path = tail.evaluate_physical(
        _repeat_anchor(anchor, path_count),
        torch.as_tensor(np.stack(path_delta), dtype=torch.float32, device=device),
        torch.as_tensor(path_z, dtype=torch.float32, device=device),
        mode="path", gate_slot=gate_slot,
    ).double()
    control = tail.evaluate_physical(
        _repeat_anchor(anchor, control_count),
        torch.as_tensor(np.stack(control_delta), dtype=torch.float32, device=device),
        torch.as_tensor(control_z, dtype=torch.float32, device=device),
        mode="control", gate_slot=gate_slot,
    ).double()
    G = (path[0] - path[1]) / (2.0 * hz)
    C = (path[0] - 2.0 * center + path[1]) / (hz * hz)
    J, HP, HC = [], [], []
    mixed_start = 2 + 2 * PROBE_FRAME_DIM
    for axis in range(PROBE_FRAME_DIM):
        J.append((path[2 + 2 * axis] - path[3 + 2 * axis]) / (2.0 * hx))
        p = path[mixed_start + 4 * axis:mixed_start + 4 * axis + 4]
        c = control[4 * axis:4 * axis + 4]
        HP.append((p[0] - p[1] - p[2] + p[3]) / (4.0 * hx * hz))
        HC.append((c[0] - c[1] - c[2] + c[3]) / (4.0 * hx * hz))
    return GateJet(
        G.cpu().numpy(), C.cpu().numpy(), torch.stack(J).cpu().numpy(),
        torch.stack(HP).cpu().numpy(), torch.stack(HC).cpu().numpy(),
    )


def _mixed_system_v13(
    tail, model, anchor, design, physical_v, contrast, epsilon_y: float,
) -> dict:
    torch = torch_module()
    device = anchor.resid_mid.device
    zero_delta = torch.zeros((1, 768), dtype=torch.float32, device=device)
    zero_z = torch.zeros(1, dtype=torch.float32, device=device)
    center = tail.evaluate_physical(
        anchor, zero_delta, zero_z, mode="path", gate_slot=0
    )[0].double()
    center_error = center - anchor.year_logits[0].double()
    center_rms = float(torch.sqrt(torch.mean(center_error**2)).item())
    center_max = float(center_error.abs().max().item())
    if center_rms > THRESHOLDS.center_rms or center_max > THRESHOLDS.center_max_abs:
        return {"theta": 0.0, "theta_full": 0.0, "theta_half": 0.0,
                "theta_error": None, "active_gates": 0, "all_valid": False,
                "bypass_disagreement": None, "gates": [], "admissible": False,
                "center_rms": center_rms, "center_max": center_max}
    gamma = model.blocks[10].ln2.w.detach().double().cpu().numpy()
    W_in = model.blocks[10].mlp.W_in.detach().double().cpu().numpy()
    residual = _selected_numpy(anchor, "resid_mid")
    gates, direct_common, bounds = [], [], []
    theta = theta_full = theta_half = 0.0
    for slot, (gate, frame) in enumerate(zip(SELECTED_GATES, design["gate_frames"])):
        hx = float(design["radius"]["h_x"]); hz = GATE_RADIUS
        full = _jet_at_radius_physical(tail, anchor, frame, slot, hx, hz, center)
        half = _jet_at_radius_physical(
            tail, anchor, frame, slot,
            hx * HALF_RADIUS_MULTIPLIER, hz * HALF_RADIUS_MULTIPLIER, center,
        )
        rich = extrapolate_gate_jet(full, half)
        gradient = layernorm_gate_gradient_formula(
            residual, gamma, W_in[:, gate], eps=float(model.cfg.eps)
        )
        wb_A = whitebox_A_coordinates(frame, gradient)
        label, values = classify_gate(
            full, half, rich, wb_A, float(np.linalg.norm(contrast)),
            float(np.linalg.norm(physical_v)), epsilon_y, hx, hz,
        )
        shift_threshold = max(
            1e-4,
            5.0 * (values["numerical"].epsilon_A[0]
                   if np.isfinite(values["numerical"].epsilon_A[0]) else math.inf),
        )
        shift_ok = values["rich"] is not None and abs(float(values["rich"].A[0])) <= shift_threshold
        if label == "active-identified" and not shift_ok:
            label = "invalid"
            values["audit"]["label"] = label
        values["audit"]["shift_coefficient"] = (
            None if values["rich"] is None else float(values["rich"].A[0])
        )
        values["audit"]["shift_threshold"] = shift_threshold if np.isfinite(shift_threshold) else None
        values["audit"]["shift_pass"] = bool(shift_ok)
        if label == "active-identified":
            for estimate, response, name in (
                (values["rich"], rich.G, "rich"),
                (values["full"], full.G, "full"),
                (values["half"], half.G, "half"),
            ):
                cotangent = reconstruct_cotangent(frame, estimate.A)
                contraction = float(contrast @ operator_action(response, cotangent, physical_v))
                if name == "rich": theta += contraction
                elif name == "full": theta_full += contraction
                else: theta_half += contraction
            direct_common.append(direct_bypass_in_common_frame(
                values["rich"].D, frame, design["common"]
            ))
            envelope_absolute = gradient_envelope_residual(frame, gradient)["absolute"]
            bounds.append(active_envelope_contraction_bound(
                float(np.linalg.norm(contrast)),
                float(np.linalg.norm(frame.T @ physical_v)),
                float(np.linalg.norm(physical_v)),
                values["numerical"].epsilon_P_F,
                float(np.linalg.norm(rich.G)), values["numerical"].epsilon_G,
                envelope_absolute,
            ))
        elif label == "certified-target-null":
            bounds.append(values["audit"]["null_bound"])
        gates.append(values["audit"])
    active = sum(row["label"] == "active-identified" for row in gates)
    complete = all(row["label"] != "invalid" for row in gates)
    bypass = None
    if direct_common:
        stack = np.stack(direct_common)
        mean = stack.mean(axis=0)
        bypass = float(
            np.sqrt(np.mean((stack - mean) ** 2))
            / max(np.sqrt(np.mean(mean**2)), 1e-12)
        )
    return {
        "theta": theta, "theta_full": theta_full, "theta_half": theta_half,
        "theta_error": sum_item_error_bounds(bounds) if complete else None,
        "active_gates": active, "all_valid": complete,
        "bypass_disagreement": bypass, "gates": gates,
        "admissible": complete and active >= THRESHOLDS.active_gates_min
                      and bypass is not None and bypass <= THRESHOLDS.bypass_disagreement_max,
        "center_rms": center_rms, "center_max": center_max,
    }


def _joint_margins_physical(tail, anchor, deltas, zs, contrast) -> np.ndarray:
    torch = torch_module()
    device = anchor.resid_mid.device
    outputs = []
    for start in range(0, len(deltas), ACTIVE_MANUAL_TAIL_BATCH_SIZE):
        stop = min(start + ACTIVE_MANUAL_TAIL_BATCH_SIZE, len(deltas))
        logits = tail.evaluate_physical(
            _repeat_anchor(anchor, stop - start),
            torch.as_tensor(deltas[start:stop], dtype=torch.float32, device=device),
            torch.as_tensor(zs[start:stop], dtype=torch.float32, device=device),
            mode="joint",
        )
        outputs.append(margin(logits, contrast).detach().cpu().numpy())
    return np.concatenate(outputs)


def _first_order_system_v13(tail, anchor, design, coefficients, contrast) -> dict:
    center = _joint_margins_physical(
        tail, anchor, np.zeros((1, 768)), np.zeros((1, 10)), contrast
    )[0]
    physical_directions = coefficients @ design["all_gate"].T
    deltas, zs, descriptors = [], [], []
    for name, rho in (("full", 1.0), ("half", HALF_RADIUS_MULTIPLIER)):
        for kind, count in (("x", FIRST_ORDER_RESIDUAL_DIRECTIONS), ("z", 10)):
            for axis in range(count):
                for sign in (1.0, -1.0):
                    delta = np.zeros(768); z = np.zeros(10)
                    if kind == "x":
                        delta = sign * rho * float(design["radius"]["h_x"]) * physical_directions[axis]
                    else:
                        z[axis] = sign * rho * GATE_RADIUS
                    deltas.append(delta); zs.append(z); descriptors.append((name, kind, axis, sign, rho))
    values = _joint_margins_physical(tail, anchor, np.stack(deltas), np.stack(zs), contrast)
    endpoints = {descriptor: value for descriptor, value in zip(descriptors, values)}
    response = {name: {"x": np.zeros(FIRST_ORDER_RESIDUAL_DIRECTIONS), "z": np.zeros(10)} for name in ("full", "half")}
    for name, rho in (("full", 1.0), ("half", HALF_RADIUS_MULTIPLIER)):
        for kind, count in (("x", FIRST_ORDER_RESIDUAL_DIRECTIONS), ("z", 10)):
            for axis in range(count):
                response[name][kind][axis] = (
                    endpoints[name, kind, axis, 1.0, rho]
                    - endpoints[name, kind, axis, -1.0, rho]
                ) / (2.0 * rho)
    return {"center": float(center), "response": response, "rich": {
        kind: (4.0 * response["half"][kind] - response["full"][kind]) / 3.0
        for kind in ("x", "z")
    }}


def _factorial_system_v13(tail, anchor, physical_v, zeta, contrast) -> dict:
    deltas, zs, descriptors = [], [], []
    for name, rho in (("full", 1.0), ("half", HALF_RADIUS_MULTIPLIER)):
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            deltas.append(sx * rho * physical_v); zs.append(sz * rho * zeta)
            descriptors.append((name, rho, sx, sz))
    values = _joint_margins_physical(tail, anchor, np.stack(deltas), np.stack(zs), contrast)
    endpoints = {(name, sx, sz): value for value, (name, _, sx, sz) in zip(values, descriptors)}
    rows = {}
    for name, rho in (("full", 1.0), ("half", HALF_RADIUS_MULTIPLIER)):
        pp, pm, mp, mm = (endpoints[name, 1, 1], endpoints[name, 1, -1], endpoints[name, -1, 1], endpoints[name, -1, -1])
        rows[name] = {
            "single": (pp - mm) / (2.0 * rho),
            "pie": (pp - pm - mp + mm) / (4.0 * rho * rho),
            "bx": (pp + pm - mp - mm) / (4.0 * rho),
            "bz": (pp - pm + mp - mm) / (4.0 * rho),
        }
    return {key: (4.0 * rows["half"][key] - rows["full"][key]) / 3.0 for key in rows["full"]} | {"raw": rows}


def _tensor_item_v13(
    model, suffix_ids, record, device, epsilon_y, coefficients, plain, design,
) -> dict:
    torch = torch_module()
    anchors = {
        system: _anchor_plain_to_device(values, device)
        for system, values in plain[record.pair_digest].items()
    }
    item = design[record.pair_digest]
    seed_frame = torch.as_tensor(item["gate_frames"][0], dtype=torch.float32, device=device)
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    tail = GreenBridgeTail(
        model,
        seed_frame,
        suffix_tensor,
        fixed_batch_size=ACTIVE_MANUAL_TAIL_BATCH_SIZE,
    )
    contrast_t = margin_vector(record.y, device)
    contrast = contrast_t.cpu().numpy()
    physical_v = item["target"]
    mixed = {}
    with torch.inference_mode():
        for system in ("tar", "pat"):
            mixed[system] = _mixed_system_v13(
                tail, model, anchors[system], item, physical_v, contrast, epsilon_y
            )
    pre_tar = _selected_numpy(anchors["tar"], "pre")[list(SELECTED_GATES)]
    pre_cor = _selected_numpy(anchors["cor"], "pre")[list(SELECTED_GATES)]
    gate_chord = pre_tar - pre_cor
    zeta = GATE_RADIUS * gate_chord / max(float(np.linalg.norm(gate_chord)), 1e-30)
    first_order, factorial = {}, {}
    with torch.inference_mode():
        for system in ("tar", "pat"):
            first_order[system] = _first_order_system_v13(
                tail, anchors[system], item, coefficients, contrast_t
            )
            factorial[system] = _factorial_system_v13(
                tail, anchors[system], physical_v, zeta, contrast_t
            )
    center_tar = float(margin(anchors["tar"].year_logits, contrast_t)[0].item())
    center_pat = float(margin(anchors["pat"].year_logits, contrast_t)[0].item())
    first_order_score = math.sqrt(
        float(np.mean((first_order["pat"]["rich"]["x"] - first_order["tar"]["rich"]["x"]) ** 2))
        + float(np.mean((first_order["pat"]["rich"]["z"] - first_order["tar"]["rich"]["z"]) ** 2))
    )
    radius_valid = bool(item["radius"]["floor_pass"] and item["radius"]["gate_floor_pass"])
    return {
        "pair_digest": record.pair_digest, "cell_id": record.cell_id,
        "split": record.split, "distance_bin": record.distance_bin,
        "orientation": record.orientation,
        "admissible": radius_valid and mixed["tar"]["admissible"] and mixed["pat"]["admissible"],
        "physical_target_norm": float(np.linalg.norm(physical_v)),
        "residual_radius": float(item["radius"]["h_x"]),
        "theta_tar": mixed["tar"]["theta"], "theta_pat": mixed["pat"]["theta"],
        "theta_full_tar": mixed["tar"]["theta_full"], "theta_full_pat": mixed["pat"]["theta_full"],
        "theta_half_tar": mixed["tar"]["theta_half"], "theta_half_pat": mixed["pat"]["theta_half"],
        "theta_error_tar": mixed["tar"].get("theta_error"),
        "theta_error_pat": mixed["pat"].get("theta_error"),
        "behavioral": abs(center_pat - center_tar),
        "single": abs(factorial["pat"]["single"] - factorial["tar"]["single"]),
        "first_order": first_order_score,
        "pie": abs(factorial["pat"]["pie"] - factorial["tar"]["pie"]),
        "cancellation_dx": factorial["pat"]["bx"] - factorial["tar"]["bx"],
        "cancellation_dz": factorial["pat"]["bz"] - factorial["tar"]["bz"],
        "mixed_audit": mixed,
    }


def _energy_item_v13(model, suffix_ids, record, device, plain, design) -> dict:
    torch = torch_module()
    anchors = {
        system: _anchor_plain_to_device(values, device)
        for system, values in plain[record.pair_digest].items()
    }
    item = design[record.pair_digest]
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    physical_v = torch.as_tensor(item["target"][None], dtype=torch.float32, device=device)
    systems = {}
    for name in ("tar", "pat", "cor"):
        anchor = anchors[name]
        target_anchor = TargetAnchor(
            resid_mid=anchor.resid_mid, pre=anchor.pre, post=anchor.post,
            final_positions=anchor.final_positions, system=name,
        )
        full = finite_path_effect(
            model, target_anchor, suffix_tensor, physical_v, [record.y], rho=1.0
        )[0]
        half = finite_path_effect(
            model, target_anchor, suffix_tensor, physical_v, [record.y],
            rho=HALF_RADIUS_MULTIPLIER,
        )[0]
        jvp = target_jvp(
            model, target_anchor, suffix_tensor, physical_v, [record.y]
        )[0]
        full_v, half_v, jvp_v = float(full.item()), float(half.item()), float(jvp.item())
        rich_v = (4.0 * half_v - full_v) / 3.0
        absolute = abs(rich_v - jvp_v)
        relative = absolute / max(abs(jvp_v), 0.05)
        locality = abs(full_v - rich_v)
        systems[name] = {
            "full": full_v, "half": half_v, "jvp": jvp_v,
            "richardson": rich_v, "jvp_absolute_error": absolute,
            "jvp_relative_error": relative, "locality_error": locality,
            "admissible": absolute <= 0.01 and relative <= 0.05
                          and locality <= max(0.02, 0.25 * abs(rich_v)),
        }
    return {
        "pair_digest": record.pair_digest, "cell_id": record.cell_id,
        "split": record.split, "distance_bin": record.distance_bin,
        "orientation": record.orientation,
        "physical_target_norm": float(np.linalg.norm(item["target"])),
        "admissible": bool(item["radius"]["floor_pass"])
                      and all(row["admissible"] for row in systems.values()),
        "systems": systems,
    }


def _run_split_v13(
    model, suffix_ids, records, split, output_root, device, epsilon_y,
    plain, design, dev_sd=None, precomputed=None,
) -> tuple[dict, float]:
    coefficients = first_order_directions()
    precomputed = {} if precomputed is None else precomputed
    tensor_rows, energy_rows = [], []
    for record in [row for row in records if row.role == "tensor"]:
        if ("tensor", record.pair_digest) in precomputed:
            tensor_rows.append(precomputed["tensor", record.pair_digest])
            continue
        batch_id = f"{split}-tensor-{record.pair_digest}"
        tensor_rows.append(_run_endpoint_batch(
            output_root, batch_id,
            {"phase": split, "items": [record.pair_digest], "endpoint_type": "tensor",
             "radii_sha256": sha256_file(output_root / f"{split}_radii.json"),
             "systems": ["tar", "pat"]},
            lambda record=record: _tensor_item_v13(
                model, suffix_ids, record, device, epsilon_y,
                coefficients, plain, design,
            ),
        ))
    for record in [row for row in records if row.role == "energy"]:
        if ("energy", record.pair_digest) in precomputed:
            energy_rows.append(precomputed["energy", record.pair_digest])
            continue
        batch_id = f"{split}-energy-{record.pair_digest}"
        energy_rows.append(_run_endpoint_batch(
            output_root, batch_id,
            {"phase": split, "items": [record.pair_digest], "endpoint_type": "energy",
             "radii_sha256": sha256_file(output_root / f"{split}_radii.json"),
             "systems": ["tar", "pat", "cor"]},
            lambda record=record: _energy_item_v13(
                model, suffix_ids, record, device, plain, design
            ),
        ))
    write_parquet(output_root / ("dev_tensor_scores.parquet" if split == "development" else "confirm_tensor_scores.parquet"), tensor_rows)
    write_parquet(output_root / ("dev_energy_targets.parquet" if split == "development" else "confirm_energy_targets.parquet"), energy_rows)
    return aggregate_cells(tensor_rows, energy_rows, dev_sd=dev_sd)


def _run_split_v135_multigpu(
    records,
    split: str,
    output_root: Path,
    epsilon_y: float,
    dev_sd=None,
) -> tuple[dict, float]:
    """Evaluate each frozen record once at exact batch one across eight GPUs."""
    physical_gpus = tuple(range(8))
    shard_root = output_root / "shards" / split
    if shard_root.exists():
        raise GreenStop("17_ENDPOINT_LEDGER", f"shard root already exists: {shard_root}")
    ordered = sorted(records, key=lambda row: (row.role, row.pair_digest))
    assignments = {
        index: [row for position, row in enumerate(ordered) if position % 8 == index]
        for index in range(8)
    }
    launch_contract_hashes = {
        name: sha256_file(output_root / name)
        for name in (
            f"{split}_anchor_cache.pt",
            f"{split}_structural_inputs.npz",
            f"{split}_structural_input_hashes.json",
            f"{split}_frames.npz",
            f"{split}_frame_audit.json",
            f"{split}_radii.json",
            f"{split}_target_vectors.npz",
        )
    }
    processes = []
    log_handles = []
    worker_roots = []
    for worker_index, physical_gpu in enumerate(physical_gpus):
        worker_root = shard_root / f"worker_{worker_index:02d}"
        worker_root.mkdir(parents=True, exist_ok=False)
        assigned = assignments[worker_index]
        write_json_atomic(worker_root / "assigned_records.json", plan_payload(assigned))
        contract = {
            "schema_version": "green-bridge-v1.3.5-worker-contract-v1",
            "worker_index": worker_index,
            "physical_gpu": physical_gpu,
            "split": split,
            "epsilon_y": float(epsilon_y),
            "record_digests": [row.pair_digest for row in assigned],
            "input_sha256": launch_contract_hashes,
            "source_sha256": source_hashes(),
        }
        write_json_atomic(worker_root / "worker_contract.json", contract)
        log_path = worker_root / "worker.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        command = [
            sys.executable,
            "src/green_bridge_multigpu_worker.py",
            "--output-root", str(output_root),
            "--worker-root", str(worker_root),
            "--split", split,
            "--worker-index", str(worker_index),
            "--physical-gpu", str(physical_gpu),
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(process)
        log_handles.append(log_handle)
        worker_roots.append(worker_root)
    return_codes = []
    for process, log_handle in zip(processes, log_handles):
        return_codes.append(process.wait())
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()
    if any(code != 0 for code in return_codes):
        raise GreenStop(
            "11_MULTIGPU_WORKER",
            str({"return_codes": return_codes, "shard_root": str(shard_root)}),
        )

    tensor_rows, energy_rows = [], []
    worker_summaries = []
    for worker_index, worker_root in enumerate(worker_roots):
        worker_result_path = worker_root / "worker_result.json"
        if not worker_result_path.is_file():
            raise GreenStop("11_MULTIGPU_WORKER", f"missing {worker_result_path}")
        worker_result = json.loads(worker_result_path.read_text(encoding="utf-8"))
        assigned = assignments[worker_index]
        if worker_result.get("record_count") != len(assigned):
            raise GreenStop("11_MULTIGPU_WORKER", f"coverage mismatch worker {worker_index}")
        committed = {
            row["batch_id"]
            for row in read_journal(worker_root / "endpoint_ledger.jsonl")
            if row.get("event") == "endpoint_batch_committed"
        }
        for record in assigned:
            batch_id = f"{split}-{record.role}-{record.pair_digest}"
            if batch_id not in committed:
                raise GreenStop("11_MULTIGPU_WORKER", f"uncommitted {batch_id}")
            artifact = worker_root / "endpoint_batches" / f"{batch_id}.json"
            row = json.loads(artifact.read_text(encoding="utf-8"))
            if row.get("pair_digest") != record.pair_digest:
                raise GreenStop("11_MULTIGPU_WORKER", f"identity mismatch {batch_id}")
            (tensor_rows if record.role == "tensor" else energy_rows).append(row)
        for journal_row in read_journal(worker_root / "endpoint_ledger.jsonl"):
            append_journal(_endpoint_ledger_path(output_root), journal_row)
        worker_summaries.append({
            "worker_index": worker_index,
            "physical_gpu": physical_gpus[worker_index],
            "record_count": len(assigned),
            "worker_result_sha256": sha256_file(worker_result_path),
            "worker_log_sha256": sha256_file(worker_root / "worker.log"),
            "elapsed_seconds": worker_result["elapsed_seconds"],
            "peak_allocated_bytes": worker_result["peak_allocated_bytes"],
        })
    tensor_rows.sort(key=lambda row: row["pair_digest"])
    energy_rows.sort(key=lambda row: row["pair_digest"])
    expected_tensor = sorted(row.pair_digest for row in records if row.role == "tensor")
    expected_energy = sorted(row.pair_digest for row in records if row.role == "energy")
    if [row["pair_digest"] for row in tensor_rows] != expected_tensor:
        raise GreenStop("11_MULTIGPU_WORKER", "tensor coverage is not exact")
    if [row["pair_digest"] for row in energy_rows] != expected_energy:
        raise GreenStop("11_MULTIGPU_WORKER", "energy coverage is not exact")
    tensor_name = "dev_tensor_scores.parquet" if split == "development" else "confirm_tensor_scores.parquet"
    energy_name = "dev_energy_targets.parquet" if split == "development" else "confirm_energy_targets.parquet"
    write_parquet(output_root / tensor_name, tensor_rows)
    write_parquet(output_root / energy_name, energy_rows)
    merge_payload = {
        "schema_version": "green-bridge-v1.3.5-multigpu-merge-v1",
        "split": split,
        "exact_batch_size": 1,
        "physical_gpus": list(physical_gpus),
        "tensor_count": len(tensor_rows),
        "energy_count": len(energy_rows),
        "input_sha256": launch_contract_hashes,
        "workers": worker_summaries,
        "tensor_parquet_sha256": sha256_file(output_root / tensor_name),
        "energy_parquet_sha256": sha256_file(output_root / energy_name),
    }
    write_json_atomic(output_root / f"{split}_multigpu_merge.json", merge_payload)
    return aggregate_cells(tensor_rows, energy_rows, dev_sd=dev_sd)


def _development_throughput_preflight(
    model, suffix_ids, records, output_root, device, epsilon_y, plain, design,
) -> dict:
    """Run the deterministic lowest-hash 2% operation mixture exactly once."""
    torch = torch_module()
    coefficients = first_order_directions()
    selected = {}
    for role in ("tensor", "energy"):
        population = sorted(
            [row for row in records if row.role == role],
            key=lambda row: sha256_text("throughput-v13|" + row.pair_digest),
        )
        count = max(1, math.ceil(0.02 * len(population)))
        selected[role] = population[:count]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    results = {}
    for role in ("tensor", "energy"):
        for record in selected[role]:
            batch_id = f"development-throughput-{role}-{record.pair_digest}"
            if role == "tensor":
                result = _run_endpoint_batch(
                    output_root, batch_id,
                    {"phase": "development", "items": [record.pair_digest],
                     "endpoint_type": "throughput_tensor", "fraction": 0.02,
                     "radii_sha256": sha256_file(output_root / "development_radii.json"),
                     "systems": ["tar", "pat"]},
                    lambda record=record: _tensor_item_v13(
                        model, suffix_ids, record, device, epsilon_y,
                        coefficients, plain, design,
                    ),
                )
            else:
                result = _run_endpoint_batch(
                    output_root, batch_id,
                    {"phase": "development", "items": [record.pair_digest],
                     "endpoint_type": "throughput_energy", "fraction": 0.02,
                     "radii_sha256": sha256_file(output_root / "development_radii.json"),
                     "systems": ["tar", "pat", "cor"]},
                    lambda record=record: _energy_item_v13(
                        model, suffix_ids, record, device, plain, design
                    ),
                )
            results[role, record.pair_digest] = result
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    elapsed = time.perf_counter() - start
    sample_units = len(selected["tensor"]) * 4180 + len(selected["energy"]) * 15
    extrapolated_seconds = elapsed * runner_units / sample_units if (runner_units := FORWARD_COUNTS["development_effective_units"]) else math.inf
    hard_cap_seconds = 24.0 * 3600.0 if "4090" in torch.cuda.get_device_name(device) else 40.0 * 3600.0
    peak_limit = 20 * 1024**3 if "4090" in torch.cuda.get_device_name(device) else 32 * 1024**3
    payload = {
        "selection": {role: [row.pair_digest for row in rows] for role, rows in selected.items()},
        "selection_rule": "lowest_sha256_rank_within_endpoint_type",
        "fraction": 0.02, "elapsed_seconds": elapsed,
        "sample_effective_units": sample_units,
        "extrapolated_development_seconds": extrapolated_seconds,
        "hard_cap_seconds": hard_cap_seconds, "peak_allocated_bytes": peak_bytes,
        "peak_limit_bytes": peak_limit,
        "outputs_reused": True,
        "passed": extrapolated_seconds <= hard_cap_seconds and peak_bytes <= peak_limit,
        "results": {
            f"{role}:{digest}": value for (role, digest), value in results.items()
        },
    }
    write_json_atomic(output_root / "development_throughput_preflight.json", payload)
    if not payload["passed"]:
        raise GreenStop("10_THROUGHPUT", str({key: payload[key] for key in ("extrapolated_development_seconds", "hard_cap_seconds", "peak_allocated_bytes", "peak_limit_bytes")}))
    return {"payload": payload, "results": results}


def _legacy_selected_projection_year_logits(
    model,
    normalized_final,
    final_positions,
    suffix_ids,
):
    """Diagnostic-only reproduction of the archived v1.3 endpoint defect."""
    torch = torch_module()
    rows = torch.arange(normalized_final.shape[0], device=normalized_final.device)
    final = normalized_final[rows, final_positions, :]
    selected = model.W_U.index_select(1, suffix_ids)
    logits = final @ selected
    if getattr(model, "b_U", None) is not None:
        logits = logits + model.b_U.index_select(0, suffix_ids)
    return logits


def _equivalence_metrics(actual, expected) -> dict:
    torch = torch_module()
    difference = actual.double() - expected.double()
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.abs().max().item()),
        "rms": float(torch.sqrt(torch.mean(difference**2)).item()),
        "shape": list(actual.shape),
    }


def derivative_equivalence_record(
    manual_plus,
    manual_minus,
    full_plus,
    full_minus,
    *,
    step: float,
) -> dict:
    torch = torch_module()
    manual_derivative = (manual_plus.double() - manual_minus.double()) / (2.0 * step)
    full_derivative = (full_plus.double() - full_minus.double()) / (2.0 * step)
    difference = manual_derivative - full_derivative
    absolute_l2 = float(torch.linalg.vector_norm(difference).item())
    absolute_max = float(difference.abs().max().item())
    reference_l2 = float(torch.linalg.vector_norm(full_derivative).item())
    if reference_l2 > TAIL_DERIVATIVE_REFERENCE_NORM_FLOOR:
        relative = absolute_l2 / reference_l2
        passed = relative <= THRESHOLDS.tail_derivative_relative
        status = "RELATIVE_APPLICABLE"
        max_bound = None
        l2_bound = None
    else:
        max_bound = THRESHOLDS.tail_max_abs / step
        l2_bound = (
            TAIL_EQUIVALENCE_OUTPUT_DIM ** 0.5
            * THRESHOLDS.tail_max_abs
            / step
        )
        relative = None
        passed = absolute_max <= max_bound and absolute_l2 <= l2_bound
        status = "NOT_APPLICABLE_NEAR_ZERO"
    return {
        "step": float(step),
        "absolute_l2": absolute_l2,
        "absolute_max": absolute_max,
        "reference_l2": reference_l2,
        "reference_norm_floor": TAIL_DERIVATIVE_REFERENCE_NORM_FLOOR,
        "relative": relative,
        "relative_threshold": THRESHOLDS.tail_derivative_relative,
        "max_bound": max_bound,
        "l2_bound": l2_bound,
        "status": status,
        "passed": bool(passed),
    }


def _prepare_exact_batch_one_and_throughput_v135(
    model,
    tokens,
    suffix_ids,
    anchor,
    frame_t,
    residual_step: float,
    device: str,
    output_root: Path,
) -> dict:
    """Validate exact batch-one endpoints and projected eight-GPU throughput."""
    torch = torch_module()
    fixed = TAIL_FIXED_BATCH_SIZE
    if fixed != 1:
        raise GreenStop("06E_EXACT_BATCH_ONE", f"batch size changed to {fixed}")
    memory_limit = 20 * 1024**3
    fixed_tail = GreenBridgeTail(
        model,
        frame_t,
        torch.as_tensor(suffix_ids, dtype=torch.long, device=device),
        fixed_batch_size=fixed,
    )

    def sync():
        torch.cuda.synchronize(device)

    coordinates = torch.zeros((1, PROBE_FRAME_DIM), dtype=torch.float32, device=device)
    z = torch.zeros(1, dtype=torch.float32, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        fixed_tail.evaluate_physical(
            anchor, coordinates @ frame_t.T, z, mode="path", gate_slot=0
        )
    sync()
    started = time.perf_counter()
    with torch.inference_mode():
        benchmark_anchor = _repeat_anchor(anchor, 64)
        benchmark_coordinates = torch.zeros(
            (64, PROBE_FRAME_DIM), dtype=torch.float32, device=device
        )
        benchmark_coordinates[:, 0] = torch.linspace(
            -residual_step, residual_step, 64, dtype=torch.float32, device=device
        )
        benchmark_z = torch.linspace(
            -GATE_RADIUS, GATE_RADIUS, 64, dtype=torch.float32, device=device
        )
        benchmark = fixed_tail.evaluate_physical(
            benchmark_anchor,
            benchmark_coordinates @ frame_t.T,
            benchmark_z,
            mode="path",
            gate_slot=0,
        )
    sync()
    elapsed = time.perf_counter() - started
    seconds_per_endpoint = elapsed / 64.0
    peak = int(torch.cuda.max_memory_allocated(device))
    if peak > memory_limit:
        raise GreenStop("06E_EXACT_BATCH_ONE", f"peak={peak}")

    with torch.inference_mode():
        benchmark_repeat = fixed_tail.evaluate_physical(
            benchmark_anchor,
            benchmark_coordinates @ frame_t.T,
            benchmark_z,
            mode="path",
            gate_slot=0,
        )
    repeat_invariance = _equivalence_metrics(benchmark, benchmark_repeat)

    batch_one_conditions = [
        ("center", "path", np.zeros(PROBE_FRAME_DIM), 0.0),
        ("x_only", "path", np.eye(PROBE_FRAME_DIM)[0] * residual_step, 0.0),
        ("z_only", "path", np.zeros(PROBE_FRAME_DIM), GATE_RADIUS),
        ("path_x_plus_z", "path", np.eye(PROBE_FRAME_DIM)[1] * residual_step, GATE_RADIUS),
        ("control_x_plus_z", "control", np.eye(PROBE_FRAME_DIM)[2] * residual_step, GATE_RADIUS),
    ]
    exact_full_records = []
    with torch.inference_mode():
        for name, mode, coordinate, z_value in batch_one_conditions:
            coordinate_t = torch.as_tensor(
                coordinate[None], dtype=torch.float32, device=device
            )
            delta_t = coordinate_t @ frame_t.T
            z_t = torch.tensor([z_value], dtype=torch.float32, device=device)
            exact = fixed_tail.evaluate_physical(
                anchor, delta_t, z_t, mode=mode, gate_slot=0
            )
            full = full_hook_endpoint_physical(
                model,
                tokens,
                suffix_ids,
                anchor,
                delta_t,
                z_t,
                mode=mode,
                gate_slot=0,
            )
            metrics = _equivalence_metrics(exact, full)
            exact_full_records.append({"condition": name, **metrics})
            if not metrics["bitwise_equal"]:
                raise GreenStop(
                    "06E_EXACT_BATCH_ONE",
                    f"batch-one endpoint {name}: {metrics}",
                )

    if not repeat_invariance["bitwise_equal"]:
        raise GreenStop("06E_EXACT_BATCH_ONE", f"repeat: {repeat_invariance}")

    plan = {
        "schema_version": "green-bridge-hardware-batch-plan-v1.3.5",
        "prepare_physical_gpu": 4,
        "worker_physical_gpus": list(range(8)),
        "worker_count": 8,
        "manual_tail_batch_size": fixed,
        "manual_tail_final_chunk_padding": False,
        "manual_tail_output_recentering": False,
        "full_model_jvp_batch_size": 1,
        "memory_limit_bytes": memory_limit,
        "peak_allocated_bytes": peak,
        "frozen_before_development": True,
    }
    write_json_atomic(output_root / "hardware_batch_plan.json", plan)
    batch_payload = {
        "schema_version": "green-bridge-exact-batch-one-equivalence-v1.3.5",
        "exact_batch_size": 1,
        "batch_one_repeat_invariance": repeat_invariance,
        "batch_one_vs_full_hook": exact_full_records,
        "manual_full_raw_equivalence_batch": 1,
        "passed": True,
    }
    write_json_atomic(
        output_root / "manual_tail_batch_equivalence_v135.json",
        batch_payload,
    )

    projected_single = FORWARD_COUNTS["tail_evaluations_total"] * seconds_per_endpoint
    projected_eight = projected_single / 8.0
    throughput = {
        "schema_version": "green-bridge-tail-throughput-v1.3.5",
        "preflight_record_only": True,
        "fixed_batch_size": fixed,
        "benchmark_endpoints": 64,
        "benchmark_seconds": elapsed,
        "seconds_per_endpoint": seconds_per_endpoint,
        "manual_tail_projected_single_gpu_seconds": projected_single,
        "manual_tail_projected_eight_gpu_seconds": projected_eight,
        "hard_cap_seconds": 24.0 * 3600.0,
        "operation_counts": FORWARD_COUNTS,
        "selected_projection_fallback": False,
        "peak_limit_bytes": memory_limit,
        "passed": projected_eight <= 24.0 * 3600.0,
    }
    write_json_atomic(output_root / "manual_tail_throughput_v135.json", throughput)
    if not throughput["passed"]:
        raise GreenStop("06G_PREPARE_THROUGHPUT", str(throughput))
    return {"plan": plan, "batch": batch_payload, "throughput": throughput}




def _tail_preflight_v135(
    model,
    tokenizer,
    suffix_ids,
    record,
    device: str,
    output_root: Path,
    preflight_design: dict,
) -> dict:
    torch = torch_module()
    anchors, clean_tokens, _ = capture_item_systems(
        model, tokenizer, suffix_ids, record, device
    )
    residuals = {name: _selected_numpy(anchor, "resid_mid") for name, anchor in anchors.items()}
    common = canonical_common_frame(residuals["tar"], residuals["pat"], residuals["cor"])
    gamma = model.blocks[10].ln2.w.detach().double().cpu().numpy()
    W_in = model.blocks[10].mlp.W_in.detach().double().cpu().numpy()
    frame = canonical_gate_frame(
        common, layernorm_gate_atom(gamma, W_in, SELECTED_GATES[0])
    )
    radius = residual_radius(residuals["tar"], residuals["pat"], residuals["cor"])
    frame_t = torch.as_tensor(frame, dtype=torch.float32, device=device)
    suffix_t = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    tail = GreenBridgeTail(model, frame_t, suffix_t)
    anchor = anchors["tar"]
    conditions = [
        ("center", "path", np.zeros(5), 0.0),
        ("x_only", "path", np.eye(5)[0] * float(radius["h_x"]), 0.0),
        ("z_only", "path", np.zeros(5), GATE_RADIUS),
        ("path_x_plus_z", "path", np.eye(5)[1] * float(radius["h_x"]), GATE_RADIUS),
        ("control_x_plus_z", "control", np.eye(5)[2] * float(radius["h_x"]), GATE_RADIUS),
    ]
    expected_legacy = [
        7.62939453125e-05,
        6.103515625e-05,
        6.103515625e-05,
        7.62939453125e-05,
        6.103515625e-05,
    ]
    legacy_errors = []
    stage_records = []
    raw_records = []
    corrected_endpoints = {}
    with torch.inference_mode():
        for name, mode, coordinate, z_value in conditions:
            coordinate_t = torch.as_tensor(coordinate[None], dtype=torch.float32, device=device)
            delta_t = coordinate_t @ frame_t.T
            z_t = torch.tensor([z_value], dtype=torch.float32, device=device)
            manual, manual_trace = tail.evaluate_physical_with_trace(
                anchor, delta_t, z_t, mode=mode, gate_slot=0
            )
            full, full_trace = full_hook_endpoint(
                model, clean_tokens, suffix_ids, anchor, frame_t, coordinate_t, z_t,
                mode=mode, gate_slot=0, return_trace=True,
            )
            legacy = _legacy_selected_projection_year_logits(
                model, manual_trace["ln_final_output"], anchor.final_positions, suffix_t
            )
            legacy_error = float((legacy.double() - full.double()).abs().max().item())
            legacy_errors.append(legacy_error)

            stage_mapping = (
                ("resid_mid_after_x", "resid_mid_after_x"),
                ("ln2_output", "ln2_output"),
                ("pre_after_z", "pre_after_z"),
                ("anchored_post", "post_after_anchor"),
                ("mlp_out", "mlp_out"),
                ("resid_post_after_subtraction", "resid_post_after_subtraction"),
                ("block11_resid_post", "block11_resid_post"),
                ("ln_final_output", "ln_final_output"),
                ("unembed_pre_softcap_full", "unembed_pre_softcap_full"),
                ("unembed_post_softcap_full", "unembed_post_softcap_full"),
            )
            comparisons = []
            for manual_name, full_name in stage_mapping:
                metrics = _equivalence_metrics(
                    manual_trace[manual_name], full_trace[full_name]
                )
                comparisons.append({
                    "stage": manual_name,
                    "manual_key": manual_name,
                    "full_key": full_name,
                    **metrics,
                })
            first_divergent = next(
                (row["stage"] for row in comparisons if not row["bitwise_equal"]),
                None,
            )
            stage_records.append({
                "condition": name,
                "comparisons": comparisons,
                "first_divergent_stage": first_divergent,
            })
            raw = _equivalence_metrics(manual, full)
            raw_records.append({"condition": name, **raw})
            corrected_endpoints[name] = (manual, full)

    legacy_payload = {
        "schema_version": "green-bridge-tail-root-cause-v1.3.5",
        "diagnostic_only": True,
        "condition_order": [row[0] for row in conditions],
        "expected_errors": expected_legacy,
        "observed_errors": legacy_errors,
        "exact_reproduction": legacy_errors == expected_legacy,
        "max_abs": max(legacy_errors),
        "active_scientific_endpoint": False,
    }
    write_json_atomic(
        output_root / "manual_tail_root_cause_reproduction_v135.json",
        legacy_payload,
    )
    if legacy_errors != expected_legacy or max(legacy_errors) <= THRESHOLDS.tail_max_abs:
        raise GreenStop("06A_LEGACY_ROOT_CAUSE_REPRODUCTION", str(legacy_payload))

    stage_payload = {
        "schema_version": "green-bridge-tail-stage-trace-v1.3.5",
        "records": stage_records,
        "first_divergent_legacy_stage": "unembedding_endpoint",
        "passed": all(
            row["first_divergent_stage"] is None for row in stage_records
        ),
    }
    write_json_atomic(output_root / "manual_tail_stage_trace_v135.json", stage_payload)
    if not stage_payload["passed"]:
        raise GreenStop("06B_MANUAL_TAIL_STAGE_TRACE", str(stage_payload))

    if any(row["max_abs"] > THRESHOLDS.tail_max_abs for row in raw_records):
        raise GreenStop("06C_MANUAL_TAIL_RAW", str(raw_records))

    stencil_specs = [
        ("path_dx_e0_at_z0", "path", "x", 0),
        ("path_dx_e1_at_z0", "path", "x", 1),
        ("path_dx_e2_at_z0", "path", "x", 2),
        ("path_dz_at_x0", "path", "z", 0),
        ("control_dx_e2_at_z0", "control", "x", 2),
        ("control_dz_at_x0", "control", "z", 0),
    ]
    signed_raw = []
    derivative_records = []
    with torch.inference_mode():
        for stencil_name, mode, direction_kind, axis in stencil_specs:
            step = float(radius["h_x"] if direction_kind == "x" else GATE_RADIUS)
            endpoints = {}
            for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                coordinate = np.zeros(PROBE_FRAME_DIM)
                z_value = 0.0
                if direction_kind == "x":
                    coordinate[axis] = sign * step
                else:
                    z_value = sign * step
                coordinate_t = torch.as_tensor(
                    coordinate[None], dtype=torch.float32, device=device
                )
                delta_t = coordinate_t @ frame_t.T
                z_t = torch.tensor([z_value], dtype=torch.float32, device=device)
                manual = tail.evaluate_physical(
                    anchor, delta_t, z_t, mode=mode, gate_slot=0
                )
                full = full_hook_endpoint(
                    model, clean_tokens, suffix_ids, anchor, frame_t,
                    coordinate_t, z_t, mode=mode, gate_slot=0,
                )
                metrics = _equivalence_metrics(manual, full)
                signed_raw.append({
                    "stencil": stencil_name,
                    "sign": sign_name,
                    **metrics,
                })
                if metrics["max_abs"] > THRESHOLDS.tail_max_abs:
                    raise GreenStop("06C_MANUAL_TAIL_RAW", str(signed_raw[-1]))
                endpoints[f"manual_{sign_name}"] = manual
                endpoints[f"full_{sign_name}"] = full
            derivative = derivative_equivalence_record(
                endpoints["manual_plus"], endpoints["manual_minus"],
                endpoints["full_plus"], endpoints["full_minus"],
                step=step,
            )
            derivative_records.append({"stencil": stencil_name, **derivative})
            if not derivative["passed"]:
                raise GreenStop("06D_MANUAL_TAIL_DERIVATIVE", str(derivative_records[-1]))

    equivalence_payload = {
        "schema_version": "green-bridge-tail-equivalence-v1.3.5",
        "quantity": "raw_100_dimensional_year_logits",
        "threshold": THRESHOLDS.tail_max_abs,
        "original_conditions": raw_records,
        "signed_endpoints": signed_raw,
        "center_retained": True,
        "passed": True,
    }
    write_json_atomic(output_root / "manual_tail_equivalence_v135.json", equivalence_payload)
    derivative_payload = {
        "schema_version": "green-bridge-tail-derivative-v1.3.5",
        "estimator": "central_finite_difference",
        "records": derivative_records,
        "near_zero_not_silently_dropped": True,
        "passed": True,
    }
    write_json_atomic(output_root / "manual_tail_derivative_v135.json", derivative_payload)

    target_anchor = TargetAnchor(
        resid_mid=anchor.resid_mid,
        pre=anchor.pre,
        post=anchor.post,
        final_positions=anchor.final_positions,
        system=anchor.system,
    )
    target_vector = torch.as_tensor(
        preflight_design[record.pair_digest]["target"][None],
        dtype=torch.float32,
        device=device,
    )
    target_records = []
    with torch.inference_mode():
        for scale in (0.0, 1.0, -1.0, 0.5, -0.5):
            physical_delta = scale * target_vector
            target = evaluate_joint_target(
                model, target_anchor, suffix_t, physical_delta
            )
            full = full_hook_endpoint_physical(
                model,
                clean_tokens,
                suffix_ids,
                anchor,
                physical_delta,
                torch.zeros((1, 10), dtype=torch.float32, device=device),
                mode="joint",
                subtract_residual_bypass=True,
            )
            metrics = _equivalence_metrics(target, full)
            target_records.append({"scale": scale, **metrics})
            if metrics["max_abs"] > THRESHOLDS.tail_max_abs:
                raise GreenStop("06F_PATH_TARGET_RAW", str(target_records[-1]))
    target_payload = {
        "schema_version": "green-bridge-path-target-equivalence-v1.3.5",
        "mode": "joint",
        "subtract_residual_bypass": True,
        "code_isolated": True,
        "records": target_records,
        "passed": True,
    }
    write_json_atomic(output_root / "path_target_equivalence_v135.json", target_payload)
    hardware = _prepare_exact_batch_one_and_throughput_v135(
        model,
        clean_tokens,
        suffix_ids,
        anchor,
        frame_t,
        float(radius["h_x"]),
        device,
        output_root,
    )
    return {
        "legacy": legacy_payload,
        "stage_trace": stage_payload,
        "raw": equivalence_payload,
        "derivative": derivative_payload,
        "path_target": target_payload,
        "hardware": hardware,
    }


def _duplicate_noise_v13(model, tokenizer, suffix_ids, records, device: str) -> dict:
    errors = []
    for record in sorted(records, key=lambda row: row.pair_digest)[:32]:
        first, _, _ = capture_item_systems(model, tokenizer, suffix_ids, record, device)
        second, _, _ = capture_item_systems(model, tokenizer, suffix_ids, record, device)
        for system in ("tar", "pat", "cor"):
            errors.append(float(
                (first[system].year_logits.double() - second[system].year_logits.double())
                .abs().max().item()
            ))
    return {"n": len(errors), "max_abs": max(errors, default=0.0), "errors": errors}


def prepare(output_root: Path, device: str) -> None:
    repository = assert_clean_repository()
    assert_empty_prepare_root(output_root)
    predecessor_archive = verify_v134_terminal_archive()
    environment = configure_runtime(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    pair_allowed = lambda first, second: token_pair_allowed(tokenizer, first, second)
    evaluation = build_evaluation_records(pair_allowed)
    validate_evaluation_plan(evaluation)
    legacy = build_legacy_donor_records(pair_allowed)
    legacy_gate04, holdout_gate04 = gate04_record_panels(legacy)
    gate04_panel = gate04_panel_metadata(legacy_gate04, holdout_gate04)
    if gate04_panel["ordered_prompt_keys_sha256"] != GATE04_ORDERED_PROMPT_HASH:
        raise GreenStop("04_HF_TL_FIDELITY", "legacy Gate-04 prompt hash changed")
    output_root.mkdir(parents=True, exist_ok=True)
    write_run_ledger(output_root, repository)
    parent_scientific_path = (
        PROJECT_ROOT / "analysis" / "archive" / "green_v13_stop_20260825"
        / "frozen_scientific_spec_v13.json"
    )
    parent_scientific = json.loads(
        parent_scientific_path.read_text(encoding="utf-8")
    )
    current_scientific_payload = _scientific_payload()
    current_scientific_hash = sha256_text(canonical_json(current_scientific_payload))
    parent_scientific_hash = parent_scientific["scientific_sha256"]
    invariance = {
        "parent_schema": "green-bridge-v1.3.4",
        "current_schema": "green-bridge-v1.3.5",
        "scientific_payload_equal": (
            parent_scientific.get("scientific_payload") == current_scientific_payload
            and parent_scientific_hash == current_scientific_hash
        ),
        "parent_scientific_sha256": parent_scientific_hash,
        "current_scientific_sha256": current_scientific_hash,
        "allowed_differences": [
            "protocol identity",
            "output root",
            "manual-tail executable endpoint",
            "target executable endpoint",
            "equivalence-audit metric implementation",
            "equivalence-audit artifacts",
            "predecessor archival metadata",
            "fixed-shape tail batching and final-chunk padding",
            "exact batch-one manual-tail execution",
            "deterministic record sharding across eight physical GPUs",
            "explicit insufficient-survival terminal decision",
            "correct GateIdentification/GateJet response pairing",
        ],
    }
    write_json_atomic(output_root / "scientific_invariance_v135.json", invariance)
    if not invariance["scientific_payload_equal"]:
        raise GreenStop("00B_V133_IDENTITY", str(invariance))
    evaluation_payload = plan_payload(evaluation)
    write_json_atomic(output_root / "splits.json", evaluation_payload)
    write_json_atomic(
        output_root / "development_splits.json",
        plan_payload([row for row in evaluation if row.split == "development"]),
    )
    write_json_atomic(output_root / "gate04_legacy_panel.json", gate04_panel)
    tokenizer, hf_model, model, cfg = load_models(device, tokenizer=tokenizer)
    suffix_ids, tokenizer_meta = validate_tokenizer(tokenizer, evaluation + legacy)
    fingerprint = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
        "config": cfg, "tokenizer": tokenizer_meta,
        "embedding_sha256": hashlib.sha256(model.W_E.detach().cpu().numpy().tobytes()).hexdigest(),
        "unembedding_sha256": hashlib.sha256(model.W_U.detach().cpu().numpy().tobytes()).hexdigest(),
    }
    write_json_atomic(output_root / "model_fingerprint.json", fingerprint)
    hf_audit = hf_tl_audit(
        tokenizer, hf_model, model, legacy_gate04, holdout_gate04,
        suffix_ids, device,
    )
    noop = no_op_audit(
        model, tokenizer, holdout_gate04, suffix_ids, device,
        hf_audit["tl_references"],
    )
    write_json_atomic(output_root / "hook_audit.json", {
        "hf_vs_tl": hf_audit, "no_op_patch": noop,
    })
    del hf_model
    torch_module().cuda.empty_cache()
    preflight_records = sorted(
        legacy_gate04 + holdout_gate04,
        key=lambda row: sha256_text("structural-preflight-v13|" + row.pair_digest),
    )[:8]
    preflight_record = preflight_records[0]
    preflight_plain = _capture_structural_inputs(
        model, tokenizer, suffix_ids, preflight_records, device,
        output_root, "preflight",
    )
    preflight_design = _construct_structural_design(
        model, preflight_records, preflight_plain, output_root, "preflight"
    )
    preflight_audits = json.loads(
        (output_root / "preflight_frame_audit.json").read_text(encoding="utf-8")
    )
    item_audits = list(preflight_audits.values())
    gate_audits = [row for item in item_audits for row in item["gates"]]
    structural_preflight = {
        "pair_digests": [row.pair_digest for row in preflight_records],
        "frame_audit_sha256": sha256_file(output_root / "preflight_frame_audit.json"),
        "common_sha256": frame_sha256(preflight_design[preflight_record.pair_digest]["common"]),
        "all_gate_sha256": frame_sha256(preflight_design[preflight_record.pair_digest]["all_gate"]),
        "common_frame_dimension": COMMON_FRAME_DIM,
        "gate_frame_dimension": PROBE_FRAME_DIM,
        "all_gate_frame_dimension": ALL_GATE_FRAME_DIM,
        "max_orthogonality_error": max(
            [value for item in item_audits for value in (item["common"]["orthogonal_max_abs"], item["all_gate"]["orthogonal_max_abs"])]
            + [row["containment"]["orthogonal_max_abs"] for row in gate_audits]
        ),
        "max_atom_residual": max(
            [value for item in item_audits for value in (item["common"]["atom_residual_relative"], item["all_gate"]["atom_residual_relative"])]
            + [row["containment"]["atom_residual_relative"] for row in gate_audits]
        ),
        "max_gradient_residual": max(
            values["envelope_relative"] for row in gate_audits
            for values in row["gradients"].values()
        ),
        "max_gradient_autograd_abs": max(
            values["autograd_max_abs"] for row in gate_audits
            for values in row["gradients"].values()
        ),
        "max_gradient_autograd_relative": max(
            values["autograd_relative"] for row in gate_audits
            for values in row["gradients"].values()
        ),
        "max_shift_null_metric": max(
            values["shift_null"] for row in gate_audits
            for values in row["gradients"].values()
        ),
        "repeated_frames_bitwise_equal": True,
        "passed": True,
    }
    write_json_atomic(output_root / "structural_frame_preflight.json", structural_preflight)
    coefficients = first_order_directions()
    coefficient_hash = hashlib.sha256(coefficients.tobytes()).hexdigest()
    if coefficient_hash != FIRST_ORDER_COEFFICIENT_SHA256:
        raise GreenStop(
            "07_FIRST_ORDER_COEFFICIENTS",
            f"coefficient hash {coefficient_hash} != {FIRST_ORDER_COEFFICIENT_SHA256}",
        )
    _atomic_np_save(output_root / "first_order_coefficients.npy", coefficients)
    tail_result = _tail_preflight_v135(
        model, tokenizer, suffix_ids, preflight_record, device,
        output_root, preflight_design,
    )
    write_json_atomic(output_root / "tail_audit.json", tail_result)
    # Preflight scratch inputs are deliberately removed before prepare commits;
    # only the authorized audit digest survives in the frozen interface.
    for name in (
        "preflight_anchor_cache.pt", "preflight_structural_inputs.npz",
        "preflight_structural_input_hashes.json", "preflight_frames.npz",
        "preflight_frame_audit.json", "preflight_radii.json",
        "preflight_target_vectors.npz",
    ):
        (output_root / name).unlink()
    required = (
        "run_ledger.json", "model_fingerprint.json", "splits.json",
        "development_splits.json", "gate04_legacy_panel.json",
        "hook_audit.json", "structural_frame_preflight.json",
        "first_order_coefficients.npy", "scientific_invariance_v135.json",
        "manual_tail_root_cause_reproduction_v135.json",
        "manual_tail_stage_trace_v135.json",
        "manual_tail_equivalence_v135.json",
        "manual_tail_derivative_v135.json",
        "manual_tail_batch_equivalence_v135.json",
        "path_target_equivalence_v135.json", "hardware_batch_plan.json",
        "manual_tail_throughput_v135.json", "tail_audit.json",
        "prepare_result.json",
    )
    prepare_result = {
        "schema_version": "green-bridge-prepare-v1.3.5",
        "verdict": "PREPARE_PASS",
        "first_failed_gate": None,
        "development_started": False,
        "confirmation_started": False,
    }
    write_json_atomic(output_root / "prepare_result.json", prepare_result)
    manifest = {
        "schema_version": "green-bridge-manifest-v1.3.5",
        "execution_commit": repository["commit"],
        "review_commit": REVIEW_COMMIT,
        "repository": {
            "url": "https://github.com/ScottBlizzard/idle_1",
            "branch": repository["branch"], "review_commit": REVIEW_COMMIT,
            "execution_commit": repository["commit"],
            "review_commit_is_ancestor": True,
            "repository_dirty_at_launch": False, "status_porcelain": "",
        },
        "run": {
            "protocol_id": PROTOCOL_ID,
            "parent_protocol_id": PARENT_PROTOCOL_ID,
            "protocol_run_id": PROTOCOL_RUN_ID,
            "amendment_id": AMENDMENT_ID,
            "attempt_index": 1, "retry_allowed": False,
            "prepare_restart_allowed": False,
            "development_restart_allowed": False,
            "confirmation_restart_allowed": False,
            "phase_all_allowed": False,
        },
        "binding_decision": {
            "document": "analysis/CODEX_GREEN_V135_GATEJET_RESPONSE_PAIRING_DECISION_20260825.md",
            "fixed_rank_donor_pca_terminated": True,
            "rank_search_allowed": False, "donor_basis_allowed": False,
            "spectral_filter_allowed": False, "learned_alignment_allowed": False,
        },
        "historical_v12_stop": {
            "execution_commit": "c40405122c779337f44b811c42850b36ba5ff850",
            "first_failed_gate": "08B_BASIS_FIT_SPECTRUM",
            "sigma5_over_sigma6": 1.0227285601080833,
            "sigma5_over_sigma1": 0.535667052214108,
            "development_responses_observed": False,
            "confirmation_responses_observed": False,
            "matrix_hash_serialization_order_defect": True,
            "defect_could_change_spectrum": False,
            "full_spectrum_preserved": False, "retry_authorized": False,
        },
        "historical_artifact_hashes": {
            "result_json": "390c5b62d5b42e216abbb15a0d6d206a55419c48117f610f34c0ac802e153747",
            "manifest_json": "ea486fe8eea798b16951fcea9394b1c4ddb4b44bbd4afb5c8b104b37aaf047be",
            "hook_audit_json": "49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99",
            "donor_plan_json": "2c8dd401b93d3864969ab941b85cae2ab5e6e983bdf39b909f33c532b480cc16",
            "run_ledger_json": "fa88911fcce749942a24c9e479c66cf89cd72ce9386b76d146262de6671b4f65",
        },
        "structural_estimand": {
            "name": "basis-free matched-bypass ambient path operator",
            "ambient_input_dimension": 768, "output_dimension": 100,
            "operator_rank_max": 1,
            "quotient_null_direction": "constant residual shift",
            "donor_pca_used": False, "eigengap_assumed": False,
        },
        "structural_frame": {
            "common_frame_dimension": COMMON_FRAME_DIM,
            "gate_frame_dimension": PROBE_FRAME_DIM,
            "all_gate_frame_dimension": ALL_GATE_FRAME_DIM,
            "orthogonality_max_abs": STRUCTURAL_FRAME_ORTHOGONAL_MAX,
            "atom_residual_max": STRUCTURAL_ATOM_RESIDUAL_MAX,
            "gradient_residual_max": STRUCTURAL_GRADIENT_RESIDUAL_MAX,
            "repeated_frame_bitwise_equal": True,
        },
        "frozen_spec": FROZEN_SPEC, "frozen_spec_sha256": frozen_spec_hash(),
        "scientific_spec_sha256": current_scientific_hash,
        "predecessor_archive_sha256": sha256_text(canonical_json(predecessor_archive)),
        "predecessor_archive": predecessor_archive,
        "source_sha256": source_hashes(),
        "requirements_sha256": sha256_file(PROJECT_ROOT / "requirements-green-bridge.lock"),
        "protocol_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in PROTOCOL_FILES},
        "evaluation_plan_sha256": evaluation_payload["records_sha256"],
        "environment": environment, "model_config": cfg,
        "forward_counts": FORWARD_COUNTS,
        "first_order": {
            "coordinate_dimension": ALL_GATE_FRAME_DIM,
            "directions": FIRST_ORDER_RESIDUAL_DIRECTIONS,
            "seed": FIRST_ORDER_COEFFICIENT_SEED,
            "coefficient_sha256": coefficient_hash,
        },
        "artifact_sha256": {name: sha256_file(output_root / name) for name in required},
        "prepare_complete": True, "confirmation_open": False,
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    finalize_hashes(output_root)


def _load_active_models_and_suffixes(output_root: Path, device: str, records):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer, hf_model, model, _ = load_models(device, tokenizer=tokenizer)
    legacy = build_legacy_donor_records(
        lambda first, second: token_pair_allowed(tokenizer, first, second)
    )
    suffix_ids, _ = validate_tokenizer(tokenizer, list(records) + legacy)
    del hf_model
    torch_module().cuda.empty_cache()
    return tokenizer, model, suffix_ids


def development_phase(output_root: Path, device: str) -> None:
    manifest = verify_freeze(output_root)
    activate_hardware_batch_plan(output_root)
    _assert_no_uncommitted_endpoint(output_root)
    claim_phase(output_root, "development")
    records = load_split_file(output_root, "development_splits.json")
    development = split_records(records, "development")
    tokenizer, model, suffix_ids = _load_active_models_and_suffixes(
        output_root, device, development
    )
    plain = _capture_structural_inputs(
        model, tokenizer, suffix_ids, development, device, output_root, "development"
    )
    design = _construct_structural_design(
        model, development, plain, output_root, "development"
    )
    noise = _duplicate_noise_v13(
        model, tokenizer, suffix_ids, development, device
    )
    epsilon_y = max(1e-7, float(noise["max_abs"]))
    noise["epsilon_y_dev"] = epsilon_y
    write_json_atomic(output_root / "noise_audit_dev.json", noise)
    del tokenizer, model, plain, design
    gc.collect()
    torch_module().cuda.empty_cache()
    payload, dev_sd = _run_split_v135_multigpu(
        development, "development", output_root, epsilon_y
    )
    write_json_atomic(output_root / "dev_cells.json", payload)
    decision = development_decision(payload)
    write_json_atomic(output_root / "dev_result.json", decision)
    if decision["n_surviving_cells"] < THRESHOLDS.development_cells_min:
        raise GreenStop("12_DEVELOPMENT_SURVIVAL", str(decision))
    if decision["n_conditioned_cells"] < THRESHOLDS.development_cells_min:
        raise GreenStop("13_DEVELOPMENT_CONDITIONING", str(decision))
    if decision["n_snr_cells"] < THRESHOLDS.development_snr_cells_min:
        raise GreenStop("14_DEVELOPMENT_SNR", str(decision))
    if decision["verdict"] != "OPEN_CONFIRMATION":
        raise GreenStop("16_DEVELOPMENT_GAIN", str(decision))
    frozen = freeze_confirmation(payload, output_root / "frozen_analysis.json")
    frozen["source_sha256"] = source_hashes()
    frozen["manifest_preconfirmation_sha256"] = sha256_file(output_root / "manifest.json")
    frozen["conditioning_dev_sd"] = dev_sd
    frozen["development_design_sha256"] = sha256_text(canonical_json({
        name: sha256_file(output_root / name) for name in (
            "development_structural_inputs.npz",
            "development_structural_input_hashes.json",
            "development_frames.npz", "development_frame_audit.json",
            "development_radii.json", "development_target_vectors.npz",
        )
    }))
    write_json_atomic(output_root / "frozen_analysis.json", frozen)
    manifest["confirmation_open"] = True
    manifest["frozen_analysis_sha256"] = sha256_file(output_root / "frozen_analysis.json")
    manifest["development_complete"] = True
    for name in (
        "development_structural_inputs.npz",
        "development_structural_input_hashes.json",
        "development_frames.npz",
        "development_frame_audit.json",
        "development_radii.json",
        "development_target_vectors.npz",
        "noise_audit_dev.json",
        "development_multigpu_merge.json",
        "dev_tensor_scores.parquet",
        "dev_energy_targets.parquet",
        "endpoint_ledger.jsonl",
        "dev_cells.json",
        "dev_result.json",
    ):
        manifest["artifact_sha256"][name] = sha256_file(output_root / name)
    manifest["artifact_sha256"]["run_ledger.json"] = sha256_file(
        output_root / "run_ledger.json"
    )
    write_json_atomic(output_root / "manifest.json", manifest)
    finalize_hashes(output_root)


def confirmation_phase(output_root: Path, device: str) -> None:
    manifest = verify_freeze(output_root, require_confirmation=True)
    activate_hardware_batch_plan(output_root)
    _assert_no_uncommitted_endpoint(output_root)
    claim_phase(output_root, "confirmation")
    frozen = json.loads((output_root / "frozen_analysis.json").read_text(encoding="utf-8"))
    if frozen["source_sha256"] != source_hashes():
        raise GreenStop("17_MANIFEST_FREEZE", "frozen source hash mismatch")
    records = load_split_file(output_root)
    confirmation = split_records(
        records, "confirmation",
        confirmation_lock=ConfirmationLock(output_root / "frozen_analysis.json"),
    )
    tokenizer, model, suffix_ids = _load_active_models_and_suffixes(
        output_root, device, confirmation
    )
    plain = _capture_structural_inputs(
        model, tokenizer, suffix_ids, confirmation, device, output_root, "confirmation"
    )
    design = _construct_structural_design(
        model, confirmation, plain, output_root, "confirmation"
    )
    noise = _duplicate_noise_v13(
        model, tokenizer, suffix_ids, confirmation, device
    )
    dev_noise = float(json.loads(
        (output_root / "noise_audit_dev.json").read_text(encoding="utf-8")
    )["epsilon_y_dev"])
    permitted = max(2.0 * dev_noise, 2e-6)
    noise["permitted_max_abs"] = permitted
    write_json_atomic(output_root / "noise_audit_confirm.json", noise)
    if noise["max_abs"] > permitted:
        raise GreenStop("18_CONFIRMATION_NOISE", f"{noise['max_abs']} > {permitted}")
    del tokenizer, model, plain, design
    gc.collect()
    torch_module().cuda.empty_cache()
    payload, _ = _run_split_v135_multigpu(
        confirmation,
        "confirmation",
        output_root,
        dev_noise,
        dev_sd=float(frozen["conditioning_dev_sd"]),
    )
    write_json_atomic(output_root / "confirm_cells.json", payload)
    result = confirmation_decision(payload, frozen)
    dev_cells = json.loads(
        (output_root / "dev_cells.json").read_text(encoding="utf-8")
    )["cells"]
    total_survival = sum(row.get("survived", False) for row in dev_cells + payload["cells"])
    result["total_surviving_cells"] = total_survival
    if total_survival < THRESHOLDS.total_cells_technical_min:
        result["verdict"] = "FAIL_TOTAL_SURVIVAL"
        result["first_failed_gate"] = "19_TOTAL_SURVIVAL"
    elif result["verdict"] != "ORAL_RESULT_PASS":
        result["first_failed_gate"] = (
            "19_CONFIRMATION_SURVIVAL"
            if result.get("n_cells", 0) < THRESHOLDS.confirmation_technical_min
            else "19_CONFIRMATION_CONDITIONING"
            if result.get("n_conditioned", 0) < THRESHOLDS.confirmation_oral_min
            else "20_CONFIRMATORY_THRESHOLD"
        )
    else:
        result["first_failed_gate"] = None
    result["schema_version"] = "green-bridge-terminal-v1.3.5"
    write_json_atomic(output_root / "result.json", result)
    manifest["confirmation_complete"] = True
    for name in (
        "confirmation_structural_inputs.npz",
        "confirmation_structural_input_hashes.json",
        "confirmation_frames.npz",
        "confirmation_frame_audit.json",
        "confirmation_radii.json",
        "confirmation_target_vectors.npz",
        "noise_audit_confirm.json",
        "confirmation_multigpu_merge.json",
        "confirm_tensor_scores.parquet",
        "confirm_energy_targets.parquet",
        "endpoint_ledger.jsonl",
        "confirm_cells.json",
        "result.json",
    ):
        manifest["artifact_sha256"][name] = sha256_file(output_root / name)
    manifest["artifact_sha256"]["run_ledger.json"] = sha256_file(
        output_root / "run_ledger.json"
    )
    write_json_atomic(output_root / "manifest.json", manifest)
    finalize_hashes(output_root)


def finalize_hashes(output_root: Path) -> None:
    paths = sorted(path for path in output_root.iterdir() if path.is_file() and path.name != "sha256sums.txt")
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    path = output_root / "sha256sums.txt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(("\n".join(lines) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "development", "confirmation"), required=True)
    parser.add_argument("--device", default="cuda:0", help="hardware placement only")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="artifact location only")
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            prepare(args.output_root, args.device)
        if args.phase == "development":
            development_phase(args.output_root, args.device)
        if args.phase == "confirmation":
            confirmation_phase(args.output_root, args.device)
    except GreenStop as exc:
        if exc.gate in {"00_REPOSITORY_CLEAN", "00_OUTPUT_ROOT_NOT_EMPTY"}:
            raise
        terminal_stop(args.output_root, exc.gate, exc.detail)


if __name__ == "__main__":
    main()
