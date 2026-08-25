"""Prepare-only coordinator and pure record contracts for GREEN v3.0.0.

Development and confirmation deliberately remain non-callable until a new
binding GPT Pro decision is supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from green_bridge_v300_dataset import (
    build_green_bridge_v300_records,
    build_green_bridge_v300_split,
    canonical_v300_split_payload,
    v300_record_plan,
)
from green_bridge_v300_directions import coefficient_serializer_status_v300
from green_bridge_v300_numerics import (
    classify_gate_v300,
    normalized_transport_error_v300,
    response_detectability_v300,
)
from green_bridge_v300_spec import (
    ATTEMPT_INDEX,
    AUTHORIZED_PHASES,
    PHASE_ALL_ALLOWED,
    POSTMORTEM_COMMIT,
    PROJECT_ROOT,
    PROTOCOL_ID,
    RESUME_ALLOWED,
    RETRY_ALLOWED,
    V200_EXECUTION_COMMIT,
    V200_FIRST_FAILED_GATE,
    V200_VERDICT,
    V300_RADIUS_CANDIDATE_SHA256,
    V300_SPLIT_SHA256,
    computed_radius_candidate_payload_sha256_v300,
    radius_candidate_payload_v300,
)


UNAUTHORIZED_PHASE = "UNAUTHORIZED_PHASE_REQUIRES_NEW_GPTPRO_DECISION"
PREPARE_SERIALIZER_STOP = "PREPARE STOP 07_PROTOCOL_HASH_SERIALIZER_UNRESOLVED"
FORBIDDEN_V300_RUNTIME_INPUTS = (
    "analysis/GREEN_V21_POSTMORTEM_20260825/",
    "analysis/GREEN_V200_DEVELOPMENT_TERMINAL_DIAGNOSTIC_20260825.json",
    "analysis/GREEN_V136_TERMINAL_AUDIT_20260825/terminal_admissibility_audit.json",
    "analysis/archive/green_v200_stop_20260825/dev_tensor_scores.parquet",
    "analysis/archive/green_v200_stop_20260825/dev_energy_targets.parquet",
)
FORBIDDEN_PREPARE_ARTIFACTS = (
    "dev_transport_scores.parquet", "dev_joint_targets.parquet", "dev_cells.json",
    "dev_result.json", "frozen_analysis.json", "confirm_transport_scores.parquet",
    "confirm_joint_targets.parquet", "confirm_cells.json", "confirm_result.json",
)
REQUIRED_PREPARE_ARTIFACTS = (
    "run_ledger.json", "predecessor_v200_terminal_manifest.json", "postmortem_manifest.json",
    "v300_split.json", "v300_record_plan.json", "v300_direction_design.json",
    "v300_direction_design.npz", "v300_radius_candidate_panel.json",
    "v300_radius_calibration.json", "v300_synthetic_theorem_suite.json",
    "v300_model_fingerprint.json", "v300_gate04_audit.json",
    "v300_manual_tail_equivalence.json", "v300_structural_frame_preflight.json",
    "v300_transport_theorem_preflight.json", "v300_joint_composition_preflight.json",
    "v300_operation_counts.json", "v300_hardware_plan.json",
    "v300_throughput_preflight.json", "prepare_result.json", "manifest.json",
    "sha256sums.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_hashes(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def verify_v200_terminal_archive_v300(
    archive: Path | None = None, postmortem: Path | None = None
) -> dict:
    archive = archive or PROJECT_ROOT / "analysis/archive/green_v200_stop_20260825"
    postmortem = postmortem or PROJECT_ROOT / "analysis/GREEN_V21_POSTMORTEM_20260825"
    expected = _parse_hashes(archive / "sha256sums.txt")
    mismatches = {
        name: {"expected": digest, "actual": _sha256(archive / name)}
        for name, digest in expected.items() if (archive / name).is_file()
        and _sha256(archive / name) != digest
    }
    result = _read_json(archive / "result.json")
    ledger = _read_json(archive / "run_ledger.json")
    manifest = _read_json(postmortem / "postmortem_manifest.json")
    checks = {
        "selected_archive_hashes": not mismatches,
        "official_verdict": result.get("verdict") == V200_VERDICT,
        "first_failed_gate": result.get("first_failed_gate") == V200_FIRST_FAILED_GATE,
        "confirmation_closed": not ledger.get("confirmation_started", True),
        "postmortem_complete": manifest.get("all_twelve_complete") is True,
        "postmortem_commit": manifest.get("postmortem_commit") == POSTMORTEM_COMMIT,
        "execution_commit": manifest.get("official_execution_commit") == V200_EXECUTION_COMMIT,
        "transport_theorem": _read_json(postmortem / "04_exact_transport_identity.json").get("theorem_failures") == 0,
        "joint_composition": _read_json(postmortem / "05_exact_joint_composition.json").get("composition_failures") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V200_TERMINAL_OR_POSTMORTEM_INTEGRITY_FAILURE: {checks} {mismatches}")
    return {"schema_version": "green-bridge-v3.0.0-predecessor-manifest-v1",
            "checks": checks, "hash_mismatches": mismatches,
            "v200_root_write_allowed": False,
            "v200_development_parquets_diagnostic_only": True,
            "fixed_rank_donor_pca_terminated": True}


def verify_v300_runtime_input_firewall(paths: Iterable[str | Path]) -> bool:
    for value in paths:
        normalized = str(value).replace("\\", "/")
        if any(forbidden.rstrip("/") in normalized for forbidden in FORBIDDEN_V300_RUNTIME_INPUTS):
            raise RuntimeError(f"V300_SOURCE_CODE_FIREWALL: {normalized}")
    return True


def select_radius_calibration_panel_v300(records: Iterable[Mapping]) -> list[dict]:
    """Select 40 behavior-blind legacy-donor strata from prompt metadata only."""
    allowed = {"pair_digest", "distance_bin", "population", "prompt_metadata"}
    candidates = []
    for record in records:
        row = dict(record)
        if row.get("population") != "legacy_donor":
            continue
        public = {name: row.get(name) for name in allowed}
        for system in ("tar", "pat"):
            for gate_slot in range(10):
                key = hashlib.sha256(
                    ("green-v300-radius-calibration-20260825|"
                     f"{public['pair_digest']}|{system}|{gate_slot}|{public['distance_bin']}").encode("utf-8")
                ).hexdigest()
                candidates.append((public["distance_bin"], system, gate_slot, key, public))
    selected = []
    for distance in ("near", "far"):
        for system in ("tar", "pat"):
            for gate_slot in range(10):
                rows = [row for row in candidates if row[:3] == (distance, system, gate_slot)]
                if not rows:
                    raise RuntimeError("RADIUS_CALIBRATION_LEGACY_STRATUM_MISSING")
                _, _, _, selection_hash, public = min(rows, key=lambda row: row[3])
                selected.append({
                    "pair_digest": public["pair_digest"], "distance_bin": distance,
                    "system": system, "gate_slot": gate_slot,
                    "selection_hash": selection_hash, "population": "legacy_donor",
                    "prompt_metadata": public.get("prompt_metadata"),
                })
    return selected


def transport_record_v300(gate_rows: Iterable[Mapping]) -> dict:
    rows = [dict(row) for row in gate_rows]
    classes = Counter(row["gate_class"] for row in rows)
    contradictions = classes["structural-contradiction"]
    invalid = classes["numerical-invalid"]
    resolved = classes["recoverable"] + classes["certified-numerical-null"]
    errors = [float(row["normalized_error"]) for row in rows if row["gate_class"] == "recoverable"]
    return {
        "all_twenty_system_gates_classified": len(rows) == 20,
        "class_counts": dict(classes),
        "technically_admissible": bool(
            len(rows) == 20 and not contradictions and not invalid and resolved / 20 >= 0.80
            and all(row.get("direct_target_certified", False) for row in rows)
        ),
        "recoverable_errors": errors,
    }


def joint_record_v300(gate_intervals: Iterable[Mapping], target: float, target_bound: float) -> dict:
    rows = [dict(row) for row in gate_intervals]
    if len(rows) != 10:
        raise ValueError("all ten gates must be accounted for")
    if any(row["gate_class"] in ("numerical-invalid", "structural-contradiction") for row in rows):
        return {"technically_admissible": False}
    center = math.fsum(float(row.get("center", 0.0)) for row in rows if row["gate_class"] == "recoverable")
    unresolved_bound = math.fsum(
        float(row.get("bound", 0.0)) for row in rows if row["gate_class"] == "unresolved"
    )
    total_bound = math.fsum(float(row.get("bound", 0.0)) for row in rows) + float(target_bound)
    return {
        "technically_admissible": math.isfinite(total_bound),
        "center": center, "bound": total_bound, "unresolved_bound": unresolved_bound,
        "target": float(target),
        "normalized_error": normalized_transport_error_v300(center, target, total_bound),
        "unresolved_is_zeroed": False,
    }


def run_split_v300(*args, phase: str, **kwargs):
    del args, kwargs
    if phase in ("development", "confirmation"):
        raise RuntimeError(UNAUTHORIZED_PHASE)
    raise ValueError("run_split_v300 accepts only future development or confirmation phases")


def development_v300(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(UNAUTHORIZED_PHASE)


def confirmation_v300(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(UNAUTHORIZED_PHASE)


def _protocol_serializer_status_v300() -> dict:
    coefficient = coefficient_serializer_status_v300()
    radius_computed = computed_radius_candidate_payload_sha256_v300()
    return {
        "coefficient": coefficient,
        "radius": {
            "binding_sha256": V300_RADIUS_CANDIDATE_SHA256,
            "computed_typed_payload_sha256": radius_computed,
            "byte_serializer_specified_by_decision": False,
            "resolved": radius_computed == V300_RADIUS_CANDIDATE_SHA256,
        },
        "all_resolved": coefficient["resolved"] and radius_computed == V300_RADIUS_CANDIDATE_SHA256,
    }


def prepare_v300(output_root: Path, device: str = "cuda:0") -> None:
    """Run the sole authorized phase after every byte-level protocol hash resolves.

    The external decision gives two binding hashes without their canonical byte
    serializers.  Failing closed here prevents burning the one-shot with a
    guessed serialization.  No formal root is created before this check.
    """
    if AUTHORIZED_PHASES != ("prepare",) or RETRY_ALLOWED or RESUME_ALLOWED or PHASE_ALL_ALLOWED:
        raise RuntimeError("V300_PHASE_IDENTITY_FAILURE")
    if device != "cuda:0":
        raise RuntimeError("PREPARE_REQUIRES_PHYSICAL_GPU_4_VISIBLE_AS_CUDA_0")
    output_root = Path(output_root)
    if output_root.exists():
        raise RuntimeError("FORMAL_V300_ROOT_ALREADY_EXISTS")
    status = _protocol_serializer_status_v300()
    if not status["all_resolved"]:
        raise RuntimeError(f"{PREPARE_SERIALIZER_STOP}: {json.dumps(status, sort_keys=True)}")
    raise RuntimeError(
        "PREPARE_IMPLEMENTATION_AWAITS_BINDING_SERIALIZER_CORRIGENDUM; "
        "formal root was not created"
    )


def synthetic_gate_class_v300(values: Mapping) -> str:
    detectability = response_detectability_v300(
        values["curvature_norm"], values["epsilon_c"], values["response_norm"],
        values["epsilon_g"], values["operator_norm"], values["epsilon_p"],
    )
    return classify_gate_v300(
        numerical_valid=values.get("numerical_valid", True),
        structural_valid=values.get("structural_valid", True),
        recoverable=detectability["recoverable"],
        exact_operator_upper=values["exact_operator_upper"],
        direct_numerical_floor=values["direct_numerical_floor"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("prepare", "development", "confirmation"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase != "prepare":
        raise RuntimeError(UNAUTHORIZED_PHASE)
    prepare_v300(args.output_root, args.device)


if __name__ == "__main__":
    main()
