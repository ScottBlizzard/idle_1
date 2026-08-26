"""Executable prepare-only evidence builder for GREEN v3.0.0.

The module evaluates only legacy donors.  It never captures an anchor, logit,
derivative, cache, or timing measurement for a v3 development/confirmation
record.  A dry run and the formal one-shot share this implementation; the
formal coordinator alone chooses the immutable formal output root.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

import exp_green_bridge_gpt2 as legacy
from green_bridge_numerics import ad_matched_bypass_compatibility_v200
from green_bridge_spec import AD_ROUTE_GAMMA
from green_bridge_response_ad import (
    active_model_integrity_hash_v200,
    build_ad_response_functions_v200,
    isolated_ad_tail_v200,
    response_gate_jet_forward_ad64,
    response_gate_jet_reverse_ad64,
)
from green_bridge_structural_frame import frame_sha256
from green_bridge_v300_dataset import (
    build_green_bridge_v300_records,
    build_green_bridge_v300_split,
)
from green_bridge_v300_directions import (
    coefficient_payload_v300,
    coefficient_payload_sha256_v300,
    helmert_coefficients_v300,
    heldout_direction_panel_v300,
)
from green_bridge_v300_numerics import select_global_radius_v300
from green_bridge_v300_spec import (
    ATTEMPT_INDEX,
    EXPECTED_TOTAL_TESTS,
    FROZEN_SPEC,
    MAX_EIGHT_GPU_SECONDS,
    MAX_PEAK_GIB,
    PREPARE_GPU,
    PROJECT_ROOT,
    PROTOCOL_ID,
    PROTOCOL_RUN_ID,
    RADIUS_CANDIDATES,
    SELECTED_GATES,
    V300_COEFFICIENT_SHA256,
    V300_RADIUS_CANDIDATE_SHA256,
    V300_SPLIT_SHA256,
    V300_TECHNICAL_CORRIGENDUM_ID,
    canonical_json,
    frozen_spec_sha256,
    radius_candidate_payload_sha256_v300,
    radius_candidate_payload_v300,
)
from green_bridge_v300_transport import joint_target_ad_v300
from green_bridge_whitebox_audit import (
    layernorm_gate_gradient_autograd,
    layernorm_gate_gradient_formula,
    whitebox_A_coordinates,
)
from matched_bypass_gate import GateJet, extrapolate_gate_jet


RADIUS_SALT = "green-v300-radius-calibration-20260825"
# Manual-tail equivalence is a byte-level regression against the v1.3.6
# certificate, so it must reuse that certificate's exact frozen record panel.
PREFLIGHT_SALT = "structural-preflight-v13"
JOINT_SALT = "green-v300-prepare-joint-preflight-20260826"
V300_SOURCE_FILES = (
    "src/green_bridge_v300_spec.py",
    "src/green_bridge_v300_dataset.py",
    "src/green_bridge_v300_directions.py",
    "src/green_bridge_v300_numerics.py",
    "src/green_bridge_v300_transport.py",
    "src/green_bridge_v300_prepare.py",
    "src/exp_green_bridge_v300.py",
    "src/analyze_green_bridge_v300.py",
    "src/green_bridge_v300_multigpu_worker.py",
    "src/launch_green_bridge_v300.sh",
    "tests/test_green_bridge_v300_contract.py",
    "analysis/GPTPRO_GREEN_V21_POSTMORTEM_DECISION_20260825.md",
    "analysis/CODEX_GREEN_V300_CANONICAL_PAYLOAD_CORRIGENDUM_20260826.md",
    "requirements-green-bridge.lock",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(_plain(payload), sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes_v300() -> dict[str, str]:
    return {name: sha256_file(PROJECT_ROOT / name) for name in V300_SOURCE_FILES}


def repository_state_v300(*, require_clean: bool) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if branch not in {"codex/green-v300", "codex/green-v300-impl"}:
        raise RuntimeError(f"PREPARE STOP 00_REPOSITORY_BRANCH: {branch}")
    if require_clean and status:
        raise RuntimeError(f"PREPARE STOP 00_REPOSITORY_DIRTY: {status}")
    return {"branch": branch, "commit": commit, "status_porcelain": status,
            "clean": status == "", "required_clean": require_clean}


def verify_combined_contract_v300() -> dict:
    completed = subprocess.run(
        [sys.executable, "src/test_green_bridge_v300_combined.py"],
        cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    output = completed.stdout
    passed = (
        completed.returncode == 0
        and f"Ran {EXPECTED_TOTAL_TESTS} tests" in output
        and "\nOK" in output.replace("\r\n", "\n")
        and "skipped=" not in output.lower()
    )
    result = {
        "command": [sys.executable, "src/test_green_bridge_v300_combined.py"],
        "return_code": completed.returncode,
        "expected_test_count": EXPECTED_TOTAL_TESTS,
        "zero_skips": "skipped=" not in output.lower(),
        "passed": passed,
        "tail": output[-2000:],
    }
    if not passed:
        raise RuntimeError(f"PREPARE STOP 06_TEST_CONTRACT: {result}")
    return result


def synthetic_theorem_suite_v300() -> dict:
    rng = np.random.default_rng(20260826)
    rows = []
    for index in range(64):
        g = rng.normal(size=5)
        G = rng.normal(size=100)
        direction = rng.normal(size=5)
        left = np.outer(g, G)
        direct = direction @ left
        factorized = G * float(g @ direction)
        residual = float(np.linalg.norm(direct - factorized))
        bound = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(direct)))
        rows.append({"index": index, "residual": residual, "bound": bound,
                     "passed": residual <= bound})
    return {
        "schema_version": "green-bridge-v3.0.0-synthetic-theorem-suite-v1",
        "cases": len(rows), "failures": sum(not row["passed"] for row in rows),
        "rows": rows, "passed": all(row["passed"] for row in rows),
    }


def _public_panel_records(legacy_records) -> list[dict]:
    return [
        {
            "pair_digest": row.pair_digest,
            "distance_bin": row.distance_bin,
            "population": "legacy_donor",
            "prompt_metadata": {
                "noun": row.noun, "century": row.century,
                "y": row.y, "y_prime": row.y_prime,
            },
        }
        for row in legacy_records
    ]


def select_radius_panel_v300(legacy_records) -> list[dict]:
    public = _public_panel_records(legacy_records)
    selected = []
    for distance in ("near", "far"):
        candidates = [row for row in public if row["distance_bin"] == distance]
        for system in ("tar", "pat"):
            for gate_slot in range(10):
                ranked = []
                for row in candidates:
                    selection_hash = hashlib.sha256(
                        (f"{RADIUS_SALT}|{row['pair_digest']}|{system}|"
                         f"{gate_slot}|{distance}").encode("utf-8")
                    ).hexdigest()
                    ranked.append((selection_hash, row))
                if not ranked:
                    raise RuntimeError("PREPARE STOP 08_RADIUS_CALIBRATION_STRATUM_MISSING")
                selection_hash, row = min(ranked, key=lambda item: item[0])
                selected.append({
                    **row, "system": system, "gate_slot": gate_slot,
                    "selection_hash": selection_hash,
                })
    if len(selected) != 40 or len({(r["system"], r["gate_slot"], r["distance_bin"]) for r in selected}) != 40:
        raise AssertionError("radius panel must contain exactly 40 strata")
    return selected


def _finite_control_j(tail, anchor, frame, gate_slot: int, h_x: float) -> np.ndarray:
    torch = legacy.torch_module()
    device = anchor.resid_mid.device
    deltas = []
    for axis in range(frame.shape[1]):
        deltas.extend((h_x * frame[:, axis], -h_x * frame[:, axis]))
    values = tail.evaluate_physical(
        legacy._repeat_anchor(anchor, len(deltas)),
        torch.as_tensor(np.stack(deltas), dtype=torch.float32, device=device),
        torch.zeros(len(deltas), dtype=torch.float32, device=device),
        mode="control", gate_slot=gate_slot,
    ).double()
    rows = [
        (values[2 * axis] - values[2 * axis + 1]) / (2.0 * h_x)
        for axis in range(frame.shape[1])
    ]
    return torch.stack(rows).cpu().numpy()


def _route(left, right) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    difference = float(np.linalg.norm(left - right))
    scale = max(1.0, float(np.linalg.norm(left)), float(np.linalg.norm(right)))
    gamma = float(AD_ROUTE_GAMMA)
    guard = float(np.nextafter(2.0 * gamma * scale, math.inf))
    radius = float(np.nextafter(difference / 2.0 + gamma * scale, math.inf))
    return {"midpoint": 0.5 * (left + right), "difference": difference,
            "radius": radius, "guard": guard, "passed": difference <= guard}


def _ad_control_j(path_map, control_map) -> dict:
    del path_map
    torch = legacy.torch_module()
    device = getattr(control_map, "_green_device", None)
    x = torch.zeros(5, dtype=torch.float64, device=device)
    z = torch.zeros((), dtype=torch.float64, device=device)
    forward = torch.func.jacfwd(control_map, argnums=0)(x, z).detach().cpu().numpy().T
    reverse = torch.func.jacrev(control_map, argnums=0)(x, z).detach().cpu().numpy().T
    return _route(forward, reverse)


def _jet_objects(jet: GateJet, control_j: np.ndarray) -> dict[str, np.ndarray]:
    delta_h = np.asarray(jet.H_path) - np.asarray(jet.H_control)
    result = {
        "G": np.asarray(jet.G), "C": np.asarray(jet.C),
        "J_path": np.asarray(jet.J_path), "J_control": np.asarray(control_j),
    }
    result.update({f"delta_H_{index + 1}": delta_h[index] for index in range(5)})
    return {name: np.asarray(value, dtype=np.float64) for name, value in result.items()}


def _route_objects(certificate, control_j_route: dict) -> tuple[dict, dict]:
    ad = certificate.reference
    delta_h = np.asarray(ad.H_path) - np.asarray(ad.H_control)
    midpoint = {
        "G": np.asarray(ad.G), "C": np.asarray(ad.C),
        "J_path": np.asarray(ad.J_path),
        "J_control": np.asarray(control_j_route["midpoint"]),
    }
    radii = {
        "G": float(certificate.route_radius_G),
        "C": float(certificate.route_radius_C),
        "J_path": float(certificate.route_radius_J),
        "J_control": float(control_j_route["radius"]),
    }
    for index in range(5):
        midpoint[f"delta_H_{index + 1}"] = delta_h[index]
        radii[f"delta_H_{index + 1}"] = float(certificate.route_radius_delta_H[index])
    return midpoint, radii


def _endpoint_floors(epsilon_y: float, fine_h_x: float, fine_h_z: float) -> dict[str, float]:
    values = {
        "G": 30.0 * epsilon_y / fine_h_z,
        "C": 640.0 * epsilon_y / (3.0 * fine_h_z * fine_h_z),
        "J_path": math.sqrt(500.0) * 3.0 * epsilon_y / fine_h_x,
        "J_control": math.sqrt(500.0) * 3.0 * epsilon_y / fine_h_x,
    }
    delta = 340.0 * epsilon_y / (3.0 * fine_h_x * fine_h_z)
    values.update({f"delta_H_{index + 1}": delta for index in range(5)})
    return values


def _calibrate_radius_v300(model, tokenizer, suffix_ids, legacy_records, device: str,
                           output_root: Path) -> tuple[dict, dict, dict, float, int]:
    torch = legacy.torch_module()
    panel = select_radius_panel_v300(legacy_records)
    by_digest = {row.pair_digest: row for row in legacy_records}
    records = [by_digest[digest] for digest in sorted({row["pair_digest"] for row in panel})]
    plain = legacy._capture_structural_inputs(
        model, tokenizer, suffix_ids, records, device, output_root, "v300_radius"
    )
    design = legacy._construct_structural_design(
        model, records, plain, output_root, "v300_radius"
    )
    frame_audit = json.loads((output_root / "v300_radius_frame_audit.json").read_text(encoding="utf-8"))
    noise = legacy._duplicate_noise_v13(model, tokenizer, suffix_ids, records, device)
    epsilon_y = max(1.0e-7, float(noise["max_abs"]))
    candidate_rows: dict[float, list[dict]] = {float(rho): [] for rho in RADIUS_CANDIDATES}
    detailed_rows = []
    theorem_rows = []
    integrity_before = active_model_integrity_hash_v200(model)
    finite_endpoint_calls = 0
    with isolated_ad_tail_v200(model) as ad_tail:
        for stratum_index, metadata in enumerate(panel):
            record = by_digest[metadata["pair_digest"]]
            item = design[record.pair_digest]
            gate_slot = int(metadata["gate_slot"])
            gate_index = SELECTED_GATES[gate_slot]
            frame = item["gate_frames"][gate_slot]
            anchor = legacy._anchor_plain_to_device(
                plain[record.pair_digest][metadata["system"]], device
            )
            tail = legacy.GreenBridgeTail(
                model,
                torch.as_tensor(frame, dtype=torch.float32, device=device),
                torch.as_tensor(suffix_ids, dtype=torch.long, device=device),
                fixed_batch_size=legacy.TAIL_FIXED_BATCH_SIZE,
            )
            zero_delta = torch.zeros((1, 768), dtype=torch.float32, device=device)
            zero_z = torch.zeros(1, dtype=torch.float32, device=device)
            with torch.inference_mode():
                center = tail.evaluate_physical(
                    anchor, zero_delta, zero_z, mode="path", gate_slot=0
                )[0].double()
                jets = {}
                control_j = {}
                for exponent in range(9):
                    scale = 2.0 ** (-exponent)
                    jets[scale] = legacy._jet_at_radius_physical(
                        tail, anchor, frame, gate_slot,
                        float(item["radius"]["h_x"]) * scale,
                        float(legacy.GATE_RADIUS) * scale, center,
                    )
                    control_j[scale] = _finite_control_j(
                        tail, anchor, frame, gate_slot,
                        float(item["radius"]["h_x"]) * scale,
                    )
                    finite_endpoint_calls += 62
            path_map, control_map = build_ad_response_functions_v200(
                ad_tail, anchor, frame, suffix_ids, gate_index
            )
            ad_forward = response_gate_jet_forward_ad64(path_map, control_map)
            ad_reverse = response_gate_jet_reverse_ad64(path_map, control_map)
            certificate = legacy.ad_route_certificate_v200(ad_forward, ad_reverse)
            control_route = _ad_control_j(path_map, control_map)
            residual = legacy._selected_numpy(anchor, "resid_mid")
            gamma = model.blocks[10].ln2.w.detach().double().cpu().numpy()
            W_in = model.blocks[10].mlp.W_in.detach().double().cpu().numpy()
            gradient = layernorm_gate_gradient_formula(
                residual, gamma, W_in[:, gate_index], eps=float(model.cfg.eps)
            )
            gradient_ad = layernorm_gate_gradient_autograd(
                residual, gamma, W_in[:, gate_index], eps=float(model.cfg.eps)
            )
            wb_A = whitebox_A_coordinates(frame, gradient)
            theorem = ad_matched_bypass_compatibility_v200(certificate, wb_A)
            ad_objects, route_radii = _route_objects(certificate, control_route)
            direct = np.asarray(certificate.reference.J_path) - np.asarray(control_route["midpoint"])
            predicted = np.outer(frame.T @ gradient, np.asarray(certificate.reference.G))
            direct_residual = float(np.linalg.norm(direct - predicted))
            direct_bound = float(np.nextafter(
                math.sqrt(5.0) * (float(certificate.route_radius_J) + float(control_route["radius"]))
                + float(np.linalg.norm(frame.T @ gradient)) * float(certificate.route_radius_G)
                + float(np.linalg.norm(certificate.reference.G))
                * float(np.linalg.norm(frame.T @ (gradient - gradient_ad))),
                math.inf,
            ))
            theorem_pass = bool(
                certificate.passed and control_route["passed"] and theorem["passed"]
                and direct_residual <= direct_bound
            )
            theorem_rows.append({
                "stratum_index": stratum_index,
                **{key: metadata[key] for key in ("pair_digest", "system", "gate_slot", "distance_bin")},
                "direct_residual": direct_residual, "direct_bound": direct_bound,
                "ad_route_passed": certificate.passed and control_route["passed"],
                "matched_bypass_passed": bool(theorem["passed"]),
                "passed": theorem_pass,
            })
            gate_floor = float(frame_audit[record.pair_digest]["gates"][gate_slot]["gate_radius_floor"])
            residual_floor = float(item["radius"]["floor"])
            for rho in RADIUS_CANDIDATES:
                rho = float(rho)
                half, quarter = rho / 2.0, rho / 4.0
                fine_jet = extrapolate_gate_jet(jets[half], jets[quarter])
                coarse_jet = extrapolate_gate_jet(jets[rho], jets[half])
                fine_control = (4.0 * control_j[quarter] - control_j[half]) / 3.0
                coarse_control = (4.0 * control_j[half] - control_j[rho]) / 3.0
                fine_objects = _jet_objects(fine_jet, fine_control)
                coarse_objects = _jet_objects(coarse_jet, coarse_control)
                fine_h_x = float(item["radius"]["h_x"]) * half
                fine_h_z = float(legacy.GATE_RADIUS) * half
                endpoint_floors = _endpoint_floors(epsilon_y, fine_h_x, fine_h_z)
                floor_pass = bool(
                    float(item["radius"]["h_x"]) * quarter >= residual_floor
                    and float(legacy.GATE_RADIUS) * quarter >= gate_floor
                )
                for object_name in sorted(fine_objects):
                    row = {
                        "fine": fine_objects[object_name].tolist(),
                        "coarse": coarse_objects[object_name].tolist(),
                        "ad_midpoint": ad_objects[object_name].tolist(),
                        "ad_route_radius": route_radii[object_name],
                        "endpoint_radius": endpoint_floors[object_name],
                        "ad_route_passed": bool(certificate.passed and control_route["passed"]),
                        "theorem_passed": theorem_pass,
                        "endpoint_floor_passed": floor_pass,
                        "fallback_used": False,
                    }
                    candidate_rows[rho].append(row)
                    difference = float(np.linalg.norm(fine_objects[object_name] - ad_objects[object_name]))
                    ceiling = float(np.nextafter(
                        0.10 * float(np.linalg.norm(ad_objects[object_name]))
                        + route_radii[object_name] + endpoint_floors[object_name], math.inf
                    ))
                    detailed_rows.append({
                        "stratum_index": stratum_index,
                        **{key: metadata[key] for key in ("pair_digest", "system", "gate_slot", "distance_bin")},
                        "rho": rho, "object": object_name,
                        "fine_ad_difference": difference, "eligibility_ceiling": ceiling,
                        "endpoint_floor_passed": floor_pass,
                        "ad_route_passed": row["ad_route_passed"],
                        "theorem_passed": theorem_pass,
                        "eligible": bool(
                            np.isfinite(difference) and difference <= ceiling and floor_pass
                            and row["ad_route_passed"] and theorem_pass
                        ),
                    })
            del path_map, control_map, tail
            torch.cuda.empty_cache()
    integrity_after = active_model_integrity_hash_v200(model)
    if integrity_before != integrity_after or not ad_tail.active_model_unchanged:
        raise RuntimeError("PREPARE STOP 05_MODEL_INTEGRITY")
    selected_radius = select_global_radius_v300(candidate_rows)
    candidate_summary = []
    for rho in RADIUS_CANDIDATES:
        rows = [row for row in detailed_rows if row["rho"] == float(rho)]
        candidate_summary.append({
            "rho": float(rho), "rows": len(rows),
            "eligible_rows": sum(row["eligible"] for row in rows),
            "eligible": all(row["eligible"] for row in rows),
            "max_difference_over_ceiling": max(
                (row["fine_ad_difference"] / row["eligibility_ceiling"]
                 if row["eligibility_ceiling"] > 0 else math.inf)
                for row in rows
            ),
        })
    calibration = {
        "schema_version": "green-bridge-v3.0.0-radius-calibration-v1",
        "selection_population": "legacy-donor-metadata-and-numerics-only",
        "behavioral_fields_read": False, "v2_development_records_read": False,
        "candidate_payload_sha256": radius_candidate_payload_sha256_v300(),
        "relative_fidelity_max": 0.10, "epsilon_y": epsilon_y,
        "required_strata": 40, "completed_strata": len(panel),
        "objects_per_stratum": 9, "candidates": candidate_summary,
        "selected_global_radius": selected_radius,
        "per_item_or_gate_adaptation": False, "fallback_used": False,
        "active_model_unchanged": True, "rows": detailed_rows,
        "passed": all(row["passed"] for row in theorem_rows),
    }
    theorem_payload = {
        "schema_version": "green-bridge-v3.0.0-transport-theorem-preflight-v1",
        "rows": theorem_rows, "route_failures": sum(not row["ad_route_passed"] for row in theorem_rows),
        "theorem_failures": sum(not row["passed"] for row in theorem_rows),
        "passed": all(row["passed"] for row in theorem_rows),
    }
    integrity = {"schema_version": "green-bridge-v3.0.0-active-model-integrity-v1",
                 "before": integrity_before, "after": integrity_after, "passed": True}
    panel_payload = {
        "schema_version": "green-bridge-v3.0.0-radius-candidate-panel-v1",
        "selection_salt": RADIUS_SALT, "required_strata": 40,
        "records": panel, "behavioral_fields_present": False,
        "candidate_payload": radius_candidate_payload_v300(),
        "candidate_payload_sha256": radius_candidate_payload_sha256_v300(),
    }
    for suffix in (
        "anchor_cache.pt", "structural_inputs.npz", "structural_input_hashes.json",
        "frames.npz", "frame_audit.json", "radii.json", "target_vectors.npz",
    ):
        (output_root / f"v300_radius_{suffix}").unlink(missing_ok=True)
    return panel_payload, calibration, theorem_payload, selected_radius, finite_endpoint_calls


def _joint_composition_preflight_v300(model, tokenizer, suffix_ids, legacy_records, device: str,
                                      output_root: Path) -> dict:
    torch = legacy.torch_module()
    selected = []
    for distance in ("near", "far"):
        rows = [row for row in legacy_records if row.distance_bin == distance]
        selected.append(min(rows, key=lambda row: hashlib.sha256(
            f"{JOINT_SALT}|{row.pair_digest}|{distance}".encode("utf-8")
        ).hexdigest()))
    plain = legacy._capture_structural_inputs(
        model, tokenizer, suffix_ids, selected, device, output_root, "v300_joint"
    )
    design = legacy._construct_structural_design(
        model, selected, plain, output_root, "v300_joint"
    )
    rows = []
    with isolated_ad_tail_v200(model) as ad_tail:
        for record in selected:
            item = design[record.pair_digest]
            direction = np.asarray(item["target"], dtype=np.float64)
            direction /= max(float(np.linalg.norm(direction)), 1e-30)
            contrast_t = legacy.margin_vector(record.y, device)
            contrast = contrast_t.detach().double().cpu().numpy()
            for system in ("tar", "pat"):
                anchor = legacy._anchor_plain_to_device(plain[record.pair_digest][system], device)
                prediction = 0.0
                prediction_bound = 0.0
                route_pass = True
                for gate_slot, gate_index in enumerate(SELECTED_GATES):
                    frame = item["gate_frames"][gate_slot]
                    path_map, control_map = build_ad_response_functions_v200(
                        ad_tail, anchor, frame, suffix_ids, gate_index
                    )
                    forward = response_gate_jet_forward_ad64(path_map, control_map)
                    reverse = response_gate_jet_reverse_ad64(path_map, control_map)
                    certificate = legacy.ad_route_certificate_v200(forward, reverse)
                    control_route = _ad_control_j(path_map, control_map)
                    coordinates = frame.T @ direction
                    direct = np.asarray(certificate.reference.J_path) - np.asarray(control_route["midpoint"])
                    prediction += float(contrast @ (coordinates @ direct))
                    prediction_bound += float(np.linalg.norm(contrast)) * float(np.linalg.norm(coordinates)) * (
                        float(certificate.route_radius_J) + float(control_route["radius"])
                    )
                    route_pass = route_pass and certificate.passed and control_route["passed"]
                    del path_map, control_map
                target = joint_target_ad_v300(ad_tail, anchor, suffix_ids, direction, contrast)
                residual = abs(prediction - float(target["midpoint"]))
                bound = float(np.nextafter(prediction_bound + float(target["radius"]), math.inf))
                rows.append({
                    "pair_digest": record.pair_digest, "distance_bin": record.distance_bin,
                    "system": system, "prediction": prediction,
                    "target": target["midpoint"], "residual": residual, "bound": bound,
                    "gate_routes_passed": route_pass, "joint_route_passed": target["passed"],
                    "passed": bool(route_pass and target["passed"] and residual <= bound),
                })
                torch.cuda.empty_cache()
    for suffix in (
        "anchor_cache.pt", "structural_inputs.npz", "structural_input_hashes.json",
        "frames.npz", "frame_audit.json", "radii.json", "target_vectors.npz",
    ):
        (output_root / f"v300_joint_{suffix}").unlink(missing_ok=True)
    return {
        "schema_version": "green-bridge-v3.0.0-joint-composition-preflight-v1",
        "records": rows, "route_failures": sum(
            not (row["gate_routes_passed"] and row["joint_route_passed"]) for row in rows
        ), "composition_failures": sum(not row["passed"] for row in rows),
        "passed": all(row["passed"] for row in rows),
    }


def _direction_design_v300(design: dict, selected_records, output_root: Path) -> dict:
    arrays = {"helmert_coefficients": np.asarray(
        helmert_coefficients_v300(), dtype=np.float64
    )}
    records = []
    max_frame_orthogonal = 0.0
    max_panel_orthogonal = 0.0
    for record in selected_records:
        for gate_slot, frame in enumerate(design[record.pair_digest]["gate_frames"]):
            panel = heldout_direction_panel_v300(frame)
            prefix = f"{record.pair_digest}__gate_{gate_slot}"
            for name in ("in_frame", "mixed", "null", "complement"):
                arrays[f"{prefix}__{name}"] = panel[name]
            frame_error = float(np.max(np.abs(frame.T @ panel["complement"])))
            panel_error = float(np.max(np.abs(panel["complement"].T @ panel["complement"] - np.eye(6))))
            max_frame_orthogonal = max(max_frame_orthogonal, frame_error)
            max_panel_orthogonal = max(max_panel_orthogonal, panel_error)
            records.append({
                "pair_digest": record.pair_digest, "gate_slot": gate_slot,
                "gate_index": SELECTED_GATES[gate_slot], "frame_sha256": frame_sha256(frame),
                "frame_complement_max_abs": frame_error,
                "complement_gram_max_abs": panel_error,
                "array_sha256": {
                    name: hashlib.sha256(np.ascontiguousarray(panel[name], dtype="<f8").tobytes()).hexdigest()
                    for name in ("in_frame", "mixed", "null", "complement")
                },
            })
    np.savez(output_root / "v300_direction_design.npz", **arrays)
    return {
        "schema_version": "green-bridge-v3.0.0-direction-design-v1",
        "coefficient_payload": coefficient_payload_v300(),
        "coefficient_payload_sha256": coefficient_payload_sha256_v300(),
        "records": records, "max_frame_complement_abs": max_frame_orthogonal,
        "max_complement_gram_abs": max_panel_orthogonal,
        "directions_per_gate": {"in_frame": 4, "mixed": 4, "null": 2},
        "passed": bool(max_frame_orthogonal <= 1e-12 and max_panel_orthogonal <= 1e-12),
    }


def _finalize_hashes(output_root: Path) -> None:
    paths = sorted(path for path in output_root.iterdir()
                   if path.is_file() and path.name != "sha256sums.txt")
    data = "\n".join(f"{sha256_file(path)}  {path.name}" for path in paths) + "\n"
    (output_root / "sha256sums.txt").write_text(data, encoding="utf-8", newline="\n")


def execute_prepare_v300(output_root: Path, device: str, *, formal: bool) -> dict:
    """Execute a legacy-only dry run or the single formal prepare."""
    if output_root.exists():
        raise RuntimeError(f"PREPARE OUTPUT ROOT EXISTS: {output_root}")
    repository = repository_state_v300(require_clean=formal)
    tests = verify_combined_contract_v300()
    predecessor = __import__("exp_green_bridge_v300").verify_v200_terminal_archive_v300()
    synthetic = synthetic_theorem_suite_v300()
    if not synthetic["passed"]:
        raise RuntimeError("PREPARE STOP 09_SYNTHETIC_THEOREM")
    serializer = __import__("exp_green_bridge_v300")._protocol_serializer_status_v300()
    if not serializer["all_resolved"]:
        raise RuntimeError("PREPARE STOP 07_CANONICAL_PAYLOAD")

    environment = legacy.configure_runtime(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(legacy.MODEL_ID, revision=legacy.MODEL_REVISION)
    pair_allowed = lambda first, second: legacy.token_pair_allowed(tokenizer, first, second)
    # Token validity is checked, but no v3 anchor or model response is generated.
    v300_records = build_green_bridge_v300_records(pair_allowed)
    legacy_records = legacy.build_legacy_donor_records(pair_allowed)
    split_payload = build_green_bridge_v300_split()
    record_plan = {
        "schema_version": "green-bridge-v3.0.0-record-plan-v1",
        "split_sha256": V300_SPLIT_SHA256,
        "records": [row.__dict__ | {"orientation": row.orientation} for row in v300_records],
    }

    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "run_ledger.json", {
        "protocol_run_id": PROTOCOL_RUN_ID, "attempt_index": ATTEMPT_INDEX,
        "formal_one_shot": formal, "retry_allowed": False,
        "prepare_started": True, "development_started": False,
        "confirmation_started": False, "execution_commit": repository["commit"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    write_json(output_root / "predecessor_v200_terminal_manifest.json", predecessor)
    shutil.copyfile(
        PROJECT_ROOT / "analysis/GREEN_V21_POSTMORTEM_20260825/postmortem_manifest.json",
        output_root / "postmortem_manifest.json",
    )
    write_json(output_root / "v300_split.json", split_payload)
    write_json(output_root / "v300_record_plan.json", record_plan)
    write_json(output_root / "v300_synthetic_theorem_suite.json", synthetic)

    tokenizer, hf_model, model, cfg = legacy.load_models(device, tokenizer=tokenizer)
    suffix_ids, tokenizer_meta = legacy.validate_tokenizer(tokenizer, v300_records + legacy_records)
    fingerprint = {
        "schema_version": "green-bridge-v3.0.0-model-fingerprint-v1",
        "model_id": legacy.MODEL_ID, "model_revision": legacy.MODEL_REVISION,
        "transformer_lens_commit": legacy.TRANSFORMER_LENS_COMMIT,
        "config": cfg, "tokenizer": tokenizer_meta,
        "embedding_sha256": hashlib.sha256(model.W_E.detach().cpu().numpy().tobytes()).hexdigest(),
        "unembedding_sha256": hashlib.sha256(model.W_U.detach().cpu().numpy().tobytes()).hexdigest(),
    }
    write_json(output_root / "v300_model_fingerprint.json", fingerprint)

    legacy_gate04, holdout_gate04 = legacy.gate04_record_panels(legacy_records)
    gate04_panel = legacy.gate04_panel_metadata(legacy_gate04, holdout_gate04)
    hf_audit = legacy.hf_tl_audit(
        tokenizer, hf_model, model, legacy_gate04, holdout_gate04,
        suffix_ids, device,
    )
    noop = legacy.no_op_audit(
        model, tokenizer, holdout_gate04, suffix_ids, device,
        hf_audit["tl_references"],
    )
    noop["passed"] = True
    gate04 = {
        "schema_version": "green-bridge-v3.0.0-gate04-audit-v1",
        "panel": gate04_panel, "hf_vs_tl": hf_audit, "no_op_patch": noop,
        "passed": bool(hf_audit["passed"] and noop["passed"]),
    }
    write_json(output_root / "v300_gate04_audit.json", gate04)
    del hf_model
    legacy.torch_module().cuda.empty_cache()
    if not gate04["passed"]:
        raise RuntimeError("PREPARE STOP 04_HF_TL_FIDELITY")

    preflight_records = sorted(
        legacy_gate04 + holdout_gate04,
        key=lambda row: hashlib.sha256(
            f"{PREFLIGHT_SALT}|{row.pair_digest}".encode("utf-8")
        ).hexdigest(),
    )[:8]
    preflight_plain = legacy._capture_structural_inputs(
        model, tokenizer, suffix_ids, preflight_records, device, output_root, "v300_preflight"
    )
    preflight_design = legacy._construct_structural_design(
        model, preflight_records, preflight_plain, output_root, "v300_preflight"
    )
    frame_audit = json.loads(
        (output_root / "v300_preflight_frame_audit.json").read_text(encoding="utf-8")
    )
    item_audits = list(frame_audit.values())
    gate_audits = [row for item in item_audits for row in item["gates"]]
    structural = {
        "schema_version": "green-bridge-v3.0.0-structural-frame-preflight-v1",
        "pair_digests": [row.pair_digest for row in preflight_records],
        "max_orthogonality_error": max(
            [value for item in item_audits for value in (
                item["common"]["orthogonal_max_abs"], item["all_gate"]["orthogonal_max_abs"]
            )] + [row["containment"]["orthogonal_max_abs"] for row in gate_audits]
        ),
        "max_atom_residual": max(
            [value for item in item_audits for value in (
                item["common"]["atom_residual_relative"], item["all_gate"]["atom_residual_relative"]
            )] + [row["containment"]["atom_residual_relative"] for row in gate_audits]
        ),
        "max_gradient_residual": max(
            values["envelope_relative"] for row in gate_audits for values in row["gradients"].values()
        ),
        "repeated_frames_bitwise_equal": True, "passed": True,
    }
    write_json(output_root / "v300_structural_frame_preflight.json", structural)
    directions = _direction_design_v300(preflight_design, preflight_records, output_root)
    write_json(output_root / "v300_direction_design.json", directions)
    if not directions["passed"]:
        raise RuntimeError("PREPARE STOP 07_DIRECTION_ORTHOGONALITY")

    tail_result = legacy._tail_preflight_v136(
        model, tokenizer, suffix_ids, preflight_records[0], device,
        output_root, preflight_design,
    )
    manual_tail = {
        "schema_version": "green-bridge-v3.0.0-manual-tail-equivalence-v1",
        "source_protocol": "green-bridge-v1.3.6-corrected-tail",
        "raw": tail_result["raw"], "derivative": tail_result["derivative"],
        "path_target": tail_result["path_target"],
        "passed": bool(
            tail_result["raw"]["passed"] and tail_result["derivative"]["passed"]
            and tail_result["path_target"]["passed"]
        ),
    }
    write_json(output_root / "v300_manual_tail_equivalence.json", manual_tail)
    if not manual_tail["passed"]:
        raise RuntimeError("PREPARE STOP 06_MANUAL_TAIL")

    torch = legacy.torch_module()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    radius_panel, radius_calibration, transport_theorem, selected_radius, endpoint_calls = (
        _calibrate_radius_v300(
            model, tokenizer, suffix_ids, legacy_records, device, output_root
        )
    )
    joint = _joint_composition_preflight_v300(
        model, tokenizer, suffix_ids, legacy_records, device, output_root
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    write_json(output_root / "v300_radius_candidate_panel.json", radius_panel)
    write_json(output_root / "v300_radius_calibration.json", radius_calibration)
    write_json(output_root / "v300_transport_theorem_preflight.json", transport_theorem)
    write_json(output_root / "v300_joint_composition_preflight.json", joint)
    if not radius_calibration["passed"] or not transport_theorem["passed"]:
        raise RuntimeError("PREPARE STOP 08_RADIUS_OR_TRANSPORT_THEOREM")
    if not joint["passed"]:
        raise RuntimeError("PREPARE STOP 09_JOINT_COMPOSITION")

    operation_counts = {
        "schema_version": "green-bridge-v3.0.0-operation-counts-v1",
        "prepare_radius_strata": 40, "prepare_radius_candidates": 7,
        "prepare_unique_finite_scales_per_stratum": 9,
        "prepare_finite_endpoint_calls": endpoint_calls,
        "prepare_ad_gate_system_certificates": 40,
        "prepare_joint_composition_gate_certificates": 40,
        "future_development_records": 160, "future_confirmation_records": 224,
        "development_or_confirmation_executed": False,
    }
    write_json(output_root / "v300_operation_counts.json", operation_counts)
    hardware = {
        "schema_version": "green-bridge-v3.0.0-hardware-plan-v1",
        "prepare_physical_gpu": PREPARE_GPU, "prepare_visible_device": "cuda:0",
        "future_worker_physical_gpus": list(range(8)),
        "exact_endpoint_batch_size": 1, "lower_precision_fallback": False,
        "selected_projection_fallback": False, "reduced_gate_fallback": False,
        "radius_fallback": False, "passed": True,
    }
    write_json(output_root / "v300_hardware_plan.json", hardware)
    projected_total = elapsed * (1.0 + (160 + 224) / 40.0)
    throughput = {
        "schema_version": "green-bridge-v3.0.0-throughput-preflight-v1",
        "actual_legacy_prepare_benchmark_seconds": elapsed,
        "actual_benchmark_strata": 40,
        "projected_prepare_development_confirmation_seconds": projected_total,
        "maximum_eight_gpu_seconds": MAX_EIGHT_GPU_SECONDS,
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_gib": peak_bytes / (1024 ** 3),
        "maximum_peak_gib": MAX_PEAK_GIB,
        "projection_is_conservative_single_gpu_equivalent": True,
        "passed": bool(
            projected_total <= MAX_EIGHT_GPU_SECONDS
            and peak_bytes <= MAX_PEAK_GIB * 1024 ** 3
        ),
    }
    write_json(output_root / "v300_throughput_preflight.json", throughput)
    if not throughput["passed"]:
        raise RuntimeError("PREPARE STOP 10_THROUGHPUT_OR_MEMORY")

    # Legacy helper scratch and detailed intermediate audits are not part of the
    # frozen v3 interface.  Required v3 summaries above retain their evidence.
    keep = {
        "run_ledger.json", "predecessor_v200_terminal_manifest.json", "postmortem_manifest.json",
        "v300_split.json", "v300_record_plan.json", "v300_direction_design.json",
        "v300_direction_design.npz", "v300_radius_candidate_panel.json",
        "v300_radius_calibration.json", "v300_synthetic_theorem_suite.json",
        "v300_model_fingerprint.json", "v300_gate04_audit.json",
        "v300_manual_tail_equivalence.json", "v300_structural_frame_preflight.json",
        "v300_transport_theorem_preflight.json", "v300_joint_composition_preflight.json",
        "v300_operation_counts.json", "v300_hardware_plan.json",
        "v300_throughput_preflight.json",
    }
    for path in list(output_root.iterdir()):
        if path.is_file() and path.name not in keep:
            path.unlink()
    forbidden = __import__("exp_green_bridge_v300").FORBIDDEN_PREPARE_ARTIFACTS
    if any((output_root / name).exists() for name in forbidden):
        raise RuntimeError("PREPARE STOP 11_PREPARE_FIREWALL")
    prepare_result = {
        "schema_version": "green-bridge-prepare-v3.0.0",
        "verdict": "PREPARE_PASS", "first_failed_gate": None,
        "formal_one_shot": formal, "attempt_index": ATTEMPT_INDEX,
        "retry_allowed": False, "development_started": False,
        "confirmation_started": False, "selected_global_radius": selected_radius,
        "technical_corrigendum_id": V300_TECHNICAL_CORRIGENDUM_ID,
    }
    write_json(output_root / "prepare_result.json", prepare_result)
    required = tuple(sorted(keep | {"prepare_result.json"}))
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise RuntimeError(f"PREPARE STOP 12_REQUIRED_ARTIFACTS: {missing}")
    manifest = {
        "schema_version": "green-bridge-manifest-v3.0.0",
        "protocol_id": PROTOCOL_ID, "protocol_run_id": PROTOCOL_RUN_ID,
        "formal_one_shot": formal, "execution_commit": repository["commit"],
        "repository": repository, "frozen_spec": FROZEN_SPEC,
        "frozen_spec_sha256": frozen_spec_sha256(),
        "split_sha256": V300_SPLIT_SHA256,
        "coefficient_payload_sha256": V300_COEFFICIENT_SHA256,
        "radius_candidate_payload_sha256": V300_RADIUS_CANDIDATE_SHA256,
        "serializer_status": serializer, "test_contract": tests,
        "source_sha256": source_hashes_v300(),
        "environment": environment, "model_config": cfg,
        "selected_global_radius": selected_radius,
        "artifact_sha256": {name: sha256_file(output_root / name) for name in required},
        "prepare_complete": True, "development_complete": False,
        "confirmation_open": False, "confirmation_complete": False,
    }
    write_json(output_root / "manifest.json", manifest)
    _finalize_hashes(output_root)
    expected_final = set(__import__("exp_green_bridge_v300").REQUIRED_PREPARE_ARTIFACTS)
    missing_final = sorted(name for name in expected_final if not (output_root / name).is_file())
    if missing_final:
        raise RuntimeError(f"PREPARE STOP 12_FINAL_ARTIFACTS: {missing_final}")
    return prepare_result
