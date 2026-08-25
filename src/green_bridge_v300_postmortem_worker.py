"""One-GPU immutable shard for exact v2 transport postmortem analyses."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

import exp_green_bridge_gpt2 as runner
from green_bridge_spec import SELECTED_GATES, sha256_file
from green_bridge_whitebox_audit import gradient_envelope_residual, layernorm_gate_gradient_formula
from green_bridge_v300_directions import heldout_direction_panel_v300
from green_bridge_v300_transport import (
    direct_path_control_ad_v300,
    heldout_transport_prediction_v300,
    joint_operator_prediction_v300,
    joint_target_ad_v300,
)


def _margin_vector(clean_suffix: int) -> np.ndarray:
    value = np.empty(100, dtype=np.float64)
    value[: clean_suffix + 1] = -1 / (clean_suffix + 1)
    value[clean_suffix + 1:] = 1 / (99 - clean_suffix)
    return value


def _load_design(root: Path, records) -> dict:
    frames = np.load(root / "development_frames.npz")
    targets = np.load(root / "development_target_vectors.npz")
    radii = json.loads((root / "development_radii.json").read_text(encoding="utf-8"))
    return {record.pair_digest: {
        "common": frames[f"{record.pair_digest}__common"],
        "gate_frames": [frames[f"{record.pair_digest}__gate_{slot}"] for slot in range(10)],
        "all_gate": frames[f"{record.pair_digest}__all_gate"],
        "target": targets[record.pair_digest], "radius": radii[record.pair_digest],
    } for record in records}


def _route_bound(result: dict, projection: float, gradient_error: float) -> float:
    G = np.asarray(result["G"]["midpoint"], dtype=np.float64)
    return math.nextafter(
        result["J_path"]["radius"] + result["J_control"]["radius"]
        + result["G"]["radius"] * abs(float(projection))
        + (float(np.linalg.norm(G)) + result["G"]["radius"]) * gradient_error,
        math.inf,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-v200-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runner.configure_runtime("cuda:0", physical_gpu=args.physical_gpu)
    runner.activate_hardware_batch_plan(args.formal_v200_root)
    records = [record for record in runner.load_split_file(args.formal_v200_root, "development_splits.json")
               if record.role == "tensor"]
    assigned = [record for index, record in enumerate(sorted(records, key=lambda row: row.pair_digest))
                if index % args.worker_count == args.worker_index]
    tokenizer, model, suffix_ids = runner._load_active_models_and_suffixes(
        args.formal_v200_root, "cuda:0", records
    )
    del tokenizer
    torch = runner.torch_module()
    plain = torch.load(args.formal_v200_root / "development_anchor_cache.pt",
                       map_location="cpu", weights_only=True)
    design = _load_design(args.formal_v200_root, assigned)
    epsilon_y = float(json.loads(
        (args.formal_v200_root / "noise_audit_dev.json").read_text(encoding="utf-8")
    )["epsilon_y_dev"])
    tensor_table = pd.read_parquet(args.formal_v200_root / "dev_tensor_scores.parquet").set_index("pair_digest")
    gamma = model.blocks[10].ln2.w.detach().double().cpu().numpy()
    W_in = model.blocks[10].mlp.W_in.detach().double().cpu().numpy()
    started = time.perf_counter()
    output_rows, transport_rows, joint_rows, ladder_rows = [], [], [], []
    with runner.isolated_ad_tail_v200(model) as ad_tail:
        for record in assigned:
            item = design[record.pair_digest]
            anchors = {system: runner._anchor_plain_to_device(values, "cuda:0")
                       for system, values in plain[record.pair_digest].items()}
            contrast = _margin_vector(record.y)
            contrast_t = runner.margin_vector(record.y, "cuda:0")
            physical_v = np.asarray(item["target"], dtype=np.float64)
            archived = tensor_table.loc[record.pair_digest]
            archived_audit = archived["mixed_audit"]
            if isinstance(archived_audit, str):
                archived_audit = json.loads(archived_audit)
            seed_frame = torch.as_tensor(
                item["gate_frames"][0], dtype=torch.float32, device="cuda:0"
            )
            finite_tail = runner.GreenBridgeTail(
                model,
                seed_frame,
                torch.as_tensor(suffix_ids, dtype=torch.long, device="cuda:0"),
                fixed_batch_size=runner.TAIL_FIXED_BATCH_SIZE,
            )
            pre_tar = runner._selected_numpy(anchors["tar"], "pre")[list(SELECTED_GATES)]
            pre_cor = runner._selected_numpy(anchors["cor"], "pre")[list(SELECTED_GATES)]
            gate_chord = pre_tar - pre_cor
            zeta = runner.GATE_RADIUS * gate_chord / max(
                float(np.linalg.norm(gate_chord)), 1e-30
            )
            system_results = {}
            all_transport_passed = True; all_routes_passed = True
            max_transport_ratio = 0.0
            for system_name in ("tar", "pat"):
                anchor = anchors[system_name]
                residual = runner._selected_numpy(anchor, "resid_mid")
                gate_responses, gradients = [], []
                gate_rows, physical_bounds = [], []
                system_ladder = {name: [] for name in (
                    "fine_response", "coarse_response", "ad_response_whitebox",
                    "fine_G_whitebox_g", "active_only_v2",
                    "all_gate_response_where_invertible", "zero_centered_v2",
                )}
                zero_delta = torch.zeros((1, 768), dtype=torch.float32, device="cuda:0")
                zero_z = torch.zeros(1, dtype=torch.float32, device="cuda:0")
                with torch.inference_mode():
                    center = finite_tail.evaluate_physical(
                        anchor, zero_delta, zero_z, mode="path", gate_slot=0
                    )[0].double()
                for slot, (gate, frame) in enumerate(zip(SELECTED_GATES, item["gate_frames"])):
                    panel = heldout_direction_panel_v300(frame)
                    analysis_directions = np.concatenate(
                        (frame, panel["in_frame"], panel["mixed"], panel["null"]), axis=1
                    )
                    direction_metadata = (
                        [("frozen_frame", index) for index in range(frame.shape[1])]
                        + [("heldout_in_frame", index) for index in range(panel["in_frame"].shape[1])]
                        + [("heldout_mixed", index) for index in range(panel["mixed"].shape[1])]
                        + [("heldout_complement_null", index) for index in range(panel["null"].shape[1])]
                    )
                    directions = np.concatenate((analysis_directions, physical_v[:, None]), axis=1)
                    exact = direct_path_control_ad_v300(ad_tail, anchor, directions, suffix_ids, gate)
                    gradient = layernorm_gate_gradient_formula(
                        residual, gamma, W_in[:, gate], eps=float(model.cfg.eps)
                    )
                    predicted = heldout_transport_prediction_v300(exact["G"]["midpoint"], gradient, directions)
                    residuals = np.linalg.norm(predicted - exact["direct"], axis=1)
                    gradient_error = math.nextafter(
                        1e-10 + gradient_envelope_residual(frame, gradient)["absolute"],
                        math.inf,
                    )
                    projections = directions.T @ gradient
                    bounds = np.asarray([_route_bound(exact, value, gradient_error) for value in projections])
                    passed = residuals <= bounds
                    ratios = residuals / np.maximum(bounds, np.finfo(float).tiny)
                    all_routes_passed &= exact["route_passed"]
                    theorem_count = len(direction_metadata)
                    all_transport_passed &= bool(np.all(passed[:theorem_count]))
                    max_transport_ratio = max(
                        max_transport_ratio, float(np.max(ratios[:theorem_count]))
                    )
                    for direction_position, (direction_class, direction_index) in enumerate(direction_metadata):
                        transport_rows.append({
                            "pair_digest": record.pair_digest, "cell_id": record.cell_id,
                            "distance_bin": record.distance_bin, "orientation": record.orientation,
                            "system": system_name, "gate_slot": slot, "gate_index": gate,
                            "direction_class": direction_class,
                            "direction_index": direction_index,
                            "residual": float(residuals[direction_position]),
                            "bound": float(bounds[direction_position]),
                            "residual_to_bound": float(ratios[direction_position]),
                            "route_passed": bool(exact["route_passed"]),
                            "theorem_passed": bool(passed[direction_position]),
                        })
                    gate_responses.append(np.asarray(exact["G"]["midpoint"], dtype=np.float64))
                    gradients.append(gradient)
                    physical_position = directions.shape[1] - 1
                    physical_bounds.append(float(bounds[physical_position]))
                    gate_rows.append({"slot": slot, "gate": gate, "route_passed": exact["route_passed"],
                                      "max_residual": float(np.max(residuals[:theorem_count])),
                                      "max_bound": float(np.max(bounds[:theorem_count])),
                                      "max_ratio": float(np.max(ratios[:theorem_count])),
                                      "passed": bool(np.all(passed[:theorem_count]))})
                    triplet = runner._gate_jet_triplet_v200(
                        finite_tail, anchor, frame, slot, float(item["radius"]["h_x"]),
                        runner.GATE_RADIUS, center, epsilon_y,
                    )
                    fine, coarse = triplet["fine_richardson"], triplet["coarse_richardson"]
                    fine_value = coarse_value = math.nan
                    fine_invertible = coarse_invertible = False
                    try:
                        fine_id = runner.identify_gate(fine)
                        fine_value = float(contrast @ runner.operator_action(
                            fine.G, runner.reconstruct_cotangent(frame, fine_id.A), physical_v
                        ))
                        fine_invertible = True
                    except ValueError:
                        pass
                    try:
                        coarse_id = runner.identify_gate(coarse)
                        coarse_value = float(contrast @ runner.operator_action(
                            coarse.G, runner.reconstruct_cotangent(frame, coarse_id.A), physical_v
                        ))
                        coarse_invertible = True
                    except ValueError:
                        pass
                    oracle_value = float(contrast @ predicted[physical_position])
                    exact_direct_value = float(contrast @ exact["direct"][physical_position])
                    fine_g_whitebox = float(
                        contrast @ runner.operator_action(fine.G, gradient, physical_v)
                    )
                    frozen_gate = archived_audit[system_name]["gates"][slot]
                    official_center = float(frozen_gate.get("contribution_center", 0.0))
                    active_center = official_center if frozen_gate["label"] == "active-identified" else 0.0
                    values = {
                        "fine_response": fine_value,
                        "coarse_response": coarse_value,
                        "ad_response_whitebox": oracle_value,
                        "fine_G_whitebox_g": fine_g_whitebox,
                        "active_only_v2": active_center,
                        "all_gate_response_where_invertible": fine_value,
                        "zero_centered_v2": official_center,
                    }
                    for estimator, value in values.items():
                        if math.isfinite(value):
                            system_ladder[estimator].append(value)
                    ladder_rows.append({
                        "pair_digest": record.pair_digest, "cell_id": record.cell_id,
                        "distance_bin": record.distance_bin, "orientation": record.orientation,
                        "system": system_name, "gate_slot": slot, "gate_index": gate,
                        "official_label": frozen_gate["label"],
                        "fine_invertible": fine_invertible,
                        "coarse_invertible": coarse_invertible,
                        "exact_direct": exact_direct_value,
                        "exact_direct_bound": float(bounds[physical_position]),
                    } | values)
                joint = joint_target_ad_v300(ad_tail, anchor, suffix_ids, physical_v, contrast)
                operator = joint_operator_prediction_v300(gate_responses, gradients, physical_v, contrast)
                operator_bound = math.fsum(physical_bounds)
                joint_bound = math.nextafter(operator_bound + joint["radius"], math.inf)
                joint_residual = abs(operator - joint["midpoint"])
                joint_passed = bool(joint["passed"] and joint_residual <= joint_bound)
                with torch.inference_mode():
                    factorial = runner._factorial_system_v13(
                        finite_tail, anchor, physical_v, zeta, contrast_t
                    )
                system_results[system_name] = {
                    "joint_target": joint["midpoint"], "joint_target_norm": abs(joint["midpoint"]),
                    "joint_prediction": operator, "joint_bound": joint_bound,
                    "joint_residual": joint_residual,
                    "joint_ratio": joint_residual / max(joint_bound, np.finfo(float).tiny),
                    "joint_passed": joint_passed, "gate_rows": gate_rows,
                    "factorial_single": float(factorial["single"]),
                    "factorial_pie": float(factorial["pie"]),
                    "ladder_item_totals": {
                        name: float(math.fsum(values)) for name, values in system_ladder.items()
                    },
                }
                joint_rows.append({
                    "pair_digest": record.pair_digest, "cell_id": record.cell_id,
                    "distance_bin": record.distance_bin, "orientation": record.orientation,
                    "system": system_name, "operator": operator,
                    "independent_target": float(joint["midpoint"]),
                    "residual": joint_residual, "bound": joint_bound,
                    "residual_to_bound": joint_residual / max(joint_bound, np.finfo(float).tiny),
                    "route_passed": bool(joint["passed"] and all_routes_passed),
                    "composition_passed": joint_passed,
                })
            signed_behavioral = float(runner.margin(anchors["pat"].year_logits,
                                                    torch.as_tensor(contrast, device="cuda:0")).item()
                                      - runner.margin(anchors["tar"].year_logits,
                                                      torch.as_tensor(contrast, device="cuda:0")).item())
            contrast_operator = system_results["pat"]["joint_prediction"] - system_results["tar"]["joint_prediction"]
            contrast_target = system_results["pat"]["joint_target"] - system_results["tar"]["joint_target"]
            contrast_residual = abs(contrast_operator - contrast_target)
            contrast_bound = math.nextafter(
                system_results["pat"]["joint_bound"] + system_results["tar"]["joint_bound"],
                math.inf,
            )
            contrast_passed = bool(contrast_residual <= contrast_bound)
            joint_rows.append({
                "pair_digest": record.pair_digest, "cell_id": record.cell_id,
                "distance_bin": record.distance_bin, "orientation": record.orientation,
                "system": "pat_minus_tar", "operator": contrast_operator,
                "independent_target": contrast_target, "residual": contrast_residual,
                "bound": contrast_bound,
                "residual_to_bound": contrast_residual / max(contrast_bound, np.finfo(float).tiny),
                "route_passed": bool(all_routes_passed),
                "composition_passed": contrast_passed,
            })
            joint_pass = all(value["joint_passed"] for value in system_results.values()) and contrast_passed
            output_rows.append({
                "pair_digest": record.pair_digest, "cell_id": record.cell_id,
                "distance_bin": record.distance_bin, "orientation": record.orientation,
                "route_passed": all_routes_passed,
                "transport_theorem_passed": all_transport_passed,
                "transport_residual_to_bound": max_transport_ratio,
                "joint_composition_passed": joint_pass,
                "joint_residual_to_bound": max(value["joint_ratio"] for value in system_results.values()),
                "joint_target_norm": max(value["joint_target_norm"] for value in system_results.values()),
                "joint_bound": max(value["joint_bound"] for value in system_results.values()),
                "signed_behavioral": signed_behavioral,
                "signed_matched_bypass": float(archived["theta_pat_center"] - archived["theta_tar_center"]),
                "signed_independent_joint_target": float(system_results["pat"]["joint_target"]
                                                         - system_results["tar"]["joint_target"]),
                "signed_pie": float(system_results["pat"]["factorial_pie"]
                                    - system_results["tar"]["factorial_pie"]),
                "signed_single": float(system_results["pat"]["factorial_single"]
                                       - system_results["tar"]["factorial_single"]),
                "error_ad_response_whitebox": max_transport_ratio,
                "error_fine_response": np.nan, "error_coarse_response": np.nan,
                "active_model_unchanged": True,
                "system_audit": json.dumps(system_results, sort_keys=True, separators=(",", ":")),
            })
        active_before = ad_tail.integrity_before
    active_after = ad_tail.integrity_after
    unchanged = bool(ad_tail.active_model_unchanged and active_before == active_after)
    for row in output_rows:
        row["active_model_unchanged"] = unchanged
    for collection in (transport_rows, joint_rows, ladder_rows):
        for row in collection:
            row["active_model_unchanged"] = unchanged
    table = pd.DataFrame(output_rows)
    table.to_parquet(args.output / "postmortem_rows.parquet", index=False)
    transport_table = pd.DataFrame(transport_rows)
    joint_table = pd.DataFrame(joint_rows)
    ladder_table = pd.DataFrame(ladder_rows)
    transport_table.to_parquet(args.output / "transport_rows.parquet", index=False)
    joint_table.to_parquet(args.output / "joint_rows.parquet", index=False)
    ladder_table.to_parquet(args.output / "ladder_gate_rows.parquet", index=False)
    payload = {"schema_version": "green-v21-postmortem-worker-v1",
               "worker_index": args.worker_index, "worker_count": args.worker_count,
               "physical_gpu": args.physical_gpu, "records": len(table),
               "elapsed_seconds": time.perf_counter() - started,
               "active_model_unchanged": unchanged,
               "rows_sha256": sha256_file(args.output / "postmortem_rows.parquet"),
               "transport_rows": len(transport_table),
               "transport_rows_sha256": sha256_file(args.output / "transport_rows.parquet"),
               "joint_rows": len(joint_table),
               "joint_rows_sha256": sha256_file(args.output / "joint_rows.parquet"),
               "ladder_gate_rows": len(ladder_table),
               "ladder_gate_rows_sha256": sha256_file(args.output / "ladder_gate_rows.parquet")}
    (args.output / "worker_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
