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
    tensor_table = pd.read_parquet(args.formal_v200_root / "dev_tensor_scores.parquet").set_index("pair_digest")
    gamma = model.blocks[10].ln2.w.detach().double().cpu().numpy()
    W_in = model.blocks[10].mlp.W_in.detach().double().cpu().numpy()
    started = time.perf_counter(); output_rows = []
    with runner.isolated_ad_tail_v200(model) as ad_tail:
        for record in assigned:
            item = design[record.pair_digest]
            anchors = {system: runner._anchor_plain_to_device(values, "cuda:0")
                       for system, values in plain[record.pair_digest].items()}
            contrast = _margin_vector(record.y)
            physical_v = np.asarray(item["target"], dtype=np.float64)
            system_results = {}
            all_transport_passed = True; all_routes_passed = True
            max_transport_ratio = 0.0
            for system_name in ("tar", "pat"):
                anchor = anchors[system_name]
                residual = runner._selected_numpy(anchor, "resid_mid")
                gate_responses, gradients = [], []
                gate_rows = []
                for slot, (gate, frame) in enumerate(zip(SELECTED_GATES, item["gate_frames"])):
                    panel = heldout_direction_panel_v300(frame)
                    directions = np.concatenate((panel["in_frame"], panel["mixed"], panel["null"]), axis=1)
                    exact = direct_path_control_ad_v300(ad_tail, anchor, directions, suffix_ids, gate)
                    gradient = layernorm_gate_gradient_formula(
                        residual, gamma, W_in[:, gate], eps=float(model.cfg.eps)
                    )
                    predicted = heldout_transport_prediction_v300(exact["G"]["midpoint"], gradient, directions)
                    residuals = np.linalg.norm(predicted - exact["direct"], axis=1)
                    gradient_error = max(1e-10, gradient_envelope_residual(frame, gradient)["absolute"])
                    projections = directions.T @ gradient
                    bounds = np.asarray([_route_bound(exact, value, gradient_error) for value in projections])
                    passed = residuals <= bounds
                    ratios = residuals / np.maximum(bounds, np.finfo(float).tiny)
                    all_routes_passed &= exact["route_passed"]
                    all_transport_passed &= bool(np.all(passed))
                    max_transport_ratio = max(max_transport_ratio, float(np.max(ratios)))
                    gate_responses.append(np.asarray(exact["G"]["midpoint"], dtype=np.float64))
                    gradients.append(gradient)
                    gate_rows.append({"slot": slot, "gate": gate, "route_passed": exact["route_passed"],
                                      "max_residual": float(np.max(residuals)),
                                      "max_bound": float(np.max(bounds)),
                                      "max_ratio": float(np.max(ratios)),
                                      "passed": bool(np.all(passed))})
                joint = joint_target_ad_v300(ad_tail, anchor, suffix_ids, physical_v, contrast)
                operator = joint_operator_prediction_v300(gate_responses, gradients, physical_v, contrast)
                operator_bound = math.fsum(
                    (float(np.linalg.norm(G)) + 1) * 1e-10 for G in gate_responses
                ) + math.fsum(row["max_bound"] for row in gate_rows)
                joint_bound = math.nextafter(operator_bound + joint["radius"], math.inf)
                joint_residual = abs(operator - joint["midpoint"])
                joint_passed = bool(joint["passed"] and joint_residual <= joint_bound)
                system_results[system_name] = {
                    "joint_target": joint["midpoint"], "joint_target_norm": abs(joint["midpoint"]),
                    "joint_prediction": operator, "joint_bound": joint_bound,
                    "joint_residual": joint_residual,
                    "joint_ratio": joint_residual / max(joint_bound, np.finfo(float).tiny),
                    "joint_passed": joint_passed, "gate_rows": gate_rows,
                }
            archived = tensor_table.loc[record.pair_digest]
            signed_behavioral = float(runner.margin(anchors["pat"].year_logits,
                                                    torch.as_tensor(contrast, device="cuda:0")).item()
                                      - runner.margin(anchors["tar"].year_logits,
                                                      torch.as_tensor(contrast, device="cuda:0")).item())
            joint_pass = all(value["joint_passed"] for value in system_results.values())
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
                "signed_pie": np.nan, "signed_single": np.nan,
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
    table = pd.DataFrame(output_rows)
    table.to_parquet(args.output / "postmortem_rows.parquet", index=False)
    payload = {"schema_version": "green-v21-postmortem-worker-v1",
               "worker_index": args.worker_index, "worker_count": args.worker_count,
               "physical_gpu": args.physical_gpu, "records": len(table),
               "elapsed_seconds": time.perf_counter() - started,
               "active_model_unchanged": unchanged,
               "rows_sha256": sha256_file(args.output / "postmortem_rows.parquet")}
    (args.output / "worker_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
