"""One deterministic GPU shard for GREEN v3.0.0 development.

The executable accepts development only. Confirmation remains sealed even if
the caller supplies a confirmation record plan.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

import exp_green_bridge_gpt2 as legacy
from green_bridge_dataset import PairRecord
from green_bridge_spec import sha256_file
from green_bridge_v300_development import (
    DEVELOPMENT_AUTHORIZATION_ID,
    _jsonable,
    _margin_vector,
    evaluate_gate_v300,
    heldout_directions_v300,
    joint_scalar_v300,
    write_json,
)
from green_bridge_response_ad import isolated_ad_tail_v200


def _load_prepared_records(prepare_root: Path, role: str) -> list[PairRecord]:
    payload = json.loads((prepare_root / "v300_record_plan.json").read_text(encoding="utf-8"))
    names = {field.name for field in fields(PairRecord)}
    rows = []
    for value in payload["records"]:
        if value.get("split") != "development" or value.get("role") != role:
            continue
        rows.append(PairRecord(**{name: value[name] for name in names}))
    if len(rows) != 80:
        raise RuntimeError(f"DEVELOPMENT_RECORD_PLAN_MISMATCH:{role}:{len(rows)}")
    return rows


def deterministic_worker_assignment_v300(phase: str, role: str, worker_index: int,
                                         worker_count: int = 8,
                                         prepare_root: Path | None = None):
    if phase != "development":
        raise RuntimeError("CONFIRMATION_REMAINS_SEALED")
    if role not in ("transport", "joint"):
        raise ValueError("invalid v3 worker role")
    if not 0 <= worker_index < worker_count:
        raise ValueError("invalid worker index")
    if prepare_root is None:
        from green_bridge_v300_dataset import build_green_bridge_v300_records
        rows = [row for row in build_green_bridge_v300_records()
                if row.split == phase and row.role == role]
    else:
        rows = _load_prepared_records(Path(prepare_root), role)
    rows.sort(key=lambda row: hashlib.sha256(
        f"green-v300-worker-assignment|{phase}|{role}|{row.pair_digest}".encode("utf-8")
    ).hexdigest())
    return [row for index, row in enumerate(rows) if index % worker_count == worker_index]


def _clean_gate_for_json(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in {
        "gradient_hat", "coarse_gradient_hat", "gate_response",
        "coarse_gate_response", "whitebox_gradient",
    }}


def _record_metadata(record: PairRecord) -> dict:
    return {
        "pair_digest": record.pair_digest,
        "cell_id": record.cell_id,
        "noun": record.noun,
        "century": record.century,
        "noun_century_group": f"{record.noun}-{record.century}",
        "distance_bin": record.distance_bin,
        "orientation": record.orientation,
        "item_index": record.item_index,
        "y": record.y,
        "y_prime": record.y_prime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("development", "confirmation"))
    parser.add_argument("--role", required=True, choices=("transport", "joint"))
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase != "development":
        raise RuntimeError("CONFIRMATION_REMAINS_SEALED")
    if args.worker_count != 8 or args.worker_index != args.physical_gpu:
        raise RuntimeError("DEVELOPMENT_REQUIRES_ONE_DETERMINISTIC_WORKER_PER_GPU")
    if args.output.exists():
        raise RuntimeError("DEVELOPMENT_WORKER_OUTPUT_EXISTS")
    args.output.mkdir(parents=True, exist_ok=False)

    prepare_result = json.loads(
        (args.prepare_root / "prepare_result.json").read_text(encoding="utf-8")
    )
    if prepare_result.get("verdict") != "PREPARE_PASS":
        raise RuntimeError("FORMAL_PREPARE_DID_NOT_PASS")
    selected_radius = float(prepare_result["selected_global_radius"])
    radius_calibration = json.loads(
        (args.prepare_root / "v300_radius_calibration.json").read_text(encoding="utf-8")
    )
    epsilon_y = float(radius_calibration["epsilon_y"])
    if radius_calibration.get("finite_response_mode") != "float64_response_only":
        raise RuntimeError("DEVELOPMENT_FINITE_MODE_MISMATCH")

    records = deterministic_worker_assignment_v300(
        args.phase, args.role, args.worker_index, args.worker_count,
        prepare_root=args.prepare_root,
    )
    if len(records) != 10:
        raise RuntimeError(f"DEVELOPMENT_WORKER_ASSIGNMENT_COUNT:{len(records)}")

    device = "cuda:0"
    environment = legacy.configure_runtime(device, physical_gpu=args.physical_gpu)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(legacy.MODEL_ID, revision=legacy.MODEL_REVISION)
    tokenizer, hf_model, model, cfg = legacy.load_models(device, tokenizer=tokenizer)
    del hf_model, cfg
    suffix_ids, tokenizer_meta = legacy.validate_tokenizer(tokenizer, records)

    scratch = args.output / "scratch"
    scratch.mkdir(parents=True, exist_ok=False)
    plain = legacy._capture_structural_inputs(
        model, tokenizer, suffix_ids, records, device, scratch, "development"
    )
    design = legacy._construct_structural_design(
        model, records, plain, scratch, "development"
    )

    started = time.perf_counter()
    transport_rows: list[dict] = []
    joint_rows: list[dict] = []
    record_rows: list[dict] = []
    torch = legacy.torch_module()
    integrity_before = legacy.active_model_integrity_hash_v200(model)
    torch.cuda.reset_peak_memory_stats(device)
    with isolated_ad_tail_v200(model) as finite_tail, isolated_ad_tail_v200(model) as ad_tail:
        if finite_tail is ad_tail:
            raise RuntimeError("FINITE_AND_AD_TAILS_MUST_BE_DISTINCT")
        for record in records:
            metadata = _record_metadata(record)
            item = design[record.pair_digest]
            physical_v = np.asarray(item["target"], dtype=np.float64)
            physical_v /= max(float(np.linalg.norm(physical_v)), np.finfo(float).tiny)
            contrast = _margin_vector(record.y)
            systems: dict[str, list[dict]] = {}
            joint_systems = {}
            for system in ("tar", "pat"):
                anchor = legacy._anchor_plain_to_device(
                    plain[record.pair_digest][system], device
                )
                gates = []
                for gate_slot, frame in enumerate(item["gate_frames"]):
                    directions, direction_classes = heldout_directions_v300(frame)
                    gate = evaluate_gate_v300(
                        model=model, finite_tail=finite_tail, ad_tail=ad_tail,
                        anchor=anchor, frame=np.asarray(frame, dtype=np.float64),
                        suffix_ids=suffix_ids, gate_slot=gate_slot,
                        directions=directions, epsilon_y=epsilon_y,
                        base_h_x=float(item["radius"]["h_x"]),
                        selected_radius=selected_radius,
                    )
                    gate["system"] = system
                    gate["direction_classes"] = direction_classes
                    gates.append(gate)
                systems[system] = gates
                if args.role == "joint":
                    joint_systems[system] = joint_scalar_v300(
                        gates, direction=physical_v, contrast=contrast,
                        ad_tail=ad_tail, anchor=anchor, suffix_ids=suffix_ids,
                    )

            all_gates = systems["tar"] + systems["pat"]
            invalid = sum(row["gate_class"] == "numerical-invalid" for row in all_gates)
            contradictions = sum(
                row["gate_class"] == "structural-contradiction" for row in all_gates
            )
            resolved = sum(
                row["gate_class"] in ("recoverable", "certified-numerical-null")
                for row in all_gates
            )
            technical = bool(
                len(all_gates) == 20 and invalid == 0 and contradictions == 0
                and resolved / 20.0 >= 0.80
            )
            record_rows.append(metadata | {
                "role": args.role, "technically_admissible": technical,
                "resolved_gate_systems": resolved,
                "numerical_invalid_units": invalid,
                "structural_contradiction_units": contradictions,
            })
            if args.role == "transport":
                for system, gates in systems.items():
                    for gate in gates:
                        baseline = gate["baseline_errors"]
                        transport_rows.append(metadata | {
                            "system": system, "gate_slot": gate["gate_slot"],
                            "gate_index": gate["gate_index"], "gate_class": gate["gate_class"],
                            "technically_admissible": technical, "nonnull": gate["nonnull"],
                            "direct_error": gate["direct_error"],
                            "coarse_direct_error": gate["coarse_direct_error"],
                            "null_leakage": gate["null_leakage"],
                            "curvature_identifiability": (
                                gate["curvature_norm"] / gate["epsilon_C"]
                                if gate["epsilon_C"] > 0 else math.inf
                            ),
                            "error_matched": baseline["matched"],
                            "error_zero": baseline["zero"],
                            "error_gate_atom_only": baseline["gate_atom_only"],
                            "error_unmatched_path_mixed": baseline["unmatched_path_mixed"],
                            "error_raw_path_jacobian": baseline["raw_path_jacobian"],
                            "bound_valid": gate["direct_theorem_passed"],
                            "gate_audit": json.dumps(
                                _jsonable(_clean_gate_for_json(gate)), sort_keys=True,
                                separators=(",", ":"),
                            ),
                        })
            else:
                pat, tar = joint_systems["pat"], joint_systems["tar"]
                center = float(pat["center"] - tar["center"])
                coarse_center = float(pat["coarse_center"] - tar["coarse_center"])
                target = float(pat["target"] - tar["target"])
                bound = math.nextafter(float(pat["bound"] + tar["bound"]), math.inf)
                target_bound = math.nextafter(
                    float(pat["target_bound"] + tar["target_bound"]), math.inf
                )
                denominator = max(abs(target), bound)
                numerator = abs(center - target)
                error = 0.0 if denominator == 0.0 and numerator == 0.0 else (
                    math.inf if denominator == 0.0 else numerator / denominator
                )
                joint_rows.append(metadata | {
                    "technically_admissible": bool(
                        technical and pat["target_certified"] and tar["target_certified"]
                        and math.isfinite(bound)
                    ),
                    "joint_center": center, "joint_coarse_center": coarse_center,
                    "joint_bound": bound,
                    "unresolved_bound": float(
                        pat["unresolved_bound"] + tar["unresolved_bound"]
                    ),
                    "joint_target": target, "target_bound": target_bound,
                    "joint_error": error,
                    "nonnull": bool(
                        target_bound == 0.0 and target != 0.0
                        or target_bound > 0.0 and abs(target) / target_bound >= 4.0
                    ),
                    "gate_classes": json.dumps(
                        {system: [row["gate_class"] for row in gates]
                         for system, gates in systems.items()},
                        sort_keys=True, separators=(",", ":"),
                    ),
                    "system_audit": json.dumps(
                        joint_systems, sort_keys=True, separators=(",", ":"),
                    ),
                })
            torch.cuda.empty_cache()
        # The context managers populate their integrity verdicts in __exit__.
        # Reading these fields inside the with-block yields None even when the
        # active model is bitwise unchanged.
        pass
    finite_integrity = finite_tail.active_model_unchanged
    ad_integrity = ad_tail.active_model_unchanged
    integrity_after = legacy.active_model_integrity_hash_v200(model)
    active_model_unchanged = bool(
        integrity_before == integrity_after and finite_integrity and ad_integrity
    )
    if not active_model_unchanged:
        raise RuntimeError("DEVELOPMENT_ACTIVE_MODEL_INTEGRITY_FAILURE")

    transport_table = pd.DataFrame(transport_rows)
    joint_table = pd.DataFrame(joint_rows)
    record_table = pd.DataFrame(record_rows)
    transport_path = args.output / "transport_rows.parquet"
    joint_path = args.output / "joint_rows.parquet"
    records_path = args.output / "record_rows.parquet"
    transport_table.to_parquet(transport_path, index=False)
    joint_table.to_parquet(joint_path, index=False)
    record_table.to_parquet(records_path, index=False)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "green-bridge-v3.0.0-development-worker-v1",
        "authorization_id": DEVELOPMENT_AUTHORIZATION_ID,
        "phase": "development", "role": args.role,
        "worker_index": args.worker_index, "worker_count": args.worker_count,
        "physical_gpu": args.physical_gpu, "records": len(records),
        "transport_rows": len(transport_table), "joint_rows": len(joint_table),
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "active_model_unchanged": active_model_unchanged,
        "finite_and_ad_model_copies_are_distinct": True,
        "point_estimator_uses_automatic_derivatives": False,
        "selected_global_radius": selected_radius, "epsilon_y": epsilon_y,
        "environment": environment, "tokenizer": tokenizer_meta,
        "transport_rows_sha256": sha256_file(transport_path),
        "joint_rows_sha256": sha256_file(joint_path),
        "record_rows_sha256": sha256_file(records_path),
    }
    write_json(args.output / "worker_result.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
