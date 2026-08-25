"""Deterministic single-GPU shard worker for exact batch-one GREEN endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import exp_green_bridge_gpt2 as runner
from green_bridge_spec import canonical_json, sha256_file, sha256_text, write_json_atomic


def load_design(output_root: Path, split: str, records) -> dict:
    frames = np.load(output_root / f"{split}_frames.npz")
    targets = np.load(output_root / f"{split}_target_vectors.npz")
    radii = json.loads(
        (output_root / f"{split}_radii.json").read_text(encoding="utf-8")
    )
    design = {}
    for record in records:
        digest = record.pair_digest
        design[digest] = {
            "common": frames[f"{digest}__common"],
            "gate_frames": [frames[f"{digest}__gate_{slot}"] for slot in range(10)],
            "all_gate": frames[f"{digest}__all_gate"],
            "target": targets[digest],
            "radius": radii[digest],
        }
    return design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.worker_root.exists() and (args.worker_root / "worker_result.json").exists():
        raise RuntimeError("worker result already exists")
    contract = json.loads(
        (args.worker_root / "worker_contract.json").read_text(encoding="utf-8")
    )
    if contract["worker_index"] != args.worker_index:
        raise RuntimeError("worker index mismatch")
    if contract["physical_gpu"] != args.physical_gpu:
        raise RuntimeError("physical GPU mismatch")
    if contract["source_sha256"] != runner.source_hashes():
        raise RuntimeError("source hash mismatch")
    for name, expected in contract["input_sha256"].items():
        actual = sha256_file(args.output_root / name)
        if actual != expected:
            raise RuntimeError(f"input hash mismatch: {name}")
    runner.configure_runtime("cuda:0", physical_gpu=args.physical_gpu)
    runner.activate_hardware_batch_plan(args.output_root)
    records = runner.load_split_file(args.worker_root, "assigned_records.json")
    tokenizer, model, suffix_ids = runner._load_active_models_and_suffixes(
        args.output_root, "cuda:0", records
    )
    del tokenizer
    torch = runner.torch_module()
    plain = torch.load(
        args.output_root / f"{args.split}_anchor_cache.pt",
        map_location="cpu",
        weights_only=True,
    )
    design = load_design(args.output_root, args.split, records)
    epsilon_y = float(contract["epsilon_y"])
    coefficients = runner.first_order_directions()
    started = time.perf_counter()
    completed = []
    for record in records:
        role = record.role
        batch_id = f"{args.split}-{role}-{record.pair_digest}"
        declaration = {
            "phase": args.split,
            "items": [record.pair_digest],
            "endpoint_type": role,
            "radii_sha256": sha256_file(
                args.output_root / f"{args.split}_radii.json"
            ),
            "systems": ["tar", "pat"] if role == "tensor" else ["tar", "pat", "cor"],
            "worker_index": args.worker_index,
            "physical_gpu": args.physical_gpu,
        }
        if role == "tensor":
            result = runner._run_endpoint_batch(
                args.worker_root,
                batch_id,
                declaration,
                lambda record=record: runner._tensor_item_v13(
                    model,
                    suffix_ids,
                    record,
                    "cuda:0",
                    epsilon_y,
                    coefficients,
                    plain,
                    design,
                ),
            )
        else:
            result = runner._run_endpoint_batch(
                args.worker_root,
                batch_id,
                declaration,
                lambda record=record: runner._energy_item_v13(
                    model, suffix_ids, record, "cuda:0", plain, design
                ),
            )
        completed.append({
            "pair_digest": record.pair_digest,
            "role": role,
            "artifact_sha256": sha256_file(
                args.worker_root / "endpoint_batches" / f"{batch_id}.json"
            ),
            "result_sha256": sha256_text(canonical_json(result)),
        })
    torch.cuda.synchronize("cuda:0")
    payload = {
        "schema_version": "green-bridge-v1.3.5-worker-v1",
        "worker_index": args.worker_index,
        "physical_gpu": args.physical_gpu,
        "split": args.split,
        "record_count": len(records),
        "completed": completed,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
        "contract_sha256": sha256_file(args.worker_root / "worker_contract.json"),
    }
    write_json_atomic(args.worker_root / "worker_result.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
