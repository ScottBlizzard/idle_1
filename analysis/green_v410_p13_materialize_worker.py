"""Materialize immutable endpoint-free P13 TensorPrograms from GPU captures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.green_v400_formal_worker import load_frozen_model
from green_bridge_v400_supervisor import atomic_publish
from green_v400_response_precision import prepare_float64_response_evaluation
from green_v410_artifacts import atomic_no_clobber_json, file_sha256
from green_v410_directions import build_row_binding, direction_vector_sha256
from green_v410_gpt2_program import (
    GRAPH_SEMANTICS_ID,
    build_green_v410_full_cone_program,
    materialize_green_v410_full_cone_store,
)
from green_v410_p13_queue import validate_p13_queue
from green_v410_protocol import BRANCH_ORDER, BRANCH_WEIGHTS, PROTOCOL_ID, load_protocol_config, sha256_canonical
from green_v410_schemas import with_artifact_self_hash, validate_artifact


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_npy(path: Path, expected_sha256: str) -> np.ndarray:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"P13 capture payload hash mismatch: {path.name}")
    return np.load(path, allow_pickle=False)


def _site_identity(
    *, job: dict[str, Any], capture: dict[str, Any], payload_hashes: dict[str, str]
) -> dict[str, Any]:
    parent_protocol = {
        "ioi": "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1",
        "greater_than": "GREEN_V400_SILENT_FAILURE_GT_REPLICATION_PREPARE_V1",
    }[job["task"]]
    return with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-site-identity-v1",
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": parent_protocol,
        "phase": "qualification",
        "task": job["task"],
        "parent_prompt_row_id": job["prompt_row_id"],
        "parent_site_row_id": job["site_row_id"],
        "layer": job["layer"],
        "hook_family": "resid_post",
        "hook_name": f"blocks.{job['layer']}.hook_resid_post",
        "token_position": job["site_position"],
        "clean_tokens_sha256": capture["clean_tokens_sha256"],
        "corrupt_tokens_sha256": capture["corrupt_tokens_sha256"],
        "clean_center_tensor_sha256": payload_hashes["clean_center.npy"],
        "pat_controlled_hook_tensor_sha256": payload_hashes["pat_controlled_hook_t0.npy"],
        "tar_controlled_hook_tensor_sha256": payload_hashes["tar_controlled_hook_t0.npy"],
        "green_direction_binding_sha256": capture["direction_binding_sha256"],
        "selected_gate_spec_sha256": job["selected_gate_spec_sha256"],
        "task_metric_spec_sha256": job["task_metric_spec_sha256"],
        "model_manifest_sha256": job["model_manifest_sha256"],
        "tokenizer_manifest_sha256": job["tokenizer_manifest_sha256"],
        "graph_semantics_id": GRAPH_SEMANTICS_ID,
    })


def _graph_manifest(
    *, job: dict[str, Any], direction: dict[str, Any], site_identity: dict[str, Any],
    program: Any, reader: Any,
) -> dict[str, Any]:
    provenance = {
        node.semantic_id: node.provenance_identity for node in program.nodes
    }
    output_ids = [program.branch_roots[name] for name in BRANCH_ORDER] + [program.output_root]
    return with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-graph-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "site_identity_sha256": site_identity["site_identity_sha256"],
        "direction_ordinal": direction["direction_ordinal"],
        "direction_payload_sha256": direction["direction_payload_sha256"],
        "graph_semantics_id": GRAPH_SEMANTICS_ID,
        "branch_order": list(BRANCH_ORDER),
        "branch_weights": list(BRANCH_WEIGHTS),
        "output_names": [*BRANCH_ORDER, "PSI"],
        "output_node_ids": output_ids,
        "node_count": len(program.nodes),
        "dependency_closure_sha256": program.resource_formula["dependency_mask_closure_sha256"],
        "provenance_map_sha256": sha256_canonical(provenance),
        "model_manifest_sha256": job["model_manifest_sha256"],
        "task_metric_spec_sha256": job["task_metric_spec_sha256"],
        "selected_gate_spec_sha256": job["selected_gate_spec_sha256"],
        "contains_endpoint_material": False,
    })


def _materialize_job(
    *, job: dict[str, Any], queue: dict[str, Any], capture_root: Path,
    graph_root: Path, model: Any, directions: np.ndarray,
) -> None:
    capture_dir = capture_root / job["job_id"]
    capture = _load(capture_dir / "capture_manifest.json")
    unhashed_capture = {k: v for k, v in capture.items() if k != "manifest_sha256"}
    if (
        capture.get("manifest_sha256") != sha256_canonical(unhashed_capture)
        or capture.get("job_id") != job["job_id"]
        or capture.get("queue_manifest_sha256") != queue["queue_manifest_sha256"]
        or capture.get("contains_endpoint_material") is not False
    ):
        raise ValueError("P13 graph materializer capture identity mismatch")
    payload_hashes = capture["payload_file_sha256s"]
    center = _load_npy(capture_dir / "clean_center.npy", payload_hashes["clean_center.npy"])
    pat = _load_npy(
        capture_dir / "pat_controlled_hook_t0.npy",
        payload_hashes["pat_controlled_hook_t0.npy"],
    )
    tar = _load_npy(
        capture_dir / "tar_controlled_hook_t0.npy",
        payload_hashes["tar_controlled_hook_t0.npy"],
    )
    if center.dtype.str != "<f8" or pat.dtype.str != "<f8" or tar.dtype.str != "<f8":
        raise ValueError("P13 capture tensors must be little-endian float64")
    site_identity = _site_identity(
        job=job, capture=capture, payload_hashes=payload_hashes
    )
    validate_artifact(site_identity)
    site_dir = graph_root / job["job_id"]
    atomic_no_clobber_json(
        site_dir / "site_identity.json", site_identity,
        job_id=f"{job['job_id']}-site-identity",
    )
    metric = capture["task_metric"]
    protocol = load_protocol_config()
    for ordinal, direction_identity in enumerate(job["direction_panel"]):
        if direction_identity["direction_ordinal"] != ordinal:
            raise ValueError("P13 direction order changed")
        vector = np.asarray(directions[ordinal], dtype="<f4")
        if direction_vector_sha256(vector) != direction_identity["direction_payload_sha256"]:
            raise ValueError("P13 direction vector differs from queue")
        final_dir = site_dir / f"direction_{ordinal:02d}"
        if final_dir.exists():
            existing = _load(final_dir / "graph_manifest.json")
            validate_artifact(existing)
            if (
                existing["site_identity_sha256"] != site_identity["site_identity_sha256"]
                or existing["direction_payload_sha256"]
                    != direction_identity["direction_payload_sha256"]
            ):
                raise ValueError("conflicting existing P13 graph artifact")
            continue
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f".{final_dir.name}.staging.", dir=final_dir.parent
        ))
        try:
            reader, dims = materialize_green_v410_full_cone_store(
                staging,
                "tensor_store",
                model,
                site_layer=job["layer"],
                site_position=job["site_position"],
                pat_controlled_hook_t0=pat,
                tar_controlled_hook_t0=tar,
                physical_direction=vector,
                suffix_token_ids=np.asarray(metric["suffix_token_ids"], dtype=np.int64),
                contrast_coefficients=np.asarray(
                    metric["contrast_coefficients"], dtype=np.float64
                ),
                contrast_coefficient_rationals=tuple(
                    tuple(value) for value in metric["contrast_coefficient_rationals"]
                ),
                selected_gates=tuple(protocol["gate"]["selected_indices"]),
            )
            program = build_green_v410_full_cone_program(
                reader, job["model_manifest_sha256"], dims
            )
            if len(program.nodes) > 2_000_000:
                raise RuntimeError("P13 graph exceeds frozen node ceiling")
            atomic_no_clobber_json(
                staging / "tensor_program.json", program.to_dict(),
                job_id=f"{job['job_id']}-direction-{ordinal}-program",
            )
            graph = _graph_manifest(
                job=job,
                direction=direction_identity,
                site_identity=site_identity,
                program=program,
                reader=reader,
            )
            validate_artifact(graph)
            atomic_no_clobber_json(
                staging / "graph_manifest.json", graph,
                job_id=f"{job['job_id']}-direction-{ordinal}-manifest",
            )
            atomic_publish(staging, final_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("P13 materializer shard coordinates are invalid")
    queue = _load(args.queue)
    validate_p13_queue(queue)
    model_manifest = _load(args.model_manifest)
    if sha256_canonical(model_manifest) != queue["model_manifest_sha256"]:
        raise ValueError("P13 materializer model manifest differs from queue")
    registry = _load(args.prepare_root / f"{queue['task']}_p13_direction_registry.json")
    direction_payload = args.prepare_root / f"{queue['task']}_p13_directions.npy"
    if file_sha256(direction_payload) != queue["direction_payload_file_sha256"]:
        raise ValueError("P13 materializer direction payload differs from queue")
    directions = np.load(direction_payload, mmap_mode="r", allow_pickle=False)
    binding_by_site = {row["site_row_id"]: row for row in registry["bindings"]}

    model = load_frozen_model(model_manifest, "cpu")
    prepare_float64_response_evaluation(
        model=model,
        model_manifest=model_manifest,
        expected_model_manifest_sha256=queue["model_manifest_sha256"],
    )
    jobs = [
        job for ordinal, job in enumerate(queue["jobs"])
        if ordinal % args.shard_count == args.shard_index
    ]
    for job in jobs:
        binding = binding_by_site[job["site_row_id"]]
        _materialize_job(
            job=job,
            queue=queue,
            capture_root=args.capture_root,
            graph_root=args.graph_root,
            model=model,
            directions=np.asarray(directions[int(binding["row_index"])], dtype="<f4"),
        )
    completion = {
        "schema_version": "green-v410-p13-materialize-shard-completion-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": queue["task"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "completed_job_ids": [job["job_id"] for job in jobs],
        "completed_job_count": len(jobs),
        "contains_endpoint_material": False,
    }
    completion["receipt_sha256"] = sha256_canonical(completion)
    atomic_no_clobber_json(
        args.graph_root / f"materialize_shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json",
        completion,
        job_id=f"p13-materialize-{queue['task']}-{args.shard_index}",
    )


if __name__ == "__main__":
    main()
