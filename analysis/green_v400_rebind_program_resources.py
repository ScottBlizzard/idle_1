"""Rebuild an outcome-blind TensorProgram against the latest resource schema."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_gpt2_program import GPT2TailDimensions, build_gpt2_joint_witness_program
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-program", required=True)
    parser.add_argument("--tensor-store", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    source_path = Path(args.source_program).resolve()
    store_path = Path(args.tensor_store).resolve()
    output_root = Path(args.output_root).resolve()
    if "/mnt/sdb/" not in output_root.as_posix() or output_root.exists():
        raise RuntimeError("rebind output must be a new directory on /mnt/sdb")
    source = TensorProgram.from_dict(json.loads(source_path.read_text(encoding="utf-8")))
    reader = TensorStoreReader(store_path)
    payload = source.resource_formula["dimensions"]
    dims = GPT2TailDimensions(
        payload["sequence_length"], payload["d_model"], payload["d_mlp"],
        payload["n_heads"], payload["d_head"], tuple(payload["selected_gates"]),
        payload["final_position"], payload["contrast_width"],
    )
    rebound = build_gpt2_joint_witness_program(reader, source.model_manifest_hash, dims)
    source_nodes = [
        (node.semantic_id, node.kernel_id, node.parent_semantic_ids, node.tensor_inputs,
         node.exact_attrs, node.output_spec, node.dependency_mask_hash)
        for node in source.nodes
    ]
    rebound_nodes = [
        (node.semantic_id, node.kernel_id, node.parent_semantic_ids, node.tensor_inputs,
         node.exact_attrs, node.output_spec, node.dependency_mask_hash)
        for node in rebound.nodes
    ]
    if source_nodes != rebound_nodes or source.branch_roots != rebound.branch_roots:
        raise RuntimeError("resource rebind changed executable TensorProgram nodes")
    output_root.mkdir(parents=True)
    program_path = output_root / "tensor_program.json"
    program_path.write_text(json.dumps(rebound.to_dict(), sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    report = {
        "schema_version": "green-v400-program-resource-rebind-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "source_program_file_sha256": sha256_file(source_path),
        "source_program_semantic_hash": source.semantic_hash(),
        "rebound_program_file_sha256": sha256_file(program_path),
        "rebound_program_semantic_hash": rebound.semantic_hash(),
        "executable_nodes_bit_identical": True,
        "node_count": len(rebound.nodes),
        "tensor_store_manifest_sha256": sha256_file(store_path),
        "tensor_store_record_closure_sha256": reader.manifest.record_closure_sha256,
        "resource_formula": rebound.resource_formula,
        "status": "PASS_RESOURCE_ONLY_REBIND",
    }
    (output_root / "rebind_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output_root": str(output_root),
                      "program_semantic_hash": rebound.semantic_hash(),
                      "node_count": len(rebound.nodes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
