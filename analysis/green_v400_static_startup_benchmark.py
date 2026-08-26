"""Outcome-blind startup audit for program/store closure and exact contrast fusion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_final_contrast_fusion import fuse_final_contrast_exact
from green_bridge_v400_gpt2_program import GPT2TailDimensions, validate_gpt2_joint_witness_program
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timed_repetitions(function, repetitions: int):
    rows, last = [], None
    for repetition in range(repetitions):
        started = time.perf_counter()
        last = function()
        rows.append({"repetition": repetition, "elapsed_seconds": time.perf_counter() - started})
    maximum = max(row["elapsed_seconds"] for row in rows)
    return rows, last, {"observed_max_seconds": maximum,
                        "guardbanded_observed_max_1p25x_seconds": 1.25 * maximum}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise RuntimeError("at least three repetitions are required")
    program_path = Path(args.program).resolve()
    store_path = Path(args.tensor_store).resolve()
    output = Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("startup benchmark output must be a new file on /mnt/sdb")

    program_rows, program, program_summary = timed_repetitions(
        lambda: TensorProgram.from_dict(json.loads(program_path.read_text(encoding="utf-8"))),
        args.repetitions,
    )
    store_rows, reader, store_summary = timed_repetitions(
        lambda: TensorStoreReader(store_path), args.repetitions,
    )
    dims_payload = program.resource_formula["dimensions"]
    dims = GPT2TailDimensions(
        dims_payload["sequence_length"], dims_payload["d_model"], dims_payload["d_mlp"],
        dims_payload["n_heads"], dims_payload["d_head"], tuple(dims_payload["selected_gates"]),
        dims_payload["final_position"], dims_payload["contrast_width"],
    )
    validate_gpt2_joint_witness_program(program, reader, dims)

    unique_refs = {ref.tensor_sha256: ref for node in program.nodes for ref in node.tensor_inputs}
    decoded_hashes = []

    def decode_program_tensors():
        digest = hashlib.sha256()
        for semantic_hash in sorted(unique_refs):
            array = reader.read_semantic(semantic_hash)
            digest.update(semantic_hash.encode("ascii")); digest.update(array.tobytes())
        decoded_hashes.append(digest.hexdigest())
        return digest.hexdigest()

    decode_rows, _, decode_summary = timed_repetitions(decode_program_tensors, args.repetitions)
    if len(set(decoded_hashes)) != 1:
        raise RuntimeError("tensor decode hash is not deterministic")
    unembed = reader.read("unembed.W_U_full")
    bias = reader.read("unembed.b_U_full")
    suffix_ids = reader.read("unembed.suffix_ids")
    coefficients = reader.read("contrast.coefficients")
    fusion_hashes = []

    def fuse():
        result = fuse_final_contrast_exact(unembed, bias, suffix_ids, coefficients)
        fusion_hashes.append(result.semantic_hash())
        return result

    fusion_rows, fusion, fusion_summary = timed_repetitions(fuse, args.repetitions)
    if len(set(fusion_hashes)) != 1:
        raise RuntimeError("exact final-contrast fusion is not deterministic")
    report = {
        "schema_version": "green-v400-static-startup-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "program_semantic_hash": program.semantic_hash(),
        "program_file_sha256": sha256_file(program_path),
        "tensor_store_manifest_sha256": sha256_file(store_path),
        "tensor_store_blob_sha256": reader.manifest.blob_sha256,
        "tensor_store_blob_nbytes": reader.manifest.blob_nbytes,
        "unique_program_tensor_count": len(unique_refs),
        "unique_program_tensor_nbytes": sum(ref.nbytes for ref in unique_refs.values()),
        "exact_final_contrast_fusion_sha256": fusion.semantic_hash(),
        "exact_final_contrast_fused_weight_count": len(fusion.weights),
        "stages": {
            "program_parse_validate_schema": {"rows": program_rows, **program_summary},
            "tensor_store_manifest_and_full_blob_hash_validate": {"rows": store_rows, **store_summary},
            "decode_and_hash_all_unique_program_tensors": {"rows": decode_rows, **decode_summary},
            "exact_rational_final_contrast_fusion": {"rows": fusion_rows, **fusion_summary},
        },
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        "status": "PASS_STATIC_STARTUP_OBSERVATION_ONLY",
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "fusion_sha256": fusion.semantic_hash(), "stages": {
                          name: value["guardbanded_observed_max_1p25x_seconds"]
                          for name, value in report["stages"].items()},
                      "peak_rss_kib": report["peak_rss_kib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
