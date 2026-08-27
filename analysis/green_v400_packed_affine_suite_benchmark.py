"""Outcome-blind actual-weight packed-affine suite observation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import Jet2
from green_bridge_v400_resident_plan import load_resident_plan_arrays
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


AFFINE_PAIRS = (
    ("block10.mlp.W_in_selected", "block10.mlp.b_in_selected"),
    ("block10.mlp.W_out_selected", "zero.d_model"),
    ("block11.attn.W_Q", "block11.attn.b_Q"),
    ("block11.attn.W_K", "block11.attn.b_K"),
    ("block11.attn.W_V", "block11.attn.b_V"),
    ("block11.attn.W_O", "block11.attn.b_O"),
    ("block11.mlp.W_in", "block11.mlp.b_in"),
    ("block11.mlp.W_out", "block11.mlp.b_out"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_inputs(width: int, precision: int) -> list[Jet2]:
    result = []
    for index in range(width):
        center = Fraction((index * 37) % 2049 - 1024, 1024)
        radius = Fraction(1, 1 << (18 + index % 7))
        result.append(Jet2(
            Interval.from_bounds(center - radius, center + radius, precision),
            Interval.point(Fraction((index * 13) % 257 - 128, 512), precision),
            Interval.point(Fraction((index * 17) % 129 - 64, 1024), precision),
        ))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise RuntimeError("at least three repetitions are required")
    library, output = Path(args.library).resolve(), Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("affine-suite output must be a new file on /mnt/sdb")
    program = TensorProgram.from_dict(json.loads(Path(args.program).read_text(encoding="utf-8")))
    reader = TensorStoreReader(Path(args.tensor_store))
    plan, arrays = load_resident_plan_arrays(Path(args.plan), program, reader)
    records = {record["name"]: record for record in plan["records"]}
    backend = CompiledMPFRBackend(library)
    rows = []
    for precision in (384, 512):
        inputs_by_width = {}
        for weight_name, bias_name in AFFINE_PAIRS:
            weight, bias = arrays[weight_name], arrays[bias_name]
            if weight.shape[0] not in inputs_by_width:
                inputs_by_width[weight.shape[0]] = synthetic_inputs(
                    weight.shape[0], precision
                )
            inputs = inputs_by_width[weight.shape[0]]
            timings, hashes = [], []
            for _ in range(args.repetitions):
                started = time.perf_counter()
                payload = backend.packed_affine_layer_jet2(weight, bias, inputs)
                timings.append(time.perf_counter() - started)
                hashes.append(sha256_canonical(payload))
            if len(set(hashes)) != 1 or len(payload["outputs"]) != weight.shape[1]:
                raise RuntimeError(f"packed affine failed closure for {weight_name}")
            rows.append({
                "precision_bits": precision,
                "weight_name": weight_name,
                "bias_name": bias_name,
                "weight_shape": list(weight.shape),
                "elapsed_seconds": timings,
                "median_seconds": statistics.median(timings),
                "observed_max_seconds": max(timings),
                "guardbanded_observed_max_1p25x_seconds": 1.25 * max(timings),
                "output_exact_payload_sha256": hashes[0],
                "weight_tensor_semantic_sha256": records[weight_name][
                    "tensor_semantic_sha256"
                ],
                "bias_tensor_semantic_sha256": records[bias_name][
                    "tensor_semantic_sha256"
                ],
            })
    report = {
        "schema_version": "green-v400-packed-actual-weight-affine-suite-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "deterministic synthetic Jet2 row with actual closed GPT-2 weights",
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "program_semantic_hash": program.semantic_hash(),
        "backend_sha256": sha256_file(library),
        "repetitions": args.repetitions,
        "rows": rows,
        "status": "PASS_ACTUAL_WEIGHT_PACKED_AFFINE_SUITE_OBSERVATION_ONLY",
        "resident_dispatcher_complete": False,
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
