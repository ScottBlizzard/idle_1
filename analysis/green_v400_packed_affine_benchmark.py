"""Outcome-blind full-size packed-weight affine correctness/performance observation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import Jet2
from green_bridge_v400_resident_plan import load_resident_plan_arrays
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


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
    library, plan_path = Path(args.library).resolve(), Path(args.plan).resolve()
    output = Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("packed affine output must be a new file on /mnt/sdb")
    program = TensorProgram.from_dict(json.loads(Path(args.program).read_text(encoding="utf-8")))
    reader = TensorStoreReader(Path(args.tensor_store))
    plan, arrays = load_resident_plan_arrays(plan_path, program, reader)
    records = {record["name"]: record for record in plan["records"]}

    weight, bias = arrays["block11.mlp.W_in"], arrays["block11.mlp.b_in"]
    backend = CompiledMPFRBackend(library)
    rows = []
    for precision in (384, 512):
        inputs = synthetic_inputs(weight.shape[0], precision)
        hashes, timings = [], []
        for repetition in range(args.repetitions):
            started = time.perf_counter()
            payload = backend.packed_affine_layer_jet2(weight, bias, inputs)
            timings.append(time.perf_counter() - started)
            hashes.append(sha256_canonical(payload))
        if len(set(hashes)) != 1:
            raise RuntimeError("packed affine outputs are not deterministic")
        rows.append({
            "precision_bits": precision,
            "elapsed_seconds": timings,
            "observed_max_seconds": max(timings),
            "guardbanded_observed_max_1p25x_seconds": 1.25 * max(timings),
            "output_exact_payload_sha256": hashes[0],
        })
    report = {
        "schema_version": "green-v400-packed-real-weight-affine-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "deterministic synthetic Jet2 inputs with actual closed GPT-2 weights",
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "program_semantic_hash": program.semantic_hash(),
        "layer": "block11.mlp.W_in",
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "weight_tensor_semantic_sha256": records["block11.mlp.W_in"]["tensor_semantic_sha256"],
        "backend_sha256": sha256_file(library),
        "rows": rows,
        "status": "PASS_ACTUAL_PACKED_WEIGHT_AFFINE_FFI_OBSERVATION_ONLY",
        "resident_dispatcher_complete": False,
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "rows": rows},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
