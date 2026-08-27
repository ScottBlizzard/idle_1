"""Outcome-blind actual-width exact fused-contrast FFI observation."""
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
        raise RuntimeError("fused-contrast output must be a new file on /mnt/sdb")
    program = TensorProgram.from_dict(json.loads(Path(args.program).read_text(encoding="utf-8")))
    reader = TensorStoreReader(Path(args.tensor_store))
    plan, _ = load_resident_plan_arrays(Path(args.plan), program, reader)
    fusion = plan["exact_final_contrast_fusion"]
    width = int(fusion["d_model"])
    backend = CompiledMPFRBackend(library)
    rows = []
    for precision in (384, 512):
        inputs = synthetic_inputs(width, precision)
        timings, hashes = [], []
        for _ in range(args.repetitions):
            started = time.perf_counter()
            payload = backend.fused_contrast_jet2(inputs, fusion)
            timings.append(time.perf_counter() - started)
            hashes.append(sha256_canonical(payload))
        if len(set(hashes)) != 1:
            raise RuntimeError("fused contrast output is nondeterministic")
        rows.append({
            "precision_bits": precision,
            "width": width,
            "elapsed_seconds": timings,
            "median_seconds": statistics.median(timings),
            "observed_max_seconds": max(timings),
            "guardbanded_observed_max_1p25x_seconds": 1.25 * max(timings),
            "output_exact_payload_sha256": hashes[0],
        })
    report = {
        "schema_version": "green-v400-actual-width-fused-contrast-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "deterministic synthetic Jet2 row with exact closed contrast fusion",
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "program_semantic_hash": program.semantic_hash(),
        "exact_final_contrast_fusion_sha256": plan["exact_final_contrast_fusion_sha256"],
        "backend_sha256": sha256_file(library),
        "repetitions": args.repetitions,
        "rows": rows,
        "status": "PASS_ACTUAL_WIDTH_FUSED_CONTRAST_FFI_OBSERVATION_ONLY",
        "resident_dispatcher_complete": False,
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
