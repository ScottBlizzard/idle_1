"""Outcome-blind actual-parameter LayerNorm suite observation."""
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


LAYERNORM_PAIRS = (
    ("block10.ln2.w", "block10.ln2.b"),
    ("block11.ln1.w", "block11.ln1.b"),
    ("block11.ln2.w", "block11.ln2.b"),
    ("ln_final.w", "ln_final.b"),
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
        raise RuntimeError("LayerNorm-suite output must be a new file on /mnt/sdb")
    program = TensorProgram.from_dict(json.loads(Path(args.program).read_text(encoding="utf-8")))
    reader = TensorStoreReader(Path(args.tensor_store))
    plan, arrays = load_resident_plan_arrays(Path(args.plan), program, reader)
    records = {record["name"]: record for record in plan["records"]}
    backend = CompiledMPFRBackend(library)
    width = int(program.resource_formula["dimensions"]["d_model"])
    epsilon = arrays["layer_norm.eps"].reshape(())
    rows = []
    for precision in (384, 512):
        inputs = synthetic_inputs(width, precision)
        for gamma_name, beta_name in LAYERNORM_PAIRS:
            gamma, beta = arrays[gamma_name], arrays[beta_name]
            timings, hashes = [], []
            for _ in range(args.repetitions):
                started = time.perf_counter()
                payload = backend.layer_norm_jet2(inputs, epsilon, gamma, beta)
                timings.append(time.perf_counter() - started)
                hashes.append(sha256_canonical(payload))
            if len(set(hashes)) != 1 or len(payload["outputs"]) != width:
                raise RuntimeError(f"LayerNorm failed closure for {gamma_name}")
            rows.append({
                "precision_bits": precision,
                "gamma_name": gamma_name,
                "beta_name": beta_name,
                "width": width,
                "elapsed_seconds": timings,
                "median_seconds": statistics.median(timings),
                "observed_max_seconds": max(timings),
                "guardbanded_observed_max_1p25x_seconds": 1.25 * max(timings),
                "output_exact_payload_sha256": hashes[0],
                "gamma_tensor_semantic_sha256": records[gamma_name][
                    "tensor_semantic_sha256"
                ],
                "beta_tensor_semantic_sha256": records[beta_name][
                    "tensor_semantic_sha256"
                ],
            })
    report = {
        "schema_version": "green-v400-actual-parameter-layernorm-suite-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "deterministic synthetic Jet2 row with actual closed LayerNorm parameters",
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "program_semantic_hash": program.semantic_hash(),
        "backend_sha256": sha256_file(library),
        "repetitions": args.repetitions,
        "rows": rows,
        "status": "PASS_ACTUAL_PARAMETER_LAYERNORM_SUITE_OBSERVATION_ONLY",
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
