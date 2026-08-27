"""Outcome-blind actual-weight resident-buffer MLP-chain observation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
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


def synthetic_jet(index: int, precision: int) -> Jet2:
    center = Fraction((index * 37) % 2049 - 1024, 1024)
    radius = Fraction(1, 1 << (18 + index % 7))
    return Jet2(
        Interval.from_bounds(center - radius, center + radius, precision),
        Interval.point(Fraction((index * 13) % 257 - 128, 512), precision),
        Interval.point(Fraction((index * 17) % 129 - 64, 1024), precision),
    )


def decode_jet(backend: CompiledMPFRBackend, payload: dict, precision: int) -> Jet2:
    return Jet2(*(
        Interval.from_bounds(
            backend.exact_fraction(payload[name]["lower"]),
            backend.exact_fraction(payload[name]["upper"]), precision,
        )
        for name in ("value", "first", "second")
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists() or args.repetitions < 3:
        raise RuntimeError("MLP benchmark requires a new /mnt/sdb output and >=3 repetitions")
    program = TensorProgram.from_dict(json.loads(
        Path(args.program).read_text(encoding="utf-8")
    ))
    reader = TensorStoreReader(Path(args.tensor_manifest))
    plan, arrays = load_resident_plan_arrays(Path(args.plan), program, reader)
    backend = CompiledMPFRBackend(Path(args.library))
    w_in, b_in = arrays["block11.mlp.W_in"], arrays["block11.mlp.b_in"]
    w_out, b_out = arrays["block11.mlp.W_out"], arrays["block11.mlp.b_out"]
    kappa, lam = arrays["gelu.kappa"].reshape(()), arrays["gelu.lambda"].reshape(())
    rows = []
    passed = True
    for precision in (384, 512):
        inputs = [synthetic_jet(index, precision) for index in range(w_in.shape[0])]
        json_times, resident_times, json_hashes, resident_hashes = [], [], [], []

        def run_json():
            started = time.perf_counter()
            pre = backend.packed_affine_layer_jet2(w_in, b_in, inputs)["outputs"]
            pre_jets = [decode_jet(backend, item, precision) for item in pre]
            post = backend.gelu_new_layer_jet2(pre_jets, kappa, lam)["outputs"]
            post_jets = [decode_jet(backend, item, precision) for item in post]
            result = backend.packed_affine_layer_jet2(w_out, b_out, post_jets)["outputs"]
            return result, time.perf_counter() - started

        def run_resident():
            buffers = []
            try:
                started = time.perf_counter()
                buffers.append(backend.resident_jet_buffer(inputs))
                buffers.append(backend.resident_packed_affine_layer_jet2(
                    buffers[-1], w_in, b_in
                ))
                buffers.append(backend.resident_gelu_new_layer_jet2(
                    buffers[-1], kappa, lam
                ))
                buffers.append(backend.resident_packed_affine_layer_jet2(
                    buffers[-1], w_out, b_out
                ))
                result = backend.export_resident_jet_buffer(buffers[-1])["outputs"]
                return result, time.perf_counter() - started
            finally:
                for buffer in reversed(buffers):
                    buffer.close()

        for repetition in range(args.repetitions):
            if repetition % 2 == 0:
                json_result, json_elapsed = run_json()
                resident_result, resident_elapsed = run_resident()
            else:
                resident_result, resident_elapsed = run_resident()
                json_result, json_elapsed = run_json()
            json_times.append(json_elapsed)
            resident_times.append(resident_elapsed)
            json_hashes.append(sha256_canonical(json_result))
            resident_hashes.append(sha256_canonical(resident_result))
            if json_result != resident_result:
                raise RuntimeError("resident MLP chain is not bit-identical to JSON path")
        deterministic = len(set(json_hashes)) == len(set(resident_hashes)) == 1
        passed = passed and deterministic and json_hashes[0] == resident_hashes[0]
        rows.append({
            "precision_bits": precision,
            "json_roundtrip_elapsed_seconds": json_times,
            "resident_buffer_elapsed_seconds": resident_times,
            "json_roundtrip_median_seconds": statistics.median(json_times),
            "resident_buffer_median_seconds": statistics.median(resident_times),
            "median_speedup_ratio": statistics.median(json_times) / statistics.median(resident_times),
            "bit_identical": True,
            "deterministic": deterministic,
        })
    report = {
        "schema_version": "green-v400-resident-mlp-chain-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS_RESIDENT_MLP_CHAIN_OBSERVATION_ONLY" if passed else "FAIL",
        "contains_scientific_outcome": False,
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
        "program_semantic_hash": program.semantic_hash(),
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "backend_sha256": backend.library_sha256,
        "dimensions": {"input_width": int(w_in.shape[0]),
                       "hidden_width": int(w_in.shape[1]),
                       "output_width": int(w_out.shape[1])},
        "input_policy": "deterministic synthetic Jet2 row; actual packed GPT-2 MLP weights",
        "timing_scope": "Python-input through final Python JSON; native intermediates are opaque",
        "rows": rows,
        "resident_dispatcher_complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "rows": rows},
                     sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
