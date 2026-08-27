"""Outcome-blind actual-shape all-head versus per-head attention FFI observation."""
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
from green_bridge_v400_schemas import sha256_canonical


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_jet(index: int, precision: int) -> Jet2:
    center = Fraction((index * 37) % 2049 - 1024, 1024)
    radius = Fraction(1, 1 << (18 + index % 7))
    return Jet2(
        Interval.from_bounds(center - radius, center + radius, precision),
        Interval.point(Fraction((index * 13) % 257 - 128, 512), precision),
        Interval.point(Fraction((index * 17) % 129 - 64, 1024), precision),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise RuntimeError("at least three repetitions are required")
    library, output = Path(args.library).resolve(), Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("attention benchmark output must be a new file on /mnt/sdb")
    d_model = args.n_heads * args.head_dim
    backend = CompiledMPFRBackend(library)
    rows = []
    for precision in (384, 512):
        query = [synthetic_jet(index, precision) for index in range(d_model)]
        keys = [[synthetic_jet(10_000 + token*d_model + index, precision)
                 for index in range(d_model)] for token in range(args.sequence_length)]
        values = [[synthetic_jet(100_000 + token*d_model + index, precision)
                   for index in range(d_model)] for token in range(args.sequence_length)]
        batch_times, individual_times, batch_hashes, individual_hashes = [], [], [], []
        def run_batch():
            started = time.perf_counter()
            result = backend.causal_attention_final_all_heads_jet2(
                query, keys, values, args.n_heads, pivot=0
            )["outputs"]
            return result, time.perf_counter() - started

        def run_individual():
            started = time.perf_counter()
            result = []
            for head in range(args.n_heads):
                start, stop = head * args.head_dim, (head + 1) * args.head_dim
                result.extend(backend.causal_attention_final_head_jet2(
                    query[start:stop], [row[start:stop] for row in keys],
                    [row[start:stop] for row in values], pivot=0,
                )["outputs"])
            return result, time.perf_counter() - started

        for repetition in range(args.repetitions):
            if repetition % 2 == 0:
                batch, batch_elapsed = run_batch()
                individual, individual_elapsed = run_individual()
            else:
                individual, individual_elapsed = run_individual()
                batch, batch_elapsed = run_batch()
            batch_times.append(batch_elapsed)
            individual_times.append(individual_elapsed)
            batch_hashes.append(sha256_canonical(batch))
            individual_hashes.append(sha256_canonical(individual))
            if batch != individual:
                raise RuntimeError("all-head attention is not bit-identical to per-head ABI")
        if len(set(batch_hashes)) != 1 or len(set(individual_hashes)) != 1:
            raise RuntimeError("attention ABI outputs are nondeterministic")
        rows.append({
            "precision_bits": precision,
            "all_heads_elapsed_seconds": batch_times,
            "per_head_elapsed_seconds": individual_times,
            "all_heads_observed_max_seconds": max(batch_times),
            "per_head_observed_max_seconds": max(individual_times),
            "observed_max_speedup_ratio": max(individual_times) / max(batch_times),
            "all_heads_median_seconds": statistics.median(batch_times),
            "per_head_median_seconds": statistics.median(individual_times),
            "median_speedup_ratio": (
                statistics.median(individual_times) / statistics.median(batch_times)
            ),
            "bit_identical": True,
            "output_exact_payload_sha256": batch_hashes[0],
        })
    report = {
        "schema_version": "green-v400-actual-shape-all-heads-attention-benchmark-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "deterministic synthetic Jet2 query/key/value tensors",
        "dimensions": {"sequence_length": args.sequence_length, "n_heads": args.n_heads,
                       "head_dim": args.head_dim, "d_model": d_model},
        "backend_sha256": sha256_file(library),
        "rows": rows,
        "status": "PASS_ACTUAL_SHAPE_ALL_HEADS_ATTENTION_FFI_OBSERVATION_ONLY",
        "execution_order_policy": "alternate all-head-first and per-head-first by repetition",
        "performance_decision": "NO_STABLE_SPEEDUP_KEEP_DISABLED_BY_DEFAULT",
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
