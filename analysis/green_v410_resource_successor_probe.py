"""Outcome-free single-cell probe for the v4.1 resource successor executor."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr_tensor_executor import (
    ResidentStaticRowCache,
    execute_tensor_program_mpfr,
    jet_exact_payload,
    preload_tensor_program_arrays,
)
from green_bridge_v400_schemas import sha256_canonical
from green_v410_resource_worker import _load_bundle


def hash_safe_exact_payload(jet) -> dict:
    """Lossless hex encoding avoids Python's decimal integer digit limit."""
    exact = jet_exact_payload(jet)
    return {
        component: {
            endpoint: [format(pair[0], "x"), format(pair[1], "x")]
            for endpoint, pair in endpoints.items()
        }
        for component, endpoints in exact.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("reference", "successor", "successor_uncached"),
        required=True,
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--fixture", default="affine")
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--trace-node-timing", action="store_true")
    args = parser.parse_args()

    _, program, reader = _load_bundle(args.bundle_dir, args.profile, args.fixture)
    backend = CompiledMPFRBackend(args.backend)
    preloaded = preload_tensor_program_arrays(program, reader)
    cache = (
        None if args.mode == "successor_uncached" else
        ResidentStaticRowCache.build_unpacked(program, backend, args.precision)
    )
    if args.cells <= 0:
        raise ValueError("probe cell count must be positive")
    elapsed_by_cell = []
    exact_by_cell = []

    def record_node_timing(event: dict) -> None:
        ordinal = int(event["ordinal"])
        node = program.nodes[ordinal]
        print(
            "GREEN_V410_NODE_DONE",
            ordinal,
            node.provenance_identity,
            node.kernel_id,
            f'{float(event["elapsed_seconds"]):.6f}',
            file=sys.stderr,
            flush=True,
        )

    try:
        for cell_index in range(args.cells):
            center = Fraction(cell_index, 4096)
            domain = Interval.from_bounds(
                center - Fraction(1, 1024), center + Fraction(1, 1024),
                args.precision,
            )
            started = time.perf_counter()
            result = execute_tensor_program_mpfr(
                program,
                reader,
                domain,
                backend,
                sparse_axis0_execution=True,
                resident_buffer_execution=args.mode != "reference",
                resident_static_row_cache=cache,
                preloaded_tensors=preloaded,
                return_runtime_metrics=True,
                successful_node_callback=(
                    record_node_timing if args.trace_node_timing else None
                ),
            )
            elapsed_by_cell.append(time.perf_counter() - started)
            exact_by_cell.append({
                name: hash_safe_exact_payload(result[name])
                for name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "output")
            })
        report = {
            "schema_version": "green-v410-resource-successor-single-cell-probe-v2",
            "mode": args.mode,
            "profile": args.profile,
            "fixture": args.fixture,
            "precision_bits": args.precision,
            "cell_count": args.cells,
            "program_semantic_hash": program.semantic_hash(),
            "backend_library_sha256": backend.library_sha256,
            "executor_source_sha256": hashlib.sha256(
                (ROOT / "src" / "green_bridge_v400_mpfr_tensor_executor.py")
                .read_bytes()
            ).hexdigest(),
            "tensor_program_source_sha256": hashlib.sha256(
                (ROOT / "src" / "green_bridge_v400_tensor_program.py").read_bytes()
            ).hexdigest(),
            "exact_outputs_sha256s": [
                sha256_canonical(exact) for exact in exact_by_cell
            ],
            "exact_output_sha256s_by_name": [
                {name: sha256_canonical(payload)
                 for name, payload in exact.items()}
                for exact in exact_by_cell
            ],
            "elapsed_seconds_by_cell": elapsed_by_cell,
            "elapsed_seconds_total": sum(elapsed_by_cell),
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "runtime_metrics": result["runtime_metrics"],
            "contains_scientific_outcome": False,
            "contains_endpoint_material": False,
        }
        print(json.dumps(report, sort_keys=True))
    finally:
        if cache is not None:
            cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
