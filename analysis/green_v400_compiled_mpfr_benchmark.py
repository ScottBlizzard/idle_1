"""Outcome-blind, tail-shaped affine benchmark for the compiled MPFR backend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend


LAYERS = (
    ("block10_selected_in", 768, 10),
    ("block10_selected_out", 10, 768),
    ("block11_qkv", 768, 2304),
    ("block11_attention_out", 768, 768),
    ("block11_mlp_in", 768, 3072),
    ("block11_mlp_out", 3072, 768),
    ("final_exact_fused_contrast", 768, 1),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("benchmark output must be a new file on /mnt/sdb")
    backend = CompiledMPFRBackend(Path(args.library))
    rows = []
    for precision in (384, 512):
        for name, input_width, output_width in LAYERS:
            row = backend.benchmark_affine_layer(precision, input_width, output_width)
            rows.append({"layer": name, **row})
    totals = {}
    for precision in (384, 512):
        selected = [row for row in rows if row["precision_bits"] == precision]
        elapsed = sum(row["elapsed_seconds"] for row in selected)
        primitives = sum(row["directed_mpfr_primitives"] for row in selected)
        totals[str(precision)] = {
            "one_branch_one_cell_affine_seconds": elapsed,
            "one_branch_one_cell_directed_mpfr_primitives": primitives,
            "effective_directed_mpfr_primitives_per_second": primitives / elapsed,
            "projected_four_branch_two_cell_affine_seconds": elapsed * 8,
        }
    report = {
        "schema_version": "green-v400-compiled-mpfr-affine-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "backend_version": backend.version,
        "scope": "tail-shaped affine kernels only; excludes nonlinear and orchestration overhead",
        "contains_scientific_outcome": False,
        "rows": rows,
        "totals": totals,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cap_decision_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_AFFINE_ONLY", "output": str(output),
                      "totals": totals}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

