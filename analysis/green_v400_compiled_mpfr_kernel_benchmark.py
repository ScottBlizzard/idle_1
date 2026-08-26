"""Outcome-blind GPT-2-tail-shaped MPFR kernel benchmark; not an integrated executor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend


D_MODEL, D_MLP, N_HEADS, D_HEAD = 768, 3072, 12, 64
SEQUENCE_LENGTH, SELECTED_GATES = 12, 10
AFFINE_LAYERS = (
    ("block10_selected_in", D_MODEL, SELECTED_GATES),
    ("block10_selected_out", SELECTED_GATES, D_MODEL),
    ("block11_qkv", D_MODEL, 3 * D_MODEL),
    ("block11_attention_out", D_MODEL, D_MODEL),
    ("block11_mlp_in", D_MODEL, D_MLP),
    ("block11_mlp_out", D_MLP, D_MODEL),
    ("final_exact_fused_contrast_shape_proxy", D_MODEL, 1),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    library, output = Path(args.library).resolve(), Path(args.output).resolve()
    if args.repetitions < 3:
        raise RuntimeError("at least three fixed repetitions are required")
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("benchmark output must be a new file on /mnt/sdb")
    backend = CompiledMPFRBackend(library)
    rows, summaries = [], {}
    for precision in (384, 512):
        totals = []
        for repetition in range(args.repetitions):
            affine_seconds = 0.0
            for name, input_width, output_width in AFFINE_LAYERS:
                result = backend.benchmark_affine_layer(precision, input_width, output_width)
                affine_seconds += result["elapsed_seconds"]
                rows.append({"kind": "pairwise_affine", "name": name,
                             "repetition": repetition, **result})
            gelu = backend.benchmark_gelu(precision, D_MLP + SELECTED_GATES)
            layer_norm = backend.benchmark_layer_norm(precision, D_MODEL, 4)
            attention = backend.benchmark_causal_attention(
                precision, SEQUENCE_LENGTH, N_HEADS, D_HEAD,
            )
            rows.extend((
                {"kind": "gelu_new", "name": "block10_plus_block11",
                 "repetition": repetition, **gelu},
                {"kind": "layer_norm", "name": "block10_plus_three_tail_norms",
                 "repetition": repetition, **layer_norm},
                {"kind": "causal_attention", "name": "final_query_all_heads",
                 "repetition": repetition, **attention},
            ))
            total = affine_seconds + gelu["elapsed_seconds"] + layer_norm["elapsed_seconds"] \
                + attention["elapsed_seconds"]
            totals.append(total)
        maximum = max(totals)
        summaries[str(precision)] = {
            "representative_conservative_branch_cell_seconds": totals,
            "maximum_seconds": maximum,
            "timing_upper_1p25x_seconds": 1.25 * maximum,
            "projected_four_branch_two_cell_kernel_upper_seconds": 8 * 1.25 * maximum,
        }
    report = {
        "schema_version": "green-v400-compiled-mpfr-kernel-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "input_policy": "fixed public GPT-2-small shapes and native exact-dyadic synthetic jets only",
        "dimensions": {"sequence_length": SEQUENCE_LENGTH, "d_model": D_MODEL,
                       "d_mlp": D_MLP, "n_heads": N_HEADS, "d_head": D_HEAD,
                       "selected_gates": SELECTED_GATES},
        "backend_version": backend.version,
        "backend_sha256": sha256_file(library),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "logical_cpu_count": os.cpu_count()},
        "repetitions": args.repetitions,
        "rows": rows,
        "summaries": summaries,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "coverage": ["pairwise_affine.v1 shape proxy", "gelu_new.v1",
                     "layer_norm.v1", "causal_attention.v1 final dynamic query"],
        "known_exclusions": [
            "resident TensorProgram dispatcher", "element-level dependency-mask closure",
            "affine_scatter.v1", "static_view.v1", "residual_add.v1",
            "real final_contrast.v1 semantics", "branch_linear_combination.v1",
            "tensor decode and serialization", "adaptive certificate orchestration",
        ],
        "claim_status": "PASS_KERNEL_LEVEL_ONLY",
        "full_tail_equivalent": False,
        "cap_decision_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["claim_status"], "output": str(output),
                      "summaries": summaries, "peak_rss_kib": report["peak_rss_kib"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
