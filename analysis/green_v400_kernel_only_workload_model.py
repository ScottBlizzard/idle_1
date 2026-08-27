"""Compose outcome-blind kernel observations over exact TensorProgram row liveness."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_mpfr_tensor_executor import tensor_program_required_axis0_rows
from green_bridge_v400_tensor_program import TensorProgram


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--affine", required=True)
    parser.add_argument("--layernorm", required=True)
    parser.add_argument("--gelu", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--contrast", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("workload-model output must be a new file on /mnt/sdb")
    program = TensorProgram.from_dict(read_json(args.program))
    plan = read_json(args.plan)
    artifacts = {
        "affine": read_json(args.affine), "layernorm": read_json(args.layernorm),
        "gelu": read_json(args.gelu), "attention": read_json(args.attention),
        "contrast": read_json(args.contrast),
    }
    if plan["program_semantic_hash"] != program.semantic_hash():
        raise RuntimeError("resident plan/program mismatch")
    for name, artifact in artifacts.items():
        if (artifact.get("contains_scientific_outcome") is not False
                or artifact.get("program_semantic_hash", program.semantic_hash())
                    != program.semantic_hash()
                or artifact.get("formal_wall_time_upper_bound") is not False
                or artifact.get("cap_decision_authorized") is not False):
            raise RuntimeError(f"ineligible {name} benchmark artifact")
    backend_hashes = {artifact["backend_sha256"] for artifact in artifacts.values()}
    if len(backend_hashes) != 1:
        raise RuntimeError("kernel benchmarks use different native backends")
    semantic_to_name = {
        record["tensor_semantic_sha256"]: record["name"] for record in plan["records"]
    }
    live_rows = tensor_program_required_axis0_rows(program)
    counts = {"affine": {}, "layernorm": {}, "gelu": {}, "attention": 0, "contrast": 0}
    for node in program.nodes:
        row_count = len(live_rows[node.semantic_id]) if node.output_spec.shape else 1
        if node.kernel_id == "pairwise_affine.v1":
            weight_name = semantic_to_name[node.tensor_inputs[0].tensor_sha256]
            counts["affine"][weight_name] = counts["affine"].get(weight_name, 0) + row_count
        elif node.kernel_id == "layer_norm.v1":
            gamma_name = semantic_to_name[node.tensor_inputs[0].tensor_sha256]
            counts["layernorm"][gamma_name] = counts["layernorm"].get(gamma_name, 0) + row_count
        elif node.kernel_id == "gelu_new.v1":
            width = node.output_spec.shape[1]
            counts["gelu"][str(width)] = counts["gelu"].get(str(width), 0) + row_count
        elif node.kernel_id == "causal_attention.v1":
            counts["attention"] += 1
        elif node.kernel_id == "final_contrast.v1":
            counts["contrast"] += 1

    def index_rows(rows: list[dict], key):
        return {(row["precision_bits"], key(row)): row for row in rows}

    affine_rows = index_rows(artifacts["affine"]["rows"], lambda row: row["weight_name"])
    layernorm_rows = index_rows(artifacts["layernorm"]["rows"], lambda row: row["gamma_name"])
    gelu_rows = index_rows(artifacts["gelu"]["rows"], lambda row: str(row["width"]))
    attention_rows = {row["precision_bits"]: row for row in artifacts["attention"]["rows"]}
    contrast_rows = {row["precision_bits"]: row for row in artifacts["contrast"]["rows"]}
    precision_rows = []
    for precision in (384, 512):
        components = []
        for kernel_group in ("affine", "layernorm", "gelu"):
            source = {"affine": affine_rows, "layernorm": layernorm_rows,
                      "gelu": gelu_rows}[kernel_group]
            for identity, count in counts[kernel_group].items():
                row = source[(precision, identity)]
                components.append({
                    "kernel_group": kernel_group,
                    "identity": identity,
                    "call_count": count,
                    "per_call_median_seconds": row["median_seconds"],
                    "per_call_guardbanded_observed_max_seconds": row[
                        "guardbanded_observed_max_1p25x_seconds"
                    ],
                    "median_contribution_seconds": count * row["median_seconds"],
                    "guardbanded_observed_max_contribution_seconds": (
                        count * row["guardbanded_observed_max_1p25x_seconds"]
                    ),
                })
        attention = attention_rows[precision]
        components.append({
            "kernel_group": "causal_attention_per_head_aggregate",
            "identity": "S12_H12_DH64",
            "call_count": counts["attention"],
            "per_call_median_seconds": attention["per_head_median_seconds"],
            "per_call_guardbanded_observed_max_seconds": (
                1.25 * attention["per_head_observed_max_seconds"]
            ),
            "median_contribution_seconds": (
                counts["attention"] * attention["per_head_median_seconds"]
            ),
            "guardbanded_observed_max_contribution_seconds": (
                counts["attention"] * 1.25 * attention["per_head_observed_max_seconds"]
            ),
        })
        contrast = contrast_rows[precision]
        components.append({
            "kernel_group": "fused_contrast",
            "identity": "d_model_768",
            "call_count": counts["contrast"],
            "per_call_median_seconds": contrast["median_seconds"],
            "per_call_guardbanded_observed_max_seconds": contrast[
                "guardbanded_observed_max_1p25x_seconds"
            ],
            "median_contribution_seconds": counts["contrast"] * contrast["median_seconds"],
            "guardbanded_observed_max_contribution_seconds": (
                counts["contrast"] * contrast["guardbanded_observed_max_1p25x_seconds"]
            ),
        })
        precision_rows.append({
            "precision_bits": precision,
            "components": components,
            "kernel_only_no_cache_median_sum_seconds": sum(
                row["median_contribution_seconds"] for row in components
            ),
            "kernel_only_no_cache_guardbanded_observed_max_sum_seconds": sum(
                row["guardbanded_observed_max_contribution_seconds"] for row in components
            ),
        })
    report = {
        "schema_version": "green-v400-kernel-only-no-cache-workload-model-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS_KERNEL_ONLY_NO_CACHE_OBSERVATIONAL_COMPOSITION",
        "contains_scientific_outcome": False,
        "program_semantic_hash": program.semantic_hash(),
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "backend_sha256": next(iter(backend_hashes)),
        "row_liveness": {
            "dense_axis0_row_slot_count": sum(
                node.output_spec.shape[0] for node in program.nodes if node.output_spec.shape
            ),
            "materialized_axis0_row_count": sum(map(len, live_rows.values())),
        },
        "call_counts_no_static_cache": counts,
        "precision_rows": precision_rows,
        "composition_policy": (
            "sum per-call medians and 1.25x observed maxima over exact row-liveness call counts; "
            "do not apply runtime static-row cache savings"
        ),
        "excluded_costs": [
            "affine-scatter and constant Jet materialization", "residual/subtract/branch arithmetic",
            "static-row cache hashing and lookup", "Python node dispatch outside timed kernel calls",
            "program/plan loading and validation", "endpoint-center and multi-radius orchestration",
            "adaptive queue and curvature integration", "certificate serialization and process-tree RSS",
        ],
        "formal_wall_time_upper_bound": False,
        "cap_decision_authorized": False,
        "numeric_cap_decision_authorized": False,
        "resident_dispatcher_complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "precision_rows": precision_rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
