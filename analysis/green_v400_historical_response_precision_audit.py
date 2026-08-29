"""Historical-only precision audit for small finite response differences.

The audit compares float32 execution with float64 execution of the exact same
float32 checkpoint values.  It serializes numerical discrepancies and runtime
only, never response vectors or untouched-v4 data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from green_bridge_spec import SELECTED_GATES
from green_v400_four_branch_baseline import (
    empirical_four_branch_interaction_response_batched,
)
from green_v400_ioi_response_adapter import (
    IOIInterventionSite,
    build_matched_bypass_four_branch_responses,
    build_target_and_patched_responses,
    capture_resid_post_center,
)
from green_v400_response_baselines import batched_response_effects
from analysis.green_v400_historical_baseline_benchmark import (
    HISTORICAL_CLEAN,
    HISTORICAL_CORRUPT,
    GREATER_THAN_PROMPT,
    MODEL_ID,
    MODEL_REVISION,
    load_model,
    one_token,
)


def checkpoint_value_hash(model) -> str:
    """Hash every state value after exact round-trip to float32 bytes."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        array = value.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def response_records(model, clean, corrupt, tokenizer, layers, directions_by_layer, *, task, dtype, ig_steps, chunk_size):
    if task == "ioi":
        io_token_id = one_token(tokenizer, "Mary")
        s_token_id = one_token(tokenizer, "John")
        positions = (clean[0] == io_token_id).nonzero(as_tuple=False).reshape(-1)
        if positions.numel() != 1:
            raise RuntimeError("historical IO token must occur exactly once")
        position = int(positions[0])
        site_builder = lambda layer: IOIInterventionSite(
            layer, position, io_token_id, s_token_id
        )
        response_builder = build_target_and_patched_responses
        center_builder = capture_resid_post_center
        branch_builder = build_matched_bypass_four_branch_responses
    else:
        from green_v400_greater_than_response_adapter import (
            GreaterThanInterventionSite,
            build_matched_bypass_four_branch_responses as gt_branch_builder,
            build_target_and_patched_responses as gt_response_builder,
            capture_resid_post_center as gt_center_builder,
        )

        suffix_values = [
            tokenizer.encode(f"{suffix:02d}", add_special_tokens=False)
            for suffix in range(100)
        ]
        if any(len(values) != 1 for values in suffix_values):
            raise RuntimeError("every Greater-Than suffix must be one token")
        suffix_ids = tuple(int(values[0]) for values in suffix_values)
        differing = (clean[0] != corrupt[0]).nonzero(as_tuple=False).reshape(-1)
        if differing.numel() != 1:
            raise RuntimeError("historical Greater-Than pair must differ at one token")
        position = int(differing[0])
        site_builder = lambda layer: GreaterThanInterventionSite(
            layer, position, 5, suffix_ids
        )
        response_builder = gt_response_builder
        center_builder = gt_center_builder
        branch_builder = gt_branch_builder
    records = {}
    for layer in layers:
        site = site_builder(layer)
        target, patched = response_builder(model, clean, corrupt, site)
        center = center_builder(model, clean, site).to(dtype=dtype)
        directions = directions_by_layer[layer].to(device=center.device, dtype=dtype)
        target_finite = batched_response_effects(
            "exact", target, center, directions, batch_chunk_size=chunk_size
        )
        patched_finite = batched_response_effects(
            "exact", patched, center, directions, batch_chunk_size=chunk_size
        )
        target_ig = batched_response_effects(
            "integrated_gradients",
            target,
            center,
            directions,
            integrated_gradients_steps=ig_steps,
            batch_chunk_size=chunk_size,
        )
        patched_ig = batched_response_effects(
            "integrated_gradients",
            patched,
            center,
            directions,
            integrated_gradients_steps=ig_steps,
            batch_chunk_size=chunk_size,
        )
        branches = branch_builder(
            model,
            clean,
            corrupt,
            site,
            center,
            selected_gates=tuple(SELECTED_GATES),
        )
        interaction = empirical_four_branch_interaction_response_batched(
            branches, center, directions
        )
        replay = empirical_four_branch_interaction_response_batched(
            branches, center, directions
        )
        records[layer] = {
            "target_finite": target_finite.detach().double().cpu(),
            "patched_finite": patched_finite.detach().double().cpu(),
            "target_ig": target_ig.detach().double().cpu(),
            "patched_ig": patched_ig.detach().double().cpu(),
            "four_branch": interaction.psi_effects.detach().double().cpu(),
            "four_branch_replay": replay.psi_effects.detach().double().cpu(),
        }
    return records


def max_pair_gap(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def run(args) -> dict:
    started = time.perf_counter()
    model, tokenizer = load_model(args.device)
    if args.task == "ioi":
        clean_text, corrupt_text = HISTORICAL_CLEAN, HISTORICAL_CORRUPT
    else:
        clean_text = GREATER_THAN_PROMPT.format(noun="invasion", cc=12, y=5)
        corrupt_text = GREATER_THAN_PROMPT.format(noun="invasion", cc=12, y=45)
    clean = tokenizer(clean_text, return_tensors="pt").input_ids.to(args.device)
    corrupt = tokenizer(corrupt_text, return_tensors="pt").input_ids.to(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    directions = {}
    for layer in args.layers:
        value = torch.randn(args.direction_count, 768, generator=generator, dtype=torch.float32)
        value = args.direction_norm * value / value.norm(dim=1, keepdim=True)
        directions[layer] = value

    float32_hash = checkpoint_value_hash(model)
    records32 = response_records(
        model,
        clean,
        corrupt,
        tokenizer,
        args.layers,
        directions,
        task=args.task,
        dtype=torch.float32,
        ig_steps=args.ig_steps,
        chunk_size=args.chunk_size,
    )
    model = model.to(torch.float64, print_details=False)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    float64_roundtrip_hash = checkpoint_value_hash(model)
    if float64_roundtrip_hash != float32_hash:
        raise RuntimeError("float64 model does not preserve exact float32 checkpoint values")
    records64 = response_records(
        model,
        clean,
        corrupt,
        tokenizer,
        args.layers,
        directions,
        task=args.task,
        dtype=torch.float64,
        ig_steps=args.ig_steps,
        chunk_size=args.chunk_size,
    )

    layers = []
    all_pass = True
    for layer in args.layers:
        r32, r64 = records32[layer], records64[layer]
        scale = max(
            float(r64["target_finite"].abs().max()),
            float(r64["patched_finite"].abs().max()),
            1e-12,
        )
        tolerance = args.absolute_tolerance + args.relative_tolerance * scale
        direct64_ig_gap = max(
            max_pair_gap(r64["target_finite"], r64["target_ig"]),
            max_pair_gap(r64["patched_finite"], r64["patched_ig"]),
        )
        direct32_ig_gap = max(
            max_pair_gap(r32["target_finite"], r32["target_ig"]),
            max_pair_gap(r32["patched_finite"], r32["patched_ig"]),
        )
        direct_cross_precision_gap = max(
            max_pair_gap(r32["target_finite"], r64["target_finite"]),
            max_pair_gap(r32["patched_finite"], r64["patched_finite"]),
        )
        four_branch_replay_gap = max_pair_gap(
            r64["four_branch"], r64["four_branch_replay"]
        )
        passed = direct64_ig_gap <= tolerance and four_branch_replay_gap <= 1e-12
        all_pass = all_pass and passed
        layers.append(
            {
                "layer": layer,
                "response_scale": scale,
                "allowed_max_absolute_error": tolerance,
                "float32_direct_vs_ig_max_absolute_gap": direct32_ig_gap,
                "float64_direct_vs_ig_max_absolute_gap": direct64_ig_gap,
                "float32_vs_float64_direct_max_absolute_gap": direct_cross_precision_gap,
                "float64_four_branch_replay_max_absolute_gap": four_branch_replay_gap,
                "passed": passed,
                "scientific_response_vectors_serialized": False,
            }
        )
    return {
        "schema_version": "green-v400-historical-response-precision-audit-v1",
        "real_outcomes_authorized": False,
        "untouched_manifest_loaded": False,
        "endpoint_fields_loaded": False,
        "scientific_response_vectors_serialized": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "task": args.task,
        "checkpoint_float32_value_hash": float32_hash,
        "float64_roundtrip_float32_value_hash": float64_roundtrip_hash,
        "checkpoint_values_preserved_exactly": True,
        "direction_count": args.direction_count,
        "direction_norm": args.direction_norm,
        "ig_steps": args.ig_steps,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "layers": layers,
        "float64_response_evaluation_required": all_pass,
        "verdict": (
            "PASS_REQUIRE_FLOAT64_SAME_CHECKPOINT_RESPONSE_EVALUATION"
            if all_pass
            else "BLOCK_RESPONSE_PRECISION"
        ),
        "total_elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=("ioi", "greater_than"), default="ioi")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--direction-count", type=int, default=8)
    parser.add_argument("--direction-norm", type=float, default=1e-3)
    parser.add_argument("--ig-steps", type=int, default=65)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
