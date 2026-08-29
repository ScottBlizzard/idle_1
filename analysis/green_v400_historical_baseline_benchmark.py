"""Actual-shape resource benchmark on the pre-existing July IOI grammar.

No untouched-v4 row, endpoint direction, endpoint outcome, or GREEN result is
loaded.  Baseline numerical values are deliberately discarded; only runtime,
memory, shape, and finiteness are reported.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import torch

from green_v400_grant_divergence import grant_divergence_panel
from green_v400_ioi_response_adapter import (
    IOIInterventionSite,
    build_matched_bypass_four_branch_responses,
    build_target_and_patched_responses,
    capture_resid_post_center,
)
from green_bridge_spec import PROMPT as GREATER_THAN_PROMPT, SELECTED_GATES
from green_v400_four_branch_baseline import (
    empirical_four_branch_interaction_response_batched,
)
from green_v400_prediction_worker import compute_normalized_mismatch_surrogate
from green_v400_response_baselines import (
    calibrate_integrated_gradients_grid,
    compare_batched_response_fields,
)


MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HISTORICAL_CLEAN = (
    "Earlier, Mary and John visited the park. Afterwards, John handed the book to"
)
HISTORICAL_CORRUPT = (
    "Earlier, John and John visited the park. Afterwards, John handed the book to"
)
METHODS = ("exact", "first_order", "integrated_gradients", "hvp", "ms_hvp")


def load_model(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().to(device)
    hf_model.config.use_cache = False
    model = HookedTransformer.from_pretrained_no_processing(
        "gpt2",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float32,
        default_prepend_bos=False,
    ).eval()
    del hf_model
    return model, tokenizer


def one_token(tokenizer, name: str) -> int:
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{name} is not one token")
    return int(ids[0])


def finite_comparison(result) -> bool:
    return bool(
        torch.isfinite(result.target_effects).all()
        and torch.isfinite(result.patched_effects).all()
        and torch.isfinite(result.discrepancies).all()
        and math.isfinite(result.rmse)
        and math.isfinite(result.normalized_rmse)
    )


def benchmark_method(
    method,
    target,
    patched,
    center,
    directions,
    *,
    integrated_gradients_steps,
    ms_hvp_segments,
    response_batch_chunk_size,
    device,
):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
    else:
        allocated_before = 0
    started = time.perf_counter()
    result = compare_batched_response_fields(
        method,
        target,
        patched,
        center,
        directions,
        integrated_gradients_steps=integrated_gradients_steps,
        ms_hvp_segments=ms_hvp_segments,
        batch_chunk_size=response_batch_chunk_size,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    analytic_record = None
    if method == "exact":
        analytic_started = time.perf_counter()
        analytic = compute_normalized_mismatch_surrogate(
            result.target_effects, result.discrepancies
        )
        analytic_record = {
            "elapsed_seconds": time.perf_counter() - analytic_started,
            "finite": all(
                math.isfinite(value)
                for value in analytic.values()
                if isinstance(value, float)
            ),
            "assumption": analytic["assumption"],
            "inferential_test_claimed": analytic["inferential_test_claimed"],
            "scientific_values_serialized": False,
        }
    record = {
        "method": method,
        "elapsed_seconds": elapsed,
        "finite": finite_comparison(result),
        "direction_count": int(directions.shape[0]),
        "activation_width": int(directions.shape[1]),
        "response_batching": True,
        "ig_steps": integrated_gradients_steps if method == "integrated_gradients" else None,
        "normalized_mismatch_surrogate": analytic_record,
        "ms_hvp_segments": ms_hvp_segments if method == "ms_hvp" else None,
        "response_batch_chunk_size": response_batch_chunk_size,
    }
    if device.startswith("cuda"):
        record.update(
            {
                "allocated_before_bytes": allocated_before,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    del result
    gc.collect()
    return record


def benchmark_four_branch(branches, center, directions, *, device):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
    else:
        allocated_before = 0
    started = time.perf_counter()
    result = empirical_four_branch_interaction_response_batched(
        branches, center, directions
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    record = {
        "method": "empirical_four_branch_interaction",
        "elapsed_seconds": time.perf_counter() - started,
        "finite": bool(
            torch.isfinite(result.psi_at_center)
            and torch.isfinite(result.psi_effects).all()
            and math.isfinite(result.rms_effect)
        ),
        "direction_count": int(directions.shape[0]),
        "activation_width": int(directions.shape[1]),
        "response_batching": True,
        "branch_order": list(result.branch_order),
        "branch_weights": list(result.branch_weights),
        "point_sampling_only": result.point_sampling_only,
        "certificate_claimed": result.certificate_claimed,
        "scientific_values_serialized": False,
    }
    if device.startswith("cuda"):
        record.update(
            {
                "allocated_before_bytes": allocated_before,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    del result
    gc.collect()
    return record


def run(args) -> dict:
    started = time.perf_counter()
    model, tokenizer = load_model(args.device)
    if args.task == "ioi":
        clean_text = HISTORICAL_CLEAN
        corrupt_text = HISTORICAL_CORRUPT
    else:
        clean_text = GREATER_THAN_PROMPT.format(noun="invasion", cc=12, y=5)
        corrupt_text = GREATER_THAN_PROMPT.format(noun="invasion", cc=12, y=45)
    clean = tokenizer(clean_text, return_tensors="pt").input_ids.to(args.device)
    corrupt = tokenizer(corrupt_text, return_tensors="pt").input_ids.to(args.device)
    if clean.shape != corrupt.shape:
        raise RuntimeError("historical clean/corrupt token shapes differ")
    if args.task == "ioi":
        io_token_id = one_token(tokenizer, "Mary")
        s_token_id = one_token(tokenizer, "John")
        positions = (clean[0] == io_token_id).nonzero(as_tuple=False).reshape(-1)
        if positions.numel() != 1:
            raise RuntimeError("historical IO token must occur exactly once")
        position = int(positions[0])
        site_builder = lambda layer: IOIInterventionSite(
            layer=layer,
            position=position,
            io_token_id=io_token_id,
            s_token_id=s_token_id,
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
        if len(set(suffix_ids)) != 100:
            raise RuntimeError("Greater-Than suffix token identifiers must be unique")
        differing = (clean[0] != corrupt[0]).nonzero(as_tuple=False).reshape(-1)
        if differing.numel() != 1:
            raise RuntimeError("historical Greater-Than pair must differ at one token")
        position = int(differing[0])
        site_builder = lambda layer: GreaterThanInterventionSite(
            layer=layer,
            position=position,
            clean_suffix=5,
            suffix_token_ids=suffix_ids,
        )
        response_builder = gt_response_builder
        center_builder = gt_center_builder
        branch_builder = gt_branch_builder
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    layer_records = []
    for layer in args.layers:
        site = site_builder(layer)
        target, patched = response_builder(model, clean, corrupt, site)
        center = center_builder(model, clean, site)
        directions = torch.randn(
            args.direction_count,
            center.numel(),
            generator=generator,
            dtype=torch.float32,
        )
        directions = directions / directions.norm(dim=1, keepdim=True)
        directions = (args.direction_norm * directions).to(args.device)
        ig_calibration = calibrate_integrated_gradients_grid(
            target,
            patched,
            center,
            directions,
            grid=tuple(args.ig_grid),
            absolute_tolerance=args.ig_absolute_tolerance,
            relative_tolerance=args.ig_relative_tolerance,
            batched=True,
            batch_chunk_size=args.response_batch_chunk_size,
            selection_rule="successive_grid_stability",
        )
        if not ig_calibration.converged:
            diagnostic = {
                "layer": layer,
                "grid": list(ig_calibration.grid),
                "absolute_tolerance": ig_calibration.absolute_tolerance,
                "relative_tolerance": ig_calibration.relative_tolerance,
                "selection_rule": ig_calibration.selection_rule,
                "records": list(ig_calibration.records),
                "scientific_values_serialized": False,
            }
            raise RuntimeError(
                "historical IG quadrature did not converge on frozen grid: "
                + json.dumps(diagnostic, sort_keys=True)
            )
        branch_build_started = time.perf_counter()
        four_branches = branch_builder(
            model,
            clean,
            corrupt,
            site,
            center,
            selected_gates=tuple(SELECTED_GATES),
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        four_branch_anchor_build_seconds = time.perf_counter() - branch_build_started
        methods = [
            benchmark_method(
                method,
                target,
                patched,
                center,
                directions,
                integrated_gradients_steps=ig_calibration.selected_steps,
                ms_hvp_segments=args.ms_hvp_segments,
                response_batch_chunk_size=args.response_batch_chunk_size,
                device=args.device,
            )
            for method in METHODS
        ]
        four_branch_record = benchmark_four_branch(
            four_branches, center, directions, device=args.device
        )
        four_branch_record["anchor_build_seconds"] = four_branch_anchor_build_seconds
        methods.append(four_branch_record)
        layer_records.append({
            "layer": layer,
            "ig_convergence": {
                "grid": list(ig_calibration.grid),
                "selected_steps": ig_calibration.selected_steps,
                "absolute_tolerance": ig_calibration.absolute_tolerance,
                "relative_tolerance": ig_calibration.relative_tolerance,
                "selection_rule": ig_calibration.selection_rule,
                "records": list(ig_calibration.records),
                "scientific_values_serialized": False,
                "quadrature_ablation_only": True,
                "direct_finite_gap_used_for_selection": False,
                "precision_risk_requires_separate_audit": True,
            },
            "methods": methods,
        })

    # Grant's cohort metric is independent of the model executor once the
    # activations are collected.  Benchmark its official actual-width path on
    # deterministic outcome-free tensors.
    cohort_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    natural = torch.randn(
        args.grant_cohort_size, 768, generator=cohort_generator
    )
    intervened = natural + 0.01 * torch.randn(
        args.grant_cohort_size, 768, generator=cohort_generator
    )
    grant_started = time.perf_counter()
    grant = grant_divergence_panel(
        natural,
        intervened,
        seed=args.seed + 2,
        sample_size=args.grant_cohort_size,
    )
    grant_elapsed = time.perf_counter() - grant_started
    grant_finite = all(
        math.isfinite(value)
        for value in grant.to_dict().values()
        if isinstance(value, float)
    )
    return {
        "schema_version": "green-v400-historical-baseline-benchmark-v1",
        "real_outcomes_authorized": False,
        "untouched_manifest_loaded": False,
        "endpoint_fields_loaded": False,
        "scientific_values_serialized": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "historical_prompt_grammar": "P0_20260712",
        "task": args.task,
        "device": args.device,
        "sequence_length": int(clean.shape[1]),
        "direction_count": args.direction_count,
        "direction_norm": args.direction_norm,
        "ig_grid": args.ig_grid,
        "ms_hvp_segments": args.ms_hvp_segments,
        "response_batch_chunk_size": args.response_batch_chunk_size,
        "layers": layer_records,
        "grant_cohort": {
            "sample_size": args.grant_cohort_size,
            "activation_width": 768,
            "elapsed_seconds": grant_elapsed,
            "finite": grant_finite,
        },
        "total_elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=("ioi", "greater_than"), default="ioi")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--direction-count", type=int, default=8)
    parser.add_argument("--direction-norm", type=float, default=1e-3)
    parser.add_argument("--ig-grid", type=int, nargs="+", default=[33, 65, 129, 257])
    parser.add_argument("--ig-absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--ig-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--ms-hvp-segments", type=int, default=8)
    parser.add_argument("--response-batch-chunk-size", type=int, default=32)
    parser.add_argument("--grant-cohort-size", type=int, default=128)
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
