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
    build_target_and_patched_responses,
    capture_resid_post_center,
)
from green_v400_prediction_worker import compute_raw_snr_analytic_power
from green_v400_response_baselines import compare_batched_response_fields


MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HISTORICAL_CLEAN = (
    "Earlier, Mary and John visited the park. Afterwards, John handed the book to"
)
HISTORICAL_CORRUPT = (
    "Earlier, John and John visited the park. Afterwards, John handed the book to"
)
METHODS = ("exact", "first_order", "integrated_gradients", "hvp")


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
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    analytic_record = None
    if method == "exact":
        analytic_started = time.perf_counter()
        analytic = compute_raw_snr_analytic_power(
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
        "raw_snr_analytic_power": analytic_record,
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
    clean = tokenizer(HISTORICAL_CLEAN, return_tensors="pt").input_ids.to(args.device)
    corrupt = tokenizer(HISTORICAL_CORRUPT, return_tensors="pt").input_ids.to(args.device)
    if clean.shape != corrupt.shape:
        raise RuntimeError("historical clean/corrupt token shapes differ")
    io_token_id = one_token(tokenizer, "Mary")
    s_token_id = one_token(tokenizer, "John")
    positions = (clean[0] == io_token_id).nonzero(as_tuple=False).reshape(-1)
    if positions.numel() != 1:
        raise RuntimeError("historical IO token must occur exactly once")
    position = int(positions[0])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    layer_records = []
    for layer in args.layers:
        site = IOIInterventionSite(
            layer=layer,
            position=position,
            io_token_id=io_token_id,
            s_token_id=s_token_id,
        )
        target, patched = build_target_and_patched_responses(
            model, clean, corrupt, site
        )
        center = capture_resid_post_center(model, clean, site)
        directions = torch.randn(
            args.direction_count,
            center.numel(),
            generator=generator,
            dtype=torch.float32,
        )
        directions = directions / directions.norm(dim=1, keepdim=True)
        directions = (args.direction_norm * directions).to(args.device)
        methods = [
            benchmark_method(
                method,
                target,
                patched,
                center,
                directions,
                integrated_gradients_steps=args.ig_steps,
                device=args.device,
            )
            for method in METHODS
        ]
        layer_records.append({"layer": layer, "methods": methods})

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
        "device": args.device,
        "sequence_length": int(clean.shape[1]),
        "direction_count": args.direction_count,
        "direction_norm": args.direction_norm,
        "ig_steps": args.ig_steps,
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
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--direction-count", type=int, default=8)
    parser.add_argument("--direction-norm", type=float, default=1e-3)
    parser.add_argument("--ig-steps", type=int, default=65)
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
