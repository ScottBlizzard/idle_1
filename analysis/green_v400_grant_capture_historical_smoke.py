"""Historical-prompt smoke for the downstream Grant capture route.

This deliberately avoids every v4 development/confirmation universe and every
GREEN/endpoint direction.  It records only execution, shape, finiteness, and
resource facts; the captured state vectors and scientific metric values are
never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from green_v400_formal_grant_runner import _capture_triplet


MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HISTORICAL_PAIRS = (
    (
        "Earlier, Mary and John visited the park. Afterwards, John handed the book to",
        "Earlier, John and John visited the park. Afterwards, John handed the book to",
    ),
    (
        "Yesterday, Alice and Bob entered the room. Later, Bob passed the cup to",
        "Yesterday, Bob and Bob entered the room. Later, Bob passed the cup to",
    ),
    (
        "Before dinner, Sarah and Michael walked home. Then, Michael gave the note to",
        "Before dinner, Michael and Michael walked home. Then, Michael gave the note to",
    ),
    (
        "At noon, Laura and David reached school. Soon, David returned the key to",
        "At noon, David and David reached school. Soon, David returned the key to",
    ),
)


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
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        refactor_factored_attn_matrices=False,
        default_prepend_bos=False,
    ).eval()
    del hf_model
    return model


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    model = load_model(args.device)
    load_seconds = time.perf_counter() - started
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    records = []
    any_contextual_change = False
    for pair_index, (clean_text, corrupt_text) in enumerate(HISTORICAL_PAIRS):
        clean = model.to_tokens(clean_text, prepend_bos=False)
        corrupt = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean.shape != corrupt.shape:
            raise RuntimeError(f"historical pair {pair_index} token lengths differ")
        candidate_position = 2
        measurement_position = clean.shape[1] - 1
        for layer in (0, 4, 8):
            run_started = time.perf_counter()
            natural, patched, control = _capture_triplet(
                model=model,
                clean_tokens=clean,
                corrupt_tokens=corrupt,
                candidate_hook=f"blocks.{layer}.hook_resid_post",
                measurement_hook="blocks.10.hook_resid_post",
                candidate_position=candidate_position,
                measurement_position=measurement_position,
            )
            any_contextual_change = any_contextual_change or bool(
                torch.any(natural != patched)
            )
            records.append(
                {
                    "historical_pair_index": pair_index,
                    "candidate_layer": layer,
                    "candidate_position": candidate_position,
                    "measurement_position": measurement_position,
                    "measurement_strictly_after_candidate": measurement_position
                    > candidate_position,
                    "vector_width": int(natural.numel()),
                    "natural_finite": bool(torch.isfinite(natural).all()),
                    "patched_finite": bool(torch.isfinite(patched).all()),
                    "unpatched_control_finite": bool(torch.isfinite(control).all()),
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )
    if not any_contextual_change:
        raise RuntimeError("historical smoke did not exercise a nondegenerate route")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_bytes = int(torch.cuda.max_memory_allocated())
    else:
        peak_bytes = 0
    report = {
        "schema_version": "green-v400-grant-capture-historical-smoke-v1",
        "contains_scientific_outcome": False,
        "uses_v4_untouched_universe": False,
        "uses_green_or_endpoint_directions": False,
        "raw_activation_serialized": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": args.device,
        "measurement_hook": "blocks.10.hook_resid_post",
        "measurement_position_rule": "final_prompt_position",
        "historical_pair_text_sha256": canonical_sha256(HISTORICAL_PAIRS),
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes_after_model_load_reset": peak_bytes,
        "nondegenerate_route_exercised": any_contextual_change,
        "capture_count": len(records),
        "captures": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
