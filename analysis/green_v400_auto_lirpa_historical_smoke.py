"""Historical-prompt auto_LiRPA trace and bound smoke test.

This script never loads the untouched IOI manifest and cannot authorize a
scientific outcome.  It establishes graph equivalence and verifier support on
the pre-existing July prompt grammar only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from green_v400_auto_lirpa_tail import (
    auto_lirpa_linf_bounds,
    build_hf_gpt2_residual_tail,
    capture_hf_resid_post,
)


MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HISTORICAL_PROMPT = (
    "Earlier, Mary and John visited the park. Afterwards, John handed the book to"
)
LAYER_NORM_FAILURE = "Only positive values are supported in BoundReciprocal"


def classify_verifier_failure(error: BaseException) -> str | None:
    if isinstance(error, AssertionError) and str(error) == LAYER_NORM_FAILURE:
        return "STANDARD_LAYER_NORM_INTERVAL_DEPENDENCY"
    message = str(error)
    if (
        isinstance(error, RuntimeError)
        and "The size of tensor a" in message
        and "must match the size of tensor b" in message
        and "non-singleton dimension 0" in message
    ):
        return "CROWN_INTERMEDIATE_BOUND_SHAPE_MISMATCH"
    return None


def single_token_id(tokenizer, name: str) -> int:
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{name} is not one GPT-2 token")
    return int(ids[0])


def run(layer: int, epsilon: float, method: str, device: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().to(device)
    model.config.use_cache = False
    tokens = tokenizer(HISTORICAL_PROMPT, return_tensors="pt").input_ids.to(device)
    io_token_id = single_token_id(tokenizer, "Mary")
    s_token_id = single_token_id(tokenizer, "John")
    positions = (tokens[0] == io_token_id).nonzero(as_tuple=False).reshape(-1)
    if positions.numel() != 1:
        raise RuntimeError("historical IO token must occur exactly once")
    position = int(positions[0])
    fixed_hidden = capture_hf_resid_post(model, tokens, layer=layer)
    tail = build_hf_gpt2_residual_tail(
        model,
        fixed_hidden,
        layer=layer,
        position=position,
        io_token_id=io_token_id,
        s_token_id=s_token_id,
    ).to(device)
    center = fixed_hidden[:, position, :]
    with torch.no_grad():
        full_logits = model(tokens, use_cache=False).logits[0, -1]
        full_value = full_logits[io_token_id] - full_logits[s_token_id]
        tail_value = tail(center).reshape(())
    equivalence_error = float((tail_value - full_value).abs().cpu())
    if equivalence_error > 1e-4:
        raise RuntimeError(f"tail equivalence error {equivalence_error} exceeds 1e-4")
    payload = {
        "schema_version": "green-v400-auto-lirpa-historical-smoke-v1",
        "real_outcomes_authorized": False,
        "untouched_manifest_loaded": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "historical_prompt_grammar": "P0_20260712",
        "layer": layer,
        "position": position,
        "sequence_length": int(tokens.shape[1]),
        "tail_equivalence_abs_error": equivalence_error,
        "auto_lirpa_source_commit": "5a098e8f9fb5786a428a024981d833d303921f2d",
    }
    try:
        bounds = auto_lirpa_linf_bounds(
            tail, center, epsilon=epsilon, method=method
        )
    except BaseException as error:
        failure = classify_verifier_failure(error)
        if failure is None:
            raise
        payload.update(
            {
                "verifier_status": "DOCUMENTED_APPLICABILITY_FAILURE",
                "failure_class": failure,
                "failure_exception": type(error).__name__,
                "failure_message": str(error),
                "bound": None,
            }
        )
    else:
        payload.update(
            {
                "verifier_status": "BOUND_COMPUTED",
                "failure_class": None,
                "bound": bounds.to_dict(),
            }
        )
    payload["elapsed_seconds"] = time.perf_counter() - started
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--method", default="backward")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(args.layer, args.epsilon, args.method, args.device)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
