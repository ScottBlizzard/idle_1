"""Historical-free smoke test of the formal float64 model-manifest contract.

This script loads the frozen model, verifies every state tensor against the
already-frozen manifest, converts the exact float32 constants to float64, and
records only the typed precision receipt and resource diagnostics.  It never
loads prompt universes, directions, endpoints, or scientific outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from green_v400_response_precision import (
    prepare_float64_response_evaluation,
    tensor_sha256,
)


MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


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
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    manifest_sha256 = canonical_sha256(manifest)
    started = time.perf_counter()
    model = load_model(args.device)
    loaded_seconds = time.perf_counter() - started
    observed_hashes = {
        name: tensor_sha256(value) for name, value in model.state_dict().items()
    }
    observed_raw_hashes = {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest()
        for name, value in model.state_dict().items()
    }
    expected_hashes = manifest.get("weight_tensor_hashes", {})
    missing_names = sorted(set(expected_hashes) - set(observed_hashes))
    extra_names = sorted(set(observed_hashes) - set(expected_hashes))
    mismatched_names = sorted(
        name
        for name in set(expected_hashes) & set(observed_hashes)
        if expected_hashes[name] != observed_hashes[name]
    )
    if missing_names or extra_names or mismatched_names:
        diagnostic = {
            "schema_version": "green-v400-model-manifest-precision-smoke-v1",
            "contains_scientific_outcome": False,
            "prompt_universe_loaded": False,
            "direction_payload_loaded": False,
            "endpoint_fields_loaded": False,
            "model_manifest_path": str(args.model_manifest),
            "model_manifest_sha256": manifest_sha256,
            "expected_full_model_hash": manifest.get("full_model_hash"),
            "observed_full_model_hash": canonical_sha256(observed_hashes),
            "observed_raw_bytes_full_model_hash": canonical_sha256(
                observed_raw_hashes
            ),
            "raw_bytes_matching_state_count": sum(
                expected_hashes.get(name) == digest
                for name, digest in observed_raw_hashes.items()
            ),
            "missing_state_names": missing_names,
            "extra_state_names": extra_names,
            "mismatched_state_names": mismatched_names,
            "mismatched_state_count": len(mismatched_names),
            "verdict": "FAIL_FLOAT32_MODEL_MANIFEST_MISMATCH",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(diagnostic, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(diagnostic, sort_keys=True, indent=2))
        raise SystemExit(2)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    verified = time.perf_counter()
    receipt = prepare_float64_response_evaluation(
        model=model,
        model_manifest=manifest,
        expected_model_manifest_sha256=manifest_sha256,
    )
    conversion_seconds = time.perf_counter() - verified
    report = {
        "schema_version": "green-v400-model-manifest-precision-smoke-v1",
        "contains_scientific_outcome": False,
        "prompt_universe_loaded": False,
        "direction_payload_loaded": False,
        "endpoint_fields_loaded": False,
        "model_manifest_path": str(args.model_manifest),
        "model_manifest_sha256": manifest_sha256,
        "model_load_seconds": loaded_seconds,
        "manifest_verification_and_float64_conversion_seconds": conversion_seconds,
        "cuda_peak_allocated_bytes_after_reset": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and args.device.startswith("cuda")
            else None
        ),
        "precision_receipt": receipt,
        "verdict": "PASS_FORMAL_FLOAT64_MODEL_MANIFEST_CONTRACT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
