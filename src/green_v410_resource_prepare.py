"""Prepare immutable actual-shape/synthetic-only GREEN v4.1 graph bundles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from green_v410_artifacts import atomic_no_clobber_json, file_sha256
from green_v410_gpt2_program import (
    build_green_v410_full_cone_program,
    materialize_green_v410_full_cone_store,
)
from green_v410_protocol import PROTOCOL_ID, sha256_canonical
from green_v410_resource_calibration import (
    FIXTURE_KINDS,
    PROFILES,
    generate_synthetic_fixture,
    load_resource_calibration_config,
)


def _safe_profile(profile: str) -> str:
    return profile.replace(":", "__")


def prepare_resource_graph_bundles(output_root: Path, model, *,
                                   model_manifest_hash: str) -> dict:
    if len(model_manifest_hash) != 64 or any(
        character not in "0123456789abcdef" for character in model_manifest_hash
    ):
        raise ValueError("resource model manifest hash is invalid")
    config = load_resource_calibration_config()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_rows = []
    for profile in PROFILES:
        task, _ = profile.split(":", 1)
        metric = config["synthetic_task_metrics"][task]
        suffix_ids = np.asarray(metric["suffix_token_ids"], dtype="<i8")
        rationals = tuple(tuple(value) for value in metric["coefficient_rationals"])
        coefficients = np.asarray(
            [numerator / denominator for numerator, denominator in rationals],
            dtype="<f8",
        )
        for fixture_kind in FIXTURE_KINDS:
            fixture = generate_synthetic_fixture(profile, fixture_kind)
            bundle_dir = output_root / _safe_profile(profile) / fixture_kind
            bundle_dir.mkdir(parents=True, exist_ok=True)
            existing_bundle_path = bundle_dir / "bundle.json"
            if existing_bundle_path.exists():
                from green_v410_resource_worker import _load_bundle
                existing, existing_program, _ = _load_bundle(
                    bundle_dir, profile, fixture_kind
                )
                if existing_program.model_manifest_hash != model_manifest_hash:
                    raise ValueError("existing resource bundle model identity mismatch")
                bundle_rows.append({
                    "profile": profile,
                    "fixture_kind": fixture_kind,
                    "relative_path": bundle_dir.relative_to(output_root).as_posix(),
                    "bundle_sha256": existing["bundle_sha256"],
                })
                continue
            reader, dims = materialize_green_v410_full_cone_store(
                bundle_dir,
                "tensor_store",
                model,
                site_layer=fixture.site_layer,
                site_position=fixture.site_position,
                pat_controlled_hook_t0=fixture.pat_controlled_hook_t0,
                tar_controlled_hook_t0=fixture.tar_controlled_hook_t0,
                physical_direction=fixture.physical_direction,
                suffix_token_ids=suffix_ids,
                contrast_coefficients=coefficients,
                contrast_coefficient_rationals=rationals,
                selected_gates=config["selected_gates"],
            )
            program = build_green_v410_full_cone_program(
                reader, model_manifest_hash, dims
            )
            program_path = bundle_dir / "program.json"
            atomic_no_clobber_json(
                program_path, program.to_dict(),
                job_id=f"prepare-{_safe_profile(profile)}-{fixture_kind}-program",
            )
            store_path = bundle_dir / "tensor_store.json"
            bundle = {
                "schema_version": "green-v410-resource-graph-bundle-v1",
                "protocol_id": PROTOCOL_ID,
                "profile": profile,
                "fixture_kind": fixture_kind,
                "fixture_sha256": fixture.semantic_hash(),
                "program_file": program_path.name,
                "program_file_sha256": file_sha256(program_path),
                "program_semantic_hash": program.semantic_hash(),
                "tensor_store_manifest_file": store_path.name,
                "tensor_store_manifest_file_sha256": file_sha256(store_path),
                "tensor_store_record_closure_sha256": (
                    reader.manifest.record_closure_sha256
                ),
                "contains_scientific_outcome": False,
                "contains_endpoint_material": False,
            }
            bundle["bundle_sha256"] = sha256_canonical(bundle)
            atomic_no_clobber_json(
                bundle_dir / "bundle.json", bundle,
                job_id=f"prepare-{_safe_profile(profile)}-{fixture_kind}-bundle",
            )
            bundle_rows.append({
                "profile": profile,
                "fixture_kind": fixture_kind,
                "relative_path": bundle_dir.relative_to(output_root).as_posix(),
                "bundle_sha256": bundle["bundle_sha256"],
            })
    manifest = {
        "schema_version": "green-v410-resource-graph-bundle-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "model_manifest_hash": model_manifest_hash,
        "resource_calibration_config_sha256": sha256_canonical(config),
        "bundle_count": len(bundle_rows),
        "bundles": bundle_rows,
        "contains_scientific_outcome": False,
        "contains_endpoint_material": False,
    }
    manifest["manifest_sha256"] = sha256_canonical(manifest)
    atomic_no_clobber_json(
        output_root / "manifest.json", manifest, job_id="resource-graph-manifest"
    )
    return manifest


def _load_model(cache_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    config = load_resource_calibration_config()["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        config["id"], revision=config["revision"], cache_dir=cache_dir,
        local_files_only=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        config["id"], revision=config["revision"], cache_dir=cache_dir,
        local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    hf_model.config.use_cache = False
    model = HookedTransformer.from_pretrained_no_processing(
        "gpt2", hf_model=hf_model, tokenizer=tokenizer, device="cpu",
        dtype=torch.float32, default_prepend_bos=False,
    ).eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_root.resolve().as_posix().startswith("/mnt/sdb/"):
        raise RuntimeError("formal server output root must be under /mnt/sdb")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = _load_model(args.model_cache)
    prepare_resource_graph_bundles(
        args.output_root, model, model_manifest_hash=file_sha256(args.model_manifest),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
