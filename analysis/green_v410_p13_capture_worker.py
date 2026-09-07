"""GPU response-capture worker for one frozen P13 queue shard."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.green_v400_formal_worker import load_frozen_model
from analysis.green_v400_greater_than_universe_prepare import _suffix_token_ids
from green_v400_response_precision import prepare_float64_response_evaluation
from green_v410_artifacts import atomic_no_clobber_bytes, atomic_no_clobber_json, file_sha256
from green_v410_directions import build_row_binding
from green_v410_p13_capture import capture_controlled_hook_t0, compute_response_only_panel
from green_v410_p13_queue import validate_p13_queue
from green_v410_protocol import PROTOCOL_ID, load_protocol_config, sha256_canonical


ALLOWED_GPUS = {"4", "5", "6", "7"}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _npy_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().cpu().contiguous().numpy()
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _runtime_preflight(device: str) -> None:
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") not in ALLOWED_GPUS:
        raise RuntimeError("P13 capture must expose exactly one physical GPU in 4 through 7")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("P13 capture requires deterministic CUBLAS workspace")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise RuntimeError("P13 capture requires frozen single-thread libraries")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _site_adapter(task: str, model: Any, prompt: dict[str, Any], layer: int):
    if task == "ioi":
        from green_v400_ioi_response_adapter import (
            IOIInterventionSite,
            build_matched_bypass_four_branch_responses,
        )
        site = IOIInterventionSite(
            layer=layer,
            position=int(prompt["signature"][1]),
            io_token_id=int(prompt["io_token_id"]),
            s_token_id=int(prompt["s_token_id"]),
        )
        metric = {
            "suffix_ids": np.asarray(
                [site.io_token_id, site.s_token_id], dtype=np.int64
            ),
            "coefficients": np.asarray([1.0, -1.0], dtype=np.float64),
            "rationals": ((1, 1), (-1, 1)),
        }
    elif task == "greater_than":
        from green_v400_greater_than_response_adapter import (
            GreaterThanInterventionSite,
            build_matched_bypass_four_branch_responses,
            greater_than_contrast,
        )
        suffix_ids = tuple(_suffix_token_ids(model.tokenizer))
        site = GreaterThanInterventionSite(
            layer=layer,
            position=int(prompt["site_position"]),
            clean_suffix=int(prompt["y"]),
            suffix_token_ids=suffix_ids,
        )
        coefficients = greater_than_contrast(
            site.clean_suffix, dtype=torch.float64, device=torch.device("cpu")
        ).numpy()
        metric = {
            "suffix_ids": np.asarray(suffix_ids, dtype=np.int64),
            "coefficients": np.asarray(coefficients, dtype=np.float64),
            "rationals": tuple(
                (-1, site.clean_suffix + 1) if index <= site.clean_suffix
                else (1, 99 - site.clean_suffix)
                for index in range(100)
            ),
        }
    else:
        raise ValueError("unknown P13 capture task")
    return site, build_matched_bypass_four_branch_responses, metric


def _capture_job(
    *,
    job: dict[str, Any],
    queue: dict[str, Any],
    model: Any,
    prompt: dict[str, Any],
    directions: np.ndarray,
    output_root: Path,
) -> None:
    final = output_root / job["job_id"] / "capture_manifest.json"
    if final.exists():
        existing = _load(final)
        if (
            existing.get("job_id") != job["job_id"]
            or existing.get("queue_manifest_sha256") != queue["queue_manifest_sha256"]
            or existing.get("manifest_sha256")
                != sha256_canonical({k: v for k, v in existing.items() if k != "manifest_sha256"})
        ):
            raise ValueError("conflicting P13 capture artifact")
        return

    device = next(model.parameters()).device
    clean_tokens = torch.tensor([prompt["clean_token_ids"]], dtype=torch.long, device=device)
    corrupt_tokens = torch.tensor([prompt["corrupt_token_ids"]], dtype=torch.long, device=device)
    site, branch_builder, metric = _site_adapter(
        queue["task"], model, prompt, int(job["layer"])
    )
    if site.position != job["site_position"] or site.hook_name != f"blocks.{job['layer']}.hook_resid_post":
        raise ValueError("P13 capture adapter identity mismatch")
    center, pat, tar = capture_controlled_hook_t0(
        model=model,
        clean_tokens=clean_tokens,
        corrupt_tokens=corrupt_tokens,
        hook_name=site.hook_name,
        site_position=site.position,
    )
    branches = branch_builder(
        model,
        clean_tokens,
        corrupt_tokens,
        site,
        center,
        selected_gates=tuple(load_protocol_config()["gate"]["selected_indices"]),
    )
    direction_tensor = torch.from_numpy(
        np.array(directions, dtype="<f4", copy=True)
    ).to(device=device, dtype=torch.float64)
    response = compute_response_only_panel(
        branches=branches, center=center, directions=direction_tensor
    )

    directory = final.parent
    publications = {}
    for name, tensor in (
        ("clean_center.npy", center),
        ("pat_controlled_hook_t0.npy", pat),
        ("tar_controlled_hook_t0.npy", tar),
    ):
        publications[name] = atomic_no_clobber_bytes(
            directory / name,
            _npy_bytes(tensor),
            job_id=f"{job['job_id']}-{name}",
        )["file_sha256"]
    metric_payload = {
        "suffix_token_ids": metric["suffix_ids"].tolist(),
        "contrast_coefficients": metric["coefficients"].tolist(),
        "contrast_coefficient_rationals": [list(value) for value in metric["rationals"]],
    }
    manifest = {
        "schema_version": "green-v410-p13-response-capture-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": queue["task"],
        "phase": "qualification",
        "job_id": job["job_id"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "prompt_row_id": job["prompt_row_id"],
        "site_row_id": job["site_row_id"],
        "layer": job["layer"],
        "site_position": job["site_position"],
        "clean_tokens_sha256": sha256_canonical(prompt["clean_token_ids"]),
        "corrupt_tokens_sha256": sha256_canonical(prompt["corrupt_token_ids"]),
        "payload_file_sha256s": publications,
        "direction_binding_sha256": job["direction_panel"][0]["direction_binding_sha256"],
        "response_panel": response,
        "task_metric": metric_payload,
        "model_manifest_sha256": job["model_manifest_sha256"],
        "tokenizer_manifest_sha256": job["tokenizer_manifest_sha256"],
        "selected_gate_spec_sha256": job["selected_gate_spec_sha256"],
        "contains_endpoint_material": False,
    }
    manifest["manifest_sha256"] = sha256_canonical(manifest)
    atomic_no_clobber_json(final, manifest, job_id=f"{job['job_id']}-capture-manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    _runtime_preflight(args.device)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("P13 capture shard coordinates are invalid")
    queue = _load(args.queue)
    validate_p13_queue(queue)
    model_manifest = _load(args.model_manifest)
    if sha256_canonical(model_manifest) != queue["model_manifest_sha256"]:
        raise ValueError("P13 capture model manifest differs from queue")
    registry = _load(args.prepare_root / f"{queue['task']}_p13_direction_registry.json")
    qualification = _load(args.prepare_root / f"{queue['task']}_p13_qualification_manifest.json")
    payload_path = args.prepare_root / f"{queue['task']}_p13_directions.npy"
    if (
        registry.get("registry_sha256") != queue["direction_registry_sha256"]
        or file_sha256(payload_path) != queue["direction_payload_file_sha256"]
        or qualification.get("manifest_sha256") != queue["qualification_manifest_sha256"]
    ):
        raise ValueError("P13 capture prepare artifacts differ from queue")
    payload = np.load(payload_path, mmap_mode="r", allow_pickle=False)
    bindings = {row["site_row_id"]: row for row in registry["bindings"]}
    prompts = {row["row_id"]: row for row in registry["selected_prompts"]}

    model = load_frozen_model(model_manifest, args.device)
    prepare_float64_response_evaluation(
        model=model,
        model_manifest=model_manifest,
        expected_model_manifest_sha256=queue["model_manifest_sha256"],
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    jobs = [
        job for ordinal, job in enumerate(queue["jobs"])
        if ordinal % args.shard_count == args.shard_index
    ]
    for job in jobs:
        binding = bindings[job["site_row_id"]]
        array = np.asarray(payload[int(binding["row_index"])], dtype="<f4")
        rebuilt = build_row_binding(
            task=queue["task"],
            site_row_id=job["site_row_id"],
            direction_domain=registry["direction_domain"] if "direction_domain" in registry
                else qualification["direction_domain"],
            array=array,
        )
        if rebuilt != {key: value for key, value in binding.items() if key != "row_index"}:
            raise ValueError("P13 capture direction binding changed")
        _capture_job(
            job=job,
            queue=queue,
            model=model,
            prompt=prompts[job["prompt_row_id"]],
            directions=array,
            output_root=args.output_root,
        )
    completion = {
        "schema_version": "green-v410-p13-capture-shard-completion-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": queue["task"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "completed_job_ids": [job["job_id"] for job in jobs],
        "completed_job_count": len(jobs),
        "contains_endpoint_material": False,
    }
    completion["receipt_sha256"] = sha256_canonical(completion)
    atomic_no_clobber_json(
        args.output_root / f"capture_shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json",
        completion,
        job_id=f"p13-capture-{queue['task']}-{args.shard_index}",
    )


if __name__ == "__main__":
    main()
