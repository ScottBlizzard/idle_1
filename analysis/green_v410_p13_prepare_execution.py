"""Freeze the P13 runtime closure and publish the two immutable queues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from green_v410_artifacts import atomic_no_clobber_json
from green_v410_p13_queue import build_p13_queue, validate_p13_queue
from green_v410_source_bundle import build_source_bundle_manifest


RUNTIME_SEEDS = (
    "analysis/green_v410_p13_capture_worker.py",
    "analysis/green_v410_p13_materialize_worker.py",
    "analysis/green_v410_p13_certificate_worker.py",
    "analysis/green_v410_p13_finalize.py",
    "analysis/green_v410_p13_prepare_execution.py",
)
FROZEN_INPUTS = (
    "analysis/GPTPRO_GREEN_V4_SUCCESSOR_CERTIFICATE_PROTOCOL_DECISION_20260830.md",
    "configs/green_v410_sfc_jwtec_protocol.json",
    "configs/green_v410_resource_calibration.json",
    "native/green_v400_mpfr_backend.cpp",
    "native/green_v400_native_plan_loader.cpp",
    "scripts/build_green_v400_mpfr_backend.sh",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-seal", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--compiled-backend", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = build_source_bundle_manifest(
        root=ROOT,
        seeds=[ROOT / value for value in RUNTIME_SEEDS],
        frozen_inputs=[ROOT / value for value in FROZEN_INPUTS],
    )
    resource = _load(args.resource_manifest)
    seal = _load(args.confirmation_seal)
    model = _load(args.model_manifest)
    if not args.compiled_backend.is_file():
        raise FileNotFoundError("compiled MPFR backend is missing")
    from green_v410_artifacts import file_sha256
    compiled_backend_sha256 = file_sha256(args.compiled_backend)
    queues = {}
    for task in ("ioi", "greater_than"):
        queue = build_p13_queue(
            task=task, prepare_root=args.prepare_root,
            resource_manifest=resource, confirmation_seal=seal,
            model_manifest=model, source_bundle_sha256=source["bundle_sha256"],
            compiled_backend_sha256=compiled_backend_sha256,
        )
        validate_p13_queue(queue)
        queues[task] = queue
    atomic_no_clobber_json(
        args.output_root / "source_bundle_manifest.json", source,
        job_id="p13-source-bundle",
    )
    for task, queue in queues.items():
        atomic_no_clobber_json(
            args.output_root / f"{task}_p13_queue.json", queue,
            job_id=f"p13-{task}-queue",
        )
    print(source["bundle_sha256"])


if __name__ == "__main__":
    main()
