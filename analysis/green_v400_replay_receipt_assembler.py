"""Assemble independent replay workers into typed layer receipts and a ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from green_v400_endpoint_calibration import merge_target_replay_stability
from green_v400_execution_receipts import (
    build_numerical_replay_layer_receipt,
    validate_model_session_for_plan,
)

from analysis.green_v400_formal_worker import FORMAL_OUTPUT_ROOT, load_json, verify_plan
from analysis.green_v400_phase_ledger import append_phase_event, validate_phase_ledger


def _atomic_no_clobber_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _worker_path(root: Path, ordinal: int, job_id: str, replay_id: str) -> Path:
    return root / f"shard_{ordinal % 4}" / f"{job_id}_{replay_id}.json"


def _validated_worker_artifact(
    *, plan: dict[str, Any], job_id: str, replay_id: str, path: Path
) -> dict[str, Any]:
    artifact = load_json(path)
    if (
        set(artifact)
        != {
            "schema_version",
            "job_id",
            "replay",
            "commitment",
            "model_session",
            "worker",
        }
        or artifact.get("schema_version")
        != "green-v400-formal-replay-worker-artifact-v1"
        or artifact.get("job_id") != job_id
        or artifact.get("replay", {}).get("replay_id_private") != replay_id
    ):
        raise ValueError("formal replay worker artifact schema or identity is invalid")
    validate_model_session_for_plan(artifact["model_session"], plan)
    worker = artifact["worker"]
    session = artifact["model_session"]
    if (
        worker.get("pid") != session.get("pid")
        or worker.get("process_start_nonce") != session.get("process_start_nonce")
    ):
        raise ValueError("replay worker process identity differs from model session")
    return artifact


def _load_replay_pair(
    *, plan: dict[str, Any], replay_root: Path, ordinal: int, job_id: str
) -> dict[str, Any]:
    a = _validated_worker_artifact(
        plan=plan,
        job_id=job_id,
        replay_id="A",
        path=_worker_path(replay_root, ordinal, job_id, "A"),
    )
    b = _validated_worker_artifact(
        plan=plan,
        job_id=job_id,
        replay_id="B",
        path=_worker_path(replay_root, ordinal, job_id, "B"),
    )
    gate, gate_commitment = merge_target_replay_stability(
        a["replay"], a["commitment"], b["replay"], b["commitment"]
    )
    return {
        "job_id": job_id,
        "replay_a": a["replay"],
        "commitment_a": a["commitment"],
        "worker_a": a["worker"],
        "replay_b": b["replay"],
        "commitment_b": b["commitment"],
        "worker_b": b["worker"],
        "gate": gate,
        "gate_commitment": gate_commitment,
    }


def assemble_replay_receipts(
    *, plan: dict[str, Any], ledger: dict[str, Any], replay_root: Path
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    verify_plan(plan)
    validate_phase_ledger(plan, ledger)
    if plan.get("development_authorized") is not True:
        raise ValueError("numerical replay requires development authorization")
    if ledger.get("numerical_replay_receipt_sha256"):
        raise ValueError("input ledger already records numerical replay receipts")
    if set(ledger.get("completed_prediction_job_ids", [])) != set(
        ledger.get("planned_prediction_job_ids", {}).get("development", [])
    ) or set(ledger.get("completed_grant_job_ids", [])) != set(
        ledger.get("planned_grant_job_ids", {}).get("development", [])
    ):
        raise ValueError("numerical replay assembly requires complete development batches")
    jobs = plan.get("queues", {}).get("endpoint_numerical_replay", [])
    if not jobs:
        raise ValueError("plan has no numerical replay jobs")
    combined = {
        job["job_id"]: _load_replay_pair(
            plan=plan,
            replay_root=replay_root,
            ordinal=ordinal,
            job_id=job["job_id"],
        )
        for ordinal, job in enumerate(jobs)
    }
    layers = sorted({int(job["layer"]) for job in jobs})
    receipts: dict[int, dict[str, Any]] = {}
    updated = ledger
    for layer in layers:
        layer_jobs = [
            combined[job["job_id"]] for job in jobs if int(job["layer"]) == layer
        ]
        receipt = build_numerical_replay_layer_receipt(
            plan=plan, layer=layer, replay_artifacts=layer_jobs
        )
        receipts[layer] = receipt
        updated = append_phase_event(
            updated,
            {
                "plan_sha256": plan["plan_sha256"],
                "previous_ledger_head_sha256": updated["ledger_head_sha256"],
                "kind": "numerical_replay_layer_complete",
                "layer": layer,
                "receipt_sha256": receipt["receipt_sha256"],
            },
        )
    validate_phase_ledger(plan, updated)
    return updated, receipts, combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase-ledger", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    formal_root = FORMAL_OUTPUT_ROOT.resolve(strict=True)
    output_parent = args.output_directory.parent.resolve(strict=True)
    output_directory = output_parent / args.output_directory.name
    if not output_directory.is_relative_to(formal_root):
        raise ValueError("replay receipt output must remain under /mnt/sdb")
    if output_directory.exists():
        raise FileExistsError(output_directory)
    updated, receipts, combined = assemble_replay_receipts(
        plan=load_json(args.plan),
        ledger=load_json(args.phase_ledger),
        replay_root=args.replay_root.resolve(strict=True),
    )
    output_directory.mkdir()
    layer_directory = output_directory / "layer_receipts"
    combined_directory = output_directory / "combined_replay_artifacts"
    layer_directory.mkdir()
    combined_directory.mkdir()
    _atomic_no_clobber_json(output_directory / "phase_ledger.json", updated)
    for layer, receipt in sorted(receipts.items()):
        _atomic_no_clobber_json(
            layer_directory / f"layer_{layer:02d}.json", receipt
        )
    for job_id, artifact in sorted(combined.items()):
        _atomic_no_clobber_json(combined_directory / f"{job_id}.json", artifact)
    summary = {
        "schema_version": "green-v400-replay-receipt-assembly-summary-v1",
        "protocol_id": updated["protocol_id"],
        "plan_sha256": updated["plan_sha256"],
        "replay_job_count": len(combined),
        "layer_receipt_count": len(receipts),
        "ledger_head_sha256": updated["ledger_head_sha256"],
        "all_replays_stable": True,
        "contains_scientific_outcome": False,
    }
    _atomic_no_clobber_json(output_directory / "assembly_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
