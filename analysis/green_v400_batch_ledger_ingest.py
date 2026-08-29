"""Validate completed formal batches and derive a hash-chained phase ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from green_v400_execution_receipts import build_grant_cohort_receipt

from analysis.green_v400_formal_batch_worker import (
    select_shard_jobs,
    validate_batch_completion_receipt,
    validate_existing_artifact,
)
from analysis.green_v400_formal_worker import FORMAL_OUTPUT_ROOT, load_json, verify_plan
from analysis.green_v400_phase_ledger import (
    append_phase_event,
    initialize_phase_ledger,
    validate_phase_ledger,
)


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


def _validated_mode_artifacts(
    *,
    plan: dict[str, Any],
    phase: str,
    mode: str,
    root: Path,
    shard_count: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    artifacts: dict[str, dict[str, Any]] = {}
    batch_receipts: dict[str, str] = {}
    for shard_index in range(shard_count):
        shard_directory = root / f"shard_{shard_index}"
        jobs = select_shard_jobs(
            plan,
            mode=mode,
            phase=phase,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        completion_path = shard_directory / (
            f"_completion_{mode}_{phase}_{shard_index:02d}_of_{shard_count:02d}.json"
        )
        completion = load_json(completion_path)
        validate_batch_completion_receipt(
            receipt=completion,
            plan=plan,
            mode=mode,
            phase=phase,
            shard_index=shard_index,
            shard_count=shard_count,
            jobs=jobs,
            output_directory=shard_directory,
        )
        digest = completion["receipt_sha256"]
        for job in jobs:
            job_id = job["job_id"]
            if job_id in artifacts:
                raise ValueError("formal job appears in more than one shard")
            artifact = load_json(shard_directory / f"{job_id}.json")
            validate_existing_artifact(
                plan=plan, mode=mode, job=job, artifact=artifact
            )
            artifacts[job_id] = artifact
            batch_receipts[job_id] = digest
    expected = {
        job["job_id"]
        for job in plan.get("queues", {}).get(
            f"{phase}_{'prediction' if mode == 'prediction' else 'grant_cohort_prediction'}",
            [],
        )
    }
    if set(artifacts) != expected or set(batch_receipts) != expected:
        raise ValueError(f"{mode} batches do not exactly cover the planned phase")
    return artifacts, batch_receipts


def derive_completed_phase_ledger(
    *,
    plan: dict[str, Any],
    phase: str,
    prediction_root: Path,
    grant_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return only validated derived records; do not analyze scientific values."""

    verify_plan(plan)
    if phase not in {"development", "confirmation"}:
        raise ValueError("phase must be development or confirmation")
    if plan.get(f"{phase}_authorized") is not True:
        raise ValueError(f"{phase} is not authorized by the activated plan")
    execution = plan.get("prediction_execution", {})
    shard_count = execution.get("shard_count")
    if shard_count != 4:
        raise ValueError("phase ledger ingestion requires the sealed four-shard topology")
    prediction_artifacts, prediction_batches = _validated_mode_artifacts(
        plan=plan,
        phase=phase,
        mode="prediction",
        root=prediction_root,
        shard_count=shard_count,
    )
    grant_artifacts, grant_batches = _validated_mode_artifacts(
        plan=plan,
        phase=phase,
        mode="grant",
        root=grant_root,
        shard_count=shard_count,
    )
    ledger = initialize_phase_ledger(plan)
    for job_id in sorted(prediction_artifacts):
        artifact = prediction_artifacts[job_id]
        commitment = artifact.get("commitment", {})
        ledger = append_phase_event(
            ledger,
            {
                "plan_sha256": plan["plan_sha256"],
                "previous_ledger_head_sha256": ledger["ledger_head_sha256"],
                "kind": "prediction_committed",
                "phase": phase,
                "job_id": job_id,
                "commitment_sha256": commitment["prediction_packet_sha256"],
                "batch_completion_receipt_sha256": prediction_batches[job_id],
            },
        )
    grant_receipts: dict[str, dict[str, Any]] = {}
    for job_id in sorted(grant_artifacts):
        receipt = build_grant_cohort_receipt(
            plan=plan, artifact=grant_artifacts[job_id]
        )
        grant_receipts[job_id] = receipt
        ledger = append_phase_event(
            ledger,
            {
                "plan_sha256": plan["plan_sha256"],
                "previous_ledger_head_sha256": ledger["ledger_head_sha256"],
                "kind": "grant_cohort_committed",
                "phase": phase,
                "job_id": job_id,
                "receipt_sha256": receipt["receipt_sha256"],
                "batch_completion_receipt_sha256": grant_batches[job_id],
            },
        )
    validate_phase_ledger(plan, ledger)
    return ledger, grant_receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--grant-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    resolved_formal_root = FORMAL_OUTPUT_ROOT.resolve(strict=True)
    output_parent = args.output_directory.parent.resolve(strict=True)
    output_directory = output_parent / args.output_directory.name
    if not output_directory.is_relative_to(resolved_formal_root):
        raise ValueError("ledger output must remain under /mnt/sdb/ccj/iclr_1_runs")
    if output_directory.exists():
        raise FileExistsError(output_directory)
    ledger, grant_receipts = derive_completed_phase_ledger(
        plan=load_json(args.plan),
        phase=args.phase,
        prediction_root=args.prediction_root.resolve(strict=True),
        grant_root=args.grant_root.resolve(strict=True),
    )
    output_directory.mkdir()
    grant_directory = output_directory / "grant_receipts"
    grant_directory.mkdir()
    _atomic_no_clobber_json(output_directory / "phase_ledger.json", ledger)
    for job_id, receipt in sorted(grant_receipts.items()):
        _atomic_no_clobber_json(grant_directory / f"{job_id}.json", receipt)
    summary = {
        "schema_version": "green-v400-batch-ledger-ingest-summary-v1",
        "protocol_id": ledger["protocol_id"],
        "plan_sha256": ledger["plan_sha256"],
        "phase": args.phase,
        "prediction_job_count": len(ledger["completed_prediction_job_ids"]),
        "grant_job_count": len(ledger["completed_grant_job_ids"]),
        "ledger_head_sha256": ledger["ledger_head_sha256"],
        "contains_scientific_outcome": False,
    }
    _atomic_no_clobber_json(output_directory / "ingest_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
