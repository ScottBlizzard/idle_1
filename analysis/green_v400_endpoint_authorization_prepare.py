"""Prepare per-job endpoint authorizations from validated development receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import green_v400_execution_receipts as execution_receipts
from green_v400_execution_receipts import build_endpoint_authorization_receipt

from analysis.green_v400_formal_batch_worker import validate_existing_artifact
from analysis.green_v400_formal_worker import FORMAL_OUTPUT_ROOT, load_json, verify_plan
from analysis.green_v400_phase_ledger import validate_phase_ledger


def _install_process_local_validation_cache() -> None:
    """Cache repeated validation of immutable objects within this process.

    Every endpoint authorization is still built by the frozen receipt builder.
    The builder otherwise re-hashes the identical execution plan, re-validates
    the identical phase ledger, and re-verifies the same typed layer/Grant
    receipts once per endpoint job. These inputs are loaded once and never
    mutated by this prepare-only process, so validating each object once is
    equivalent while avoiding quadratic canonicalization work.
    """

    if getattr(
        execution_receipts, "_endpoint_prepare_validation_cache_installed", False
    ):
        return

    original_plan_hash = execution_receipts._plan_hash
    original_validate_ledger = execution_receipts.validate_phase_ledger
    original_verify_receipt = execution_receipts.verify_receipt
    plan_cache: dict[int, tuple[dict[str, Any], str]] = {}
    ledger_cache: dict[
        tuple[int, int], tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    receipt_cache: dict[
        tuple[int, str], tuple[dict[str, Any], str]
    ] = {}

    def cached_plan_hash(plan: dict[str, Any]) -> str:
        cached = plan_cache.get(id(plan))
        if cached is not None and cached[0] is plan:
            return cached[1]
        digest = original_plan_hash(plan)
        plan_cache[id(plan)] = (plan, digest)
        return digest

    def cached_validate_ledger(
        plan: dict[str, Any], ledger: dict[str, Any]
    ) -> None:
        key = (id(plan), id(ledger))
        cached = ledger_cache.get(key)
        if cached is not None and cached[0] is plan and cached[1] is ledger:
            return
        original_validate_ledger(plan, ledger)
        ledger_cache[key] = (plan, ledger)

    def cached_verify_receipt(receipt: dict[str, Any], schema: str) -> None:
        key = (id(receipt), schema)
        cached = receipt_cache.get(key)
        if cached is not None and cached[0] is receipt:
            return
        original_verify_receipt(receipt, schema)
        receipt_cache[key] = (receipt, schema)

    execution_receipts._plan_hash = cached_plan_hash
    execution_receipts.validate_phase_ledger = cached_validate_ledger
    execution_receipts.verify_receipt = cached_verify_receipt
    execution_receipts._endpoint_prepare_validation_cache_installed = True


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


def _prediction_artifacts(
    *,
    plan: dict[str, Any],
    phase: str,
    prediction_root: Path,
) -> dict[str, dict[str, Any]]:
    queue = plan.get("queues", {}).get(f"{phase}_prediction", [])
    artifacts: dict[str, dict[str, Any]] = {}
    for ordinal, job in enumerate(queue):
        path = prediction_root / f"shard_{ordinal % 4}" / f"{job['job_id']}.json"
        artifact = load_json(path)
        validate_existing_artifact(
            plan=plan, mode="prediction", job=job, artifact=artifact
        )
        site_row_id = job["site_row_id"]
        if site_row_id in artifacts:
            raise ValueError("prediction queue has duplicate site rows")
        artifacts[site_row_id] = artifact
    return artifacts


def prepare_endpoint_authorizations(
    *,
    plan: dict[str, Any],
    phase: str,
    ledger: dict[str, Any],
    universe: dict[str, Any],
    prediction_root: Path,
    grant_receipt_directory: Path,
    replay_receipt_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    verify_plan(plan)
    validate_phase_ledger(plan, ledger)
    _install_process_local_validation_cache()
    if phase not in {"development", "confirmation"}:
        raise ValueError("endpoint phase is invalid")
    if plan.get(f"{phase}_authorized") is not True:
        raise ValueError(f"{phase} endpoint execution is not authorized")
    endpoints = plan.get("queues", {}).get(f"{phase}_endpoint", [])
    if not endpoints:
        raise ValueError("endpoint phase queue is empty")
    predictions = _prediction_artifacts(
        plan=plan, phase=phase, prediction_root=prediction_root
    )
    grant_jobs = plan.get("queues", {}).get(
        f"{phase}_grant_cohort_prediction", []
    )
    grant_receipts = [
        load_json(grant_receipt_directory / f"{job['job_id']}.json")
        for job in grant_jobs
    ]
    replay_receipts = {
        int(layer): load_json(replay_receipt_directory / f"layer_{int(layer):02d}.json")
        for layer in ledger.get("planned_replay_layers", [])
    }
    universe_rows = {row.get("row_id"): row for row in universe.get("rows", [])}
    if len(universe_rows) != len(universe.get("rows", [])):
        raise ValueError("universe contains duplicate row identifiers")
    adapter_source = (
        "src/green_v400_greater_than_response_adapter.py"
        if "GT_REPLICATION" in plan.get("protocol_id", "")
        else "src/green_v400_ioi_response_adapter.py"
    )
    authorizations: dict[str, dict[str, Any]] = {}
    commitments: dict[str, dict[str, Any]] = {}
    for endpoint in endpoints:
        endpoint_job_id = endpoint["job_id"]
        prediction_artifact = predictions.get(endpoint["site_row_id"])
        universe_row = universe_rows.get(endpoint["prompt_row_id"])
        replay_receipt = replay_receipts.get(int(endpoint["layer"]))
        if prediction_artifact is None or universe_row is None or replay_receipt is None:
            raise ValueError("endpoint prerequisite does not resolve exactly once")
        authorization = build_endpoint_authorization_receipt(
            plan=plan,
            endpoint_job_id=endpoint_job_id,
            prediction_packet=prediction_artifact["prediction"],
            prediction_commitment=prediction_artifact["commitment"],
            replay_layer_receipt=replay_receipt,
            phase_ledger=ledger,
            universe_row=universe_row,
            response_adapter_source_path=adapter_source,
            grant_cohort_receipts=grant_receipts,
        )
        authorizations[endpoint_job_id] = authorization
        commitments[endpoint_job_id] = prediction_artifact["commitment"]
    return authorizations, commitments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--phase-ledger", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--grant-receipt-directory", type=Path, required=True)
    parser.add_argument("--replay-receipt-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    formal_root = FORMAL_OUTPUT_ROOT.resolve(strict=True)
    output_parent = args.output_directory.parent.resolve(strict=True)
    output_directory = output_parent / args.output_directory.name
    if not output_directory.is_relative_to(formal_root):
        raise ValueError("endpoint authorization output must remain under /mnt/sdb")
    if output_directory.exists():
        raise FileExistsError(output_directory)
    authorizations, commitments = prepare_endpoint_authorizations(
        plan=load_json(args.plan),
        phase=args.phase,
        ledger=load_json(args.phase_ledger),
        universe=load_json(args.universe),
        prediction_root=args.prediction_root.resolve(strict=True),
        grant_receipt_directory=args.grant_receipt_directory.resolve(strict=True),
        replay_receipt_directory=args.replay_receipt_directory.resolve(strict=True),
    )
    output_directory.mkdir()
    authorization_directory = output_directory / "endpoint_authorizations"
    commitment_directory = output_directory / "prediction_commitments"
    authorization_directory.mkdir()
    commitment_directory.mkdir()
    for job_id in sorted(authorizations):
        _atomic_no_clobber_json(
            authorization_directory / f"{job_id}.json", authorizations[job_id]
        )
        _atomic_no_clobber_json(
            commitment_directory / f"{job_id}.json", commitments[job_id]
        )
    summary = {
        "schema_version": "green-v400-endpoint-authorization-prepare-summary-v1",
        "protocol_id": next(iter(authorizations.values()))["protocol_id"],
        "plan_sha256": next(iter(authorizations.values()))["plan_sha256"],
        "phase": args.phase,
        "endpoint_authorization_count": len(authorizations),
        "phase_ledger_head_sha256": next(iter(authorizations.values()))[
            "phase_ledger_head_sha256"
        ],
        "contains_endpoint_outcome": False,
    }
    _atomic_no_clobber_json(output_directory / "prepare_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
