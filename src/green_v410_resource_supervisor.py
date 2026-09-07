"""Serial, resumable supervisor for the frozen GREEN v4.1 calibration queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from green_v410_artifacts import atomic_no_clobber_json
from green_v410_protocol import PROTOCOL_ID, sha256_canonical
from green_v410_resource_calibration import (
    CANDIDATES,
    FIXTURE_KINDS,
    PRECISIONS,
    PROFILES,
    build_minimum_budget_failfast_receipt,
)
from green_v410_resource_finalize import _load_raw, raw_artifact_path


MAX_COLD_PROCESSES_PER_PHASE = len(PROFILES) * len(FIXTURE_KINDS)
DEFAULT_MAX_WORKERS = 8


def build_resource_queue(
    bundle_root: Path, raw_root: Path, max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict:
    if type(max_workers) is not int or not 1 <= max_workers <= MAX_COLD_PROCESSES_PER_PHASE:
        raise ValueError("resource calibration max_workers is out of range")
    jobs = []
    ordinal = 0
    for candidate in CANDIDATES:
        # Phase-major order is binding: every official child precedes every
        # audit child for the same candidate.
        for precision in PRECISIONS:
            for profile in PROFILES:
                for fixture in FIXTURE_KINDS:
                    mode = "official" if precision == 384 else "audit"
                    jobs.append({
                        "ordinal": ordinal,
                        "candidate_leaf_budget": candidate,
                        "precision_bits": precision,
                        "mode": mode,
                        "profile": profile,
                        "fixture_kind": fixture,
                        "bundle_dir": (
                            bundle_root / profile.replace(":", "__") / fixture
                        ).as_posix(),
                        "output_path": raw_artifact_path(
                            raw_root, candidate, precision, profile, fixture
                        ).as_posix(),
                        "official_artifact_path": (
                            None if precision == 384 else raw_artifact_path(
                                raw_root, candidate, 384, profile, fixture
                            ).as_posix()
                        ),
                    })
                    ordinal += 1
    payload = {
        "schema_version": "green-v410-resource-calibration-queue-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "scheduler": "phase_major_bounded_parallel_independent_cold_processes_v1",
        "max_workers": max_workers,
        "job_count": len(jobs),
        "jobs": jobs,
        "contains_scientific_outcome": False,
        "contains_endpoint_material": False,
    }
    payload["queue_sha256"] = sha256_canonical(payload)
    return payload


def _publish_failfast_stop(raw_root: Path, payload: dict) -> None:
    atomic_no_clobber_json(
        raw_root / "resource_failfast_stop.json",
        payload,
        job_id="resource-minimum-budget-failfast-stop",
    )


def _worker_command(job: dict, backend: Path) -> list[str]:
    command = [
        sys.executable, "-m", "green_v410_resource_worker",
        "--mode", job["mode"], "--candidate", str(job["candidate_leaf_budget"]),
        "--profile", job["profile"], "--fixture", job["fixture_kind"],
        "--bundle-dir", job["bundle_dir"], "--backend", str(backend),
        "--output", job["output_path"],
    ]
    if job["official_artifact_path"] is not None:
        command.extend(["--official-artifact", job["official_artifact_path"]])
    return command


def _run_job(job: dict, backend: Path) -> subprocess.CompletedProcess:
    return subprocess.run(_worker_command(job, backend), check=False)


def run_supervisor(
    bundle_root: Path, backend: Path, raw_root: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> str:
    queue = build_resource_queue(bundle_root, raw_root, max_workers=max_workers)
    atomic_no_clobber_json(
        raw_root / "queue.json", queue, job_id="resource-calibration-queue"
    )
    # Candidate-major and precision-major barriers preserve the binding
    # official-before-audit order.  Rows inside one phase are independent
    # cold processes, so they may run concurrently and publish canonically to
    # disjoint no-clobber paths.
    for candidate in CANDIDATES:
        for precision in PRECISIONS:
            phase_jobs = [
                job for job in queue["jobs"]
                if job["candidate_leaf_budget"] == candidate
                and job["precision_bits"] == precision
            ]
            missing = []
            for job in phase_jobs:
                output = Path(job["output_path"])
                if not output.exists():
                    missing.append(job)
                    continue
                record = _load_raw(
                    output, candidate=candidate, precision=precision,
                    profile=job["profile"], fixture=job["fixture_kind"],
                )
                stop = build_minimum_budget_failfast_receipt(record)
                if stop is not None:
                    _publish_failfast_stop(raw_root, stop)
                    return stop["decision"]

            if not missing:
                continue
            stop_receipt = None
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(missing)),
                thread_name_prefix="green-v410-resource-cold",
            ) as pool:
                futures = {pool.submit(_run_job, job, backend): job for job in missing}
                for future in as_completed(futures):
                    job = futures[future]
                    completed = future.result()
                    if completed.returncode != 0:
                        raise RuntimeError(
                            "resource calibration child failed at ordinal "
                            f"{job['ordinal']}"
                        )
                    record = _load_raw(
                        Path(job["output_path"]), candidate=candidate,
                        precision=precision, profile=job["profile"],
                        fixture=job["fixture_kind"],
                    )
                    stop = build_minimum_budget_failfast_receipt(record)
                    if stop is not None:
                        stop_receipt = stop
            if stop_receipt is not None:
                _publish_failfast_stop(raw_root, stop_receipt)
                return stop_receipt["decision"]
    return "CALIBRATION_QUEUE_COMPLETE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        choices=range(1, MAX_COLD_PROCESSES_PER_PHASE + 1),
    )
    args = parser.parse_args()
    for path in (args.bundle_root, args.backend, args.raw_root):
        if not path.resolve().as_posix().startswith("/mnt/sdb/"):
            raise RuntimeError("formal server calibration paths must be under /mnt/sdb")
    run_supervisor(
        args.bundle_root, args.backend, args.raw_root,
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
