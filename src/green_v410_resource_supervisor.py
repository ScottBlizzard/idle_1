"""Serial, resumable supervisor for the frozen GREEN v4.1 calibration queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from green_v410_artifacts import atomic_no_clobber_json
from green_v410_protocol import PROTOCOL_ID, sha256_canonical
from green_v410_resource_calibration import CANDIDATES, FIXTURE_KINDS, PRECISIONS, PROFILES
from green_v410_resource_finalize import _load_raw, raw_artifact_path


def build_resource_queue(bundle_root: Path, raw_root: Path) -> dict:
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
        "scheduler": "strict_serial_one_cold_process_at_a_time",
        "job_count": len(jobs),
        "jobs": jobs,
        "contains_scientific_outcome": False,
        "contains_endpoint_material": False,
    }
    payload["queue_sha256"] = sha256_canonical(payload)
    return payload


def run_supervisor(bundle_root: Path, backend: Path, raw_root: Path) -> None:
    queue = build_resource_queue(bundle_root, raw_root)
    atomic_no_clobber_json(
        raw_root / "queue.json", queue, job_id="resource-calibration-queue"
    )
    for job in queue["jobs"]:
        candidate = job["candidate_leaf_budget"]
        precision = job["precision_bits"]
        profile = job["profile"]
        fixture = job["fixture_kind"]
        output = Path(job["output_path"])
        if output.exists():
            _load_raw(
                output, candidate=candidate, precision=precision,
                profile=profile, fixture=fixture,
            )
            continue
        command = [
            sys.executable, "-m", "green_v410_resource_worker",
            "--mode", job["mode"], "--candidate", str(candidate),
            "--profile", profile, "--fixture", fixture,
            "--bundle-dir", job["bundle_dir"], "--backend", str(backend),
            "--output", str(output),
        ]
        if job["official_artifact_path"] is not None:
            command.extend(["--official-artifact", job["official_artifact_path"]])
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"resource calibration child failed at ordinal {job['ordinal']}"
            )
        _load_raw(
            output, candidate=candidate, precision=precision,
            profile=profile, fixture=fixture,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.bundle_root, args.backend, args.raw_root):
        if not path.resolve().as_posix().startswith("/mnt/sdb/"):
            raise RuntimeError("formal server calibration paths must be under /mnt/sdb")
    run_supervisor(args.bundle_root, args.backend, args.raw_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
