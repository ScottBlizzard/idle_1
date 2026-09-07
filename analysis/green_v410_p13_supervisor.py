"""Resume-safe, outcome-blind materialization and certificate fleet supervisor."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from green_v410_artifacts import atomic_no_clobber_json, configure_exact_integer_io
from green_v410_p13_runtime import (
    load_object, progress_counts, valid_capture, valid_certificate_record,
    valid_graph, verify_runtime_closure,
    require_unstarted_certificate_unit,
)
from green_v410_protocol import PROTOCOL_ID, sha256_canonical


TRANSIENT_MATERIALIZE_MARKERS = (
    "resource temporarily unavailable", "input/output error", "stale file handle",
)


def _mutable_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _environment(source_root: Path) -> dict[str, str]:
    value = dict(os.environ)
    value.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
        "PYTHONINTMAXSTRDIGITS": "0",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join((
            str(source_root / "src"), str(source_root),
            "/mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages",
        )),
    })
    return value


def _transient(returncode: int, log_path: Path) -> bool:
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:].lower()
    except OSError:
        return False
    return any(marker in tail for marker in TRANSIENT_MATERIALIZE_MARKERS)


def _run_fleet(
    *, units: list[tuple[str, list[str]]], parallelism: int,
    log_root: Path, status_path: Path, environment: dict[str, str],
    retry_delay_seconds: int,
    allow_transient_retries: bool,
) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(units)
    active: dict[str, tuple[subprocess.Popen, Any, Path, list[str]]] = {}
    complete = 0
    retries: dict[str, int] = {}
    failures: list[str] = []
    # Keep the supervisor lock while already-launched children finish.  A
    # failed child stops NEW work, not the accounting of its live siblings.
    while active or (pending and not failures):
        while pending and not failures and len(active) < parallelism:
            unit_id, command = pending.pop(0)
            attempt = retries.get(unit_id, 0) + 1
            log_path = log_root / f"{unit_id}.attempt_{attempt:03d}.log"
            handle = log_path.open("ab", buffering=0)
            try:
                process = subprocess.Popen(
                    command, stdout=handle, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=environment,
                    start_new_session=True,
                )
            except OSError as exc:
                handle.close()
                failures.append(f"WORKER_LAUNCH_FAILED unit={unit_id} error={exc}")
                break
            active[unit_id] = (process, handle, log_path, command)
        changed = False
        for unit_id, (process, handle, log_path, command) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            changed = True
            handle.close()
            del active[unit_id]
            if returncode == 0:
                complete += 1
            elif allow_transient_retries and _transient(returncode, log_path):
                retries[unit_id] = retries.get(unit_id, 0) + 1
                pending.append((unit_id, command))
                time.sleep(retry_delay_seconds)
            else:
                failures.append(
                    f"HARD_WORKER_FAILURE unit={unit_id} code={returncode} log={log_path}"
                )
        _mutable_status(status_path, {
            "state": "FAILED_DRAINING" if failures and active else (
                "FAILED" if failures else "RUNNING"),
            "total_units": len(units), "completed_units": complete,
            "active_units": sorted(active), "pending_units": len(pending),
            "active_pids": {key: row[0].pid for key, row in active.items()},
            "failures": failures,
            "transient_retries": retries, "updated_unix_seconds": time.time(),
        })
        if not changed:
            time.sleep(5)
    if failures:
        raise RuntimeError("; ".join(failures))


def _materialize_units(args: argparse.Namespace, queue: dict[str, Any]) -> list[tuple[str, list[str]]]:
    capture_root = args.run_root / "capture" / args.task
    for job in queue["jobs"]:
        if not valid_capture(
            capture_root / job["job_id"] / "capture_manifest.json",
            queue_sha256=queue["queue_manifest_sha256"], job_id=job["job_id"],
        ):
            raise ValueError(f"missing or invalid formal capture: {job['job_id']}")
    graph_root = args.run_root / "graphs" / args.task
    units = []
    for shard in range(args.materialize_shards):
        receipt = graph_root / f"materialize_shard_{shard:02d}_of_{args.materialize_shards:02d}.json"
        if receipt.is_file():
            continue
        units.append((f"shard_{shard:02d}", [
            args.python, str(args.source_root / "analysis" / "green_v410_p13_materialize_worker.py"),
            "--queue", str(args.queue), "--prepare-root", str(args.prepare_root),
            "--capture-root", str(capture_root), "--graph-root", str(graph_root),
            "--model-manifest", str(args.model_manifest), "--shard-index", str(shard),
            "--shard-count", str(args.materialize_shards),
        ]))
    return units


def _certificate_units(args: argparse.Namespace, queue: dict[str, Any]) -> list[tuple[str, list[str]]]:
    graph_root = args.run_root / "graphs" / args.task
    output_root = args.run_root / "certificates" / args.task
    units = []
    for job in queue["jobs"]:
        for direction in job["direction_panel"]:
            ordinal = direction["direction_ordinal"]
            graph = graph_root / job["job_id"] / f"direction_{ordinal:02d}" / "graph_manifest.json"
            if not valid_graph(
                graph, site_row_id=job["site_row_id"], direction_ordinal=ordinal,
                direction_payload_sha256=direction["direction_payload_sha256"],
            ):
                raise ValueError(f"missing or invalid formal graph: {job['job_id']}:{ordinal}")
            record = output_root / job["job_id"] / f"direction_{ordinal:02d}" / "p13_record.json"
            if valid_certificate_record(
                record, queue_sha256=queue["queue_manifest_sha256"],
                job_id=job["job_id"], site_row_id=job["site_row_id"],
                direction_ordinal=ordinal,
            ):
                continue
            unit_id = f"{job['job_id']}_{ordinal:02d}"
            require_unstarted_certificate_unit(
                args.run_root / "logs" / "certificate" / args.task, unit_id,
            )
            units.append((unit_id, [
                args.python, str(args.source_root / "analysis" / "green_v410_p13_certificate_worker.py"),
                "--queue", str(args.queue), "--capture-root", str(args.run_root / "capture" / args.task),
                "--graph-root", str(graph_root), "--backend", str(args.compiled_backend),
                "--job-id", job["job_id"], "--direction-ordinal", str(ordinal),
                "--output-root", str(output_root),
            ]))
    return units


def main() -> None:
    configure_exact_integer_io()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("materialize", "certificate"), required=True)
    parser.add_argument("--task", choices=("ioi", "greater_than"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--compiled-backend", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--parallelism", type=int, required=True)
    parser.add_argument("--materialize-shards", type=int, default=16)
    parser.add_argument("--retry-delay-seconds", type=int, default=60)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.parallelism < 1 or args.materialize_shards < 1:
        raise ValueError("parallelism and shard count must be positive")
    args.queue = args.run_root / "queues" / f"{args.task}_p13_queue.json"
    queue = load_object(args.queue)
    source = load_object(args.run_root / "queues" / "source_bundle_manifest.json")
    verify_runtime_closure(
        source_root=args.source_root, queue=queue,
        source_manifest=source, compiled_backend=args.compiled_backend,
    )
    lock_path = args.run_root / "status" / f"{args.stage}_{args.task}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another identical P13 supervisor is already active") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
    lock_handle.flush()
    status_path = args.run_root / "status" / f"{args.stage}_{args.task}.json"
    units = (
        _materialize_units(args, queue) if args.stage == "materialize"
        else _certificate_units(args, queue)
    )
    if args.plan_only:
        print(json.dumps({
            "stage": args.stage, "task": args.task,
            "planned_units": len(units), "parallelism": args.parallelism,
            "runtime_closure": "PASS", "execution_started": False,
        }, sort_keys=True, separators=(",", ":")))
        return
    try:
        _run_fleet(
            units=units, parallelism=args.parallelism,
            log_root=args.run_root / "logs" / args.stage / args.task,
            status_path=status_path, environment=_environment(args.source_root),
            retry_delay_seconds=args.retry_delay_seconds,
            allow_transient_retries=args.stage == "materialize",
        )
        counts = progress_counts(queue=queue, run_root=args.run_root)
        expected = 864 if args.stage == "certificate" else 864
        observed = counts["certificates" if args.stage == "certificate" else "graphs"]
        if observed != expected:
            raise RuntimeError(f"P13 stage completed with incomplete artifacts: {observed}/{expected}")
        receipt_payload = {
            "schema_version": "green-v410-p13-stage-completion-v1",
            "protocol_id": PROTOCOL_ID, "attempt_index": 1,
            "stage": args.stage, "task": args.task,
            "queue_manifest_sha256": queue["queue_manifest_sha256"],
            "expected_records": expected, "observed_records": observed,
            "runtime_closure": "PASS", "terminal_state": "PASS",
        }
        receipt = receipt_payload | {"receipt_sha256": sha256_canonical(receipt_payload)}
        atomic_no_clobber_json(
            args.run_root / "status" / f"{args.stage}_{args.task}_completion.json",
            receipt, job_id=f"p13-{args.stage}-{args.task}-completion",
        )
        _mutable_status(status_path, {"state": "COMPLETE", "counts": counts})
    except BaseException as error:
        _mutable_status(status_path, {"state": "FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
