"""Finalize immutable P13 direction records into task and global receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from green_v410_artifacts import atomic_no_clobber_json, configure_exact_integer_io
from green_v410_p13_finalize import finalize_task, finalize_two_tasks


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _records(queue: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for job in queue["jobs"]:
        for ordinal in range(8):
            path = root / job["job_id"] / f"direction_{ordinal:02d}" / "p13_record.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen P13 record: {path}")
            rows.append(_load(path))
    return rows


def main() -> None:
    configure_exact_integer_io()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ioi-queue", type=Path, required=True)
    parser.add_argument("--gt-queue", type=Path, required=True)
    parser.add_argument("--certificate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipts = []
    for task, queue_path in (("ioi", args.ioi_queue), ("greater_than", args.gt_queue)):
        queue = _load(queue_path)
        site_manifest, receipt = finalize_task(
            queue=queue, direction_records=_records(queue, args.certificate_root / task),
        )
        task_root = args.output_root / task
        atomic_no_clobber_json(
            task_root / "site_ratio_manifest.json", site_manifest,
            job_id=f"p13-{task}-site-ratios",
        )
        atomic_no_clobber_json(
            task_root / "p13_task_receipt.json", receipt,
            job_id=f"p13-{task}-receipt",
        )
        receipts.append(receipt)
    global_receipt = finalize_two_tasks(receipts)
    atomic_no_clobber_json(
        args.output_root / "p13_two_task_receipt.json", global_receipt,
        job_id="p13-two-task-receipt",
    )
    print(global_receipt["terminal_state"])


if __name__ == "__main__":
    main()
