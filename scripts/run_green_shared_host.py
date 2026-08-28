"""Run a routine GREEN experiment under the no-root shared-host envelope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_shared_host import (
    SharedHostResourcePolicy, run_shared_host_command,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-directory", required=True)
    parser.add_argument("--storage-root", default="/mnt/sdb")
    parser.add_argument("--cwd", default=str(ROOT))
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--address-space-gib", type=float, required=True)
    parser.add_argument("--observed-tree-gib", type=float, required=True)
    parser.add_argument("--sample-seconds", type=float, default=0.25)
    parser.add_argument("--allow-descendants", action="store_true")
    parser.add_argument("--hard-single-process", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    storage_root = Path(args.storage_root).resolve(strict=True)
    attempt_directory = Path(args.attempt_directory).resolve()
    try:
        relative_attempt = attempt_directory.relative_to(storage_root)
    except ValueError:
        parser.error("attempt-directory must be below storage-root")
    if not relative_attempt.parts:
        parser.error("attempt-directory may not equal storage-root")
    policy = SharedHostResourcePolicy(
        wall_deadline_seconds=args.wall_seconds,
        per_process_address_space_bytes=int(args.address_space_gib * (1 << 30)),
        observed_tree_memory_bytes=int(args.observed_tree_gib * (1 << 30)),
        sample_interval_seconds=args.sample_seconds,
        allow_descendant_processes=args.allow_descendants,
        hard_single_process=args.hard_single_process,
    )
    result = run_shared_host_command(
        command, cwd=Path(args.cwd),
        attempt_directory=attempt_directory, policy=policy,
    )
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0 if result.status == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
