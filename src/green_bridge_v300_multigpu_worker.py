"""Future deterministic v3 split worker; inferential phases remain unauthorized."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from exp_green_bridge_v300 import UNAUTHORIZED_PHASE
from green_bridge_v300_dataset import build_green_bridge_v300_records


def deterministic_worker_assignment_v300(phase: str, role: str, worker_index: int,
                                         worker_count: int = 8):
    if phase not in ("development", "confirmation") or role not in ("transport", "joint"):
        raise ValueError("invalid v3 worker phase or role")
    rows = [row for row in build_green_bridge_v300_records()
            if row.split == phase and row.role == role]
    rows.sort(key=lambda row: hashlib.sha256(
        f"green-v300-worker-assignment|{phase}|{role}|{row.pair_digest}".encode("utf-8")
    ).hexdigest())
    return [row for index, row in enumerate(rows) if index % worker_count == worker_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("development", "confirmation"))
    parser.add_argument("--role", required=True, choices=("transport", "joint"))
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.parse_args()
    raise RuntimeError(UNAUTHORIZED_PHASE)


if __name__ == "__main__":
    main()
