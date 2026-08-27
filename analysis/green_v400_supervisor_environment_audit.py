"""Outcome-blind host readiness audit for the external resource supervisor."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_supervisor import probe_cgroup_v2
from green_v400_native_adaptive_policy_audit import _git


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        relative = output.relative_to(Path("/mnt/sdb").resolve())
    except ValueError as error:
        raise RuntimeError("supervisor environment output must be below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("supervisor environment output must be new below /mnt/sdb")

    availability = probe_cgroup_v2()
    libc = ctypes.CDLL(None)
    runtime = {
        "pidfd_open_available": hasattr(os, "pidfd_open"),
        "timerfd_create_symbol_available": hasattr(libc, "timerfd_create"),
        "timerfd_settime_symbol_available": hasattr(libc, "timerfd_settime"),
        "renameat2_symbol_available": hasattr(libc, "renameat2"),
    }
    passed = availability.hard_memory_gate_ready and all(runtime.values())
    report = {
        "schema_version": "green-v400-supervisor-environment-audit-v1",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "status": (
            "PASS_SUPERVISOR_HOST_PRIMITIVES" if passed
            else "BLOCK_CGROUP_V2_MEMORY_UNAVAILABLE"
        ),
        "platform": platform.platform(),
        "cgroup_v2": availability.to_dict(),
        "linux_runtime_primitives": runtime,
        "frozen_memory_policy": {
            "memory_enforcement": "cgroup_v2_memory.max",
            "swap_enforcement": "cgroup_v2_memory.swap.max=0",
            "v1_fallback_allowed": False,
        },
        "production_execution_authorized": False,
        "required_host_action": (
            None if passed else
            "provide a pure/delegated cgroup v2 subtree with the memory controller "
            "and writable memory.max, memory.swap.max, memory.events, cgroup.procs"
        ),
        "provenance": {
            "repository_commit": _git("rev-parse", "HEAD"),
            "repository_clean": not bool(_git(
                "status", "--porcelain=v1", "--untracked-files=all"
            )),
        },
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "report_semantic_hash": report["report_semantic_hash"],
        "cgroup_v2": availability.to_dict(),
        "runtime": runtime,
    }, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
