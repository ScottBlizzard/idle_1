"""Minimal exec shim that applies inherited Linux rlimits before exec."""
from __future__ import annotations

import os
import resource
import sys


CAP_SYS_ADMIN = 21
CAP_SYS_RESOURCE = 24
SINGLE_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def _linux_status_fields() -> dict[str, str]:
    fields = {}
    with open("/proc/self/status", encoding="ascii", errors="strict") as stream:
        for line in stream:
            key, separator, value = line.partition(":")
            if separator:
                fields[key] = value.strip()
    return fields


def _apply_hard_single_process_limit() -> None:
    if os.getuid() == 0 or os.geteuid() == 0:
        raise SystemExit("hard single-process mode forbids root identities")
    fields = _linux_status_fields()
    try:
        threads = int(fields["Threads"])
        effective_capabilities = int(fields["CapEff"], 16)
    except (KeyError, ValueError) as error:
        raise SystemExit("cannot verify Linux task/capability state") from error
    if threads != 1:
        raise SystemExit("hard single-process mode requires one initial task")
    forbidden_capabilities = (
        (1 << CAP_SYS_ADMIN) | (1 << CAP_SYS_RESOURCE)
    )
    if effective_capabilities & forbidden_capabilities:
        raise SystemExit(
            "hard single-process mode forbids CAP_SYS_ADMIN/CAP_SYS_RESOURCE"
        )
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    if resource.getrlimit(resource.RLIMIT_NPROC) != (1, 1):
        raise SystemExit("RLIMIT_NPROC hard lock verification failed")
    os.environ.update(SINGLE_THREAD_ENVIRONMENT)


def main() -> int:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: shared_host_exec.py ADDRESS_BYTES HARD_SINGLE_PROCESS "
            "COMMAND [ARG ...]"
        )
    maximum = int(sys.argv[1])
    if maximum <= 0:
        raise SystemExit("ADDRESS_BYTES must be positive")
    if sys.argv[2] not in {"0", "1"}:
        raise SystemExit("HARD_SINGLE_PROCESS must be 0 or 1")
    hard_single_process = sys.argv[2] == "1"
    command = sys.argv[3:]
    if hard_single_process:
        _apply_hard_single_process_limit()
    resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
