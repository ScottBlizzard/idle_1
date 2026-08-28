"""Minimal exec shim that applies inherited Linux rlimits before exec."""
from __future__ import annotations

import os
import resource
import sys


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: shared_host_exec.py ADDRESS_BYTES COMMAND [ARG ...]")
    maximum = int(sys.argv[1])
    if maximum <= 0:
        raise SystemExit("ADDRESS_BYTES must be positive")
    command = sys.argv[2:]
    resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
