"""Strict outcome-free machine/concurrency identity for GREEN v4 manifests."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import sys

from green_bridge_v400_schemas import sha256_canonical


SCHEMA_VERSION = "green-v400-machine-concurrency-identity-v1"
_SHORT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_CPU_MODEL = re.compile(r"[ -~]{1,200}\Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _below_mnt_sdb(value: str) -> bool:
    pure = PurePosixPath(value)
    return len(pure.parts) > 3 and pure.parts[:3] == ("/", "mnt", "sdb")


def _cpu_model_linux() -> str:
    for line in Path("/proc/cpuinfo").read_text(
            encoding="ascii", errors="strict").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Processor", "Hardware"}:
            model = " ".join(value.strip().split())
            if _CPU_MODEL.fullmatch(model):
                return model
    raise RuntimeError("MACHINE_IDENTITY_CPU_MODEL_UNAVAILABLE")


def collect_machine_concurrency_manifest(
    *, max_workers: int, absolute_max_workers: int,
    wall_seconds_per_process: float, per_process_address_space_bytes: int,
    observed_tree_memory_bytes: int, sample_interval_seconds: float,
    gpu_environment: dict, backend_kind: str, backend_path: Path,
    backend_opened_by_workload: bool,
) -> dict:
    if os.name != "posix" or not Path("/proc/cpuinfo").is_file():
        raise RuntimeError("machine identity collection requires Linux /proc")
    backend = backend_path.resolve(strict=True)
    if not _below_mnt_sdb(backend.as_posix()):
        raise RuntimeError("machine identity backend must resolve below /mnt/sdb")
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
    core = {
        "schema_version": SCHEMA_VERSION,
        "host": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "kernel_release": platform.release(),
        },
        "cpu": {
            "model": _cpu_model_linux(),
            "logical_cpu_count": os.cpu_count(),
        },
        "memory": {
            "page_size_bytes": page_size,
            "physical_memory_bytes": page_size * physical_pages,
        },
        "gpu_environment": dict(gpu_environment),
        "concurrency": {
            "max_workers": max_workers,
            "absolute_max_workers": absolute_max_workers,
        },
        "per_process_limits": {
            "wall_seconds": wall_seconds_per_process,
            "address_space_bytes": per_process_address_space_bytes,
            "observed_tree_memory_bytes": observed_tree_memory_bytes,
            "sample_interval_seconds": sample_interval_seconds,
        },
        "backend_identity": {
            "kind": backend_kind,
            "path": backend.as_posix(),
            "sha256": _sha256_file(backend),
            "opened_by_workload": backend_opened_by_workload,
        },
        "python_identity": {
            "implementation": platform.python_implementation(),
            "version": [sys.version_info.major, sys.version_info.minor,
                        sys.version_info.micro],
            "executable_sha256": _sha256_file(Path(sys.executable).resolve(strict=True)),
        },
    }
    payload = {**core, "machine_manifest_semantic_hash": sha256_canonical(core)}
    validate_machine_concurrency_manifest(payload)
    return payload


def validate_machine_concurrency_manifest(
    payload: dict, *, expected_max_workers: int | None = None,
    expected_absolute_max_workers: int | None = None,
    expected_wall_seconds: float | None = None,
    expected_address_space_bytes: int | None = None,
    expected_observed_tree_memory_bytes: int | None = None,
    expected_sample_interval_seconds: float | None = None,
    expected_gpu_environment: dict | None = None,
    expected_backend_kind: str | None = None,
    expected_backend_path: str | None = None,
    expected_backend_sha256: str | None = None,
    expected_backend_opened: bool | None = None,
) -> None:
    required = {
        "schema_version", "host", "cpu", "memory", "gpu_environment",
        "concurrency", "per_process_limits", "backend_identity",
        "python_identity", "machine_manifest_semantic_hash",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("machine identity fields mismatch")
    stored_hash = payload["machine_manifest_semantic_hash"]
    core = {key: value for key, value in payload.items()
            if key != "machine_manifest_semantic_hash"}
    if (payload["schema_version"] != SCHEMA_VERSION
            or not isinstance(stored_hash, str) or len(stored_hash) != 64
            or stored_hash != sha256_canonical(core)):
        raise ValueError("machine identity semantic hash invalid")
    host = payload["host"]
    if (not isinstance(host, dict)
            or set(host) != {"hostname", "machine", "kernel_release"}
            or any(not isinstance(host[name], str) or not _SHORT_ID.fullmatch(host[name])
                   for name in host)):
        raise ValueError("machine host identity invalid")
    cpu = payload["cpu"]
    if (not isinstance(cpu, dict) or set(cpu) != {"model", "logical_cpu_count"}
            or not isinstance(cpu["model"], str)
            or not _CPU_MODEL.fullmatch(cpu["model"])
            or type(cpu["logical_cpu_count"]) is not int
            or cpu["logical_cpu_count"] <= 0):
        raise ValueError("machine CPU identity invalid")
    memory = payload["memory"]
    if (not isinstance(memory, dict)
            or set(memory) != {"page_size_bytes", "physical_memory_bytes"}
            or type(memory["page_size_bytes"]) is not int
            or memory["page_size_bytes"] <= 0
            or memory["page_size_bytes"] & (memory["page_size_bytes"] - 1)
            or type(memory["physical_memory_bytes"]) is not int
            or memory["physical_memory_bytes"] < memory["page_size_bytes"]
            or memory["physical_memory_bytes"] % memory["page_size_bytes"]):
        raise ValueError("machine memory identity invalid")
    gpu = payload["gpu_environment"]
    if (not isinstance(gpu, dict)
            or gpu != {"CUDA_VISIBLE_DEVICES": "",
                       "NVIDIA_VISIBLE_DEVICES": "none", "gpu_used": False}
            or (expected_gpu_environment is not None
                and gpu != expected_gpu_environment)):
        raise ValueError("machine GPU identity invalid")
    concurrency = payload["concurrency"]
    if (not isinstance(concurrency, dict)
            or set(concurrency) != {"max_workers", "absolute_max_workers"}
            or type(concurrency["max_workers"]) is not int
            or type(concurrency["absolute_max_workers"]) is not int
            or not 1 <= concurrency["max_workers"] <= concurrency["absolute_max_workers"]
            or (expected_max_workers is not None
                and concurrency["max_workers"] != expected_max_workers)
            or (expected_absolute_max_workers is not None
                and concurrency["absolute_max_workers"]
                != expected_absolute_max_workers)):
        raise ValueError("machine concurrency identity invalid")
    limits = payload["per_process_limits"]
    if (not isinstance(limits, dict)
            or set(limits) != {"wall_seconds", "address_space_bytes",
                               "observed_tree_memory_bytes", "sample_interval_seconds"}
            or type(limits["wall_seconds"]) not in {int, float}
            or not math.isfinite(limits["wall_seconds"])
            or limits["wall_seconds"] <= 0
            or type(limits["address_space_bytes"]) is not int
            or limits["address_space_bytes"] <= 0
            or type(limits["observed_tree_memory_bytes"]) is not int
            or limits["observed_tree_memory_bytes"] <= 0
            or type(limits["sample_interval_seconds"]) not in {int, float}
            or not math.isfinite(limits["sample_interval_seconds"])
            or not 0.01 <= limits["sample_interval_seconds"] <= 60
            or (expected_wall_seconds is not None
                and limits["wall_seconds"] != expected_wall_seconds)
            or (expected_address_space_bytes is not None
                and limits["address_space_bytes"] != expected_address_space_bytes)
            or (expected_observed_tree_memory_bytes is not None
                and limits["observed_tree_memory_bytes"]
                != expected_observed_tree_memory_bytes)
            or (expected_sample_interval_seconds is not None
                and limits["sample_interval_seconds"]
                != expected_sample_interval_seconds)):
        raise ValueError("machine process limits invalid")
    backend = payload["backend_identity"]
    if (not isinstance(backend, dict)
            or set(backend) != {"kind", "path", "sha256", "opened_by_workload"}
            or not isinstance(backend["kind"], str)
            or not _SHORT_ID.fullmatch(backend["kind"])
            or not isinstance(backend["path"], str)
            or not _below_mnt_sdb(backend["path"])
            or not isinstance(backend["sha256"], str)
            or len(backend["sha256"]) != 64
            or any(character not in "0123456789abcdef"
                   for character in backend["sha256"])
            or type(backend["opened_by_workload"]) is not bool
            or (expected_backend_kind is not None
                and backend["kind"] != expected_backend_kind)
            or (expected_backend_path is not None
                and backend["path"] != expected_backend_path)
            or (expected_backend_sha256 is not None
                and backend["sha256"] != expected_backend_sha256)
            or (expected_backend_opened is not None
                and backend["opened_by_workload"] is not expected_backend_opened)):
        raise ValueError("machine backend identity invalid")
    python = payload["python_identity"]
    version = python.get("version") if isinstance(python, dict) else None
    if (not isinstance(python, dict)
            or set(python) != {"implementation", "version", "executable_sha256"}
            or not isinstance(python["implementation"], str)
            or not _SHORT_ID.fullmatch(python["implementation"])
            or not isinstance(version, list) or len(version) != 3
            or any(type(value) is not int or value < 0 for value in version)
            or not isinstance(python["executable_sha256"], str)
            or len(python["executable_sha256"]) != 64
            or any(character not in "0123456789abcdef"
                   for character in python["executable_sha256"])):
        raise ValueError("machine Python identity invalid")


def verify_current_machine_concurrency_manifest(payload: dict) -> None:
    """Fail closed if the frozen identity does not describe this process host."""
    validate_machine_concurrency_manifest(payload)
    concurrency = payload["concurrency"]
    limits = payload["per_process_limits"]
    backend = payload["backend_identity"]
    observed = collect_machine_concurrency_manifest(
        max_workers=concurrency["max_workers"],
        absolute_max_workers=concurrency["absolute_max_workers"],
        wall_seconds_per_process=limits["wall_seconds"],
        per_process_address_space_bytes=limits["address_space_bytes"],
        observed_tree_memory_bytes=limits["observed_tree_memory_bytes"],
        sample_interval_seconds=limits["sample_interval_seconds"],
        gpu_environment=payload["gpu_environment"],
        backend_kind=backend["kind"], backend_path=Path(backend["path"]),
        backend_opened_by_workload=backend["opened_by_workload"],
    )
    if observed != payload:
        raise RuntimeError("FROZEN_MACHINE_CONCURRENCY_IDENTITY_MISMATCH")


def fixture_machine_concurrency_manifest(
    *, max_workers: int, absolute_max_workers: int,
    wall_seconds_per_process: float, per_process_address_space_bytes: int,
    observed_tree_memory_bytes: int, sample_interval_seconds: float,
    backend_kind: str, backend_path: str, backend_sha256: str,
    backend_opened_by_workload: bool,
) -> dict:
    """Deterministic valid fixture; never used by production entry points."""
    core = {
        "schema_version": SCHEMA_VERSION,
        "host": {"hostname": "fixture-host", "machine": "x86_64",
                 "kernel_release": "6.8.0-fixture"},
        "cpu": {"model": "Fixture CPU", "logical_cpu_count": 32},
        "memory": {"page_size_bytes": 4096,
                   "physical_memory_bytes": 64 * (1 << 30)},
        "gpu_environment": {"CUDA_VISIBLE_DEVICES": "",
                            "NVIDIA_VISIBLE_DEVICES": "none", "gpu_used": False},
        "concurrency": {"max_workers": max_workers,
                        "absolute_max_workers": absolute_max_workers},
        "per_process_limits": {
            "wall_seconds": wall_seconds_per_process,
            "address_space_bytes": per_process_address_space_bytes,
            "observed_tree_memory_bytes": observed_tree_memory_bytes,
            "sample_interval_seconds": sample_interval_seconds,
        },
        "backend_identity": {
            "kind": backend_kind, "path": backend_path,
            "sha256": backend_sha256,
            "opened_by_workload": backend_opened_by_workload,
        },
        "python_identity": {"implementation": "CPython", "version": [3, 11, 0],
                            "executable_sha256": "e" * 64},
    }
    payload = {**core, "machine_manifest_semantic_hash": sha256_canonical(core)}
    validate_machine_concurrency_manifest(payload)
    return payload
