"""Fail-closed resource accounting and two-phase publication for GREEN v4.

The production launcher remains disabled while CertificateResourceLock rejects
``production_authorized=True``.  This module supplies the outcome-blind core
that can be audited without opening a scientific result.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import select
import signal
import threading
import time

from green_bridge_v400_schemas import CertificateResourceLock, canonical_json


SHA256_RE = re.compile(r"[0-9a-f]{64}")
PUBLISHABLE_STATUSES = ("INTERVAL_COMPUTED", "RESOURCE_INCONCLUSIVE")


class SupervisorViolation(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass(frozen=True)
class WorkerWaitResult:
    deadline_reached: bool
    exit_code: int | None
    termination_signal: int | None


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class _Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", _Timespec), ("it_value", _Timespec)]


class LinuxMonotonicDeadline:
    """pidfd + absolute CLOCK_MONOTONIC timer retained through publication."""

    CLOCK_MONOTONIC = 1
    TFD_TIMER_ABSTIME = 1
    SYS_PIDFD_OPEN = 434

    def __init__(self, deadline_seconds: float, *, started_at: float | None = None):
        if os.name != "posix" or deadline_seconds <= 0:
            raise SupervisorViolation(
                "SUPERVISOR_INFRASTRUCTURE_INVALID", "Linux positive deadline required",
            )
        self.started_at = time.monotonic() if started_at is None else float(started_at)
        now = time.monotonic()
        if self.started_at > now:
            raise SupervisorViolation(
                "SUPERVISOR_INFRASTRUCTURE_INVALID", "deadline start is in the future",
            )
        self.deadline_at = self.started_at + float(deadline_seconds)
        self._libc = ctypes.CDLL(None, use_errno=True)
        create = getattr(self._libc, "timerfd_create", None)
        settime = getattr(self._libc, "timerfd_settime", None)
        if create is None or settime is None:
            raise SupervisorViolation(
                "SUPERVISOR_INFRASTRUCTURE_INVALID", "timerfd symbols unavailable",
            )
        create.argtypes = [ctypes.c_int, ctypes.c_int]
        create.restype = ctypes.c_int
        self.timer_fd = create(
            self.CLOCK_MONOTONIC, os.O_CLOEXEC | os.O_NONBLOCK,
        )
        if self.timer_fd < 0:
            raise SupervisorViolation(
                "SUPERVISOR_INFRASTRUCTURE_INVALID",
                f"timerfd_create errno={ctypes.get_errno()}",
            )
        seconds = int(self.deadline_at)
        nanoseconds = int((self.deadline_at - seconds) * 1_000_000_000)
        specification = _Itimerspec(_Timespec(0, 0), _Timespec(seconds, nanoseconds))
        settime.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Itimerspec),
            ctypes.POINTER(_Itimerspec),
        ]
        settime.restype = ctypes.c_int
        if settime(
            self.timer_fd, self.TFD_TIMER_ABSTIME,
            ctypes.byref(specification), None,
        ) != 0:
            error = ctypes.get_errno()
            os.close(self.timer_fd)
            self.timer_fd = -1
            raise SupervisorViolation(
                "SUPERVISOR_INFRASTRUCTURE_INVALID", f"timerfd_settime errno={error}",
            )
        self._expired = False

    def _pidfd_open(self, pid: int) -> int:
        if hasattr(os, "pidfd_open"):
            return os.pidfd_open(pid, 0)
        descriptor = self._libc.syscall(self.SYS_PIDFD_OPEN, pid, 0)
        if descriptor < 0:
            raise SupervisorViolation(
                "SUPERVISOR_INFRASTRUCTURE_INVALID",
                f"pidfd_open syscall errno={ctypes.get_errno()}",
            )
        return descriptor

    @staticmethod
    def _decode_wait_status(status: int) -> tuple[int | None, int | None]:
        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status), None
        if os.WIFSIGNALED(status):
            return None, os.WTERMSIG(status)
        return None, None

    def wait_worker(self, pid: int) -> WorkerWaitResult:
        pidfd = self._pidfd_open(pid)
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        poller.register(self.timer_fd, select.POLLIN)
        try:
            events = {descriptor for descriptor, _ in poller.poll()}
            if self.timer_fd in events:
                self._expired = True
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status = os.waitpid(pid, 0)
                exit_code, termination_signal = self._decode_wait_status(status)
                return WorkerWaitResult(True, exit_code, termination_signal)
            if pidfd not in events:
                raise SupervisorViolation(
                    "SUPERVISOR_INFRASTRUCTURE_INVALID", "poll returned no pidfd/timerfd",
                )
            _, status = os.waitpid(pid, 0)
            exit_code, termination_signal = self._decode_wait_status(status)
            return WorkerWaitResult(False, exit_code, termination_signal)
        finally:
            os.close(pidfd)

    def deadline_expired(self) -> bool:
        if self._expired:
            return True
        poller = select.poll()
        poller.register(self.timer_fd, select.POLLIN)
        if poller.poll(0):
            self._expired = True
        return self._expired

    def assert_publish_window(self) -> None:
        if self.deadline_expired():
            raise SupervisorViolation(
                "WALL_DEADLINE_REACHED", "deadline expired before atomic publication",
            )

    def close(self) -> None:
        if self.timer_fd >= 0:
            os.close(self.timer_fd)
            self.timer_fd = -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@dataclass(frozen=True)
class AdmissionRecord:
    ordinal: int
    token_id: str
    attempt_id: str
    exact_domain_sha256: str
    precision_bits: int
    charged_tokens: int
    cumulative_tokens: int


class AdmissionLedger:
    """Charge an exact native pass before execution; never refund it."""

    def __init__(self, resource_lock: CertificateResourceLock, ledger_path: Path):
        self.resource_lock = resource_lock
        self._records: list[AdmissionRecord] = []
        self._records_by_token: dict[str, AdmissionRecord] = {}
        self._dispatch_state_by_token: dict[str, str] = {}
        self._charged_tokens = 0
        self._phase = "OFFICIAL_384"
        self._mutex = threading.RLock()
        self.ledger_path = Path(ledger_path).absolute()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("xb") as stream:
            stream.write((canonical_json({
                "event": "LEDGER_OPENED",
                "resource_lock_semantic_hash": resource_lock.semantic_hash(),
                "phase": self._phase,
            }) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())

    def _append(self, payload: dict) -> None:
        with self.ledger_path.open("ab") as stream:
            stream.write((canonical_json(payload) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())

    @property
    def charged_tokens(self) -> int:
        return self._charged_tokens

    @property
    def remaining_tokens(self) -> int:
        return self.resource_lock.token_budget - self._charged_tokens

    @property
    def records(self) -> tuple[AdmissionRecord, ...]:
        return tuple(self._records)

    def admit(
        self, precision_bits: int, *, token_id: str, attempt_id: str,
        exact_domain_sha256: str,
    ) -> AdmissionRecord:
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", token_id)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", attempt_id)
                or not SHA256_RE.fullmatch(exact_domain_sha256)):
            raise SupervisorViolation(
                "RESOURCE_ACCOUNTING_INVALID", "admission identity invalid",
            )
        with self._mutex:
            existing = self._records_by_token.get(token_id)
            if existing is not None:
                if (existing.precision_bits != precision_bits
                        or existing.attempt_id != attempt_id
                        or existing.exact_domain_sha256 != exact_domain_sha256):
                    raise SupervisorViolation(
                        "RESOURCE_ACCOUNTING_INVALID", "token id identity collision",
                    )
                return existing
            return self._admit_new(
                precision_bits, token_id, attempt_id, exact_domain_sha256,
            )

    def _admit_new(
        self, precision_bits: int, token_id: str, attempt_id: str,
        exact_domain_sha256: str,
    ) -> AdmissionRecord:
        if precision_bits == self.resource_lock.official_precision:
            if self._phase != "OFFICIAL_384":
                raise SupervisorViolation(
                    "RESOURCE_ACCOUNTING_INVALID", "official phase is already frozen",
                )
            charge = self.resource_lock.token_weight_384
        elif precision_bits == self.resource_lock.audit_precision:
            if self._phase != "AUDIT_512":
                raise SupervisorViolation(
                    "RESOURCE_ACCOUNTING_INVALID", "audit admission before official freeze",
                )
            charge = self.resource_lock.token_weight_512
        else:
            raise SupervisorViolation(
                "RESOURCE_ACCOUNTING_INVALID", "unsupported precision admission",
            )
        if self._charged_tokens + charge > self.resource_lock.token_budget:
            raise SupervisorViolation(
                "RESOURCE_ACCOUNTING_INVALID", "token budget cannot admit another pass",
            )
        record = AdmissionRecord(
            len(self._records), token_id, attempt_id, exact_domain_sha256,
            precision_bits, charge, self._charged_tokens + charge,
        )
        self._append({
            "event": "PASS_ADMITTED", "phase": self._phase,
            **record.__dict__,
        })
        self._charged_tokens += charge
        self._records.append(record)
        self._records_by_token[token_id] = record
        self._dispatch_state_by_token[token_id] = "ADMITTED"
        return record

    def freeze_official_phase(
        self, *, completed_radius_count: int,
        official_partition_manifest_sha256: str,
    ) -> None:
        with self._mutex:
            if self._phase != "OFFICIAL_384":
                raise SupervisorViolation(
                    "RESOURCE_ACCOUNTING_INVALID", "official phase freeze is not unique",
                )
            if (completed_radius_count != self.resource_lock.radii_count
                    or not SHA256_RE.fullmatch(official_partition_manifest_sha256)
                    or any(state != "FINISHED" for state in self._dispatch_state_by_token.values())
                    or len(self._records) < self.resource_lock.radii_count * 5
                    or any(record.precision_bits != self.resource_lock.official_precision
                           for record in self._records)):
                raise SupervisorViolation(
                    "RESOURCE_ACCOUNTING_INVALID", "official freeze evidence incomplete",
                )
            self._append({
                "event": "OFFICIAL_PARTITIONS_FROZEN",
                "completed_radius_count": completed_radius_count,
                "official_partition_manifest_sha256": official_partition_manifest_sha256,
                "admitted_pass_count": len(self._records),
                "charged_tokens": self._charged_tokens,
            })
            self._phase = "AUDIT_512"

    def mark_dispatch_started(self, token_id: str) -> None:
        with self._mutex:
            if self._dispatch_state_by_token.get(token_id) != "ADMITTED":
                raise SupervisorViolation(
                    "RESOURCE_ACCOUNTING_INVALID", "dispatch token is not uniquely admitted",
                )
            self._append({"event": "DISPATCH_STARTED", "token_id": token_id})
            self._dispatch_state_by_token[token_id] = "STARTED"

    def mark_dispatch_finished(self, token_id: str, *, success: bool) -> None:
        with self._mutex:
            if self._dispatch_state_by_token.get(token_id) != "STARTED":
                raise SupervisorViolation(
                    "RESOURCE_ACCOUNTING_INVALID", "dispatch token is not active",
                )
            self._append({
                "event": "DISPATCH_FINISHED", "token_id": token_id,
                "success": bool(success), "refund": False,
            })
            self._dispatch_state_by_token[token_id] = "FINISHED"

    def failure_without_refund(self, ordinal: int) -> None:
        with self._mutex:
            if ordinal < 0 or ordinal >= len(self._records):
                raise SupervisorViolation(
                    "RESOURCE_ACCOUNTING_INVALID", "unknown admitted-pass ordinal",
                )
            self._append({
                "event": "ADMITTED_PASS_FAILED_NO_REFUND", "ordinal": ordinal,
                "charged_tokens": self._charged_tokens,
            })
            # Deliberately no accounting mutation: an admitted pass remains charged.

    def to_dict(self) -> dict:
        return {
            "schema_version": "green-v400-admission-ledger-v1",
            "resource_lock_semantic_hash": self.resource_lock.semantic_hash(),
            "token_budget": self.resource_lock.token_budget,
            "charged_tokens": self.charged_tokens,
            "remaining_tokens": self.remaining_tokens,
            "phase": self._phase,
            "dispatch_states": dict(sorted(self._dispatch_state_by_token.items())),
            "records": [record.__dict__ for record in self._records],
            "failed_dispatch_refund": False,
        }


@dataclass(frozen=True)
class CgroupV2Availability:
    mount_path: str | None
    process_path: str | None
    controllers: tuple[str, ...]
    memory_controller_available: bool
    memory_max_available: bool
    swap_max_available: bool
    memory_events_available: bool
    delegated_subtree_writable: bool

    @property
    def hard_memory_gate_ready(self) -> bool:
        return all((
            self.memory_controller_available,
            self.memory_max_available,
            self.swap_max_available,
            self.memory_events_available,
            self.delegated_subtree_writable,
        ))

    def to_dict(self) -> dict:
        return self.__dict__ | {"controllers": list(self.controllers)}


def _cgroup2_mount(mountinfo: str) -> Path | None:
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator or not right.startswith("cgroup2 "):
            continue
        fields = left.split()
        if len(fields) >= 5:
            return Path(fields[4])
    return None


def _unified_process_path(cgroup_text: str) -> PurePosixPath | None:
    for line in cgroup_text.splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            return PurePosixPath(path)
    return None


def probe_cgroup_v2(
    *, mountinfo_path: Path = Path("/proc/self/mountinfo"),
    cgroup_path: Path = Path("/proc/self/cgroup"),
) -> CgroupV2Availability:
    mount = _cgroup2_mount(mountinfo_path.read_text(encoding="utf-8"))
    process_relative = _unified_process_path(
        cgroup_path.read_text(encoding="utf-8")
    )
    if mount is None or process_relative is None:
        return CgroupV2Availability(None, None, (), False, False, False, False, False)
    process_directory = mount.joinpath(*process_relative.parts[1:])
    controller_file = process_directory / "cgroup.controllers"
    controllers = tuple(sorted(
        controller_file.read_text(encoding="ascii").split()
    )) if controller_file.is_file() else ()
    return CgroupV2Availability(
        str(mount), str(process_directory), controllers,
        "memory" in controllers,
        (process_directory / "memory.max").is_file(),
        (process_directory / "memory.swap.max").is_file(),
        (process_directory / "memory.events").is_file(),
        os.access(process_directory, os.W_OK),
    )


def assert_cgroup_v2_hard_memory_gate(availability: CgroupV2Availability) -> None:
    if not availability.hard_memory_gate_ready:
        raise SupervisorViolation(
            "MEMORY_ENFORCEMENT_UNAVAILABLE",
            "cgroup v2 memory.max/swap.max/events with writable delegation is required",
        )


def configure_worker_cgroup(directory: Path, memory_max_bytes: int) -> dict:
    directory = Path(directory).resolve(strict=True)
    if memory_max_bytes <= 0:
        raise SupervisorViolation("RESOURCE_ACCOUNTING_INVALID", "invalid memory.max")
    required = {
        "memory.max": str(memory_max_bytes),
        "memory.swap.max": "0",
    }
    for name, value in required.items():
        target = directory / name
        if not target.is_file() or target.is_symlink():
            raise SupervisorViolation(
                "MEMORY_ENFORCEMENT_UNAVAILABLE", f"missing regular {name}",
            )
        target.write_text(value + "\n", encoding="ascii")
        if target.read_text(encoding="ascii").strip() != value:
            raise SupervisorViolation(
                "MEMORY_ENFORCEMENT_UNAVAILABLE", f"{name} readback mismatch",
            )
    events = directory / "memory.events"
    procs = directory / "cgroup.procs"
    if any(not path.is_file() or path.is_symlink() for path in (events, procs)):
        raise SupervisorViolation(
            "MEMORY_ENFORCEMENT_UNAVAILABLE", "memory.events/cgroup.procs missing",
        )
    return {
        "schema_version": "green-v400-cgroup-v2-worker-config-v1",
        "directory": str(directory),
        "memory_max_bytes": memory_max_bytes,
        "swap_max_bytes": 0,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_readonly(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_relative_file(staging: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or pure.is_absolute() or ".." in pure.parts
            or any(part in {"", "."} for part in pure.parts)):
        raise SupervisorViolation("PUBLICATION_INVALID", "unsafe artifact path")
    target = staging.joinpath(*pure.parts)
    if target.is_symlink() or not target.is_file():
        raise SupervisorViolation(
            "PUBLICATION_INVALID", "artifact is missing, nonregular, or a symlink",
        )
    resolved = target.resolve(strict=True)
    try:
        resolved.relative_to(staging.resolve(strict=True))
    except ValueError as error:
        raise SupervisorViolation("PUBLICATION_INVALID", "artifact escapes staging") from error
    return resolved


def validate_staged_commit(
    staging_directory: Path, resource_lock: CertificateResourceLock,
) -> dict:
    staging = Path(staging_directory).resolve(strict=True)
    manifest_path = staging / "supervisor_commit_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SupervisorViolation("PUBLICATION_INVALID", "commit manifest missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "resource_lock_semantic_hash", "attempt_id",
        "status", "resource_reason", "scientific_threshold_applied", "files",
    }
    if set(payload) != expected_fields:
        raise SupervisorViolation("PUBLICATION_INVALID", "manifest fields mismatch")
    if payload["schema_version"] != "green-v400-supervisor-commit-manifest-v1":
        raise SupervisorViolation("PUBLICATION_INVALID", "manifest schema mismatch")
    if payload["resource_lock_semantic_hash"] != resource_lock.semantic_hash():
        raise SupervisorViolation("PUBLICATION_INVALID", "resource lock identity mismatch")
    if (not isinstance(payload["attempt_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", payload["attempt_id"])):
        raise SupervisorViolation("PUBLICATION_INVALID", "attempt id invalid")
    if payload["status"] not in PUBLISHABLE_STATUSES:
        raise SupervisorViolation("PUBLICATION_INVALID", "status is not publishable")
    if payload["scientific_threshold_applied"] is not False:
        raise SupervisorViolation(
            "PUBLICATION_INVALID", "worker read scientific threshold before commit",
        )
    reason = payload["resource_reason"]
    if payload["status"] == "INTERVAL_COMPUTED" and reason is not None:
        raise SupervisorViolation("PUBLICATION_INVALID", "completed interval has reason")
    if (payload["status"] == "RESOURCE_INCONCLUSIVE"
            and reason not in resource_lock.reachable_primary_reasons):
        raise SupervisorViolation("PUBLICATION_INVALID", "resource reason invalid")
    files = payload["files"]
    if (not isinstance(files, dict) or not files
            or any(not isinstance(key, str) or not isinstance(value, dict)
                   for key, value in files.items())):
        raise SupervisorViolation("PUBLICATION_INVALID", "artifact table invalid")
    observed = {}
    for relative, identity in sorted(files.items()):
        if set(identity) != {"sha256", "nbytes"}:
            raise SupervisorViolation("PUBLICATION_INVALID", "artifact identity fields invalid")
        expected_hash = identity["sha256"]
        expected_nbytes = identity["nbytes"]
        if not SHA256_RE.fullmatch(expected_hash):
            raise SupervisorViolation("PUBLICATION_INVALID", "artifact hash invalid")
        if type(expected_nbytes) is not int or expected_nbytes < 0:
            raise SupervisorViolation("PUBLICATION_INVALID", "artifact size invalid")
        target = _safe_relative_file(staging, relative)
        observed[relative] = {
            "sha256": _sha256_file(target), "nbytes": target.stat().st_size,
        }
        if observed[relative] != identity:
            raise SupervisorViolation("PUBLICATION_INVALID", "artifact hash mismatch")
        _fsync_readonly(target)
    expected_files = set(files) | {"supervisor_commit_manifest.json"}
    actual_files = set()
    for candidate in staging.rglob("*"):
        if candidate.is_symlink():
            raise SupervisorViolation("PUBLICATION_INVALID", "staging contains a symlink")
        if candidate.is_file():
            actual_files.add(candidate.relative_to(staging).as_posix())
        elif not candidate.is_dir():
            raise SupervisorViolation("PUBLICATION_INVALID", "staging contains a special file")
    if actual_files != expected_files:
        raise SupervisorViolation("PUBLICATION_INVALID", "staging contains unexpected files")
    _fsync_readonly(manifest_path)
    _fsync_readonly(staging)
    canonical_manifest = canonical_json(payload).encode("utf-8")
    return {
        "schema_version": "green-v400-supervisor-validated-commit-v1",
        "attempt_id": payload["attempt_id"],
        "status": payload["status"],
        "resource_reason": reason,
        "resource_lock_semantic_hash": resource_lock.semantic_hash(),
        "manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
        "artifact_hashes": observed,
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.name != "posix":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SupervisorViolation("PUBLICATION_INVALID", "renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    status = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1,
    )
    if status != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise SupervisorViolation("PUBLICATION_INVALID", "publish target already exists")
        raise SupervisorViolation(
            "PUBLICATION_INVALID", f"renameat2 failed with errno {error}",
        )


def atomic_publish(staging_directory: Path, publish_directory: Path) -> None:
    staging = Path(staging_directory).resolve(strict=True)
    publish = Path(publish_directory).absolute()
    if publish.exists() or publish.is_symlink():
        raise SupervisorViolation("PUBLICATION_INVALID", "publish target already exists")
    publish.parent.mkdir(parents=True, exist_ok=True)
    if staging.stat().st_dev != publish.parent.stat().st_dev:
        raise SupervisorViolation("PUBLICATION_INVALID", "publish rename crosses filesystem")
    _rename_noreplace(staging, publish)
    _fsync_readonly(publish.parent)


def authorize_supervised_execution(resource_lock: CertificateResourceLock) -> None:
    if not resource_lock.production_authorized:
        raise SupervisorViolation(
            "PRODUCTION_EXECUTION_UNAUTHORIZED",
            "resource lock remains prepare-only",
        )
