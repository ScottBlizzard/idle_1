"""No-root resource envelope for trusted non-certificate GREEN jobs.

This module deliberately does not impersonate the stricter cgroup-v2 certificate
lock.  It supplies controls and measurements for explicitly scoped jobs on a
shared university server without changing host configuration:

* an inherited per-process ``RLIMIT_AS`` ceiling;
* optional kernel-enforced single-process mode via ``RLIMIT_NPROC``;
* an external monotonic wall-clock deadline;
* cleanup and empty verification of the initial worker process group;
* observed aggregate process-tree RSS/swap and descendant-policy checks; and
* a machine-readable report that states which guarantees are and are not hard.

The official formal-certificate supervisor remains separate and binding for its
artifact. Scientific outputs must never claim aggregate cgroup enforcement or
complete process-tree containment from a report produced here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence

from green_bridge_v400_schemas import canonical_json, sha256_canonical
from green_bridge_v400_supervisor import LinuxMonotonicDeadline


class SharedHostViolation(RuntimeError):
    """Invalid or unavailable shared-host experiment control."""


@dataclass(frozen=True)
class SharedHostResourcePolicy:
    wall_deadline_seconds: float
    per_process_address_space_bytes: int
    observed_tree_memory_bytes: int
    sample_interval_seconds: float = 0.25
    allow_descendant_processes: bool = False
    hard_single_process: bool = False

    def __post_init__(self) -> None:
        if self.wall_deadline_seconds <= 0:
            raise ValueError("wall deadline must be positive")
        if self.per_process_address_space_bytes <= 0:
            raise ValueError("address-space limit must be positive")
        if self.observed_tree_memory_bytes <= 0:
            raise ValueError("observed tree-memory limit must be positive")
        if not 0.01 <= self.sample_interval_seconds <= 60.0:
            raise ValueError("sample interval must lie in [0.01, 60] seconds")
        if type(self.allow_descendant_processes) is not bool:
            raise ValueError("allow_descendant_processes must be a bool")
        if type(self.hard_single_process) is not bool:
            raise ValueError("hard_single_process must be a bool")
        if self.hard_single_process and self.allow_descendant_processes:
            raise ValueError(
                "hard_single_process is incompatible with allowed descendants"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProcessTreeSample:
    monotonic_seconds: float
    process_count: int
    descendant_count: int
    rss_bytes: int
    swap_bytes: int
    pids: tuple[int, ...]
    pid_starttimes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SharedHostRunResult:
    status: str
    exit_code: int | None
    termination_signal: int | None
    elapsed_seconds: float
    peak_tree_rss_bytes: int
    peak_tree_swap_bytes: int
    peak_process_count: int
    report_path: str
    report_semantic_hash: str


def _parse_proc_status(text: str) -> tuple[int, int, int]:
    """Return (ppid, rss bytes, swap bytes) from Linux /proc/<pid>/status."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    try:
        ppid = int(fields["PPid"].split()[0])
        rss_kib = int(fields.get("VmRSS", "0 kB").split()[0])
        swap_kib = int(fields.get("VmSwap", "0 kB").split()[0])
    except (KeyError, ValueError, IndexError) as error:
        raise SharedHostViolation("malformed Linux process status") from error
    return ppid, rss_kib * 1024, swap_kib * 1024


def _linux_process_table(proc_root: Path = Path("/proc")) -> dict[int, tuple[int, int, int]]:
    table: dict[int, tuple[int, int, int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            table[int(entry.name)] = _parse_proc_status(
                (entry / "status").read_text(encoding="ascii", errors="replace")
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            # A process may exit while /proc is being sampled.
            continue
    return table


def _tree_pids(root_pid: int, table: Mapping[int, tuple[int, int, int]]) -> set[int]:
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _, _) in table.items():
            if pid not in descendants and parent in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _proc_starttime(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    try:
        text = (proc_root / str(pid) / "stat").read_text(
            encoding="ascii", errors="replace"
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    closing = text.rfind(")")
    fields = text[closing + 1:].split() if closing >= 0 else []
    try:
        return int(fields[19])  # field 22 overall; fields begins at state (3)
    except (IndexError, ValueError):
        return None


def sample_linux_process_tree(
    root_pid: int, *, proc_root: Path = Path("/proc"),
) -> ProcessTreeSample:
    table = _linux_process_table(proc_root)
    pids = _tree_pids(root_pid, table)
    live = [table[pid] for pid in pids if pid in table]
    identities = tuple(sorted(
        (pid, starttime) for pid in pids if pid in table
        for starttime in [_proc_starttime(pid, proc_root)] if starttime is not None
    ))
    return ProcessTreeSample(
        monotonic_seconds=time.monotonic(),
        process_count=len(live),
        descendant_count=max(0, len(live) - (1 if root_pid in table else 0)),
        rss_bytes=sum(row[1] for row in live),
        swap_bytes=sum(row[2] for row in live),
        pids=tuple(sorted(pid for pid in pids if pid in table)),
        pid_starttimes=identities,
    )


def _decode_return_code(return_code: int | None) -> tuple[int | None, int | None]:
    if return_code is None:
        return None, None
    if return_code < 0:
        return None, -return_code
    return return_code, None


def _identity_is_live(pid: int, starttime: int) -> bool:
    return _proc_starttime(pid) == starttime


def _cleanup_worker_group(
    worker: subprocess.Popen, observed_identities: set[tuple[int, int]],
) -> dict:
    kill_requested_at = None
    cleanup_errors: list[str] = []
    try:
        os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as error:
        cleanup_errors.append(f"killpg:{error.errno}")
    else:
        kill_requested_at = time.monotonic()
    for pid, starttime in sorted(observed_identities):
        if not _identity_is_live(pid, starttime):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            cleanup_errors.append(f"kill_observed_{pid}:{error.errno}")
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        cleanup_errors.append("leader_wait_timeout")
    cleanup_deadline = time.monotonic() + 5.0
    group_empty = False
    while True:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            group_empty = True
        except OSError as error:
            cleanup_errors.append(f"killpg_verify:{error.errno}")
            break
        else:
            group_empty = False
        for pid, starttime in sorted(observed_identities):
            if _identity_is_live(pid, starttime):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    cleanup_errors.append(f"kill_observed_verify_{pid}:{error.errno}")
        remaining_now = [
            (pid, starttime) for pid, starttime in observed_identities
            if _identity_is_live(pid, starttime)
        ]
        if group_empty and not remaining_now:
            break
        if time.monotonic() >= cleanup_deadline:
            if not group_empty:
                cleanup_errors.append("initial_process_group_not_empty")
            if remaining_now:
                cleanup_errors.append("observed_descendant_identity_not_gone")
            break
        time.sleep(0.01)
    remaining_observed = [
        [pid, starttime] for pid, starttime in sorted(observed_identities)
        if _identity_is_live(pid, starttime)
    ]
    return {
        "kill_requested_at_monotonic": kill_requested_at,
        "cleanup_completed_at_monotonic": time.monotonic(),
        "leader_reaped": worker.poll() is not None,
        "initial_process_group_empty": group_empty,
        "cleanup_verified": (
            group_empty and worker.poll() is not None and not remaining_observed
        ),
        "remaining_observed_pid_starttimes": remaining_observed,
        "cleanup_errors": cleanup_errors,
    }


def run_shared_host_command(
    argv: Sequence[str], *, cwd: Path, attempt_directory: Path,
    policy: SharedHostResourcePolicy,
    environment: Mapping[str, str] | None = None,
) -> SharedHostRunResult:
    """Run one command under the explicitly limited shared-host envelope."""
    if os.name != "posix" or not Path("/proc/self/status").is_file():
        raise SharedHostViolation("shared-host runner requires Linux /proc")
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise SharedHostViolation("argv must be a nonempty sequence of strings")
    working_directory = Path(cwd).resolve(strict=True)
    attempt = Path(attempt_directory).absolute()
    attempt.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt / "worker.stdout.log"
    stderr_path = attempt / "worker.stderr.log"
    report_path = attempt / "shared_host_resource_report.json"
    samples: list[ProcessTreeSample] = []
    observed_identities: set[tuple[int, int]] = set()
    status = "WORKER_FAILED"
    infrastructure_error = None
    trigger_sample: ProcessTreeSample | None = None
    exit_observed_at = None
    deadline_detected_at = None
    cleanup = None

    child_environment = dict(os.environ)
    if environment is not None:
        child_environment.update(environment)

    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        exec_shim = Path(__file__).with_name("green_bridge_v400_shared_host_exec.py")
        worker = subprocess.Popen(
            (sys.executable, str(exec_shim),
             str(policy.per_process_address_space_bytes),
             "1" if policy.hard_single_process else "0", *argv),
            cwd=working_directory, env=child_environment,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True, close_fds=True,
        )
        started = time.monotonic()
        deadline_at = started + policy.wall_deadline_seconds
        deadline_guard = None
        pidfd = None
        try:
            deadline_guard = LinuxMonotonicDeadline(
                policy.wall_deadline_seconds, started_at=started,
            )
            pidfd = deadline_guard._pidfd_open(worker.pid)
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            poller.register(deadline_guard.timer_fd, select.POLLIN)
            while True:
                events = {descriptor for descriptor, _ in poller.poll(0)}
                if deadline_guard.timer_fd in events:
                    status = "WALL_DEADLINE_REACHED"
                    deadline_detected_at = time.monotonic()
                    break
                if pidfd in events:
                    worker.wait()
                    exit_observed_at = time.monotonic()
                    if (deadline_guard.deadline_expired()
                            or exit_observed_at >= deadline_at):
                        status = "WALL_DEADLINE_REACHED"
                        deadline_detected_at = exit_observed_at
                    else:
                        status = (
                            "COMPLETED" if worker.returncode == 0
                            else "WORKER_FAILED"
                        )
                    break

                sample = sample_linux_process_tree(worker.pid)
                samples.append(sample)
                observed_identities.update(sample.pid_starttimes)
                # Deadline takes precedence if sampling crossed the boundary.
                if deadline_guard.deadline_expired():
                    status = "WALL_DEADLINE_REACHED"
                    deadline_detected_at = time.monotonic()
                    trigger_sample = sample
                    break
                if (not policy.allow_descendant_processes
                        and sample.descendant_count > 0):
                    status = "DESCENDANT_PROCESS_POLICY_REACHED"
                    trigger_sample = sample
                    break
                if (sample.rss_bytes + sample.swap_bytes
                        >= policy.observed_tree_memory_bytes):
                    status = "OBSERVED_TREE_MEMORY_REACHED"
                    trigger_sample = sample
                    break
                poller.poll(max(1, int(policy.sample_interval_seconds * 1000)))
        except Exception as error:  # cleanup still runs before reporting failure
            status = "SUPERVISOR_INFRASTRUCTURE_FAILED"
            infrastructure_error = f"{type(error).__name__}: {error}"
        finally:
            close_errors: list[str] = []
            try:
                if pidfd is not None:
                    try:
                        os.close(pidfd)
                    except OSError as error:
                        close_errors.append(f"pidfd_close:{error.errno}")
                if deadline_guard is not None:
                    try:
                        deadline_guard.close()
                    except OSError as error:
                        close_errors.append(f"timerfd_close:{error.errno}")
            finally:
                cleanup = _cleanup_worker_group(worker, observed_identities)
            if close_errors:
                status = "SUPERVISOR_INFRASTRUCTURE_FAILED"
                detail = ",".join(close_errors)
                infrastructure_error = (
                    detail if infrastructure_error is None
                    else f"{infrastructure_error}; {detail}"
                )

        if (not cleanup["cleanup_verified"]
                and status not in {"SUPERVISOR_INFRASTRUCTURE_FAILED"}):
            status = "SUPERVISOR_CLEANUP_FAILED"
        elif (status in {"COMPLETED", "WORKER_FAILED"}
              and cleanup["kill_requested_at_monotonic"] is not None
              and not policy.allow_descendant_processes):
            # The leader exited but its initial process group was still alive.
            status = "DESCENDANT_PROCESS_POLICY_REACHED"

        return_code = worker.poll()
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())

    ended = time.monotonic()
    exit_code, termination_signal = _decode_return_code(return_code)
    report = {
        "schema_version": "green-v400-shared-host-resource-report-v1",
        "report_contains_scientific_outcome": False,
        "supervisor_applied_scientific_threshold": False,
        "status": status,
        "command_sha256": sha256_canonical(list(argv)),
        "working_directory_sha256": sha256_canonical(str(working_directory)),
        "policy": policy.to_dict(),
        "observations": {
            "elapsed_seconds": ended - started,
            "exit_code": exit_code,
            "termination_signal": termination_signal,
            "sample_count": len(samples),
            "deadline_started_after_successful_spawn": True,
            "deadline_at_monotonic": deadline_at,
            "deadline_detected_at_monotonic": deadline_detected_at,
            "deadline_detection_latency_seconds": (
                None if deadline_detected_at is None else
                max(0.0, deadline_detected_at - deadline_at)
            ),
            "exit_observed_at_monotonic": exit_observed_at,
            "infrastructure_error": infrastructure_error,
            "peak_tree_rss_bytes": max((row.rss_bytes for row in samples), default=0),
            "peak_tree_swap_bytes": max((row.swap_bytes for row in samples), default=0),
            "peak_sampled_rss_plus_swap_bytes": max(
                (row.rss_bytes + row.swap_bytes for row in samples), default=0
            ),
            "peak_process_count": max((row.process_count for row in samples), default=0),
            "peak_descendant_count": max(
                (row.descendant_count for row in samples), default=0
            ),
            "maximum_sample_gap_seconds": max(
                (later.monotonic_seconds - earlier.monotonic_seconds
                 for earlier, later in zip(samples, samples[1:])), default=0.0,
            ),
            "trigger_sample": (
                None if trigger_sample is None else asdict(trigger_sample)
            ),
            "cleanup": cleanup,
        },
        "guarantee_scope": {
            "hard_per_process_virtual_address_space_limit": True,
            "hard_single_process_creation_limit": policy.hard_single_process,
            "numeric_thread_environment_forced_to_one": (
                policy.hard_single_process
            ),
            "hard_aggregate_user_space_address_space_upper_bound": (
                policy.hard_single_process
            ),
            "supervisor_live_timerfd_deadline_for_leader_and_initial_group": True,
            "scoped_cleanup_verified": cleanup["cleanup_verified"],
            "aggregate_process_tree_memory_is_sampled_not_hard_capped": (
                not policy.hard_single_process
            ),
            "complete_process_tree_containment_claimed": (
                policy.hard_single_process
            ),
            "unexpected_descendant_detection_is_observational": (
                not policy.hard_single_process
            ),
            "sampling_can_miss_short_lived_processes_or_memory_peaks": True,
            "cgroup_v2_enforcement_claimed": False,
        },
        "permitted_job_scope": (
            "trusted_hard_single_process_resource_lock_candidate"
            if policy.hard_single_process else
            "trusted_non_certificate_experiment_resource_record"
        ),
        "logs": {
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
        },
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    with report_path.open("xb") as stream:
        stream.write((canonical_json(report) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return SharedHostRunResult(
        status=status,
        exit_code=exit_code,
        termination_signal=termination_signal,
        elapsed_seconds=ended - started,
        peak_tree_rss_bytes=report["observations"]["peak_tree_rss_bytes"],
        peak_tree_swap_bytes=report["observations"]["peak_tree_swap_bytes"],
        peak_process_count=report["observations"]["peak_process_count"],
        report_path=str(report_path),
        report_semantic_hash=report["report_semantic_hash"],
    )
