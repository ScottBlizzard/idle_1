"""Observed Linux process-tree resources for outcome-blind certificate runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import threading
import time

try:
    import resource
except ImportError:  # pragma: no cover - the formal runtime is Linux.
    resource = None


@dataclass(frozen=True)
class ProcessObservation:
    pid: int
    start_ticks: int
    parent_pid: int
    command_sha256: str
    peak_rss_kib: int
    last_rss_kib: int
    user_ticks: int
    system_ticks: int
    samples: int


@dataclass(frozen=True)
class ProcessTreeResourceRecord:
    schema_version: str
    measurement_scope: str
    is_formal_upper_bound: bool
    root_pid: int
    root_start_ticks: int
    started_at_utc: str
    finished_at_utc: str
    wall_seconds: float
    sample_interval_seconds: float
    sample_count: int
    missed_processes_possible: bool
    peak_sampled_tree_rss_kib: int
    root_peak_rss_kib: int
    descendant_identity_count: int
    self_ru_maxrss_before_kib: int
    self_ru_maxrss_after_kib: int
    children_ru_maxrss_before_kib: int
    children_ru_maxrss_after_kib: int
    process_observations: tuple[ProcessObservation, ...]
    platform: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "measurement_scope": self.measurement_scope,
            "is_formal_upper_bound": self.is_formal_upper_bound,
            "root_pid": self.root_pid,
            "root_start_ticks": self.root_start_ticks,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "wall_seconds": self.wall_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
            "sample_count": self.sample_count,
            "missed_processes_possible": self.missed_processes_possible,
            "peak_sampled_tree_rss_kib": self.peak_sampled_tree_rss_kib,
            "root_peak_rss_kib": self.root_peak_rss_kib,
            "descendant_identity_count": self.descendant_identity_count,
            "self_ru_maxrss_before_kib": self.self_ru_maxrss_before_kib,
            "self_ru_maxrss_after_kib": self.self_ru_maxrss_after_kib,
            "children_ru_maxrss_before_kib": self.children_ru_maxrss_before_kib,
            "children_ru_maxrss_after_kib": self.children_ru_maxrss_after_kib,
            "process_observations": [item.__dict__ for item in self.process_observations],
            "platform": self.platform,
        }


@dataclass
class _MutableObservation:
    pid: int
    start_ticks: int
    parent_pid: int
    command_sha256: str
    peak_rss_kib: int = 0
    last_rss_kib: int = 0
    user_ticks: int = 0
    system_ticks: int = 0
    samples: int = 0


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def _process_snapshot(pid: int) -> tuple[_MutableObservation, tuple[int, ...]] | None:
    process = Path("/proc") / str(pid)
    try:
        stat = _read_text(process / "stat")
        close = stat.rfind(")")
        if close < 0:
            return None
        fields = stat[close + 2:].split()
        parent_pid = int(fields[1])
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        start_ticks = int(fields[19])
        status = _read_text(process / "status")
        values: dict[str, int] = {}
        for line in status.splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, raw = line.split(":", 1)
                values[key] = int(raw.split()[0])
        command = (process / "cmdline").read_bytes()
        children_path = process / "task" / str(pid) / "children"
        children = tuple(int(value) for value in _read_text(children_path).split())
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    observation = _MutableObservation(
        pid=pid,
        start_ticks=start_ticks,
        parent_pid=parent_pid,
        command_sha256=hashlib.sha256(command).hexdigest(),
        peak_rss_kib=values.get("VmHWM", values.get("VmRSS", 0)),
        last_rss_kib=values.get("VmRSS", 0),
        user_ticks=user_ticks,
        system_ticks=system_ticks,
        samples=1,
    )
    return observation, children


class ProcessTreeResourceRecorder:
    """Sample one Linux process tree without reading certificate outcomes.

    `/proc` sampling can miss a descendant that starts and exits between samples,
    so the resulting peak is deliberately labelled an observation, never a bound.
    """

    def __init__(self, *, root_pid: int | None = None,
                 sample_interval_seconds: float = 0.01):
        if (resource is None or platform.system() != "Linux"
                or not Path("/proc/self/stat").exists()):
            raise RuntimeError("PROCESS_TREE_RESOURCE_RECORD_REQUIRES_LINUX_PROCFS")
        if sample_interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        self.root_pid = os.getpid() if root_pid is None else int(root_pid)
        self.sample_interval_seconds = float(sample_interval_seconds)
        self._observations: dict[tuple[int, int], _MutableObservation] = {}
        self._peak_tree_rss_kib = 0
        self._sample_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._record: ProcessTreeResourceRecord | None = None

    def _sample(self) -> None:
        queue = [self.root_pid]
        seen_pids: set[int] = set()
        tree_rss = 0
        while queue:
            pid = queue.pop()
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            result = _process_snapshot(pid)
            if result is None:
                continue
            current, children = result
            queue.extend(children)
            tree_rss += current.last_rss_kib
            identity = (current.pid, current.start_ticks)
            prior = self._observations.get(identity)
            if prior is None:
                self._observations[identity] = current
            else:
                prior.parent_pid = current.parent_pid
                prior.peak_rss_kib = max(prior.peak_rss_kib, current.peak_rss_kib)
                prior.last_rss_kib = current.last_rss_kib
                prior.user_ticks = max(prior.user_ticks, current.user_ticks)
                prior.system_ticks = max(prior.system_ticks, current.system_ticks)
                prior.samples += 1
        self._peak_tree_rss_kib = max(self._peak_tree_rss_kib, tree_rss)
        self._sample_count += 1

    def _sampling_loop(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self._sample()

    def __enter__(self) -> "ProcessTreeResourceRecorder":
        if self._thread is not None:
            raise RuntimeError("resource recorder cannot be reused")
        self._started_wall = time.monotonic()
        self._started_utc = datetime.now(timezone.utc)
        self._self_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._children_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        self._sample()
        self._thread = threading.Thread(
            target=self._sampling_loop, name="green-v400-process-tree-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self._sample()
        finished_wall = time.monotonic()
        finished_utc = datetime.now(timezone.utc)
        root = next(
            (item for key, item in self._observations.items() if key[0] == self.root_pid),
            None,
        )
        if root is None:
            raise RuntimeError("PROCESS_TREE_ROOT_NOT_OBSERVED")
        observations = tuple(
            ProcessObservation(**item.__dict__)
            for _, item in sorted(self._observations.items())
        )
        self._record = ProcessTreeResourceRecord(
            "green-v400-process-tree-resource-v1",
            "sampled_root_and_observed_descendants",
            False,
            self.root_pid,
            root.start_ticks,
            self._started_utc.isoformat().replace("+00:00", "Z"),
            finished_utc.isoformat().replace("+00:00", "Z"),
            finished_wall - self._started_wall,
            self.sample_interval_seconds,
            self._sample_count,
            True,
            self._peak_tree_rss_kib,
            root.peak_rss_kib,
            len(observations) - 1,
            self._self_before,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            self._children_before,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            observations,
            platform.platform(),
        )

    @property
    def record(self) -> ProcessTreeResourceRecord:
        if self._record is None:
            raise RuntimeError("resource record is available only after the measured scope exits")
        return self._record
