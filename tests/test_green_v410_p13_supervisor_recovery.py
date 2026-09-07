"""Orchestration must stop new work but retain ownership of live children."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("fcntl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "p13_supervisor_recovery", ROOT / "analysis/green_v410_p13_supervisor.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def test_hard_failure_drains_live_sibling_without_launching_next(tmp_path, monkeypatch):
    launches, snapshots = [], []
    class Process:
        def __init__(self, unit):
            self.unit, self.polls, self.pid = unit, 0, 100 + len(launches)
        def poll(self):
            self.polls += 1
            if self.unit == "bad":
                return 1
            return None if self.polls == 1 else 0
    def launch(command, **kwargs):
        launches.append(command[0])
        return Process(command[0])
    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    monkeypatch.setattr(supervisor, "_mutable_status", lambda p, value: snapshots.append(value))
    with pytest.raises(RuntimeError, match="HARD_WORKER_FAILURE"):
        supervisor._run_fleet(
            units=[(key, [key]) for key in ("bad", "live", "not_started")],
            parallelism=2, log_root=tmp_path / "logs", status_path=tmp_path / "status.json",
            environment={}, retry_delay_seconds=0, allow_transient_retries=False)
    assert launches == ["bad", "live"]
    assert snapshots[0]["state"] == "FAILED_DRAINING"
    assert snapshots[0]["active_units"] == ["live"]
    assert snapshots[-1]["state"] == "FAILED"
    assert snapshots[-1]["active_units"] == []
    assert snapshots[-1]["completed_units"] == 1
    assert snapshots[-1]["pending_units"] == 1
    assert not (tmp_path / "logs/not_started.attempt_001.log").exists()


def test_spawn_failure_does_not_launch_following_units(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("temporarily unavailable")
    monkeypatch.setattr(supervisor.subprocess, "Popen", fail)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="WORKER_LAUNCH_FAILED"):
        supervisor._run_fleet(
            units=[("first", ["first"]), ("next", ["next"])], parallelism=2,
            log_root=tmp_path / "logs", status_path=tmp_path / "status.json",
            environment={}, retry_delay_seconds=0, allow_transient_retries=False)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "FAILED"
    assert status["pending_units"] == 1
    assert not (tmp_path / "logs/next.attempt_001.log").exists()
