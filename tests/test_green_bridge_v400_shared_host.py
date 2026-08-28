from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import green_bridge_v400_shared_host as shared_host
from green_bridge_v400_shared_host import (
    SharedHostResourcePolicy, _parse_proc_status, _tree_pids,
    run_shared_host_command,
)


def test_shared_host_policy_rejects_invalid_limits():
    with pytest.raises(ValueError, match="wall deadline"):
        SharedHostResourcePolicy(0, 1024, 1024)
    with pytest.raises(ValueError, match="address-space"):
        SharedHostResourcePolicy(1, 0, 1024)
    with pytest.raises(ValueError, match="tree-memory"):
        SharedHostResourcePolicy(1, 1024, 0)
    with pytest.raises(ValueError, match="sample interval"):
        SharedHostResourcePolicy(1, 1024, 1024, 0.001)
    with pytest.raises(ValueError, match="hard_single_process"):
        SharedHostResourcePolicy(
            1, 1024, 1024, allow_descendant_processes=True,
            hard_single_process=True,
        )


def test_proc_status_and_tree_parsing():
    assert _parse_proc_status(
        "Name:\tpython\nPPid:\t42\nVmRSS:\t11 kB\nVmSwap:\t3 kB\n"
    ) == (42, 11 * 1024, 3 * 1024)
    table = {
        10: (1, 100, 0),
        11: (10, 50, 2),
        12: (11, 25, 0),
        99: (1, 999, 0),
    }
    assert _tree_pids(10, table) == {10, 11, 12}


LINUX_PROC = os.name == "posix" and Path("/proc/self/status").is_file()


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_shared_host_runner_records_completed_command(tmp_path):
    result = run_shared_host_command(
        [sys.executable, "-c", "print('ok')"], cwd=tmp_path,
        attempt_directory=tmp_path / "run",
        policy=SharedHostResourcePolicy(5, 512 << 20, 256 << 20, 0.01),
    )
    report = json.loads(Path(result.report_path).read_text())
    assert result.status == "COMPLETED"
    assert report["guarantee_scope"]["cgroup_v2_enforcement_claimed"] is False
    assert report["guarantee_scope"]["complete_process_tree_containment_claimed"] is False
    assert report["permitted_job_scope"] == "trusted_non_certificate_experiment_resource_record"
    assert report["observations"]["cleanup"]["cleanup_verified"] is True


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_hard_single_process_runner_blocks_fork_and_threads(tmp_path):
    evidence_path = tmp_path / "kernel-lock-evidence.json"
    program = (
        "import errno,json,os,threading; "
        "out={'fork_blocked':False,'thread_blocked':False}; "
        "\ntry:\n pid=os.fork()\n"
        "except OSError as e:\n out['fork_blocked']=e.errno==errno.EAGAIN\n"
        "else:\n"
        "\n if pid==0: os._exit(7)\n"
        "\n os.waitpid(pid,0)\n"
        "\ntry:\n t=threading.Thread(target=lambda:None);t.start();t.join()\n"
        "except RuntimeError:\n out['thread_blocked']=True\n"
        f"\nopen({str(evidence_path)!r},'w').write(json.dumps(out))\n"
        "assert out=={'fork_blocked':True,'thread_blocked':True}"
    )
    result = run_shared_host_command(
        [sys.executable, "-c", program], cwd=tmp_path,
        attempt_directory=tmp_path / "hard-single-process",
        policy=SharedHostResourcePolicy(
            5, 512 << 20, 256 << 20, 0.01,
            hard_single_process=True,
        ),
    )
    report = json.loads(Path(result.report_path).read_text())
    assert result.status == "COMPLETED"
    assert json.loads(evidence_path.read_text()) == {
        "fork_blocked": True, "thread_blocked": True,
    }
    guarantees = report["guarantee_scope"]
    assert guarantees["hard_single_process_creation_limit"] is True
    assert guarantees["hard_aggregate_user_space_address_space_upper_bound"] is True
    assert guarantees["complete_process_tree_containment_claimed"] is True
    assert guarantees["cgroup_v2_enforcement_claimed"] is False
    assert report["permitted_job_scope"] == (
        "trusted_hard_single_process_resource_lock_candidate"
    )


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_shared_host_runner_kills_at_monotonic_deadline(tmp_path):
    result = run_shared_host_command(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
        attempt_directory=tmp_path / "timeout",
        policy=SharedHostResourcePolicy(0.08, 512 << 20, 256 << 20, 0.01),
    )
    assert result.status == "WALL_DEADLINE_REACHED"
    assert result.termination_signal == signal.SIGKILL


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_shared_host_runner_rejects_descendant_process(tmp_path):
    result = run_shared_host_command(
        [sys.executable, "-c", (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
            "time.sleep(30)"
        )],
        cwd=tmp_path, attempt_directory=tmp_path / "descendant",
        policy=SharedHostResourcePolicy(5, 512 << 20, 256 << 20, 0.01),
    )
    assert result.status == "DESCENDANT_PROCESS_POLICY_REACHED"
    assert result.termination_signal == signal.SIGKILL


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_shared_host_runner_cleans_child_when_root_exits_between_samples(tmp_path):
    result = run_shared_host_command(
        [sys.executable, "-c", (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])"
        )],
        cwd=tmp_path, attempt_directory=tmp_path / "fast-descendant",
        policy=SharedHostResourcePolicy(5, 512 << 20, 256 << 20, 1.0),
    )
    assert result.status == "DESCENDANT_PROCESS_POLICY_REACHED"


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_shared_host_runner_cleans_observed_setsid_descendant(tmp_path):
    result = run_shared_host_command(
        [sys.executable, "-c", (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
            "start_new_session=True); time.sleep(30)"
        )],
        cwd=tmp_path, attempt_directory=tmp_path / "setsid-descendant",
        policy=SharedHostResourcePolicy(5, 512 << 20, 256 << 20, 0.01),
    )
    report = json.loads(Path(result.report_path).read_text())
    assert result.status == "DESCENDANT_PROCESS_POLICY_REACHED"
    assert report["observations"]["cleanup"]["cleanup_verified"] is True


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_shared_host_runner_deadline_not_hidden_by_long_sample_interval(tmp_path):
    result = run_shared_host_command(
        [sys.executable, "-c", "import time;time.sleep(.3)"], cwd=tmp_path,
        attempt_directory=tmp_path / "timerfd-boundary",
        policy=SharedHostResourcePolicy(0.1, 512 << 20, 256 << 20, 1.0),
    )
    assert result.status == "WALL_DEADLINE_REACHED"
    assert result.elapsed_seconds < 0.5


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_shared_host_runner_cleans_worker_when_sampler_fails(tmp_path, monkeypatch):
    def fail_sample(_pid):
        raise OSError("injected sampler failure")

    monkeypatch.setattr(shared_host, "sample_linux_process_tree", fail_sample)
    result = run_shared_host_command(
        [sys.executable, "-c", "import time;time.sleep(30)"], cwd=tmp_path,
        attempt_directory=tmp_path / "sampler-failure",
        policy=SharedHostResourcePolicy(5, 512 << 20, 256 << 20, 0.01),
    )
    report = json.loads(Path(result.report_path).read_text())
    assert result.status == "SUPERVISOR_INFRASTRUCTURE_FAILED"
    assert result.termination_signal == signal.SIGKILL
    assert report["observations"]["cleanup"]["cleanup_verified"] is True


@pytest.mark.skipif(not LINUX_PROC, reason="Linux /proc-only shared-host runner")
def test_shared_host_runner_reports_exec_failure_and_cleans(tmp_path):
    result = run_shared_host_command(
        ["/definitely/not/a/real/green-command"], cwd=tmp_path,
        attempt_directory=tmp_path / "exec-failure",
        policy=SharedHostResourcePolicy(5, 512 << 20, 256 << 20, 0.01),
    )
    report = json.loads(Path(result.report_path).read_text())
    assert result.status == "WORKER_FAILED"
    assert result.exit_code not in {None, 0}
    assert report["observations"]["cleanup"]["cleanup_verified"] is True
