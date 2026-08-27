from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_schemas import CertificateResourceLock, RESOURCE_REASONS
from green_bridge_v400_supervisor import (
    AdmissionLedger, SupervisorViolation, atomic_publish,
    authorize_supervised_execution, configure_worker_cgroup,
    LinuxMonotonicDeadline, probe_cgroup_v2, validate_staged_commit,
)


def _lock():
    return CertificateResourceLock(
        "green-v400-certificate-resource-lock-v1", "0"*64, "9"*64,
        "a"*64, 17, "ALL_384_THEN_REPLAY_SAME_PARTITION_512", 384, 512,
        24, 14, False, 3, True, False, "green-v400-fte-pass-v1",
        "green-v400-directed-primitives-v1", 352_275_450, 90, 100,
        75_600, 10_800, 86_400, 68_719_476_736, False, False, 1,
        "cgroup_v2_memory.max", "cgroup_v2_memory.swap.max=0",
        "external_monotonic_supervisor_v1",
        "outside_worker_cgroup_pidfd_timerfd",
        "pre_exec_validation_through_atomic_publish",
        "TWO_PHASE_SUPERVISOR_COMMIT", RESOURCE_REASONS,
        ("MAX_FINAL_LEAVES_PER_RADIUS_REACHED", "WALL_DEADLINE_REACHED",
         "MEMORY_MAX_REACHED"), "b"*40, "c"*64, "d"*64, "e"*64,
        "1"*64, "2"*64, "3"*64, "4"*64, "5"*64, "6"*64,
        "7"*64, "8"*64, False,
    )


def test_admission_charges_before_execution_and_never_refunds(tmp_path):
    ledger = AdmissionLedger(_lock(), tmp_path / "ledger.jsonl")
    first = ledger.admit(384)
    ledger.freeze_official_phase()
    second = ledger.admit(512)
    ledger.failure_without_refund(first.ordinal)
    assert (first.charged_tokens, second.charged_tokens) == (90, 100)
    assert ledger.charged_tokens == 190
    assert ledger.remaining_tokens == 75_410
    assert ledger.to_dict()["failed_dispatch_refund"] is False
    assert "ADMITTED_PASS_FAILED_NO_REFUND" in ledger.ledger_path.read_text()


def test_admission_rejects_precision_and_budget_exhaustion(tmp_path):
    ledger = AdmissionLedger(_lock(), tmp_path / "ledger.jsonl")
    with pytest.raises(SupervisorViolation, match="unsupported precision"):
        ledger.admit(256)
    ledger.freeze_official_phase()
    for _ in range(756):
        ledger.admit(512)
    with pytest.raises(SupervisorViolation, match="cannot admit"):
        ledger.admit(512)


def _stage(tmp_path, lock, **changes):
    staging = tmp_path / "stage"
    staging.mkdir()
    artifact = staging / "certificate.json"
    artifact.write_text('{"contains_scientific_outcome":false}\n', encoding="utf-8")
    payload = {
        "schema_version": "green-v400-supervisor-commit-manifest-v1",
        "resource_lock_semantic_hash": lock.semantic_hash(),
        "attempt_id": "synthetic-audit-1", "status": "INTERVAL_COMPUTED",
        "resource_reason": None, "scientific_threshold_applied": False,
        "files": {"certificate.json": {
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "nbytes": artifact.stat().st_size,
        }},
    } | changes
    (staging / "supervisor_commit_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    return staging


def test_two_phase_validate_and_atomic_publish(tmp_path):
    lock = _lock()
    staging = _stage(tmp_path, lock)
    validated = validate_staged_commit(staging, lock)
    assert validated["status"] == "INTERVAL_COMPUTED"
    published = tmp_path / "published"
    atomic_publish(staging, published)
    assert not staging.exists()
    assert (published / "certificate.json").is_file()


@pytest.mark.parametrize("changes", [
    {"scientific_threshold_applied": True},
    {"resource_lock_semantic_hash": "f"*64},
    {"status": "CERTIFIED"},
    {"status": "RESOURCE_INCONCLUSIVE", "resource_reason": "MAX_DEPTH_REACHED"},
    {"files": {"../escape": {"sha256": "0"*64, "nbytes": 1}}},
    {"files": {"certificate.json": {"sha256": "0"*64, "nbytes": 39}}},
])
def test_commit_validation_fails_closed(tmp_path, changes):
    lock = _lock()
    staging = _stage(tmp_path, lock, **changes)
    with pytest.raises(SupervisorViolation):
        validate_staged_commit(staging, lock)


def test_cgroup_probe_detects_hybrid_without_v2_memory_controller(tmp_path):
    mount = tmp_path / "unified"
    process = mount / "user.slice" / "session.scope"
    process.mkdir(parents=True)
    (process / "cgroup.controllers").write_text("\n", encoding="ascii")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"1 0 0:1 / {mount} rw - cgroup2 cgroup2 rw\n", encoding="utf-8",
    )
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/user.slice/session.scope\n", encoding="utf-8")
    availability = probe_cgroup_v2(
        mountinfo_path=mountinfo, cgroup_path=cgroup,
    )
    assert availability.controllers == ()
    assert not availability.hard_memory_gate_ready


def test_configure_worker_cgroup_requires_readback_and_swap_zero(tmp_path):
    for name, value in {
        "memory.max": "max\n", "memory.swap.max": "max\n",
        "memory.events": "oom 0\noom_kill 0\n", "cgroup.procs": "",
    }.items():
        (tmp_path / name).write_text(value, encoding="ascii")
    record = configure_worker_cgroup(tmp_path, 4096)
    assert record["memory_max_bytes"] == 4096
    assert (tmp_path / "memory.max").read_text().strip() == "4096"
    assert (tmp_path / "memory.swap.max").read_text().strip() == "0"


def test_prepare_only_lock_cannot_authorize_worker():
    with pytest.raises(SupervisorViolation, match="prepare-only"):
        authorize_supervised_execution(_lock())


@pytest.mark.skipif(os.name != "posix" or not hasattr(os, "fork"), reason="Linux only")
def test_pidfd_timerfd_observes_clean_worker_exit():
    with LinuxMonotonicDeadline(1.0) as deadline:
        pid = os.fork()
        if pid == 0:
            os._exit(7)
        result = deadline.wait_worker(pid)
        assert not result.deadline_reached
        assert result.exit_code == 7
        assert result.termination_signal is None
        deadline.assert_publish_window()


@pytest.mark.skipif(os.name != "posix" or not hasattr(os, "fork"), reason="Linux only")
def test_pidfd_timerfd_kills_worker_at_absolute_deadline():
    with LinuxMonotonicDeadline(0.05) as deadline:
        pid = os.fork()
        if pid == 0:
            import time
            time.sleep(10)
            os._exit(0)
        result = deadline.wait_worker(pid)
        assert result.deadline_reached
        assert result.exit_code is None
        assert result.termination_signal == 9
        with pytest.raises(SupervisorViolation, match="before atomic publication"):
            deadline.assert_publish_window()
