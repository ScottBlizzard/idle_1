from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_schemas import RESOURCE_REASONS
from green_bridge_v400_strict_resource_lock import (
    StrictSingleProcessResourceLock,
)
from green_bridge_v400_supervisor import (
    AdmissionLedger, SupervisorViolation, authorize_supervised_execution,
)


def _lock(**changes) -> StrictSingleProcessResourceLock:
    payload = {
        "schema_version": "green-v400-certificate-resource-lock-v2",
        "row_hash": "a" * 64,
        "certificate_plan_semantic_hash": "b" * 64,
        "radii_order_sha256": "c" * 64,
        "radii_count": 17,
        "phase_order": "ALL_384_THEN_FULL_HISTORY_512",
        "official_precision": 384,
        "audit_precision": 512,
        "audit_history_policy": (
            "COMPLETE_FROZEN_OFFICIAL_SPLIT_HISTORY_INDEPENDENT_RECURRENCE"
        ),
        "max_depth": 24,
        "max_final_leaves_per_radius": 14,
        "center_reuse": False,
        "charge_on_admission": True,
        "failed_dispatch_refund": False,
        "token_weight_384": 90,
        "token_weight_512": 100,
        "token_budget": 100000,
        "orchestration_reserve_seconds": 1000,
        "wall_deadline_seconds": 101000,
        "user_address_space_max_bytes": 4 << 30,
        "worker_concurrency": 1,
        "memory_enforcement": "RLIMIT_AS_HARD_EQUAL_SOFT",
        "process_creation_enforcement": "RLIMIT_NPROC_HARD_EQUAL_SOFT_ONE",
        "numeric_thread_environment_sha256": "d" * 64,
        "swap_policy": "WITHIN_HARD_ADDRESS_SPACE_NOT_SEPARATELY_DISABLED",
        "deadline_enforcement": "EXTERNAL_MONOTONIC_PIDFD_TIMERFD_V2",
        "supervisor_process_scope": "OUTSIDE_WORKER_PROCESS_GROUP",
        "deadline_scope": "PRE_EXEC_VALIDATION_THROUGH_ATOMIC_PUBLISH",
        "publication_policy": "TWO_PHASE_SUPERVISOR_COMMIT",
        "resource_reasons": RESOURCE_REASONS,
        "reachable_primary_reasons": (
            "MAX_FINAL_LEAVES_PER_RADIUS_REACHED",
            "WALL_DEADLINE_REACHED",
            "MEMORY_MAX_REACHED",
        ),
        "repository_commit": "e" * 40,
        "source_closure_sha256": "f" * 64,
        "strict_wrapper_report_semantic_hash": "1" * 64,
        "strict_numerics_report_semantic_hash": "2" * 64,
        "backend_sha256": "3" * 64,
        "descriptor_sha256": "4" * 64,
        "blob_sha256": "5" * 64,
        "program_execution_sha256": "6" * 64,
        "dispatch_sha256": "7" * 64,
        "fusion_sha256": "8" * 64,
        "rounding_environment_sha256": "9" * 64,
        "hardware_manifest_sha256": "0" * 64,
        "production_authorized": False,
    }
    return StrictSingleProcessResourceLock(**(payload | changes))


def test_strict_resource_lock_uses_full_history_cost_and_roundtrips():
    lock = _lock()
    assert lock.worst_case_passes_384 == 493
    assert lock.worst_case_passes_512 == 493
    assert lock.worst_case_total_passes == 986
    assert lock.worst_case_token_charge == 93670
    assert StrictSingleProcessResourceLock.from_dict(lock.to_dict()) == lock
    assert len(lock.semantic_hash()) == 64


@pytest.mark.parametrize("field,value", [
    ("production_authorized", True),
    ("phase_order", "ALL_384_THEN_REPLAY_SAME_PARTITION_512"),
    ("audit_history_policy", "CONSTRUCTIVE_INTERSECTION"),
    ("center_reuse", True),
    ("worker_concurrency", 2),
    ("memory_enforcement", "cgroup_v2_memory.max"),
    ("process_creation_enforcement", "PROC_SAMPLING"),
    ("swap_policy", "SWAP_DISABLED_CLAIMED"),
])
def test_strict_resource_lock_rejects_protocol_drift(field, value):
    with pytest.raises(ValueError):
        _lock(**{field: value})


def test_strict_resource_lock_rejects_underbudgeted_full_history():
    with pytest.raises(ValueError, match="exceeds token budget"):
        _lock(token_budget=90000, wall_deadline_seconds=91000)


def test_strict_resource_lock_is_immutable():
    lock = _lock()
    with pytest.raises(Exception):
        replace(lock, production_authorized=True)


def test_strict_resource_lock_drives_supervisor_ledger_but_not_launch(tmp_path):
    lock = _lock(radii_count=1, max_final_leaves_per_radius=2)
    ledger = AdmissionLedger(lock, tmp_path / "strict-ledger.jsonl")
    for ordinal in range(5):
        token = f"official-{ordinal}"
        record = ledger.admit(
            384, token_id=token, attempt_id="strict-attempt",
            exact_domain_sha256=f"{ordinal + 1:064x}",
        )
        assert record.charged_tokens == 90
        ledger.mark_dispatch_started(token)
        ledger.mark_dispatch_finished(token, success=True)
    ledger.freeze_official_phase(
        completed_radius_count=1,
        official_partition_manifest_sha256="a" * 64,
    )
    audit = ledger.admit(
        512, token_id="audit-0", attempt_id="strict-attempt",
        exact_domain_sha256="f" * 64,
    )
    assert audit.charged_tokens == 100
    assert ledger.charged_tokens == 550
    with pytest.raises(
            SupervisorViolation, match="resource lock remains prepare-only"):
        authorize_supervised_execution(lock)
