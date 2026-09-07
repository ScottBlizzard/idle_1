from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_protocol import PROTOCOL_ID, sha256_canonical
from green_v410_resource_calibration import build_minimum_budget_failfast_receipt
import green_v410_resource_supervisor as supervisor
from green_v410_resource_supervisor import build_resource_queue


def test_resource_queue_is_complete_bounded_parallel_and_phase_major(tmp_path):
    queue = build_resource_queue(
        tmp_path / "bundles", tmp_path / "raw", max_workers=8,
    )
    assert queue["job_count"] == 240
    assert [row["ordinal"] for row in queue["jobs"]] == list(range(240))
    for offset in range(0, 240, 60):
        candidate_jobs = queue["jobs"][offset:offset + 60]
        assert {row["precision_bits"] for row in candidate_jobs[:30]} == {384}
        assert {row["precision_bits"] for row in candidate_jobs[30:]} == {512}
        for official, audit in zip(candidate_jobs[:30], candidate_jobs[30:]):
            assert (official["profile"], official["fixture_kind"]) == (
                audit["profile"], audit["fixture_kind"]
            )
            assert audit["official_artifact_path"] == official["output_path"]
    assert queue["scheduler"] == (
        "phase_major_bounded_parallel_independent_cold_processes_v1"
    )
    assert queue["max_workers"] == 8


def _minimum_raw(*, rss_bytes: int) -> dict:
    payload = {
        "schema_version": "green-v410-resource-cold-process-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "mode": "official",
        "candidate_leaf_budget": 4,
        "profile": "ioi:layer0",
        "fixture_kind": "affine",
        "precision_bits": 384,
        "child_seed_uint64": 0,
        "fixture_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "program_semantic_hash": "3" * 64,
        "backend_library_sha256": "4" * 64,
        "dispatch_count": 9,
        "process_wall_seconds": 1.0,
        "single_pass_wall_seconds": 1.0 / 9.0,
        "process_tree_peak_rss_bytes": rss_bytes,
        "process_tree_resource_record": {"test_fixture": True},
        "max_depth": 3,
        "graph_nodes_metric": "root_only_peak_live_dependent_scalar_outputs_v1",
        "graph_nodes": 1000,
        "dependent_scalar_outputs_total": 5000,
        "executor_source_sha256": "5" * 64,
        "theorem_checks_pass": True,
        "nesting_checks_pass": None,
        "deterministic_replay": True,
        "schedule": {"dispatch_count": 9},
        "contains_scientific_outcome": False,
        "contains_endpoint_material": False,
    }
    payload["run_artifact_sha256"] = sha256_canonical(payload)
    return payload


def test_minimum_budget_guarded_memory_failure_is_terminal():
    receipt = build_minimum_budget_failfast_receipt(
        _minimum_raw(rss_bytes=57_530_040_320)
    )
    assert receipt["decision"] == "STOP_RESOURCE_LOCK_INFEASIBLE"
    assert receipt["failure_code"] == "MEMORY_GUARDBAND_EXCEEDED"
    assert receipt["larger_candidates_scheduled"] is False


def test_minimum_budget_within_guard_has_no_failfast_receipt():
    assert build_minimum_budget_failfast_receipt(
        _minimum_raw(rss_bytes=1024)
    ) is None


def test_supervisor_publishes_stop_before_scheduling_larger_candidates(
    tmp_path, monkeypatch,
):
    raw_root = tmp_path / "raw"
    existing = raw_root / "L4" / "384" / "ioi__layer0__affine.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("already validated by frozen loader", encoding="utf-8")
    queue = {
        "schema_version": "green-v410-resource-calibration-queue-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "scheduler": "strict_serial_one_cold_process_at_a_time",
        "job_count": 1,
        "jobs": [{
            "ordinal": 0,
            "candidate_leaf_budget": 4,
            "precision_bits": 384,
            "mode": "official",
            "profile": "ioi:layer0",
            "fixture_kind": "affine",
            "bundle_dir": (tmp_path / "bundle").as_posix(),
            "output_path": existing.as_posix(),
            "official_artifact_path": None,
        }],
        "contains_scientific_outcome": False,
        "contains_endpoint_material": False,
    }
    queue["queue_sha256"] = sha256_canonical(queue)
    monkeypatch.setattr(
        supervisor, "build_resource_queue", lambda *_args, **_kwargs: queue,
    )
    monkeypatch.setattr(
        supervisor, "_load_raw",
        lambda *_args, **_kwargs: _minimum_raw(rss_bytes=57_530_040_320),
    )

    decision = supervisor.run_supervisor(
        tmp_path / "bundles", tmp_path / "backend.so", raw_root,
    )

    assert decision == "STOP_RESOURCE_LOCK_INFEASIBLE"
    stop = json.loads((raw_root / "resource_failfast_stop.json").read_text())
    assert stop["failure_code"] == "MEMORY_GUARDBAND_EXCEEDED"
    assert stop["larger_candidates_scheduled"] is False
