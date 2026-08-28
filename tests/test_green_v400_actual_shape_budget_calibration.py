from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import sha256_canonical
from green_v400_machine_concurrency_identity import (
    fixture_machine_concurrency_manifest,
)
import green_v400_actual_shape_budget_calibration as calibration


def _manifest(**changes):
    values = {
        "output_root": "/mnt/sdb/ccj/budget-calibration-fixture",
        "library_path": "/mnt/sdb/ccj/native/libgreen.so",
        "library_sha256": "a" * 64,
        "descriptor_path": "/mnt/sdb/ccj/native/plan.desc",
        "blob_path": "/mnt/sdb/ccj/native/resident.bin",
        "max_depth": 24,
        "wall_seconds_per_budget": 7200.0,
        "address_space_bytes": 4 << 30,
        "observed_tree_memory_bytes": 3 << 30,
        "sample_interval_seconds": 0.05,
        "repository_commit": "b" * 40,
        "repository_clean": True,
        "source_sha256": {
            path: "c" * 64 for path in calibration.SOURCE_RELATIVE_PATHS
        },
        "api_preflight": calibration.inspect_required_api_capabilities(),
    }
    values.update(changes)
    values.setdefault(
        "machine_concurrency_manifest",
        fixture_machine_concurrency_manifest(
            max_workers=1, absolute_max_workers=1,
            wall_seconds_per_process=values["wall_seconds_per_budget"],
            per_process_address_space_bytes=values["address_space_bytes"],
            observed_tree_memory_bytes=values["observed_tree_memory_bytes"],
            sample_interval_seconds=values["sample_interval_seconds"],
            backend_kind="compiled-mpfr-native",
            backend_path=values["library_path"],
            backend_sha256=values["library_sha256"],
            backend_opened_by_workload=True,
        ),
    )
    return calibration.build_manifest(**values)


def test_current_api_preflight_authorizes_only_the_closed_calibration_interfaces():
    preflight = calibration.inspect_required_api_capabilities()
    assert preflight["official_anytime_state_api_present"] is True
    assert preflight["compiled_native_explicit_synthetic_only_boundary"] is True
    assert preflight["public_512_frozen_partition_replay_and_nesting_api"] is True
    assert preflight["execution_ready"] is True
    assert preflight["blockers"] == []


def test_four_budgets_and_worst_case_17_radius_counts_are_frozen():
    manifest = _manifest()
    assert manifest["candidate_final_leaf_budgets"] == [4, 8, 16, 32]
    assert [job["leaf_budget"] for job in manifest["jobs"]] == [4, 8, 16, 32]
    expected = {
        4: (153, 153, 306),
        8: (289, 289, 578),
        16: (561, 561, 1122),
        32: (1105, 1105, 2210),
    }
    for job in manifest["jobs"]:
        assert (
            job["maximum_charged_passes_384"],
            job["maximum_charged_passes_512"],
            job["maximum_charged_passes_total"],
        ) == expected[job["leaf_budget"]]
    assert calibration.exact_no_cache_counts([14] * 17) == {
        "384": 493, "512": 493, "total": 986,
    }
    assert calibration._exact_ratio_payload(deepcopy(calibration.RADIUS)) == [1, 16384]


def test_dual_track_phase_replay_and_artifact_separation_are_explicit():
    manifest = _manifest()
    assert manifest["dual_track_policy"] == {
        "continuation_to_32_anchor_radius": [1, 16384],
        "continuation_prefix_leaf_counts": [4, 8, 16, 32],
        "continuation_same_path_required": True,
        "standalone_fresh_process_per_budget": True,
        "standalone_all_17_radii": True,
    }
    accounting = manifest["accounting_policy"]
    assert accounting["per_budget_all_17_radii_384_before_any_512"] is True
    assert accounting["per_radius_no_cache_count_formula"] == (
        "384=2L+1;512=2L+1;total=4L+2"
    )
    assert accounting["any_384_failure_launches_zero_512"] is True
    assert accounting["memoization"] is False
    assert accounting["center_reuse"] is False
    separation = manifest["artifact_separation_policy"]
    assert separation["selector_safe_resource_artifact"][
        "interval_width_forbidden"
    ] is True
    assert separation["selector_inaccessible_numerics_artifact"][
        "jet_payload_forbidden"
    ] is True
    assert manifest["continuation_job"] == {
        "leaf_budget": 32,
        "radius": [1, 16384],
        "checkpoint_leaf_counts": [4, 8, 16, 32],
        "attempt_relative_path": "attempts/continuation_to_L32",
        "numerics_relative_path": (
            "attempts/continuation_to_L32/numerics_audit/continuation_report.json"
        ),
    }


@pytest.mark.parametrize("mutation", [
    "budget", "radius", "memo", "phase", "selector", "api", "machine",
    "descriptor", "outside",
])
def test_manifest_mutations_fail_closed(mutation):
    manifest = deepcopy(_manifest())
    if mutation == "budget":
        manifest["candidate_final_leaf_budgets"] = [4, 8, 16, 31]
    elif mutation == "radius":
        manifest["closed_synthetic_identity"]["continuation_anchor_radius"] = [1, 8192]
    elif mutation == "memo":
        manifest["accounting_policy"]["memoization"] = True
    elif mutation == "phase":
        manifest["accounting_policy"]["any_384_failure_launches_zero_512"] = False
    elif mutation == "selector":
        manifest["selector_policy"]["width_sign_p13_or_numerics_input_forbidden"] = False
    elif mutation == "api":
        manifest["api_preflight"]["execution_ready"] = False
    elif mutation == "machine":
        manifest["machine_concurrency_manifest"]["cpu"]["logical_cpu_count"] = 31
    elif mutation == "descriptor":
        manifest["native_artifacts"]["descriptor_sha256"] = "d" * 64
    elif mutation == "outside":
        manifest["output_root"] = "/tmp/calibration"
    with pytest.raises(ValueError):
        calibration.validate_manifest(manifest)


def test_manifest_freezes_before_attempts_and_blocker_precedes_launch(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(calibration, "_below_mnt_sdb", lambda _value: True)
    output_root = tmp_path / "calibration"
    blocked = calibration.inspect_required_api_capabilities() | {
        "compiled_native_explicit_synthetic_only_boundary": False,
        "blockers": [calibration.BLOCKER_SYNTHETIC_BOUNDARY],
        "execution_ready": False,
    }
    manifest = _manifest(
        output_root=output_root.as_posix(), api_preflight=blocked,
    )
    path, semantic_hash = calibration.freeze_manifest(output_root, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.pop("manifest_semantic_hash") == semantic_hash
    assert sha256_canonical(payload) == semantic_hash
    assert not (output_root / "attempts").exists()
    monkeypatch.setattr(calibration, "_verify_frozen_resources", lambda _manifest: None)
    monkeypatch.setattr(
        calibration, "verify_current_machine_concurrency_manifest",
        lambda _manifest: None,
    )
    report_path = calibration.write_blocker_report(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED_BEFORE_ANY_BUDGET_PROCESS"
    assert report["budget_processes_launched"] == 0
    assert report["charged_native_dispatches"] == 0
    assert not (output_root / "attempts").exists()


def _resource_records(manifest):
    machine_hash = manifest["machine_concurrency_manifest"][
        "machine_manifest_semantic_hash"
    ]
    records = [
        {
            "budget": budget,
            "job_ordinal": ordinal,
            "manifest_semantic_hash": calibration.manifest_semantic_hash(manifest),
            "charged_pass_counts": {"384": 100, "512": 50, "total": 150},
            "fault_reason": None,
            "timing_seconds": 100.0 * budget,
            "rss_bytes": 1 << 30,
            "machine_manifest_hash": machine_hash,
            "wrapper_report_semantic_hash": "d" * 64,
            "numerics_report_semantic_hash": "e" * 64,
        }
        for ordinal, budget in enumerate((4, 8, 16, 32))
    ]
    for record in records:
        record["record_semantic_hash"] = sha256_canonical(record)
    return records


def test_selector_uses_only_resource_fields_and_largest_guardband_safe_budget():
    manifest = _manifest(wall_seconds_per_budget=3000.0)
    records = _resource_records(manifest)
    assert calibration.select_largest_resource_safe_budget(manifest, records) == 16
    records[2]["interval_width"] = [1, 2]
    with pytest.raises(ValueError, match="fields invalid"):
        calibration.select_largest_resource_safe_budget(manifest, records)


@pytest.mark.parametrize("fault", [
    "WORKER_FAILED", "SUPERVISOR_INFRASTRUCTURE_FAILED",
    "SUPERVISOR_CLEANUP_FAILED",
])
def test_nonresource_fault_is_fail_closed_on_new_or_resumed_record(fault):
    with pytest.raises(RuntimeError, match="NONRESOURCE_FAILURE_FAIL_CLOSED"):
        calibration._raise_on_nonresource_fault({"fault_reason": fault})
    calibration._raise_on_nonresource_fault({"fault_reason": "WALL_DEADLINE_REACHED"})
    calibration._raise_on_nonresource_fault({"fault_reason": None})


def test_planned_budget_command_is_fresh_supervised_and_has_split_outputs():
    manifest = _manifest()
    command = calibration.budget_worker_command(
        manifest, calibration.manifest_semantic_hash(manifest), manifest["jobs"][0]
    )
    joined = " ".join(command)
    assert "run_green_shared_host.py" in joined
    assert "--budget-worker" in command
    assert "--selector-safe-resource-output" in command
    assert "--selector-inaccessible-numerics-output" in command
    assert "threshold" not in joined.lower()
    assert "--allow-descendants" not in command
    continuation = calibration.continuation_worker_command(
        manifest, calibration.manifest_semantic_hash(manifest)
    )
    assert "--continuation-worker" in continuation
    assert "--continuation-numerics-output" in continuation
    assert "--allow-descendants" not in continuation
