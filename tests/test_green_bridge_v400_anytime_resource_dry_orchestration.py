from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import sha256_canonical
import green_v400_anytime_resource_dry_orchestration as dry
from green_v400_machine_concurrency_identity import (
    fixture_machine_concurrency_manifest,
)


def _manifest(*, mode="binding-semantic-dry", leaves=(14,) * 17, **changes):
    values = {
        "mode": mode,
        "output_root": "/mnt/sdb/ccj/resource-dry-fixture",
        "leaves_per_radius": leaves,
        "max_workers": 4,
        "wall_seconds": 30.0,
        "address_space_bytes": 1 << 30,
        "observed_tree_memory_bytes": 1 << 29,
        "sample_interval_seconds": 0.05,
        "fixture_seconds": 0.0,
        "fixture_memory_bytes": 0,
        "injected_failure_ordinal": None,
        "repository_commit": "a" * 40,
        "repository_clean": True,
        "source_sha256": {
            relative: "b" * 64 for relative in dry.SOURCE_RELATIVE_PATHS
        },
        "backend_path": "/mnt/sdb/ccj/native/libgreen.so",
        "backend_sha256": "c" * 64,
    }
    values.update(changes)
    values.setdefault(
        "machine_concurrency_manifest",
        fixture_machine_concurrency_manifest(
            max_workers=values["max_workers"],
            absolute_max_workers=dry.MAX_WORKERS,
            wall_seconds_per_process=values["wall_seconds"],
            per_process_address_space_bytes=values["address_space_bytes"],
            observed_tree_memory_bytes=values["observed_tree_memory_bytes"],
            sample_interval_seconds=values["sample_interval_seconds"],
            backend_kind="compiled-mpfr-native",
            backend_path=values["backend_path"],
            backend_sha256=values["backend_sha256"],
            backend_opened_by_workload=False,
        ),
    )
    return dry.build_manifest(**values)


def test_radii_are_exact_two_to_negative_zero_through_sixteen():
    manifest = _manifest(leaves=(2,) * 17)
    assert manifest["radius_exponents"] == list(range(17))
    assert [Fraction(*value) for value in manifest["radii"]] == [
        Fraction(1, 2**exponent) for exponent in range(17)
    ]
    assert manifest["center_reuse"] is False
    assert manifest["memoization"] is False


def test_worst_case_l14_is_nonbinding_and_exactly_493_plus_289():
    manifest = _manifest(mode="worst-case-L14-resource-stress")
    assert manifest["nonbinding"] is True
    assert manifest["binding_resource_failure_semantics"] is False
    assert manifest["expected_pass_counts"] == {
        "384": 493, "512": 289, "total": 782,
        "formula": "N384=sum(2L_r+1); N512=sum(L_r+3)",
    }
    jobs = manifest["jobs"]
    assert len(jobs) == 782
    assert [job["ordinal"] for job in jobs] == list(range(782))
    assert all(job["precision_bits"] == 384 for job in jobs[:493])
    assert all(job["precision_bits"] == 512 for job in jobs[493:])


def test_512_replays_the_same_frozen_exact_partition():
    leaves = tuple(2 + (index % 13) for index in range(17))
    manifest = _manifest(leaves=leaves)
    for radius_index, leaf_count in enumerate(leaves):
        radius = Fraction(1, 2**radius_index)
        _, frozen = dry._frozen_partition(radius, leaf_count)
        observed = [
            (Fraction(*job["lower"]), Fraction(*job["upper"]), job["depth"])
            for job in manifest["jobs"]
            if job["precision_bits"] == 512
            and job["radius_index"] == radius_index
            and job["pass_kind"] == "frozen_partition_leaf"
        ]
        assert observed == [
            (row["lower"], row["upper"], row["depth"]) for row in frozen
        ]


@pytest.mark.parametrize("mutation", [
    "outside_storage", "enable_center_reuse", "enable_memo", "interleave_phase",
    "drop_job", "too_many_workers", "stress_leaf_13", "binding_mark_nonbinding",
    "machine_cpu", "machine_workers", "machine_backend",
])
def test_manifest_validation_fails_closed_on_protocol_mutation(mutation):
    mode = "worst-case-L14-resource-stress" if mutation == "stress_leaf_13" else "binding-semantic-dry"
    manifest = deepcopy(_manifest(mode=mode))
    if mutation == "outside_storage":
        manifest["output_root"] = "/tmp/dry"
    elif mutation == "enable_center_reuse":
        manifest["center_reuse"] = True
    elif mutation == "enable_memo":
        manifest["memoization"] = True
    elif mutation == "interleave_phase":
        manifest["jobs"][0]["precision_bits"] = 512
    elif mutation == "drop_job":
        manifest["jobs"].pop()
    elif mutation == "too_many_workers":
        manifest["execution_policy"]["max_workers"] = 17
    elif mutation == "stress_leaf_13":
        manifest["leaves_per_radius"][0] = 13
    elif mutation == "binding_mark_nonbinding":
        manifest["nonbinding"] = True
    elif mutation == "machine_cpu":
        manifest["machine_concurrency_manifest"]["cpu"]["model"] = "bad\nmodel"
    elif mutation == "machine_workers":
        manifest["machine_concurrency_manifest"]["concurrency"]["max_workers"] = 3
    elif mutation == "machine_backend":
        manifest["machine_concurrency_manifest"]["backend_identity"]["sha256"] = "d" * 64
    with pytest.raises(ValueError):
        dry.validate_manifest(manifest)


def test_manifest_is_frozen_before_attempt_directory_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(dry, "_below_mnt_sdb", lambda _value: True)
    output_root = tmp_path / "dry"
    manifest = _manifest(output_root=output_root.as_posix())
    path, semantic_hash = dry.freeze_manifest(output_root, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.pop("manifest_semantic_hash") == semantic_hash
    assert sha256_canonical(payload) == semantic_hash
    assert not (output_root / "attempts").exists()


def _pass_row(job, *, status="PASS_DRY_ORCHESTRATION"):
    return {
        "ordinal": job["ordinal"], "pass_id": job["pass_id"],
        "precision_bits": job["precision_bits"], "status": status,
        "reason": None if status == "PASS_DRY_ORCHESTRATION" else "FIXTURE_FAILURE",
        "charged_passes": 1,
    }


def test_binding_resource_failure_stops_admission_and_launches_no_512():
    manifest = _manifest(leaves=(2,) * 17, max_workers=1)
    launched = []

    def runner(job):
        launched.append(job)
        return _pass_row(
            job, status=("RESOURCE_INCONCLUSIVE" if job["ordinal"] == 3
                         else "PASS_DRY_ORCHESTRATION"),
        )

    official, audit = dry.execute_schedule(manifest, runner)
    assert [row["ordinal"] for row in official] == [0, 1, 2, 3]
    assert audit == []
    assert all(job["precision_bits"] == 384 for job in launched)


def test_nonbinding_stress_executes_all_782_and_commits_canonically():
    manifest = _manifest(
        mode="worst-case-L14-resource-stress", max_workers=8,
    )
    official, audit = dry.execute_schedule(manifest, lambda job: _pass_row(job))
    assert len(official) == 493
    assert len(audit) == 289
    rows = official + audit
    assert [row["ordinal"] for row in rows] == list(range(782))


def test_worker_command_uses_external_supervisor_and_contains_no_scientific_input():
    manifest = _manifest(leaves=(2,) * 17)
    manifest_hash = dry.manifest_semantic_hash(manifest)
    command = dry._worker_command(manifest, manifest_hash, manifest["jobs"][0])
    joined = " ".join(command)
    assert "run_green_shared_host.py" in joined
    assert "green_v400_anytime_resource_dry_orchestration.py" in joined
    assert "--allow-descendants" not in command
    assert "threshold" not in joined.lower()
    assert "jet" not in joined.lower()


def test_receipt_rejects_any_extra_payload_field():
    manifest = _manifest(leaves=(2,) * 17)
    job = manifest["jobs"][0]
    report = {
        "schema_version": "green-v400-resource-dry-pass-v1",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "status": "PASS_RESOURCE_DRY_FIXTURE",
        "manifest_sha256": dry.manifest_semantic_hash(manifest),
        "pass_id": job["pass_id"], "ordinal": job["ordinal"],
        "precision_bits": job["precision_bits"], "lower": job["lower"],
        "upper": job["upper"], "gpu_environment": dry.GPU_ENVIRONMENT,
        "native_backend_loaded": False, "jet_evaluated": False,
        "scientific_threshold_read": False,
        "scientific_outcome_read_or_retained": False,
        "fixture_observation": {"elapsed_seconds": 0.0, "allocated_bytes": 0},
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    dry._validate_receipt(report, report["manifest_sha256"], job)
    report["numeric_payload"] = [1, 2, 3]
    report["report_semantic_hash"] = sha256_canonical({
        key: value for key, value in report.items() if key != "report_semantic_hash"
    })
    with pytest.raises(RuntimeError, match="RECEIPT_INVALID"):
        dry._validate_receipt(report, dry.manifest_semantic_hash(manifest), job)
