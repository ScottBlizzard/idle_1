from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_schemas import sha256_canonical
import green_v400_native_cold_matrix as cold
import green_v400_native_cold_identity as identity
from green_v400_machine_concurrency_identity import (
    fixture_machine_concurrency_manifest,
)


def _manifest(**changes):
    values = {
        "output_root": "/mnt/sdb/ccj/cold-matrix-test",
        "library_path": "/mnt/sdb/ccj/native/libgreen.so",
        "library_sha256": "a" * 64,
        "descriptor_path": "/mnt/sdb/ccj/native/plan.desc",
        "blob_path": "/mnt/sdb/ccj/native/resident.bin",
        "max_workers": 8,
        "per_sample_wall_seconds": 300.0,
        "address_space_bytes": 2 << 30,
        "observed_tree_memory_bytes": 1 << 30,
        "sample_interval_seconds": 0.05,
        "repository_commit": "b" * 40,
        "repository_clean": True,
        "source_sha256": {
            relative_path: "c" * 64 for relative_path in cold.SOURCE_RELATIVE_PATHS
        },
    }
    values.update(changes)
    values.setdefault(
        "machine_concurrency_manifest",
        fixture_machine_concurrency_manifest(
            max_workers=values["max_workers"],
            absolute_max_workers=cold.MAX_CONCURRENT_PROCESSES,
            wall_seconds_per_process=values["per_sample_wall_seconds"],
            per_process_address_space_bytes=values["address_space_bytes"],
            observed_tree_memory_bytes=values["observed_tree_memory_bytes"],
            sample_interval_seconds=values["sample_interval_seconds"],
            backend_kind="compiled-mpfr-native",
            backend_path=values["library_path"],
            backend_sha256=values["library_sha256"],
            backend_opened_by_workload=True,
        ),
    )
    return cold.build_manifest(**values)


def test_frozen_matrix_is_exactly_30_per_precision_in_phase_order():
    manifest = _manifest()
    jobs = manifest["jobs"]
    assert len(jobs) == 60
    assert [row["ordinal"] for row in jobs] == list(range(60))
    assert [row["precision_bits"] for row in jobs[:30]] == [384] * 30
    assert [row["precision_bits"] for row in jobs[30:]] == [512] * 30
    for precision in (384, 512):
        observed = Counter(
            row["domain_class"] for row in jobs
            if row["precision_bits"] == precision
        )
        assert observed == Counter(cold.CATEGORY_COUNTS)
    assert manifest["gpu_environment"] == {
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "gpu_used": False,
    }
    assert manifest["execution_policy"]["max_workers"] == 8
    assert manifest["execution_policy"]["all_384_complete_before_512_launch"] is True


def test_domains_are_exact_fractions_and_cover_all_seven_classes():
    templates = cold._domain_templates()
    assert len(templates) == 30
    assert {row["domain_class"] for row in templates} == set(cold.DOMAIN_CLASSES)
    for row in templates:
        lower = Fraction(*row["lower"])
        upper = Fraction(*row["upper"])
        assert lower <= upper
        assert row["lower"] == [lower.numerator, lower.denominator]
        assert row["upper"] == [upper.numerator, upper.denominator]
    deep_positive = [
        row for row in templates if row["domain_class"] == "deep_positive_dyadic"
    ]
    for row, exponent in zip(deep_positive, (0, 4, 8, 12, 16)):
        lower, upper = Fraction(*row["lower"]), Fraction(*row["upper"])
        h = Fraction(1, 2**exponent)
        assert upper == h
        assert upper - lower == h / (2**24)


@pytest.mark.parametrize("mutation", [
    "interleave_precision", "drop_job", "show_gpu", "outside_output",
    "too_many_workers", "change_category", "drop_source",
    "machine_hostname", "machine_workers", "machine_backend",
])
def test_manifest_validation_rejects_scope_or_design_mutation(mutation):
    manifest = deepcopy(_manifest())
    if mutation == "interleave_precision":
        manifest["jobs"][0]["precision_bits"] = 512
    elif mutation == "drop_job":
        manifest["jobs"].pop()
    elif mutation == "show_gpu":
        manifest["gpu_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif mutation == "outside_output":
        manifest["output_root"] = "/tmp/cold"
    elif mutation == "too_many_workers":
        manifest["execution_policy"]["max_workers"] = 17
    elif mutation == "change_category":
        manifest["jobs"][0]["domain_class"] = "positive_endpoint"
    elif mutation == "drop_source":
        manifest["provenance"]["source_sha256"].pop(
            "analysis/green_v400_native_cold_identity.py"
        )
    elif mutation == "machine_hostname":
        manifest["machine_concurrency_manifest"]["host"]["hostname"] = "bad\ntext"
    elif mutation == "machine_workers":
        manifest["machine_concurrency_manifest"]["concurrency"]["max_workers"] = 7
    elif mutation == "machine_backend":
        manifest["machine_concurrency_manifest"]["backend_identity"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        cold.validate_manifest(manifest)


def test_manifest_is_frozen_with_hash_before_any_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(cold, "_below_mnt_sdb", lambda _value: True)
    root = tmp_path / "cold"
    manifest = _manifest(output_root=root.as_posix())
    path, semantic_hash = cold._freeze_manifest(root, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.pop("manifest_semantic_hash") == semantic_hash
    assert sha256_canonical(payload) == semantic_hash
    assert not (root / "samples").exists()


def _sample_report(manifest, job):
    payload = {
        "schema_version": "green-v400-native-cold-cell-sample-v1",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "report_contains_scientific_outcome": False,
        "supervisor_applied_scientific_threshold": False,
        "status": "PASS_NATIVE_COLD_CELL_SAMPLE",
        "manifest_sha256": cold.manifest_semantic_hash(manifest),
        "sample": {
            "sample_id": job["sample_id"], "ordinal": job["ordinal"],
            "precision_bits": job["precision_bits"],
            "domain_class": job["domain_class"],
            "lower": job["lower"], "upper": job["upper"],
        },
        "process_identity": {"pid": 1234, "start_ticks": 5678},
        "gpu_environment": cold.GPU_ENVIRONMENT,
        "native_identity": {
            "backend_sha256": manifest["native_inputs"]["library_sha256"],
            "backend_version": "fixture",
            "descriptor_sha256": cold.DESCRIPTOR_SHA,
            "program_execution_sha256": cold.PROGRAM_SHA,
            "dispatch_sha256": cold.DISPATCH_SHA,
            "blob_sha256": cold.BLOB_SHA,
            "fusion_sha256": cold.FUSION_SHA,
            "kernel_tags_sha256": manifest["native_inputs"]["kernel_tags_sha256"],
        },
        "observations": {
            "envelope_open_seconds": 1.0, "context_build_seconds": 2.0,
            "cell_dispatch_seconds": 3.0, "total_seconds": 6.0,
            "process_peak_rss_before_kib": 1,
            "process_peak_rss_after_kib": 2,
            "process_peak_rss_delta_kib": 1,
        },
        "root_payload_sha256": {name: "d" * 64 for name in cold.ROOT_NAMES},
        "numeric_jet_payload_retained": False,
        "physical_native_dispatch_count": 1,
        "claim_scope": "hash-only fixture",
    }
    payload["report_semantic_hash"] = sha256_canonical(payload)
    return payload


def test_sample_validation_accepts_hash_only_and_rejects_numeric_payload():
    manifest = _manifest()
    job = manifest["jobs"][0]
    manifest_hash = cold.manifest_semantic_hash(manifest)
    report = _sample_report(manifest, job)
    assert cold._validate_sample_report(
        report, manifest, manifest_hash, job,
    ) == (1234, 5678)
    report["jet_payload"] = {"value": [1, 2]}
    report["report_semantic_hash"] = sha256_canonical({
        key: value for key, value in report.items() if key != "report_semantic_hash"
    })
    with pytest.raises(RuntimeError, match="FIELDS_INVALID"):
        cold._validate_sample_report(report, manifest, manifest_hash, job)


def test_job_command_binds_one_precision_domain_and_manifest_hash():
    manifest = _manifest()
    job = manifest["jobs"][31]
    manifest_hash = cold.manifest_semantic_hash(manifest)
    command = cold._job_command(manifest, manifest_hash, job)
    assert command.count("--precision") == 1
    assert command[command.index("--precision") + 1] == "512"
    assert command[command.index("--manifest-sha256") + 1] == manifest_hash
    assert "green_v400_native_cold_cell_sample.py" in " ".join(command)
    assert "--allow-descendants" not in command


def _literal_assignments(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_cold_identity_matches_existing_native_audit_constants():
    payload = _literal_assignments(
        ROOT / "analysis" / "green_v400_native_payload_parser_audit.py"
    )
    typed = _literal_assignments(
        ROOT / "analysis" / "green_v400_native_typed_plan_audit.py"
    )
    for name in (
        "BLOB_NBYTES", "BLOB_SHA", "DESCRIPTOR_SHA", "DISPATCH_SHA",
        "FUSION_SHA", "PROGRAM_SHA",
    ):
        assert getattr(identity, name) == payload[name]
    assert identity.EXPECTED_KERNEL_TAGS == typed["EXPECTED_KERNEL_TAGS"]
