from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_resource_calibration import (
    CANDIDATES,
    build_candidate_receipt,
    calibration_config_sha256,
    derive_child_seed,
    expected_run_identities,
    generate_synthetic_fixture,
    load_resource_calibration_config,
    pass_counts,
    select_resource_manifest,
    synthetic_modifier_jet,
)


def _records(budget, *, rss=1024, wall384=1.0, wall512=2.0, fault="NONE"):
    result = []
    for identity in expected_run_identities(budget):
        record = dict(identity) | {
            "theorem_checks_pass": True,
            "nesting_checks_pass": True,
            "max_depth": 4,
            "graph_nodes": 1000,
            "process_tree_rss_bytes": rss,
            "single_pass_wall_seconds": wall384 if identity["precision_bits"] == 384 else wall512,
            "deterministic_replay": True,
            "fault_code": fault,
            "contains_scientific_outcome": False,
            "contains_endpoint_material": False,
            "run_artifact_sha256": f"{identity['child_seed_uint64']:064x}"[-64:],
        }
        result.append(record)
    return result


def test_config_and_seed_derivation_are_frozen():
    config = load_resource_calibration_config()
    assert len(calibration_config_sha256()) == 64
    assert config["cold_processes_per_precision_and_candidate"] == 30
    assert derive_child_seed("ioi:layer0", "affine") == 16294457557233561567
    assert derive_child_seed("greater_than:layer8", "deep_dyadic") == 8102724552382121248
    assert len(expected_run_identities(4)) == 60
    assert pass_counts(32) == {"official": 195, "audit": 105, "total": 300}


def test_synthetic_actual_shape_fixture_is_golden_and_outcome_blind():
    ioi = generate_synthetic_fixture("ioi:layer0", "affine")
    gt = generate_synthetic_fixture("greater_than:layer8", "deep_dyadic")
    assert (ioi.sequence_length, ioi.site_position, ioi.site_layer) == (17, 4, 0)
    assert (gt.sequence_length, gt.site_position, gt.site_layer) == (13, 8, 8)
    assert ioi.semantic_hash() == "62bd88cff3650e91ac98f3175de2f4a9d6fbcb0b20d28f2cf5aadacd376a87f8"
    assert gt.semantic_hash() == "33c40af08390517090d8c40c60714e33bd9d36bf7ea652b96ce41c1664b9efb8"
    assert ioi.identity_payload()["contains_scientific_outcome"] is False
    assert gt.identity_payload()["contains_endpoint_material"] is False


def test_synthetic_modifier_has_exact_polynomial_jet():
    pytest.importorskip("gmpy2")
    from green_bridge_v400_interval import Interval

    jet = synthetic_modifier_jet(
        Interval.point("1/2", 128),
        (__import__("fractions").Fraction(1, 16), __import__("fractions").Fraction(-1, 16)),
    )
    assert jet.value.lower == jet.value.upper
    assert str(jet.value.lower) == "0.0078125"
    assert str(jet.first.lower) == "0.015625"
    assert str(jet.second.lower) == "-0.0625"


def test_candidate_receipt_projects_complete_direction_and_applies_guardband():
    receipt = build_candidate_receipt(8, _records(8, wall384=1.5, wall512=2.5))
    counts = pass_counts(8)
    assert receipt["projected_complete_direction_wall_seconds"] == (
        1.5 * counts["official"] + 2.5 * counts["audit"]
    )
    assert receipt["candidate_pass"] is True
    failed = build_candidate_receipt(8, _records(8, rss=60_000_000_000))
    assert failed["candidate_pass"] is False
    assert failed["first_failure_code"] == "MEMORY_GUARDBAND_EXCEEDED"


def test_selector_chooses_largest_passing_candidate_only():
    receipts = [build_candidate_receipt(value, _records(value)) for value in CANDIDATES]
    failing = deepcopy(receipts[-1])
    failing_records = _records(32, fault="TIMEOUT")
    receipts[-1] = build_candidate_receipt(32, failing_records)
    manifest = select_resource_manifest(receipts)
    assert manifest["selected_leaf_budget"] == 16
    assert manifest["candidate_receipt_sha256s"] == [
        receipt["receipt_sha256"] for receipt in receipts
    ]


def test_selector_stops_when_no_candidate_passes():
    receipts = [
        build_candidate_receipt(value, _records(value, fault="CGROUP_OOM"))
        for value in CANDIDATES
    ]
    with pytest.raises(RuntimeError, match="STOP_RESOURCE_LOCK_INFEASIBLE"):
        select_resource_manifest(receipts)


def test_run_panel_rejects_reordering_or_scientific_fields():
    records = _records(4)
    records[0], records[1] = records[1], records[0]
    with pytest.raises(ValueError, match="identity mismatch"):
        build_candidate_receipt(4, records)
    records = _records(4)
    records[0]["scientific_score"] = 0.9
    with pytest.raises(ValueError, match="field mismatch"):
        build_candidate_receipt(4, records)
