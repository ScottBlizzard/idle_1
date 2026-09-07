from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_protocol import PROTOCOL_ID, sha256_canonical
from green_v410_resource_calibration import (
    CANDIDATES, FIXTURE_KINDS, PRECISIONS, PROFILES, derive_child_seed,
)
from green_v410_resource_finalize import (
    finalize_resource_calibration,
    raw_artifact_path,
)


def _raw(candidate, precision, profile, fixture):
    mode = "official" if precision == 384 else "audit"
    dispatches = 2 * candidate + 1 if precision == 384 else candidate + 3
    value = {
        "schema_version": "green-v410-resource-cold-process-v2",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "mode": mode,
        "candidate_leaf_budget": candidate,
        "profile": profile,
        "fixture_kind": fixture,
        "precision_bits": precision,
        "child_seed_uint64": derive_child_seed(profile, fixture),
        "fixture_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "program_semantic_hash": "3" * 64,
        "backend_library_sha256": "4" * 64,
        "dispatch_count": dispatches,
        "process_wall_seconds": float(dispatches),
        "single_pass_wall_seconds": 1.0,
        "process_tree_peak_rss_bytes": 1024,
        "process_tree_resource_record": {"test_fixture": True},
        "max_depth": 3,
        "graph_nodes_metric": "root_only_peak_live_dependent_scalar_outputs_v1",
        "graph_nodes": 1000,
        "dependent_scalar_outputs_total": 5000,
        "executor_source_sha256": "5" * 64,
        "theorem_checks_pass": True,
        "nesting_checks_pass": None if precision == 384 else True,
        "deterministic_replay": True,
        "schedule": {"dispatch_count": dispatches},
        "contains_scientific_outcome": False,
        "contains_endpoint_material": False,
    }
    value["run_artifact_sha256"] = sha256_canonical(value)
    return value


def test_finalizer_pairs_precisions_and_selects_largest_candidate(tmp_path):
    raw_root, output_root = tmp_path / "raw", tmp_path / "final"
    for candidate in CANDIDATES:
        for precision in PRECISIONS:
            for profile in PROFILES:
                for fixture in FIXTURE_KINDS:
                    path = raw_artifact_path(
                        raw_root, candidate, precision, profile, fixture
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(
                        _raw(candidate, precision, profile, fixture), sort_keys=True
                    ), encoding="utf-8")
    manifest = finalize_resource_calibration(raw_root, output_root)
    assert manifest["selected_leaf_budget"] == 32
    assert (output_root / "candidate_L4.json").is_file()
    assert (output_root / "resource_manifest.json").is_file()
