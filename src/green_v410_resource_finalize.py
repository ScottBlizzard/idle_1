"""Validate cold-process outputs and freeze GREEN v4.1 resource receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from green_v410_artifacts import atomic_no_clobber_json
from green_v410_protocol import PROTOCOL_ID, sha256_canonical, strict_fields
from green_v410_resource_calibration import (
    CANDIDATES,
    FIXTURE_KINDS,
    PRECISIONS,
    PROFILES,
    build_candidate_receipt,
    derive_child_seed,
    expected_run_identities,
    select_resource_manifest,
)


RAW_FIELDS = {
    "schema_version", "protocol_id", "attempt_index", "mode",
    "candidate_leaf_budget", "profile", "fixture_kind", "precision_bits",
    "child_seed_uint64", "fixture_sha256", "bundle_sha256",
    "program_semantic_hash", "backend_library_sha256", "dispatch_count",
    "process_wall_seconds", "single_pass_wall_seconds",
    "process_tree_peak_rss_bytes", "process_tree_resource_record",
    "max_depth", "graph_nodes_metric", "graph_nodes",
    "dependent_scalar_outputs_total", "executor_source_sha256",
    "theorem_checks_pass", "nesting_checks_pass", "deterministic_replay",
    "schedule", "contains_scientific_outcome", "contains_endpoint_material",
    "run_artifact_sha256",
}


def raw_artifact_path(root: Path, candidate: int, precision: int,
                      profile: str, fixture: str) -> Path:
    return root / f"L{candidate}" / str(precision) / (
        f"{profile.replace(':', '__')}__{fixture}.json"
    )


def _load_raw(path: Path, *, candidate: int, precision: int,
              profile: str, fixture: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    strict_fields(payload, RAW_FIELDS, "resource cold-process artifact")
    unhashed = dict(payload)
    observed = unhashed.pop("run_artifact_sha256")
    mode = "official" if precision == 384 else "audit"
    expected_dispatches = 2 * candidate + 1 if precision == 384 else candidate + 3
    if (
        payload["schema_version"] != "green-v410-resource-cold-process-v2"
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["attempt_index"] != 1
        or payload["mode"] != mode
        or payload["candidate_leaf_budget"] != candidate
        or payload["profile"] != profile
        or payload["fixture_kind"] != fixture
        or payload["precision_bits"] != precision
        or payload["child_seed_uint64"] != derive_child_seed(profile, fixture)
        or payload["dispatch_count"] != expected_dispatches
        or payload["schedule"].get("dispatch_count") != expected_dispatches
        or payload["graph_nodes_metric"]
            != "root_only_peak_live_dependent_scalar_outputs_v1"
        or type(payload["dependent_scalar_outputs_total"]) is not int
        or payload["dependent_scalar_outputs_total"] < payload["graph_nodes"]
        or payload["contains_scientific_outcome"] is not False
        or payload["contains_endpoint_material"] is not False
        or observed != sha256_canonical(unhashed)
    ):
        raise ValueError("resource cold-process artifact identity mismatch")
    return payload


def finalize_resource_calibration(raw_root: Path, output_root: Path) -> dict:
    receipts = []
    for candidate in CANDIDATES:
        raw = {
            (precision, profile, fixture): _load_raw(
                raw_artifact_path(raw_root, candidate, precision, profile, fixture),
                candidate=candidate, precision=precision, profile=profile,
                fixture=fixture,
            )
            for precision in PRECISIONS
            for profile in PROFILES
            for fixture in FIXTURE_KINDS
        }
        records = []
        for identity in expected_run_identities(candidate):
            key = (
                identity["precision_bits"], identity["profile"], identity["fixture_kind"]
            )
            item = raw[key]
            pair = raw[(
                512 if identity["precision_bits"] == 384 else 384,
                identity["profile"], identity["fixture_kind"],
            )]
            audit = item if item["precision_bits"] == 512 else pair
            same_graph = all(
                item[field] == pair[field]
                for field in (
                    "fixture_sha256", "bundle_sha256", "program_semantic_hash",
                    "backend_library_sha256", "executor_source_sha256",
                    "graph_nodes_metric", "graph_nodes",
                    "dependent_scalar_outputs_total",
                )
            )
            nesting = audit["nesting_checks_pass"] is True
            fault = "NONE"
            if not same_graph:
                fault = "MALFORMED_CHECKPOINT"
            elif not item["deterministic_replay"]:
                fault = "NONDETERMINISTIC_REPLAY"
            elif item["theorem_checks_pass"] is not True:
                fault = "THEOREM_FAILURE"
            elif not nesting:
                fault = "NESTING_FAILURE"
            records.append(dict(identity) | {
                "theorem_checks_pass": item["theorem_checks_pass"] is True and same_graph,
                "nesting_checks_pass": nesting and same_graph,
                "max_depth": max(item["max_depth"], pair["max_depth"]),
                "graph_nodes_metric": item["graph_nodes_metric"],
                "graph_nodes": item["graph_nodes"],
                "dependent_scalar_outputs_total": item[
                    "dependent_scalar_outputs_total"
                ],
                "executor_source_sha256": item["executor_source_sha256"],
                "process_tree_rss_bytes": item["process_tree_peak_rss_bytes"],
                "single_pass_wall_seconds": item["single_pass_wall_seconds"],
                "deterministic_replay": item["deterministic_replay"],
                "fault_code": fault,
                "contains_scientific_outcome": False,
                "contains_endpoint_material": False,
                "run_artifact_sha256": item["run_artifact_sha256"],
            })
        receipt = build_candidate_receipt(candidate, records)
        atomic_no_clobber_json(
            output_root / f"candidate_L{candidate}.json", receipt,
            job_id=f"resource-finalize-L{candidate}",
        )
        receipts.append(receipt)
    manifest = select_resource_manifest(receipts)
    atomic_no_clobber_json(
        output_root / "resource_manifest.json", manifest,
        job_id="resource-finalize-manifest",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    finalize_resource_calibration(args.raw_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
