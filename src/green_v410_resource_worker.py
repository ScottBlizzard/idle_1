"""One cold-process GREEN v4.1 synthetic resource-calibration worker."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gc
import json
from pathlib import Path
import time

from green_bridge_v400_resources import ProcessTreeResourceRecorder
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader
from green_v410_artifacts import atomic_no_clobber_json, file_sha256
from green_v410_gpt2_program import (
    V410GPT2Dimensions,
    validate_green_v410_full_cone_program,
)
from green_v410_protocol import PROTOCOL_ID, sha256_canonical, strict_fields
from green_v410_resource_calibration import (
    CANDIDATES,
    FIXTURE_KINDS,
    PRECISIONS,
    PROFILES,
    SyntheticModifierEvaluator,
    TensorProgramCalibrationEvaluator,
    derive_child_seed,
    generate_synthetic_fixture,
    load_resource_calibration_config,
)
from green_v410_resource_schedule import (
    run_frozen_partition_audit,
    run_official_schedule,
)


def _dimensions(payload: dict) -> V410GPT2Dimensions:
    strict_fields(payload, {
        "site_layer", "site_position", "sequence_length", "d_model", "d_mlp",
        "n_heads", "d_head", "selected_gates", "final_position", "contrast_width",
        "contrast_coefficient_rationals",
    }, "v4.1 resource dimensions")
    return V410GPT2Dimensions(
        payload["site_layer"], payload["site_position"], payload["sequence_length"],
        payload["d_model"], payload["d_mlp"], payload["n_heads"], payload["d_head"],
        tuple(payload["selected_gates"]), payload["final_position"],
        payload["contrast_width"],
        tuple(tuple(value) for value in payload["contrast_coefficient_rationals"]),
    )


def _load_bundle(path: Path, profile: str, fixture_kind: str):
    bundle_path = path / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    strict_fields(bundle, {
        "schema_version", "protocol_id", "profile", "fixture_kind", "fixture_sha256",
        "program_file", "program_file_sha256", "program_semantic_hash",
        "tensor_store_manifest_file", "tensor_store_manifest_file_sha256",
        "tensor_store_record_closure_sha256", "contains_scientific_outcome",
        "contains_endpoint_material", "bundle_sha256",
    }, "resource graph bundle")
    unhashed = dict(bundle)
    observed_hash = unhashed.pop("bundle_sha256")
    if (
        bundle["schema_version"] != "green-v410-resource-graph-bundle-v1"
        or bundle["protocol_id"] != PROTOCOL_ID
        or bundle["profile"] != profile
        or bundle["fixture_kind"] != fixture_kind
        or bundle["fixture_sha256"]
        != generate_synthetic_fixture(profile, fixture_kind).semantic_hash()
        or bundle["contains_scientific_outcome"] is not False
        or bundle["contains_endpoint_material"] is not False
        or observed_hash != sha256_canonical(unhashed)
    ):
        raise ValueError("resource graph bundle identity mismatch")
    program_path = path / bundle["program_file"]
    store_path = path / bundle["tensor_store_manifest_file"]
    if (
        program_path.parent.resolve() != path.resolve()
        or store_path.parent.resolve() != path.resolve()
        or file_sha256(program_path) != bundle["program_file_sha256"]
        or file_sha256(store_path) != bundle["tensor_store_manifest_file_sha256"]
    ):
        raise ValueError("resource graph bundle file closure mismatch")
    program = TensorProgram.from_dict(json.loads(program_path.read_text(encoding="utf-8")))
    reader = TensorStoreReader(store_path)
    if (
        program.semantic_hash() != bundle["program_semantic_hash"]
        or reader.manifest.record_closure_sha256
        != bundle["tensor_store_record_closure_sha256"]
    ):
        raise ValueError("resource graph bundle semantic closure mismatch")
    dims = _dimensions(program.resource_formula["dimensions"])
    validate_green_v410_full_cone_program(program, reader, dims)
    return bundle, program, reader


def _new_evaluator(program, reader, backend_path: Path, fixture):
    return SyntheticModifierEvaluator(
        TensorProgramCalibrationEvaluator(program, reader, backend_path), fixture
    )


def run_worker(*, mode: str, candidate: int, profile: str, fixture_kind: str,
               bundle_dir: Path, backend_path: Path,
               official_artifact: Path | None = None) -> dict:
    if (
        mode not in {"official", "audit"}
        or candidate not in CANDIDATES
        or profile not in PROFILES
        or fixture_kind not in FIXTURE_KINDS
        or (mode == "official") != (official_artifact is None)
    ):
        raise ValueError("resource worker identity mismatch")
    config = load_resource_calibration_config()
    bundle, program, reader = _load_bundle(bundle_dir, profile, fixture_kind)
    fixture = generate_synthetic_fixture(profile, fixture_kind)
    config_fixture = next(row for row in config["fixtures"] if row["kind"] == fixture_kind)
    domain = tuple(Fraction(value) for value in config_fixture["domain"])
    precision = 384 if mode == "official" else 512
    if precision not in PRECISIONS:
        raise RuntimeError("resource precision panel drift")
    if not backend_path.is_file():
        raise FileNotFoundError("compiled MPFR backend is missing")

    official = None
    if mode == "audit":
        official_envelope = json.loads(Path(official_artifact).read_text(encoding="utf-8"))
        if (
            official_envelope.get("mode") != "official"
            or official_envelope.get("candidate_leaf_budget") != candidate
            or official_envelope.get("profile") != profile
            or official_envelope.get("fixture_kind") != fixture_kind
        ):
            raise ValueError("audit worker official artifact identity mismatch")
        official = official_envelope["schedule"]

    evaluator = _new_evaluator(program, reader, backend_path, fixture)
    with ProcessTreeResourceRecorder(sample_interval_seconds=0.01) as rss:
        started = time.perf_counter()
        schedule = (
            run_official_schedule(evaluator, domain, candidate)
            if mode == "official"
            else run_frozen_partition_audit(evaluator, official)
        )
        wall = time.perf_counter() - started
        evaluator.close()
        del evaluator
        gc.collect()
        replay_evaluator = _new_evaluator(program, reader, backend_path, fixture)
        replay = (
            run_official_schedule(replay_evaluator, domain, candidate)
            if mode == "official"
            else run_frozen_partition_audit(replay_evaluator, official)
        )
        replay_evaluator.close()
    deterministic = replay == schedule
    dispatches = schedule["dispatch_count"]
    payload = {
        "schema_version": "green-v410-resource-cold-process-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "mode": mode,
        "candidate_leaf_budget": candidate,
        "profile": profile,
        "fixture_kind": fixture_kind,
        "precision_bits": precision,
        "child_seed_uint64": derive_child_seed(profile, fixture_kind),
        "fixture_sha256": fixture.semantic_hash(),
        "bundle_sha256": bundle["bundle_sha256"],
        "program_semantic_hash": program.semantic_hash(),
        "backend_library_sha256": file_sha256(backend_path),
        "dispatch_count": dispatches,
        "process_wall_seconds": wall,
        "single_pass_wall_seconds": wall / dispatches,
        "process_tree_peak_rss_bytes": rss.record.peak_sampled_tree_rss_kib * 1024,
        "process_tree_resource_record": rss.record.to_dict(),
        "max_depth": (
            schedule["max_depth"] if mode == "official" else official["max_depth"]
        ),
        "graph_nodes": program.resource_formula["dependent_scalar_outputs_total"],
        "theorem_checks_pass": True,
        "nesting_checks_pass": (
            None if mode == "official" else schedule["all_nested"]
        ),
        "deterministic_replay": deterministic,
        "schedule": schedule,
        "contains_scientific_outcome": False,
        "contains_endpoint_material": False,
    }
    payload["run_artifact_sha256"] = sha256_canonical(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("official", "audit"), required=True)
    parser.add_argument("--candidate", type=int, choices=CANDIDATES, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--fixture", choices=FIXTURE_KINDS, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--official-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_worker(
        mode=args.mode, candidate=args.candidate, profile=args.profile,
        fixture_kind=args.fixture, bundle_dir=args.bundle_dir,
        backend_path=args.backend, official_artifact=args.official_artifact,
    )
    atomic_no_clobber_json(
        args.output, payload,
        job_id=f"resource-{args.mode}-{args.candidate}-{args.profile}-{args.fixture}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
