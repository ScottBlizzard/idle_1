"""Deterministic, outcome-blind GREEN v4.1 resource calibration contracts."""
from __future__ import annotations

from fractions import Fraction
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterable, Mapping

import numpy as np

from green_v410_protocol import PROTOCOL_ID, sha256_canonical, strict_fields
from green_v410_schemas import with_artifact_self_hash


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "green_v410_resource_calibration.json"
PROFILES = (
    "ioi:layer0", "ioi:layer4", "ioi:layer8",
    "greater_than:layer0", "greater_than:layer4", "greater_than:layer8",
)
FIXTURE_KINDS = (
    "affine", "positive_curvature", "negative_curvature",
    "signed_cancellation", "deep_dyadic",
)
CANDIDATES = (4, 8, 16, 32)
PRECISIONS = (384, 512)
DOMAIN_SEPARATOR = "GREEN_V410_RESOURCE_CALIBRATION_V1"
MASTER_SEED = 41020260830


def load_resource_calibration_config(path: Path = CONFIG_PATH) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    strict_fields(payload, {
        "schema_version", "protocol_id", "attempt_index", "execution_scope",
        "generator", "master_seed_uint64", "domain_separator", "model",
        "selected_gates", "synthetic_task_metrics", "profiles",
        "profile_shapes", "fixtures", "candidate_leaf_budgets", "precision_bits", "direction_width",
        "direction_nominal_norm", "controlled_hook_dtype", "direction_dtype",
        "cold_processes_per_precision_and_candidate", "radius_count", "pass_formulas",
        "limits", "selection", "real_artifact_access_allowed",
        "endpoint_material_allowed",
    }, "resource calibration config")
    if (
        payload["schema_version"] != "green-v410-sfc-jwtec-resource-calibration-config-v1"
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["attempt_index"] != 1
        or payload["execution_scope"] != "outcome_blind_synthetic_only"
        or payload["generator"] != "numpy-PCG64DXSM"
        or payload["master_seed_uint64"] != MASTER_SEED
        or payload["domain_separator"] != DOMAIN_SEPARATOR
        or payload["model"] != {
            "id": "openai-community/gpt2",
            "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            "checkpoint_dtype": "float32",
        }
        or payload["selected_gates"] != [
            2326, 1138, 2287, 606, 2848, 2305, 46, 2659, 946, 1616,
        ]
        or payload["profiles"] != list(PROFILES)
        or payload["profile_shapes"] != {
            "ioi": {
                "sequence_length": 17,
                "site_position": 4,
                "qualification_causal_cone_manifest_sha256": "7cdb53950452dcfefd478f59d9a2d2526ffecff617c60ecc9f43700eb0c311e3",
            },
            "greater_than": {
                "sequence_length": 13,
                "site_position": 8,
                "qualification_causal_cone_manifest_sha256": "22a7ab6605dfad9373a00eae2882d7b26ad07be8a5e640f891835dada15e4b36",
            },
        }
        or payload["candidate_leaf_budgets"] != list(CANDIDATES)
        or payload["precision_bits"] != list(PRECISIONS)
        or payload["direction_width"] != 768
        or Fraction(payload["direction_nominal_norm"]) != Fraction(1, 1000)
        or payload["controlled_hook_dtype"] != "float64-little-endian"
        or payload["direction_dtype"] != "float32-little-endian"
        or payload["cold_processes_per_precision_and_candidate"] != 30
        or payload["radius_count"] != 3
        or payload["pass_formulas"] != {
            "official": "3*(2*L+1)", "audit": "3*(L+3)", "total": "9*L+12",
        }
        or payload["limits"] != {
            "max_depth": 24,
            "max_graph_nodes": 2_000_000,
            "memory_max_bytes": 68_719_476_736,
            "direction_wall_max_seconds": 85_800,
            "guardband": "5/4",
        }
        or payload["selection"] != "largest passing candidate"
        or payload["real_artifact_access_allowed"] is not False
        or payload["endpoint_material_allowed"] is not False
    ):
        raise ValueError("resource calibration config differs from the binding decision")
    metrics = payload["synthetic_task_metrics"]
    if set(metrics) != {"ioi", "greater_than"}:
        raise ValueError("resource calibration synthetic task metric domain mismatch")
    expected_metrics = {
        "ioi": (
            [0, 1], [[1, 1], [-1, 1]],
            "fixed_two_column_synthetic_logit_difference",
        ),
        "greater_than": (
            list(range(100)), [[-1, 50]] * 50 + [[1, 50]] * 50,
            "fixed_balanced_hundred_column_synthetic_greater_than_contrast",
        ),
    }
    for task, (ids, coefficients, semantics) in expected_metrics.items():
        if metrics[task] != {
            "suffix_token_ids": ids,
            "coefficient_rationals": coefficients,
            "semantics": semantics,
        }:
            raise ValueError("resource calibration synthetic task metric mismatch")
    fixtures = payload["fixtures"]
    if not isinstance(fixtures, list) or [row.get("kind") for row in fixtures] != list(FIXTURE_KINDS):
        raise ValueError("resource calibration fixture order mismatch")
    expected = {
        "affine": ((Fraction(-1), Fraction(0)), (Fraction(0), Fraction(0))),
        "positive_curvature": ((Fraction(0), Fraction(1)), (Fraction(1, 16), Fraction(0))),
        "negative_curvature": ((Fraction(-1), Fraction(-1, 2)), (Fraction(-1, 16), Fraction(0))),
        "signed_cancellation": ((Fraction(1, 2), Fraction(1)), (Fraction(1, 16), Fraction(-1, 16))),
        "deep_dyadic": ((Fraction(511, 1024), Fraction(513, 1024)), (Fraction(1, 32), Fraction(1, 64))),
    }
    for row in fixtures:
        strict_fields(row, {"kind", "domain", "modifier_coefficients"}, "fixture")
        observed = (
            tuple(Fraction(value) for value in row["domain"]),
            tuple(Fraction(value) for value in row["modifier_coefficients"]),
        )
        if observed != expected[row["kind"]]:
            raise ValueError("resource calibration fixture content mismatch")
    return payload


def calibration_config_sha256(path: Path = CONFIG_PATH) -> str:
    return sha256_canonical(load_resource_calibration_config(path))


def derive_child_seed(profile: str, fixture_kind: str) -> int:
    if profile not in PROFILES or fixture_kind not in FIXTURE_KINDS:
        raise ValueError("unknown calibration seed domain")
    payload = (
        DOMAIN_SEPARATOR.encode("ascii") + b"\0" + struct.pack("<Q", MASTER_SEED)
        + b"\0" + profile.encode("ascii") + b"\0" + fixture_kind.encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _tensor_semantic_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    if value.dtype.itemsize > 1:
        value = value.astype(value.dtype.newbyteorder("<"), copy=False)
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape), "layout": "C"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class SyntheticCalibrationFixture:
    profile: str
    fixture_kind: str
    task: str
    site_layer: int
    sequence_length: int
    site_position: int
    child_seed_uint64: int
    pat_controlled_hook_t0: np.ndarray
    tar_controlled_hook_t0: np.ndarray
    clean_center: np.ndarray
    physical_direction: np.ndarray
    modifier_coefficients: tuple[Fraction, Fraction]

    def identity_payload(self) -> dict:
        return {
            "schema_version": "green-v410-synthetic-calibration-fixture-v1",
            "protocol_id": PROTOCOL_ID,
            "execution_scope": "outcome_blind_synthetic_only",
            "profile": self.profile,
            "fixture_kind": self.fixture_kind,
            "task": self.task,
            "site_layer": self.site_layer,
            "sequence_length": self.sequence_length,
            "site_position": self.site_position,
            "child_seed_uint64": self.child_seed_uint64,
            "pat_controlled_hook_t0_sha256": _tensor_semantic_sha256(
                self.pat_controlled_hook_t0
            ),
            "tar_controlled_hook_t0_sha256": _tensor_semantic_sha256(
                self.tar_controlled_hook_t0
            ),
            "clean_center_sha256": _tensor_semantic_sha256(self.clean_center),
            "physical_direction_sha256": _tensor_semantic_sha256(self.physical_direction),
            "modifier_coefficients": [
                f"{value.numerator}/{value.denominator}" for value in self.modifier_coefficients
            ],
            "modifier_semantics": "PAT_J_plus_a_t_squared_plus_b_t_cubed_test_wrapper_only",
            "contains_scientific_outcome": False,
            "contains_endpoint_material": False,
        }

    def semantic_hash(self) -> str:
        return sha256_canonical(self.identity_payload())


def generate_synthetic_fixture(profile: str, fixture_kind: str) -> SyntheticCalibrationFixture:
    config = load_resource_calibration_config()
    if profile not in PROFILES or fixture_kind not in FIXTURE_KINDS:
        raise ValueError("unknown synthetic calibration fixture")
    task, layer_text = profile.split(":", 1)
    layer = int(layer_text.removeprefix("layer"))
    shape = config["profile_shapes"][task]
    seed = derive_child_seed(profile, fixture_kind)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    tensor_shape = (shape["sequence_length"], config["direction_width"])
    pat = np.ascontiguousarray(rng.standard_normal(tensor_shape), dtype="<f8")
    tar = np.ascontiguousarray(rng.standard_normal(tensor_shape), dtype="<f8")
    center = np.ascontiguousarray(
        rng.standard_normal(config["direction_width"]), dtype="<f8"
    )
    pat[shape["site_position"]] = center
    tar[shape["site_position"]] = center
    raw_direction = rng.standard_normal(config["direction_width"])
    direction = np.ascontiguousarray(
        raw_direction / np.linalg.norm(raw_direction) * 0.001, dtype="<f4"
    )
    norm = float(np.linalg.norm(direction.astype(np.float64)))
    if abs(norm - 0.001) > max(1e-8, 0.001 * 2e-6):
        raise RuntimeError("synthetic direction failed frozen binding tolerance")
    fixture = next(row for row in config["fixtures"] if row["kind"] == fixture_kind)
    result = SyntheticCalibrationFixture(
        profile, fixture_kind, task, layer, shape["sequence_length"],
        shape["site_position"], seed, pat, tar, center, direction,
        tuple(Fraction(value) for value in fixture["modifier_coefficients"]),
    )
    if (
        result.pat_controlled_hook_t0.dtype != np.dtype("<f8")
        or result.tar_controlled_hook_t0.dtype != np.dtype("<f8")
        or result.clean_center.dtype != np.dtype("<f8")
        or result.physical_direction.dtype != np.dtype("<f4")
        or not np.array_equal(result.pat_controlled_hook_t0[result.site_position], center)
        or not np.array_equal(result.tar_controlled_hook_t0[result.site_position], center)
    ):
        raise RuntimeError("synthetic fixture construction drift")
    return result


def synthetic_modifier_jet(domain, coefficients: tuple[Fraction, Fraction]):
    """Return the exact ``a*t^2+b*t^3`` closed-fixture modifier jet."""
    from green_bridge_v400_interval import Interval
    from green_bridge_v400_interval_jet import (
        add_jet, affine_control_jet, mul_jet, square_jet,
    )
    from green_bridge_v400_transformer_ops import scale_jet

    precision = domain.precision_bits
    t = affine_control_jet(Interval.point(0, precision), Interval.point(1, precision), domain)
    squared = square_jet(t)
    cubed = mul_jet(squared, t)
    return add_jet(scale_jet(squared, coefficients[0]), scale_jet(cubed, coefficients[1]))


class SyntheticModifierEvaluator:
    """Calibration-only evaluator wrapper; production graphs never import it."""

    def __init__(self, base_evaluator, fixture: SyntheticCalibrationFixture):
        self.base_evaluator = base_evaluator
        self.fixture = fixture
        self.contains_scientific_outcome = False
        self.synthetic_only = True

    @property
    def evaluator_identity_sha256(self) -> str:
        return sha256_canonical({
            "schema_version": "green-v410-synthetic-modifier-evaluator-v1",
            "base_evaluator_identity_sha256": self.base_evaluator.evaluator_identity_sha256,
            "fixture_sha256": self.fixture.semantic_hash(),
            "modifier_semantics": "PAT_J_plus_a_t_squared_plus_b_t_cubed_test_wrapper_only",
        })

    def evaluate_interval(self, domain):
        from green_bridge_v400_interval_jet import add_jet
        return add_jet(
            self.base_evaluator.evaluate_interval(domain),
            synthetic_modifier_jet(domain, self.fixture.modifier_coefficients),
        )

    def close(self) -> None:
        close = getattr(self.base_evaluator, "close", None)
        if callable(close):
            close()


class TensorProgramCalibrationEvaluator:
    """Repeated-cell evaluator over one hash-closed synthetic TensorProgram."""

    contains_scientific_outcome = False
    synthetic_only = True

    def __init__(self, program, reader, compiled_backend=None):
        from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
        from green_bridge_v400_mpfr_tensor_executor import (
            ResidentStaticRowCache,
            preload_tensor_program_arrays,
        )

        if isinstance(compiled_backend, (str, Path)):
            compiled_backend = CompiledMPFRBackend(Path(compiled_backend))
        self.program = program
        self.reader = reader
        self.compiled_backend = compiled_backend
        self.preloaded_tensors = preload_tensor_program_arrays(program, reader)
        self._static_caches = {}
        self._cache_type = ResidentStaticRowCache
        backend_identity = (
            compiled_backend.library_sha256 if compiled_backend is not None else "python-reference"
        )
        self.evaluator_identity_sha256 = sha256_canonical({
            "schema_version": "green-v410-resource-tensor-evaluator-v1",
            "program_semantic_hash": program.semantic_hash(),
            "tensor_store_record_closure_sha256": reader.manifest.record_closure_sha256,
            "backend_identity": backend_identity,
            "sparse_axis0_execution": True,
            "preloaded_tensor_closure": True,
            "cross_cell_static_cache": compiled_backend is not None,
            "contains_scientific_outcome": False,
        })

    def evaluate_interval(self, domain):
        from green_bridge_v400_mpfr_tensor_executor import execute_tensor_program_mpfr

        cache = None
        if self.compiled_backend is not None:
            cache = self._static_caches.get(domain.precision_bits)
            if cache is None:
                cache = self._cache_type.build_unpacked(
                    self.program, self.compiled_backend, domain.precision_bits
                )
                self._static_caches[domain.precision_bits] = cache
        result = execute_tensor_program_mpfr(
            self.program,
            self.reader,
            domain,
            self.compiled_backend,
            sparse_axis0_execution=True,
            resident_static_row_cache=cache,
            preloaded_tensors=self.preloaded_tensors,
        )
        return result["output"]

    def close(self) -> None:
        for cache in self._static_caches.values():
            cache.close()
        self._static_caches.clear()


def expected_run_identities(candidate_leaf_budget: int) -> tuple[dict, ...]:
    if candidate_leaf_budget not in CANDIDATES:
        raise ValueError("unknown candidate leaf budget")
    return tuple({
        "candidate_leaf_budget": candidate_leaf_budget,
        "profile": profile,
        "fixture_kind": fixture,
        "precision_bits": precision,
        "child_seed_uint64": derive_child_seed(profile, fixture),
    } for precision in PRECISIONS for profile in PROFILES for fixture in FIXTURE_KINDS)


RUN_FIELDS = {
    "candidate_leaf_budget", "profile", "fixture_kind", "precision_bits",
    "child_seed_uint64", "theorem_checks_pass", "nesting_checks_pass",
    "max_depth", "graph_nodes", "process_tree_rss_bytes", "single_pass_wall_seconds",
    "deterministic_replay", "fault_code", "contains_scientific_outcome",
    "contains_endpoint_material", "run_artifact_sha256",
}


def _validate_run_record(record: Mapping, expected: Mapping) -> None:
    strict_fields(record, RUN_FIELDS, "resource calibration run record")
    if any(record[key] != value for key, value in expected.items()):
        raise ValueError("resource calibration run identity mismatch")
    if (
        type(record["max_depth"]) is not int or record["max_depth"] < 0
        or type(record["graph_nodes"]) is not int or record["graph_nodes"] < 1
        or type(record["process_tree_rss_bytes"]) is not int
        or record["process_tree_rss_bytes"] < 0
        or type(record["single_pass_wall_seconds"]) not in {int, float}
        or not 0 <= record["single_pass_wall_seconds"] < float("inf")
        or record["fault_code"] not in {
            "NONE", "CGROUP_OOM", "TIMEOUT", "MALFORMED_CHECKPOINT",
            "NONDETERMINISTIC_REPLAY", "THEOREM_FAILURE", "NESTING_FAILURE",
        }
        or record["contains_scientific_outcome"] is not False
        or record["contains_endpoint_material"] is not False
        or not isinstance(record["run_artifact_sha256"], str)
        or len(record["run_artifact_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in record["run_artifact_sha256"])
    ):
        raise ValueError("malformed resource calibration run record")


def pass_counts(leaf_budget: int) -> dict[str, int]:
    if leaf_budget not in CANDIDATES:
        raise ValueError("unknown candidate leaf budget")
    official = 3 * (2 * leaf_budget + 1)
    audit = 3 * (leaf_budget + 3)
    return {"official": official, "audit": audit, "total": official + audit}


def build_candidate_receipt(
    candidate_leaf_budget: int,
    run_records: Iterable[Mapping],
) -> dict:
    config = load_resource_calibration_config()
    records = [dict(record) for record in run_records]
    expected = expected_run_identities(candidate_leaf_budget)
    if len(records) != len(expected):
        raise ValueError("candidate requires exactly 60 cold-process run records")
    for record, identity in zip(records, expected):
        _validate_run_record(record, identity)
    max_depth = max(record["max_depth"] for record in records)
    max_nodes = max(record["graph_nodes"] for record in records)
    max_rss = max(record["process_tree_rss_bytes"] for record in records)
    max_wall = {
        precision: max(
            record["single_pass_wall_seconds"] for record in records
            if record["precision_bits"] == precision
        ) for precision in PRECISIONS
    }
    counts = pass_counts(candidate_leaf_budget)
    projected_wall = max_wall[384] * counts["official"] + max_wall[512] * counts["audit"]
    all_theorem = all(
        record["theorem_checks_pass"] is True
        and record["nesting_checks_pass"] is True
        for record in records
    )
    deterministic = all(record["deterministic_replay"] is True for record in records)
    no_fault = all(record["fault_code"] == "NONE" for record in records)
    limits = config["limits"]
    guardband = Fraction(limits["guardband"])
    checks = (
        (all_theorem, "THEOREM_OR_NESTING_CHECK_FAILED"),
        (max_depth <= limits["max_depth"], "MAX_DEPTH_EXCEEDED"),
        (max_nodes <= limits["max_graph_nodes"], "MAX_GRAPH_NODES_EXCEEDED"),
        (guardband * max_rss <= limits["memory_max_bytes"], "MEMORY_GUARDBAND_EXCEEDED"),
        (guardband * Fraction.from_float(float(projected_wall))
         <= limits["direction_wall_max_seconds"], "WALL_GUARDBAND_EXCEEDED"),
        (deterministic, "NONDETERMINISTIC_REPLAY"),
        (no_fault, "CALIBRATION_CHILD_FAULT"),
    )
    first_failure = next((reason for passed, reason in checks if not passed), "NONE")
    payload = {
        "schema_version": "green-v410-sfc-jwtec-resource-calibration-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "candidate_leaf_budget": candidate_leaf_budget,
        "profiles": list(PROFILES),
        "fixture_kinds": list(FIXTURE_KINDS),
        "precision_bits": list(PRECISIONS),
        "run_records": records,
        "all_theorem_checks_pass": all_theorem,
        "max_depth": max_depth,
        "max_graph_nodes": max_nodes,
        "max_process_tree_rss_bytes": max_rss,
        "projected_complete_direction_wall_seconds": projected_wall,
        "deterministic_replay": deterministic,
        "candidate_pass": first_failure == "NONE",
        "first_failure_code": first_failure,
    }
    return with_artifact_self_hash(payload)


def select_resource_manifest(candidate_receipts: Iterable[Mapping]) -> dict:
    config = load_resource_calibration_config()
    receipts = [dict(value) for value in candidate_receipts]
    if [row.get("candidate_leaf_budget") for row in receipts] != list(CANDIDATES):
        raise ValueError("resource selector requires four ordered candidate receipts")
    from green_v410_schemas import validate_artifact
    for receipt in receipts:
        validate_artifact(receipt)
    passing = [row["candidate_leaf_budget"] for row in receipts if row["candidate_pass"] is True]
    if not passing:
        raise RuntimeError("STOP_RESOURCE_LOCK_INFEASIBLE")
    limits = config["limits"]
    payload = {
        "schema_version": "green-v410-sfc-jwtec-resource-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "selected_leaf_budget": max(passing),
        "selection_rule": "largest passing candidate",
        "candidate_receipt_sha256s": [row["receipt_sha256"] for row in receipts],
        "max_depth": limits["max_depth"],
        "max_graph_nodes": limits["max_graph_nodes"],
        "memory_max_bytes": limits["memory_max_bytes"],
        "direction_wall_max_seconds": limits["direction_wall_max_seconds"],
        "guardband": limits["guardband"],
        "max_process_launches": 2,
        "official_precision_bits": 384,
        "audit_precision_bits": 512,
        "radii": ["1", "1/2", "1/4"],
    }
    return with_artifact_self_hash(payload)
