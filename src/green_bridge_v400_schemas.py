"""Canonical schemas for GREEN v4 static formal prepare."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
import copy
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_fields(payload: dict, expected: set[str], schema_name: str) -> None:
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise ValueError(
            f"{schema_name} field mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_exact_rational_pair(value: object, name: str) -> None:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(item) is not int for item in value)):
        raise ValueError(f"{name} must be an exact rational pair")
    numerator, denominator = value
    if denominator <= 0:
        raise ValueError(f"{name} denominator must be positive")
    reduced = Fraction(numerator, denominator)
    if reduced.numerator != numerator or reduced.denominator != denominator:
        raise ValueError(f"{name} must be canonically reduced")


def _validate_interval_payload(value: object, precision_bits: int, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an interval payload")
    _strict_fields(value, {"precision_bits", "lower", "upper"}, name)
    if type(value["precision_bits"]) is not int or value["precision_bits"] != precision_bits:
        raise ValueError(f"{name} precision mismatch")
    _validate_exact_rational_pair(value["lower"], f"{name}.lower")
    _validate_exact_rational_pair(value["upper"], f"{name}.upper")
    lower = Fraction(*value["lower"])
    upper = Fraction(*value["upper"])
    if lower > upper:
        raise ValueError(f"{name} lower endpoint exceeds upper endpoint")


@dataclass(frozen=True)
class Dyadic:
    numerator: int
    exponent: int

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator) * (
            Fraction(2**self.exponent) if self.exponent >= 0
            else Fraction(1, 2**(-self.exponent))
        )

    def to_dict(self) -> dict:
        return {"numerator": self.numerator, "exponent": self.exponent}

    @classmethod
    def from_dict(cls, payload: dict) -> "Dyadic":
        _strict_fields(payload, {"numerator", "exponent"}, "Dyadic")
        return cls(int(payload["numerator"]), int(payload["exponent"]))


@dataclass(frozen=True)
class JointWitnessRowSpec:
    schema_version: str
    row_hash: str
    split: str
    model_hash: str
    token_hash: str
    hook_spec_hash: str
    control_ast_hash: str
    contrast_hash: str
    branch_order: tuple[str, ...]
    graph_payload: dict
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.branch_order != ("PAT_J", "PAT_B", "TAR_J", "TAR_B"):
            raise ValueError("binding four-branch order mismatch")
        if self.split not in {"formal_prepare_pool", "synthetic"}:
            raise ValueError("row split is not authorized in static formal prepare")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "JointWitnessRowSpec":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _strict_fields(payload, expected, "JointWitnessRowSpec")
        return cls(**(payload | {"branch_order": tuple(payload["branch_order"])}))


@dataclass(frozen=True)
class CertificatePlan:
    schema_version: str
    row_hash: str
    exact_dyadic_amplitudes: tuple[Dyadic, ...]
    initial_partition: str
    split_policy: str
    absolute_width_tolerance: str
    relative_width_tolerance: str
    max_depth: int
    max_cells: int
    official_precision: int
    audit_precision: int
    expected_artifact_paths: tuple[str, ...]
    execution_authorized: bool = False

    def __post_init__(self):
        if self.execution_authorized:
            raise ValueError("real-row certificate execution is not authorized")
        if (self.initial_partition != "[-h,0],[0,h]" or
                self.split_policy != "curvature-weighted width priority dyadic bisection"):
            raise ValueError("unsupported frozen partition policy")
        if not self.exact_dyadic_amplitudes or any(
                radius.as_fraction() <= 0 for radius in self.exact_dyadic_amplitudes):
            raise ValueError("certificate radii must be nonempty and positive")
        if len({radius.as_fraction() for radius in self.exact_dyadic_amplitudes}) != len(
                self.exact_dyadic_amplitudes):
            raise ValueError("certificate radii must be unique")
        if self.official_precision < 2 or self.audit_precision <= self.official_precision:
            raise ValueError("audit precision must exceed valid official precision")
        if self.max_depth < 0 or self.max_cells < 2:
            raise ValueError("invalid certificate resource limits")
        for name, value in (("absolute", self.absolute_width_tolerance),
                            ("relative", self.relative_width_tolerance)):
            try:
                parsed = float.fromhex(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid {name} width tolerance") from error
            if not 0 < parsed < float("inf"):
                raise ValueError(f"invalid {name} width tolerance")

    @property
    def radii(self) -> tuple[Dyadic, ...]:
        return self.exact_dyadic_amplitudes

    @property
    def official_precision_bits(self) -> int:
        return self.official_precision

    @property
    def audit_precision_bits(self) -> int:
        return self.audit_precision

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "row_hash": self.row_hash,
            "exact_dyadic_amplitudes": [radius.to_dict() for radius in self.exact_dyadic_amplitudes],
            "initial_partition": self.initial_partition,
            "split_policy": self.split_policy,
            "absolute_width_tolerance": self.absolute_width_tolerance,
            "relative_width_tolerance": self.relative_width_tolerance,
            "max_depth": self.max_depth,
            "max_cells": self.max_cells,
            "official_precision": self.official_precision,
            "audit_precision": self.audit_precision,
            "expected_artifact_paths": list(self.expected_artifact_paths),
            "execution_authorized": self.execution_authorized,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CertificatePlan":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _strict_fields(payload, expected, "CertificatePlan")
        return cls(**(payload | {
            "exact_dyadic_amplitudes": tuple(
                Dyadic.from_dict(row) for row in payload["exact_dyadic_amplitudes"]),
            "expected_artifact_paths": tuple(payload["expected_artifact_paths"]),
        }))


@dataclass(frozen=True)
class AnytimeCellState:
    """One immutable, identity-closed leaf in a synthetic anytime partition."""

    schema_version: str
    evaluator_identity_sha256: str
    precision_bits: int
    lower: tuple[int, int]
    upper: tuple[int, int]
    depth: int
    priority: tuple[int, int]
    jet_payload: dict
    jet_semantic_hash: str
    result_source: str
    cache_entry_semantic_hash: str

    def __post_init__(self):
        if self.schema_version != "green-v400-anytime-cell-state-v1":
            raise ValueError("anytime cell schema version mismatch")
        if not _is_lower_sha256(self.evaluator_identity_sha256):
            raise ValueError("anytime cell evaluator identity is not lowercase SHA-256")
        if type(self.precision_bits) is not int or self.precision_bits < 2:
            raise ValueError("anytime cell precision is invalid")
        _validate_exact_rational_pair(self.lower, "anytime cell lower")
        _validate_exact_rational_pair(self.upper, "anytime cell upper")
        if Fraction(*self.lower) >= Fraction(*self.upper):
            raise ValueError("anytime cell bounds are invalid")
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("anytime cell depth is invalid")
        _validate_exact_rational_pair(self.priority, "anytime cell priority")
        if Fraction(*self.priority) < 0:
            raise ValueError("anytime cell priority must be nonnegative")
        if not isinstance(self.jet_payload, dict):
            raise ValueError("anytime cell Jet2 payload is invalid")
        _strict_fields(
            self.jet_payload, {"value", "first", "second"},
            "anytime cell Jet2 payload",
        )
        for component in ("value", "first", "second"):
            _validate_interval_payload(
                self.jet_payload[component], self.precision_bits,
                f"anytime cell Jet2 {component}",
            )
        if (not _is_lower_sha256(self.jet_semantic_hash)
                or self.jet_semantic_hash != sha256_canonical(self.jet_payload)):
            raise ValueError("anytime cell Jet2 semantic hash mismatch")
        if self.result_source not in {"COMPUTED", "EXACT_CACHE_HIT"}:
            raise ValueError("anytime cell result source is invalid")
        if not _is_lower_sha256(self.cache_entry_semantic_hash):
            raise ValueError("anytime cell cache identity is not lowercase SHA-256")
        expected_cache_hash = sha256_canonical({
            "schema_version": "green-v400-anytime-cache-entry-identity-v1",
            "evaluator_identity_sha256": self.evaluator_identity_sha256,
            "precision_bits": self.precision_bits,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "jet_semantic_hash": self.jet_semantic_hash,
        })
        if self.cache_entry_semantic_hash != expected_cache_hash:
            raise ValueError("anytime cell cache semantic hash mismatch")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "evaluator_identity_sha256": self.evaluator_identity_sha256,
            "precision_bits": self.precision_bits,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "depth": self.depth,
            "priority": list(self.priority),
            "jet_payload": copy.deepcopy(self.jet_payload),
            "jet_semantic_hash": self.jet_semantic_hash,
            "result_source": self.result_source,
            "cache_entry_semantic_hash": self.cache_entry_semantic_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AnytimeCellState":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _strict_fields(payload, expected, "AnytimeCellState")
        return cls(**(payload | {
            "lower": tuple(payload["lower"]),
            "upper": tuple(payload["upper"]),
            "priority": tuple(payload["priority"]),
        }))


@dataclass(frozen=True)
class MonotoneAnytimeCertificateState:
    """Strict synthetic-only checkpoint for one radius and one precision."""

    schema_version: str
    execution_scope: str
    row_hash: str
    certificate_plan_semantic_hash: str
    resource_lock_semantic_hash: str
    evaluator_identity_sha256: str
    radius: tuple[int, int]
    precision_bits: int
    phase: str
    checkpoint_index: int
    parent_state_semantic_hash: str
    logical_evaluations: int
    admitted_native_dispatches: int
    completed_native_dispatches: int
    exact_cache_hits: int
    leaves: tuple[AnytimeCellState, ...]
    endpoint_payload: dict
    raw_curvature_positive: dict
    raw_curvature_negative: dict
    raw_curvature_accounting: dict
    monotone_curvature_positive: dict
    monotone_curvature_negative: dict
    monotone_residual_positive: dict
    monotone_residual_negative: dict
    raw_witness: dict
    monotone_witness: dict
    computation_status: str
    resource_reason: str | None
    scientific_threshold_applied: bool
    _construction_integrity_hash: str = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self):
        if self.schema_version != "green-v400-monotone-anytime-state-v1":
            raise ValueError("anytime state schema version mismatch")
        if self.execution_scope != "outcome_blind_synthetic_only":
            raise ValueError("anytime state is restricted to outcome-blind synthetic execution")
        for name, value in (
            ("row", self.row_hash),
            ("plan", self.certificate_plan_semantic_hash),
            ("resource lock", self.resource_lock_semantic_hash),
            ("evaluator", self.evaluator_identity_sha256),
            ("parent state", self.parent_state_semantic_hash),
        ):
            if not _is_lower_sha256(value):
                raise ValueError(f"anytime {name} identity is not lowercase SHA-256")
        _validate_exact_rational_pair(self.radius, "anytime radius")
        if Fraction(*self.radius) <= 0:
            raise ValueError("anytime radius must be positive")
        if type(self.precision_bits) is not int or self.precision_bits < 2:
            raise ValueError("anytime state precision is invalid")
        if self.phase not in {"SYNTHETIC_OFFICIAL", "SYNTHETIC_AUDIT"}:
            raise ValueError("anytime state phase is invalid")
        counters = (
            self.checkpoint_index, self.logical_evaluations,
            self.admitted_native_dispatches, self.completed_native_dispatches,
            self.exact_cache_hits,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("anytime state counters must be nonnegative integers")
        if self.logical_evaluations != (
                self.admitted_native_dispatches + self.exact_cache_hits):
            raise ValueError("anytime logical evaluation accounting does not close")
        if self.completed_native_dispatches > self.admitted_native_dispatches:
            raise ValueError("anytime completed dispatches exceed admissions")
        if not self.leaves:
            raise ValueError("anytime state partition is empty")
        if any(
            leaf.evaluator_identity_sha256 != self.evaluator_identity_sha256
            or leaf.precision_bits != self.precision_bits
            for leaf in self.leaves
        ):
            raise ValueError("anytime leaf identity/precision mismatch")
        ordered = tuple(sorted(self.leaves, key=lambda leaf: Fraction(*leaf.lower)))
        if ordered != self.leaves:
            raise ValueError("anytime leaves are not in canonical endpoint order")
        h = Fraction(*self.radius)
        if Fraction(*self.leaves[0].lower) != -h or Fraction(*self.leaves[-1].upper) != h:
            raise ValueError("anytime leaves do not cover the radius")
        if not any(Fraction(*leaf.upper) == 0 for leaf in self.leaves):
            raise ValueError("anytime partition is not split at zero")
        for left, right in zip(self.leaves, self.leaves[1:]):
            if Fraction(*left.upper) != Fraction(*right.lower):
                raise ValueError("anytime partition has a gap or overlap")
        for leaf in self.leaves:
            lower, upper = Fraction(*leaf.lower), Fraction(*leaf.upper)
            if lower >= 0:
                weight = h * (upper - lower) - (upper * upper - lower * lower) / 2
            elif upper <= 0:
                weight = h * (upper - lower) + (upper * upper - lower * lower) / 2
            else:
                raise ValueError("anytime leaf crosses zero")
            second = leaf.jet_payload["second"]
            width = Fraction(*second["upper"]) - Fraction(*second["lower"])
            if Fraction(*leaf.priority) != weight * width:
                raise ValueError("anytime leaf exact priority mismatch")
        endpoint_expected = {"h", "negative", "center", "positive", "slope"}
        if not isinstance(self.endpoint_payload, dict):
            raise ValueError("anytime endpoint payload is invalid")
        _strict_fields(self.endpoint_payload, endpoint_expected, "anytime endpoint payload")
        _validate_exact_rational_pair(self.endpoint_payload["h"], "anytime endpoint h")
        if Fraction(*self.endpoint_payload["h"]) != h:
            raise ValueError("anytime endpoint radius mismatch")
        for name in ("negative", "center", "positive", "slope"):
            _validate_interval_payload(
                self.endpoint_payload[name], self.precision_bits,
                f"anytime endpoint {name}",
            )
        interval_fields = (
            "raw_curvature_positive", "raw_curvature_negative",
            "monotone_curvature_positive", "monotone_curvature_negative",
            "monotone_residual_positive", "monotone_residual_negative",
            "raw_witness", "monotone_witness",
        )
        for name in interval_fields:
            _validate_interval_payload(getattr(self, name), self.precision_bits, name)
        accounting_fields = {
            "positive_midpoint", "negative_midpoint",
            "positive_radius", "negative_radius",
            "positive_weight_rounding", "negative_weight_rounding",
            "positive_summation_rounding", "negative_summation_rounding",
        }
        if not isinstance(self.raw_curvature_accounting, dict):
            raise ValueError("raw curvature accounting must remain separate and explicit")
        _strict_fields(
            self.raw_curvature_accounting, accounting_fields,
            "raw curvature accounting",
        )
        for name, value in self.raw_curvature_accounting.items():
            _validate_exact_rational_pair(value, f"raw curvature accounting {name}")
        def nested(inner: dict, outer: dict) -> bool:
            return (
                Fraction(*outer["lower"]) <= Fraction(*inner["lower"])
                <= Fraction(*inner["upper"]) <= Fraction(*outer["upper"])
            )
        if not nested(self.monotone_curvature_positive, self.raw_curvature_positive):
            raise ValueError("monotone positive curvature is not inside current raw enclosure")
        if not nested(self.monotone_curvature_negative, self.raw_curvature_negative):
            raise ValueError("monotone negative curvature is not inside current raw enclosure")
        if not nested(self.monotone_residual_positive, self.monotone_curvature_positive):
            raise ValueError("positive residual is not inside monotone curvature")
        if not nested(self.monotone_residual_negative, self.monotone_curvature_negative):
            raise ValueError("negative residual is not inside monotone curvature")
        if not nested(self.monotone_witness, self.raw_witness):
            raise ValueError("monotone witness is not inside current raw witness")
        if self.computation_status not in {"PROVISIONAL", "RESOURCE_INCONCLUSIVE"}:
            raise ValueError("anytime computation status is invalid")
        if self.computation_status == "PROVISIONAL" and self.resource_reason is not None:
            raise ValueError("provisional anytime state cannot have a resource reason")
        if self.computation_status == "RESOURCE_INCONCLUSIVE" and self.resource_reason not in RESOURCE_REASONS:
            raise ValueError("anytime resource reason is invalid")
        if type(self.scientific_threshold_applied) is not bool or self.scientific_threshold_applied:
            raise ValueError("anytime synthetic state may not apply a scientific threshold")

        object.__setattr__(
            self, "_construction_integrity_hash",
            self._compute_integrity_hash_unchecked(),
        )

    def _to_dict_unchecked(self) -> dict:
        payload = {
            name: copy.deepcopy(getattr(self, name))
            for name, definition in self.__dataclass_fields__.items()
            if definition.init
        }
        payload["radius"] = list(self.radius)
        payload["leaves"] = [leaf.to_dict() for leaf in self.leaves]
        return payload

    def _compute_integrity_hash_unchecked(self) -> str:
        encoded = (
            b"GREEN-MONOTONE-ANYTIME-V1\0"
            + canonical_json(self._to_dict_unchecked()).encode("utf-8")
        )
        return hashlib.sha256(encoded).hexdigest()

    def assert_integrity(self) -> None:
        if self._compute_integrity_hash_unchecked() != self._construction_integrity_hash:
            raise RuntimeError("ANYTIME_STATE_INTEGRITY_INVALID")

    def to_dict(self) -> dict:
        self.assert_integrity()
        return self._to_dict_unchecked()

    def semantic_hash(self) -> str:
        self.assert_integrity()
        return self._construction_integrity_hash

    @classmethod
    def from_dict(cls, payload: dict) -> "MonotoneAnytimeCertificateState":
        expected = {
            field.name for field in cls.__dataclass_fields__.values() if field.init
        }
        _strict_fields(payload, expected, "MonotoneAnytimeCertificateState")
        return cls(**(payload | {
            "radius": tuple(payload["radius"]),
            "leaves": tuple(AnytimeCellState.from_dict(leaf) for leaf in payload["leaves"]),
        }))


RESOURCE_REASONS = (
    "MAX_DEPTH_REACHED",
    "MAX_FINAL_LEAVES_PER_RADIUS_REACHED",
    "WORK_TOKEN_BUDGET_EXHAUSTED",
    "WALL_DEADLINE_REACHED",
    "MEMORY_MAX_REACHED",
)


@dataclass(frozen=True)
class CertificateResourceLock:
    """Hash-closed candidate lock; external enforcement remains mandatory."""

    schema_version: str
    row_hash: str
    certificate_plan_semantic_hash: str
    radii_order_sha256: str
    radii_count: int
    phase_order: str
    official_precision: int
    audit_precision: int
    max_depth: int
    max_final_leaves_per_radius: int
    center_reuse: bool
    endpoint_passes_per_radius_precision: int
    charge_on_admission: bool
    failed_dispatch_refund: bool
    fte_formula_version: str
    primitive_taxonomy_version: str
    primitive_charge_per_dispatch: int
    token_weight_384: int
    token_weight_512: int
    token_budget: int
    orchestration_reserve_seconds: int
    wall_deadline_seconds: int
    memory_max_bytes: int
    partial_success_allowed: bool
    scientific_threshold_reads_before_interval_complete: bool
    worker_concurrency: int
    memory_enforcement: str
    swap_enforcement: str
    deadline_enforcement: str
    supervisor_process_scope: str
    deadline_scope: str
    publication_policy: str
    resource_reasons: tuple[str, ...]
    reachable_primary_reasons: tuple[str, ...]
    repository_commit: str
    python_source_manifest_sha256: str
    supervisor_executable_sha256: str
    resource_corrigendum_sha256: str
    backend_sha256: str
    descriptor_sha256: str
    blob_sha256: str
    program_execution_sha256: str
    dispatch_sha256: str
    fusion_sha256: str
    rounding_environment_sha256: str
    hardware_manifest_sha256: str
    production_authorized: bool = False

    def __post_init__(self):
        if self.schema_version != "green-v400-certificate-resource-lock-v1":
            raise ValueError("resource lock schema version mismatch")
        if self.production_authorized:
            raise ValueError("production resource lock is not yet authorized")
        integer_fields = (
            self.radii_count, self.official_precision, self.audit_precision,
            self.max_depth, self.max_final_leaves_per_radius,
            self.endpoint_passes_per_radius_precision,
            self.primitive_charge_per_dispatch, self.token_weight_384,
            self.token_weight_512, self.token_budget,
            self.orchestration_reserve_seconds, self.wall_deadline_seconds,
            self.memory_max_bytes, self.worker_concurrency,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ValueError("resource lock integers must be exact JSON integers")
        boolean_fields = (
            self.center_reuse, self.charge_on_admission,
            self.failed_dispatch_refund, self.partial_success_allowed,
            self.scientific_threshold_reads_before_interval_complete,
            self.production_authorized,
        )
        if any(type(value) is not bool for value in boolean_fields):
            raise ValueError("resource lock booleans must be exact JSON booleans")
        if self.phase_order != "ALL_384_THEN_REPLAY_SAME_PARTITION_512":
            raise ValueError("resource lock must use official-first phase order")
        if self.radii_count <= 0 or self.max_depth < 0:
            raise ValueError("resource lock has invalid radius/depth counts")
        if (self.official_precision, self.audit_precision) != (384, 512):
            raise ValueError("resource lock precision policy mismatch")
        if self.max_final_leaves_per_radius < 2:
            raise ValueError("resource lock needs both zero-split half-cells")
        if self.center_reuse:
            raise ValueError("center reuse is not enabled in the frozen cost formula")
        if self.endpoint_passes_per_radius_precision != 3:
            raise ValueError("resource lock endpoint-pass formula mismatch")
        if not self.charge_on_admission:
            raise ValueError("resource passes must be charged on admission")
        if self.failed_dispatch_refund:
            raise ValueError("failed dispatches may not refund admitted work")
        if self.fte_formula_version != "green-v400-fte-pass-v1":
            raise ValueError("resource lock FTE formula mismatch")
        if self.primitive_taxonomy_version != "green-v400-directed-primitives-v1":
            raise ValueError("resource lock primitive taxonomy version mismatch")
        if self.primitive_charge_per_dispatch != 352_275_450:
            raise ValueError("resource lock primitive taxonomy charge mismatch")
        if min(self.token_weight_384, self.token_weight_512, self.token_budget,
               self.wall_deadline_seconds, self.memory_max_bytes,
               self.worker_concurrency) <= 0:
            raise ValueError("resource lock contains a nonpositive limit")
        if self.token_budget + self.orchestration_reserve_seconds != self.wall_deadline_seconds:
            raise ValueError("token and orchestration budgets must close the wall deadline")
        if self.partial_success_allowed:
            raise ValueError("partial execution may not publish certificate success")
        if self.scientific_threshold_reads_before_interval_complete:
            raise ValueError("resource allocation may not read scientific thresholds")
        if self.worker_concurrency != 1:
            raise ValueError("candidate resource lock freezes one worker per row")
        if self.memory_enforcement != "cgroup_v2_memory.max":
            raise ValueError("resource lock requires cgroup-v2 hard memory enforcement")
        if self.swap_enforcement != "cgroup_v2_memory.swap.max=0":
            raise ValueError("resource lock requires disabled worker swap")
        if self.deadline_enforcement != "external_monotonic_supervisor_v1":
            raise ValueError("resource lock requires an external monotonic deadline")
        if self.supervisor_process_scope != "outside_worker_cgroup_pidfd_timerfd":
            raise ValueError("resource supervisor must survive worker OOM/deadline")
        if self.deadline_scope != "pre_exec_validation_through_atomic_publish":
            raise ValueError("resource deadline scope mismatch")
        if self.publication_policy != "TWO_PHASE_SUPERVISOR_COMMIT":
            raise ValueError("resource lock requires atomic two-phase publication")
        if self.resource_reasons != RESOURCE_REASONS:
            raise ValueError("resource lock reason vocabulary mismatch")
        expected_reachable = (
            "MAX_FINAL_LEAVES_PER_RADIUS_REACHED",
            "WALL_DEADLINE_REACHED",
            "MEMORY_MAX_REACHED",
        )
        if self.reachable_primary_reasons != expected_reachable:
            raise ValueError("resource lock reachable-reason set mismatch")
        identities = (
            self.row_hash, self.certificate_plan_semantic_hash,
            self.radii_order_sha256,
            self.python_source_manifest_sha256,
            self.supervisor_executable_sha256,
            self.resource_corrigendum_sha256, self.backend_sha256,
            self.descriptor_sha256, self.blob_sha256,
            self.program_execution_sha256, self.dispatch_sha256,
            self.fusion_sha256, self.rounding_environment_sha256,
            self.hardware_manifest_sha256,
        )
        if any(len(value) != 64 or any(character not in "0123456789abcdef"
                                       for character in value) for value in identities):
            raise ValueError("resource lock identity is not lowercase SHA-256")
        if (len(self.repository_commit) not in {40, 64}
                or any(character not in "0123456789abcdef"
                       for character in self.repository_commit)):
            raise ValueError("resource lock repository commit is not a full git object id")
        leaves = self.max_final_leaves_per_radius
        passes_384 = self.radii_count * (2 * leaves + 1)
        passes_512 = self.radii_count * (leaves + 3)
        worst_tokens = (
            passes_384 * self.token_weight_384
            + passes_512 * self.token_weight_512
        )
        if worst_tokens > self.token_budget:
            raise ValueError("resource lock leaf cap exceeds its token budget")

    @property
    def worst_case_passes_384(self) -> int:
        return self.radii_count * (2 * self.max_final_leaves_per_radius + 1)

    @property
    def worst_case_passes_512(self) -> int:
        return self.radii_count * (self.max_final_leaves_per_radius + 3)

    @property
    def worst_case_total_passes(self) -> int:
        return self.worst_case_passes_384 + self.worst_case_passes_512

    @property
    def worst_case_charged_primitives(self) -> int:
        return self.worst_case_total_passes * self.primitive_charge_per_dispatch

    def to_dict(self) -> dict:
        return asdict(self) | {
            "resource_reasons": list(self.resource_reasons),
            "reachable_primary_reasons": list(self.reachable_primary_reasons),
        }

    def semantic_hash(self) -> str:
        encoded = b"GREEN-RESOURCELOCK-V1\0" + canonical_json(self).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict) -> "CertificateResourceLock":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _strict_fields(payload, expected, "CertificateResourceLock")
        return cls(**(payload | {
            "resource_reasons": tuple(payload["resource_reasons"]),
            "reachable_primary_reasons": tuple(payload["reachable_primary_reasons"]),
        }))
