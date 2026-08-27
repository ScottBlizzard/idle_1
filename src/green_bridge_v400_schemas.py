"""Canonical schemas for GREEN v4 static formal prepare."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
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
