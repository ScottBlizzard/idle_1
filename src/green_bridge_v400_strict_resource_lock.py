"""Prepare-only v2 resource lock for the corrected full-history audit path."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re

from green_bridge_v400_schemas import RESOURCE_REASONS, canonical_json


SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class StrictSingleProcessResourceLock:
    """Hash-closed strict-lock candidate; real execution remains disabled."""

    schema_version: str
    row_hash: str
    certificate_plan_semantic_hash: str
    radii_order_sha256: str
    radii_count: int
    phase_order: str
    official_precision: int
    audit_precision: int
    audit_history_policy: str
    max_depth: int
    max_final_leaves_per_radius: int
    center_reuse: bool
    charge_on_admission: bool
    failed_dispatch_refund: bool
    token_weight_384: int
    token_weight_512: int
    token_budget: int
    orchestration_reserve_seconds: int
    wall_deadline_seconds: int
    user_address_space_max_bytes: int
    worker_concurrency: int
    memory_enforcement: str
    process_creation_enforcement: str
    numeric_thread_environment_sha256: str
    swap_policy: str
    deadline_enforcement: str
    supervisor_process_scope: str
    deadline_scope: str
    publication_policy: str
    resource_reasons: tuple[str, ...]
    reachable_primary_reasons: tuple[str, ...]
    repository_commit: str
    source_closure_sha256: str
    strict_wrapper_report_semantic_hash: str
    strict_numerics_report_semantic_hash: str
    backend_sha256: str
    descriptor_sha256: str
    blob_sha256: str
    program_execution_sha256: str
    dispatch_sha256: str
    fusion_sha256: str
    rounding_environment_sha256: str
    hardware_manifest_sha256: str
    production_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != "green-v400-certificate-resource-lock-v2":
            raise ValueError("strict resource lock schema version mismatch")
        if self.production_authorized:
            raise ValueError("strict production resource lock is not yet authorized")
        integers = (
            self.radii_count, self.official_precision, self.audit_precision,
            self.max_depth, self.max_final_leaves_per_radius,
            self.token_weight_384, self.token_weight_512, self.token_budget,
            self.orchestration_reserve_seconds, self.wall_deadline_seconds,
            self.user_address_space_max_bytes, self.worker_concurrency,
        )
        if any(type(value) is not int for value in integers):
            raise ValueError("strict resource lock integers must be exact")
        booleans = (
            self.center_reuse, self.charge_on_admission,
            self.failed_dispatch_refund, self.production_authorized,
        )
        if any(type(value) is not bool for value in booleans):
            raise ValueError("strict resource lock booleans must be exact")
        if self.radii_count <= 0 or self.max_depth < 0:
            raise ValueError("strict resource lock radius/depth invalid")
        if self.max_final_leaves_per_radius < 2:
            raise ValueError("strict resource lock requires both initial cells")
        if (self.official_precision, self.audit_precision) != (384, 512):
            raise ValueError("strict resource lock precision policy mismatch")
        if self.phase_order != "ALL_384_THEN_FULL_HISTORY_512":
            raise ValueError("strict resource lock phase order mismatch")
        if self.audit_history_policy != (
                "COMPLETE_FROZEN_OFFICIAL_SPLIT_HISTORY_INDEPENDENT_RECURRENCE"):
            raise ValueError("strict resource lock audit history mismatch")
        if self.center_reuse:
            raise ValueError("strict resource lock forbids center reuse")
        if not self.charge_on_admission or self.failed_dispatch_refund:
            raise ValueError("strict resource accounting must charge without refund")
        if min(
            self.token_weight_384, self.token_weight_512, self.token_budget,
            self.wall_deadline_seconds, self.user_address_space_max_bytes,
            self.worker_concurrency,
        ) <= 0:
            raise ValueError("strict resource lock contains nonpositive limit")
        if self.token_budget + self.orchestration_reserve_seconds != (
                self.wall_deadline_seconds):
            raise ValueError("strict token and wall budgets do not close")
        if self.worker_concurrency != 1:
            raise ValueError("strict resource lock requires one worker")
        expected_strings = {
            "memory_enforcement": "RLIMIT_AS_HARD_EQUAL_SOFT",
            "process_creation_enforcement": "RLIMIT_NPROC_HARD_EQUAL_SOFT_ONE",
            "swap_policy": "WITHIN_HARD_ADDRESS_SPACE_NOT_SEPARATELY_DISABLED",
            "deadline_enforcement": "EXTERNAL_MONOTONIC_PIDFD_TIMERFD_V2",
            "supervisor_process_scope": "OUTSIDE_WORKER_PROCESS_GROUP",
            "deadline_scope": "PRE_EXEC_VALIDATION_THROUGH_ATOMIC_PUBLISH",
            "publication_policy": "TWO_PHASE_SUPERVISOR_COMMIT",
        }
        for field, expected in expected_strings.items():
            if getattr(self, field) != expected:
                raise ValueError(f"strict resource lock {field} mismatch")
        if self.resource_reasons != RESOURCE_REASONS:
            raise ValueError("strict resource reason vocabulary mismatch")
        expected_reachable = (
            "MAX_FINAL_LEAVES_PER_RADIUS_REACHED",
            "WALL_DEADLINE_REACHED",
            "MEMORY_MAX_REACHED",
        )
        if self.reachable_primary_reasons != expected_reachable:
            raise ValueError("strict reachable resource reasons mismatch")
        hashes = (
            self.row_hash, self.certificate_plan_semantic_hash,
            self.radii_order_sha256, self.numeric_thread_environment_sha256,
            self.source_closure_sha256,
            self.strict_wrapper_report_semantic_hash,
            self.strict_numerics_report_semantic_hash, self.backend_sha256,
            self.descriptor_sha256, self.blob_sha256,
            self.program_execution_sha256, self.dispatch_sha256,
            self.fusion_sha256, self.rounding_environment_sha256,
            self.hardware_manifest_sha256,
        )
        if any(SHA256_RE.fullmatch(value) is None for value in hashes):
            raise ValueError("strict resource identity is not SHA-256")
        if (
            len(self.repository_commit) not in {40, 64}
            or any(character not in "0123456789abcdef"
                   for character in self.repository_commit)
        ):
            raise ValueError("strict repository commit is not a full object id")
        if self.worst_case_token_charge > self.token_budget:
            raise ValueError("strict leaf cap exceeds token budget")

    @property
    def worst_case_passes_384(self) -> int:
        return self.radii_count * (2 * self.max_final_leaves_per_radius + 1)

    @property
    def worst_case_passes_512(self) -> int:
        return self.radii_count * (2 * self.max_final_leaves_per_radius + 1)

    @property
    def worst_case_total_passes(self) -> int:
        return self.worst_case_passes_384 + self.worst_case_passes_512

    @property
    def worst_case_token_charge(self) -> int:
        return (
            self.worst_case_passes_384 * self.token_weight_384
            + self.worst_case_passes_512 * self.token_weight_512
        )

    def to_dict(self) -> dict:
        return asdict(self) | {
            "resource_reasons": list(self.resource_reasons),
            "reachable_primary_reasons": list(self.reachable_primary_reasons),
        }

    def semantic_hash(self) -> str:
        encoded = b"GREEN-RESOURCELOCK-V2\0" + canonical_json(self).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict) -> "StrictSingleProcessResourceLock":
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("strict resource lock fields mismatch")
        return cls(**(payload | {
            "resource_reasons": tuple(payload["resource_reasons"]),
            "reachable_primary_reasons": tuple(
                payload["reachable_primary_reasons"]),
        }))
