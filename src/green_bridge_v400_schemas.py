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
