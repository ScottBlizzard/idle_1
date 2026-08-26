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
    radii: tuple[Dyadic, ...]
    official_precision_bits: int
    audit_precision_bits: int
    max_depth: int
    max_cells: int
    execution_authorized: bool = False

    def __post_init__(self):
        if self.execution_authorized:
            raise ValueError("real-row certificate execution is not authorized")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "row_hash": self.row_hash,
            "radii": [radius.to_dict() for radius in self.radii],
            "official_precision_bits": self.official_precision_bits,
            "audit_precision_bits": self.audit_precision_bits,
            "max_depth": self.max_depth,
            "max_cells": self.max_cells,
            "execution_authorized": self.execution_authorized,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CertificatePlan":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _strict_fields(payload, expected, "CertificatePlan")
        return cls(**(payload | {"radii": tuple(Dyadic.from_dict(row) for row in payload["radii"])}))
