"""Canonical binding and verification for frozen direction tensors."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_float32_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return platform-independent little-endian contiguous float32 bytes."""

    if tensor.dtype != torch.float32:
        raise ValueError("direction tensor dtype must be torch.float32")
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError("direction tensor must have a nonempty [count, width] shape")
    if not torch.isfinite(tensor).all():
        raise ValueError("direction tensor must be finite")
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return array.tobytes(order="C")


def direction_tensor_sha256(tensor: torch.Tensor) -> str:
    header = {
        "schema_version": "green-v400-direction-tensor-v1",
        "dtype": "float32-little-endian",
        "shape": list(tensor.shape),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header))
    digest.update(b"\0")
    digest.update(canonical_float32_tensor_bytes(tensor))
    return digest.hexdigest()


def build_direction_binding(
    *,
    protocol_id: str,
    row_id: str,
    panel_kind: str,
    tensor: torch.Tensor,
    direction_norm: float,
    generator_spec: str,
) -> dict[str, Any]:
    if panel_kind not in {"green", "endpoint"}:
        raise ValueError("panel_kind must be green or endpoint")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("protocol_id must be nonempty")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("row_id must be nonempty")
    if not math.isfinite(direction_norm) or direction_norm <= 0:
        raise ValueError("direction_norm must be finite and positive")
    if not isinstance(generator_spec, str) or not generator_spec:
        raise ValueError("generator_spec must be nonempty")
    canonical_float32_tensor_bytes(tensor)
    norms = torch.linalg.vector_norm(tensor.detach().cpu().to(torch.float64), dim=1)
    tolerance = max(1e-8, direction_norm * 2e-6)
    if not torch.all(torch.abs(norms - direction_norm) <= tolerance):
        raise ValueError("direction rows do not have the declared norm")
    return {
        "schema_version": "green-v400-direction-binding-v1",
        "protocol_id": protocol_id,
        "row_id": row_id,
        "panel_kind": panel_kind,
        "dtype": "float32-little-endian",
        "shape": list(tensor.shape),
        "direction_norm": direction_norm,
        "generator_spec": generator_spec,
        "tensor_sha256": direction_tensor_sha256(tensor),
    }


def binding_sha256(binding: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(binding)).hexdigest()


def verify_direction_binding(
    *,
    tensor: torch.Tensor,
    binding: dict[str, Any],
    expected_binding_sha256: str,
    protocol_id: str,
    row_id: str,
    panel_kind: str,
) -> None:
    if binding_sha256(binding) != expected_binding_sha256:
        raise ValueError("direction binding commitment mismatch")
    if binding.get("protocol_id") != protocol_id or binding.get("row_id") != row_id:
        raise ValueError("direction binding identity mismatch")
    if binding.get("panel_kind") != panel_kind:
        raise ValueError("direction binding panel mismatch")
    if binding.get("shape") != list(tensor.shape):
        raise ValueError("direction binding shape mismatch")
    if binding.get("dtype") != "float32-little-endian":
        raise ValueError("direction binding dtype declaration mismatch")
    if binding.get("tensor_sha256") != direction_tensor_sha256(tensor):
        raise ValueError("direction tensor payload hash mismatch")
    rebuilt = build_direction_binding(
        protocol_id=protocol_id,
        row_id=row_id,
        panel_kind=panel_kind,
        tensor=tensor,
        direction_norm=float(binding.get("direction_norm")),
        generator_spec=binding.get("generator_spec"),
    )
    if rebuilt != binding:
        raise ValueError("direction binding metadata does not match tensor")
