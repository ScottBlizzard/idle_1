"""Deterministic public P13 qualification direction payloads."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import numpy as np

from green_v410_protocol import PROTOCOL_ID, canonical_json_bytes, sha256_canonical


GENERATOR_SPEC = "numpy-PCG64DXSM-rowwise-normal-v1"
DIRECTION_WIDTH = 768
DIRECTION_COUNT = 8
DIRECTION_NORM = 0.001


def qualification_master_seed(direction_domain: str) -> bytes:
    if direction_domain not in {
        "GREEN_V410_P13_IOI_GREEN_PANEL",
        "GREEN_V410_P13_GT_GREEN_PANEL",
    }:
        raise ValueError("unknown P13 qualification direction domain")
    payload = (
        f"{PROTOCOL_ID}\0{direction_domain}\0A1\0"
        "OUTCOME_BLIND_PUBLIC_QUALIFICATION"
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _row_seed(master_seed: bytes, site_row_id: str, direction_domain: str) -> int:
    if len(master_seed) != 32:
        raise ValueError("qualification master seed must contain 32 bytes")
    message = (
        f"{PROTOCOL_ID}\0{site_row_id}\0{direction_domain}\0{GENERATOR_SPEC}"
    ).encode("utf-8")
    return int.from_bytes(hmac.new(master_seed, message, hashlib.sha256).digest(), "little")


def generate_site_directions(site_row_id: str, direction_domain: str) -> np.ndarray:
    if not isinstance(site_row_id, str) or not site_row_id:
        raise ValueError("site row ID must be nonempty")
    seed = qualification_master_seed(direction_domain)
    generator = np.random.Generator(np.random.PCG64DXSM(
        _row_seed(seed, site_row_id, direction_domain)
    ))
    values = generator.standard_normal((DIRECTION_COUNT, DIRECTION_WIDTH), dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise RuntimeError("direction generator produced a degenerate vector")
    return np.ascontiguousarray((values / norms * DIRECTION_NORM).astype("<f4"))


def tensor_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.dtype != np.dtype("<f4") or value.ndim != 2:
        raise ValueError("direction tensor must be rank-2 little-endian float32")
    if not np.isfinite(value).all():
        raise ValueError("direction tensor must be finite")
    header = {
        "schema_version": "green-v410-direction-tensor-v1",
        "dtype": "float32-little-endian",
        "shape": list(value.shape),
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(value, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def build_row_binding(
    *, task: str, site_row_id: str, direction_domain: str, array: np.ndarray,
) -> dict[str, Any]:
    if task not in {"ioi", "greater_than"}:
        raise ValueError("qualification direction task is invalid")
    if array.shape != (DIRECTION_COUNT, DIRECTION_WIDTH):
        raise ValueError("qualification direction tensor shape is invalid")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    tolerance = max(1e-8, DIRECTION_NORM * 2e-6)
    if not np.all(np.abs(norms - DIRECTION_NORM) <= tolerance):
        raise ValueError("qualification direction norm mismatch")
    binding = {
        "schema_version": "green-v410-p13-direction-binding-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": task,
        "site_row_id": site_row_id,
        "direction_domain": direction_domain,
        "generator_spec": GENERATOR_SPEC,
        "master_seed_derivation": "sha256(protocol_id\\0direction_domain\\0A1\\0OUTCOME_BLIND_PUBLIC_QUALIFICATION)",
        "master_seed_sha256": hashlib.sha256(qualification_master_seed(direction_domain)).hexdigest(),
        "dtype": "float32-little-endian",
        "shape": [DIRECTION_COUNT, DIRECTION_WIDTH],
        "direction_norm": DIRECTION_NORM,
        "tensor_sha256": tensor_sha256(array),
    }
    binding["binding_sha256"] = sha256_canonical(binding)
    return binding

