"""Frozen held-out direction design for GREEN v3.0.0."""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

from green_bridge_v300_spec import V300_COEFFICIENT_SHA256, canonical_json


def helmert_coefficients_v300() -> np.ndarray:
    return np.asarray([
        [1 / math.sqrt(5)] * 5,
        [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0, 0],
        [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6), 0, 0],
        [1 / math.sqrt(12), 1 / math.sqrt(12), 1 / math.sqrt(12), -3 / math.sqrt(12), 0],
    ], dtype=np.float64)


def coefficient_payload_v300() -> dict:
    """Typed payload; its protocol hash is frozen separately by Pro.

    The decision supplies an identifier but omits the byte serializer that
    generated it.  We preserve both the typed values and the binding hash ID;
    prepare refuses to claim the one-shot until the serializer is resolved.
    """
    return {
        "schema": "green-bridge-v3.0.0-helmert-coefficients-v1",
        "rows": helmert_coefficients_v300().tolist(),
        "binding_hash_id": V300_COEFFICIENT_SHA256,
    }


def computed_coefficient_payload_sha256_v300() -> str:
    return hashlib.sha256(canonical_json(coefficient_payload_v300()).encode("utf-8")).hexdigest()


def coefficient_payload_sha256_v300() -> str:
    """Return the binding hash supplied by the external protocol decision."""
    return V300_COEFFICIENT_SHA256


def coefficient_serializer_status_v300() -> dict:
    computed = computed_coefficient_payload_sha256_v300()
    return {
        "binding_sha256": V300_COEFFICIENT_SHA256,
        "computed_typed_payload_sha256": computed,
        "byte_serializer_specified_by_decision": False,
        "resolved": computed == V300_COEFFICIENT_SHA256,
    }


def deterministic_complement_v300(frame: np.ndarray, count: int = 6) -> np.ndarray:
    q = np.asarray(frame, dtype=np.float64)
    if q.shape != (768, 5):
        raise ValueError(f"frame must have shape (768,5), got {q.shape}")
    accepted: list[np.ndarray] = []
    for index in range(768):
        value = np.zeros(768, dtype=np.float64)
        value[index] = 1.0
        for _ in range(2):
            value -= q @ (q.T @ value)
            for prior in accepted:
                value -= prior * float(prior @ value)
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            continue
        value /= norm
        pivot = int(np.argmax(np.abs(value)))
        if value[pivot] < 0:
            value = -value
        accepted.append(value)
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise RuntimeError("unable to construct six complement directions")
    result = np.stack(accepted, axis=1)
    if np.max(np.abs(q.T @ result)) > 1e-12:
        raise AssertionError("complement is not orthogonal to frame")
    if np.max(np.abs(result.T @ result - np.eye(count))) > 1e-12:
        raise AssertionError("complement is not orthonormal")
    return result


def heldout_direction_panel_v300(frame: np.ndarray) -> dict[str, np.ndarray]:
    q = np.asarray(frame, dtype=np.float64)
    a = helmert_coefficients_v300()
    in_frame = q @ a.T
    complement = deterministic_complement_v300(q)
    mixed = (in_frame + complement[:, :4]) / math.sqrt(2)
    null = complement[:, 4:6]
    for name, values in (("in_frame", in_frame), ("mixed", mixed), ("null", null)):
        norms = np.linalg.norm(values, axis=0)
        if not np.allclose(norms, 1.0, atol=1e-12, rtol=0):
            raise AssertionError(f"{name} directions are not unit norm")
    return {"in_frame": in_frame, "mixed": mixed, "null": null, "complement": complement}


def direction_design_sha256_v300(frame: np.ndarray) -> str:
    panel = heldout_direction_panel_v300(frame)
    digest = hashlib.sha256()
    for name in ("in_frame", "mixed", "null", "complement"):
        value = np.ascontiguousarray(panel[name], dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()
