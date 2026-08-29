"""Plan-bound conversion to high-precision response evaluation.

The checkpoint remains the frozen float32 artifact.  Float64 execution uses the
same exactly representable constants and is required only to avoid cancellation
when differencing nearby scalar responses.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Match the byte-only tensor hashes in the frozen v4 model manifest."""

    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def precision_receipt_sha256(receipt: dict[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def prepare_float64_response_evaluation(
    *,
    model: Any,
    model_manifest: dict[str, Any],
    expected_model_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify float32 checkpoint tensors, convert, and verify exact round-trip."""

    if hashlib.sha256(_canonical(model_manifest)).hexdigest() != expected_model_manifest_sha256:
        raise ValueError("response precision model manifest hash mismatch")
    expected = model_manifest.get("weight_tensor_hashes")
    state = model.state_dict()
    if not isinstance(expected, dict) or set(expected) != set(state):
        raise ValueError("response precision state keys differ from model manifest")
    floating_names = [name for name, value in state.items() if value.is_floating_point()]
    if not floating_names or any(state[name].dtype != torch.float32 for name in floating_names):
        raise ValueError("response precision conversion requires a fresh float32 model")
    before = {name: tensor_sha256(value) for name, value in state.items()}
    if before != expected:
        raise ValueError("response precision input model differs from frozen float32 weights")

    to_method = getattr(model, "to", None)
    if not callable(to_method):
        raise ValueError("response precision model does not support dtype conversion")
    try:
        model = to_method(torch.float64, print_details=False)
    except TypeError:
        model = to_method(dtype=torch.float64)
    converted = model.state_dict()
    if any(converted[name].dtype != torch.float64 for name in floating_names):
        raise ValueError("response precision model did not convert every floating tensor")
    roundtrip = {
        name: tensor_sha256(
            value.float() if name in floating_names else value
        )
        for name, value in converted.items()
    }
    if roundtrip != expected:
        raise ValueError("float64 conversion changed a frozen checkpoint value")
    receipt = {
        "schema_version": "green-v400-response-evaluation-precision-receipt-v1",
        "model_manifest_sha256": expected_model_manifest_sha256,
        "checkpoint_storage_dtype": "float32",
        "response_evaluation_dtype": "float64",
        "model_manifest_tensor_hash_scheme": (
            "sha256-contiguous-numpy-native-bytes-v1"
        ),
        "floating_tensor_count": len(floating_names),
        "all_manifest_tensor_hashes_matched_before_conversion": True,
        "all_float64_values_roundtrip_to_manifest_float32_exactly": True,
        "scientific_outcome_evaluated": False,
    }
    receipt["receipt_sha256"] = precision_receipt_sha256(receipt)
    return receipt


def verify_precision_receipt(receipt: dict[str, Any], manifest_sha256: str) -> None:
    if receipt.get("schema_version") != (
        "green-v400-response-evaluation-precision-receipt-v1"
    ):
        raise ValueError("response precision receipt schema mismatch")
    if receipt.get("receipt_sha256") != precision_receipt_sha256(receipt):
        raise ValueError("response precision receipt self hash mismatch")
    if receipt.get("model_manifest_sha256") != manifest_sha256:
        raise ValueError("response precision receipt manifest mismatch")
    if receipt.get("response_evaluation_dtype") != "float64":
        raise ValueError("response precision receipt does not require float64")
    if receipt.get("model_manifest_tensor_hash_scheme") != (
        "sha256-contiguous-numpy-native-bytes-v1"
    ):
        raise ValueError("response precision receipt tensor hash scheme mismatch")
    if receipt.get("all_float64_values_roundtrip_to_manifest_float32_exactly") is not True:
        raise ValueError("response precision receipt lacks exact checkpoint preservation")
