import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_response_precision import (
    prepare_float64_response_evaluation,
    tensor_sha256,
    verify_precision_receipt,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def test_float64_conversion_preserves_frozen_float32_values_exactly():
    model = torch.nn.Linear(3, 2).float()
    manifest = {
        "weight_tensor_hashes": {
            name: tensor_sha256(value) for name, value in model.state_dict().items()
        }
    }
    receipt = prepare_float64_response_evaluation(
        model=model,
        model_manifest=manifest,
        expected_model_manifest_sha256=digest(manifest),
    )
    assert all(value.dtype == torch.float64 for value in model.state_dict().values())
    assert receipt["all_float64_values_roundtrip_to_manifest_float32_exactly"] is True
    verify_precision_receipt(receipt, digest(manifest))


def test_manifest_tensor_hash_is_exactly_the_frozen_raw_byte_scheme():
    tensor = torch.tensor([1.0, -2.5], dtype=torch.float32)
    expected = hashlib.sha256(
        tensor.numpy().tobytes(order="C")
    ).hexdigest()
    assert tensor_sha256(tensor) == expected


def test_precision_conversion_rejects_changed_or_already_converted_model():
    model = torch.nn.Linear(2, 1).float()
    manifest = {
        "weight_tensor_hashes": {
            name: tensor_sha256(value) for name, value in model.state_dict().items()
        }
    }
    with torch.no_grad():
        model.weight[0, 0] += 1.0
    with pytest.raises(ValueError, match="differs"):
        prepare_float64_response_evaluation(
            model=model,
            model_manifest=manifest,
            expected_model_manifest_sha256=digest(manifest),
        )
    model = torch.nn.Linear(2, 1).double()
    manifest = {
        "weight_tensor_hashes": {
            name: tensor_sha256(value.float())
            for name, value in model.state_dict().items()
        }
    }
    with pytest.raises(ValueError, match="fresh float32"):
        prepare_float64_response_evaluation(
            model=model,
            model_manifest=manifest,
            expected_model_manifest_sha256=digest(manifest),
        )
