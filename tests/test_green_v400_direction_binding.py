from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_v400_direction_binding import (
    binding_sha256,
    build_direction_binding,
    direction_tensor_sha256,
    verify_direction_binding,
)


def directions():
    return torch.tensor([[0.0006, 0.0008], [-0.0008, 0.0006]], dtype=torch.float32)


def test_actual_float32_tensor_bytes_are_bound_and_verified():
    tensor = directions()
    binding = build_direction_binding(
        protocol_id="P",
        row_id="row",
        panel_kind="endpoint",
        tensor=tensor,
        direction_norm=1e-3,
        generator_spec="test-v1",
    )
    verify_direction_binding(
        tensor=tensor,
        binding=binding,
        expected_binding_sha256=binding_sha256(binding),
        protocol_id="P",
        row_id="row",
        panel_kind="endpoint",
    )


def test_tensor_substitution_breaks_payload_hash_even_when_shape_matches():
    tensor = directions()
    binding = build_direction_binding(
        protocol_id="P",
        row_id="row",
        panel_kind="endpoint",
        tensor=tensor,
        direction_norm=1e-3,
        generator_spec="test-v1",
    )
    changed = tensor.clone()
    changed[0] = -changed[0]
    assert direction_tensor_sha256(changed) != direction_tensor_sha256(tensor)
    with pytest.raises(ValueError, match="payload hash"):
        verify_direction_binding(
            tensor=changed,
            binding=binding,
            expected_binding_sha256=binding_sha256(binding),
            protocol_id="P",
            row_id="row",
            panel_kind="endpoint",
        )


def test_binding_substitution_and_wrong_dtype_fail_closed():
    tensor = directions()
    binding = build_direction_binding(
        protocol_id="P",
        row_id="row",
        panel_kind="endpoint",
        tensor=tensor,
        direction_norm=1e-3,
        generator_spec="test-v1",
    )
    changed = dict(binding)
    changed["row_id"] = "other"
    with pytest.raises(ValueError, match="commitment"):
        verify_direction_binding(
            tensor=tensor,
            binding=changed,
            expected_binding_sha256=binding_sha256(binding),
            protocol_id="P",
            row_id="row",
            panel_kind="endpoint",
        )
    with pytest.raises(ValueError, match="dtype"):
        direction_tensor_sha256(tensor.to(torch.float64))
