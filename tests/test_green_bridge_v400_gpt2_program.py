from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_gpt2_program import (
    GPT2TailDimensions, build_gpt2_joint_witness_program,
    execute_tensor_program_numpy, execute_tensor_program_torch, program_identity_payload,
)
from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr_tensor_executor import (
    execute_tensor_program_mpfr, jet_exact_payload, tensor_program_required_axis0_rows,
)
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import (
    TensorProgram, tensor_program_dispatch_signature, tensor_program_native_trace,
)
from green_bridge_v400_tensor_store import TensorStoreReader, write_tensor_store
from green_bridge_v400_resident_plan import build_resident_plan, load_resident_plan_arrays


def _layer_norm(x, weight, bias, epsilon):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    return centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + epsilon) * weight + bias


def _fixture(tmp_path):
    rng = np.random.default_rng(17)
    dims = GPT2TailDimensions(3, 4, 8, 2, 2, (1, 6), 2, 3)

    def random(shape, scale=0.1):
        return (rng.standard_normal(shape) * scale).astype("<f4")

    eps = np.asarray(1e-5, dtype="<f4")
    kappa = np.asarray(math.sqrt(2.0 / math.pi), dtype="<f4")
    lam = np.asarray(0.044715, dtype="<f4")
    ln10_w, ln10_b = random((4,), 0.02) + 1, random((4,), 0.02)
    w10_in, b10_in = random((4, 2)), random((2,))
    w10_out = random((2, 4))

    tensors = [
        ("physical_direction", random((4,), 0.03)),
        ("layer_norm.eps", eps),
        ("gelu.kappa", kappa),
        ("gelu.lambda", lam),
        ("zero.d_model", np.zeros((4,), dtype="<f4")),
        ("block10.ln2.w", ln10_w.astype("<f4")),
        ("block10.ln2.b", ln10_b),
        ("block10.mlp.W_in_selected", w10_in),
        ("block10.mlp.b_in_selected", b10_in),
        ("block10.mlp.W_out_selected", w10_out),
    ]
    for layer in ("ln1", "ln2"):
        tensors.extend((
            (f"block11.{layer}.w", (random((4,), 0.02) + 1).astype("<f4")),
            (f"block11.{layer}.b", random((4,), 0.02)),
        ))
    for name in ("Q", "K", "V", "O"):
        tensors.extend((
            (f"block11.attn.W_{name}", random((4, 4))),
            (f"block11.attn.b_{name}", random((4,))),
        ))
    tensors.extend((
        ("block11.mlp.W_in", random((4, 8))),
        ("block11.mlp.b_in", random((8,))),
        ("block11.mlp.W_out", random((8, 4))),
        ("block11.mlp.b_out", random((4,))),
        ("ln_final.w", (random((4,), 0.02) + 1).astype("<f4")),
        ("ln_final.b", random((4,), 0.02)),
        ("unembed.W_U_full", random((4, 7))),
        ("unembed.b_U_full", random((7,))),
        ("unembed.suffix_ids", np.asarray([0, 3, 6], dtype="<i8")),
        ("contrast.coefficients", np.asarray([1.0, -0.5, 0.25], dtype="<f8")),
    ))
    for condition in ("PAT", "TAR"):
        resid_mid = random((3, 4), 0.2)
        normalized = _layer_norm(resid_mid, ln10_w, ln10_b, eps)
        selected_pre = normalized @ w10_in + b10_in
        selected_post = 0.5 * selected_pre * (
            1 + np.tanh(kappa * (selected_pre + lam * selected_pre**3))
        )
        tensors.extend((
            (f"{condition}.resid_mid", resid_mid),
            (f"{condition}.resid_post", random((3, 4), 0.2)),
            (f"{condition}.selected_post", selected_post[dims.final_position].astype("<f4")),
        ))
    write_tensor_store(tmp_path, "fixture", tensors)
    reader = TensorStoreReader(tmp_path / "fixture.json")
    program = build_gpt2_joint_witness_program(reader, "a" * 64, dims)
    return reader, dims, program


def test_gpt2_program_is_closed_replayable_and_four_branch(tmp_path):
    reader, dims, program = _fixture(tmp_path)
    assert TensorProgram.from_dict(program.to_dict()) == program
    identity = program_identity_payload(program, dims, reader)
    assert identity["contains_scientific_outcome"] is False
    assert set(program.branch_roots) == {"PAT_J", "PAT_B", "TAR_J", "TAR_B"}
    assert len(program.nodes) > 50
    assert program.nodes[-1].exact_attrs["weights"] == [1, -1, -1, 1]
    for node in program.nodes:
        mask = node.exact_attrs["dependency_mask_spec"]
        if not node.exact_attrs["depends_on_t"]:
            assert mask["kind"] == "empty"
            assert mask["dependent_scalar_count"] == 0
        elif node.output_spec.shape:
            assert mask["kind"] == "axis0_rows"
            assert mask["axis0_indices"] == [dims.final_position]
            assert mask["dependent_scalar_count"] == math.prod(node.output_spec.shape[1:])
        else:
            assert mask["kind"] == "dense"
            assert mask["dependent_scalar_count"] == 1
    assert len(program.resource_formula["dependency_mask_closure_sha256"]) == 64
    signature = tensor_program_dispatch_signature(program.nodes)
    assert signature["node_count"] == 81
    assert signature["kernel_counts"] == {
        "affine_scatter.v1": 4,
        "branch_linear_combination.v1": 1,
        "causal_attention.v1": 4,
        "final_contrast.v1": 4,
        "gelu_new.v1": 8,
        "layer_norm.v1": 16,
        "pairwise_affine.v1": 30,
        "residual_add.v1": 10,
        "static_view.v1": 4,
    }
    assert program.resource_formula["dispatcher_signature_sha256"] == sha256_canonical(signature)
    assert program.resource_formula["directed_arithmetic_dominates_dense_lower_bound"] is True
    assert len(program.resource_formula["exact_final_contrast_fusion_sha256"]) == 64
    assert tensor_program_native_trace(program.nodes) == {
        "event_count": 81, "fnv1a_u64": "e0f23d0f4c4df894",
    }


def test_scalar_output_row_liveness_keeps_attention_history(tmp_path):
    _, dims, program = _fixture(tmp_path)
    rows = tensor_program_required_axis0_rows(program)
    assert len(rows) == len([node for node in program.nodes if node.output_spec.shape])
    for node in program.nodes:
        if node.kernel_id == "causal_attention.v1":
            assert rows[node.semantic_id] == (dims.final_position,)
            q, k, v = node.parent_semantic_ids
            assert rows[q] == (dims.final_position,)
            assert rows[k] == tuple(range(dims.final_position + 1))
            assert rows[v] == tuple(range(dims.final_position + 1))
        if node.provenance_identity.endswith("block11.mlp.pre"):
            assert rows[node.semantic_id] == (dims.final_position,)


def test_gpt2_program_zero_control_joint_equals_bypass(tmp_path):
    reader, _, program = _fixture(tmp_path)
    replay = execute_tensor_program_numpy(program, reader, 0.0)
    assert replay["PAT_J"].tobytes() == replay["PAT_B"].tobytes()
    assert replay["TAR_J"].tobytes() == replay["TAR_B"].tobytes()
    assert float(replay["output"]) == 0.0


def test_gpt2_dimensions_reject_nonfinal_causal_control():
    with pytest.raises(ValueError, match="final unpadded causal token"):
        GPT2TailDimensions(4, 4, 8, 2, 2, (1, 6), 2, 3)


def test_gpt2_program_nonzero_control_is_deterministic(tmp_path):
    reader, _, program = _fixture(tmp_path)
    first = execute_tensor_program_numpy(program, reader, 0.125)
    second = execute_tensor_program_numpy(program, reader, 0.125)
    assert np.isfinite(first["output"])
    for key in first:
        assert np.asarray(first[key]).tobytes() == np.asarray(second[key]).tobytes()


def test_gpt2_program_torch_replay_matches_numpy_fixture(tmp_path):
    pytest.importorskip("torch")
    reader, _, program = _fixture(tmp_path)
    numpy_replay = execute_tensor_program_numpy(program, reader, 0.03125)
    torch_replay = execute_tensor_program_torch(program, reader, 0.03125, "cpu")
    for key in numpy_replay:
        assert float(torch_replay[key].item()) == pytest.approx(
            float(numpy_replay[key]), abs=2e-5
        )


def test_gpt2_program_rejects_tensor_store_outside_closure(tmp_path):
    reader, dims, program = _fixture(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    write_tensor_store(other, "fixture", [("x", np.asarray([1], dtype="<i4"))])
    with pytest.raises(ValueError, match="outside the store closure"):
        TensorStoreReader(other / "fixture.json").validate_ref(program.nodes[0].tensor_inputs[0])


@pytest.mark.parametrize("precision", [384, 512])
def test_complete_four_branch_mpfr_program_is_bit_identical_compiled(tmp_path, precision):
    library = os.environ.get("GREEN_V400_MPFR_BACKEND")
    if not library:
        pytest.skip("compiled MPFR backend is not configured")
    reader, _, program = _fixture(tmp_path)
    domain = Interval.from_bounds(-2.0**-14, 2.0**-14, precision)
    reference = execute_tensor_program_mpfr(
        program, reader, domain, return_dispatch_trace=True
    )
    compiled = execute_tensor_program_mpfr(
        program, reader, domain, CompiledMPFRBackend(Path(library))
    )
    resident_root = tmp_path / f"resident_{precision}"
    resident_root.mkdir()
    resident_plan = build_resident_plan(
        resident_root, "tiny", program, reader
    )
    resident_plan, resident_arrays = load_resident_plan_arrays(
        resident_root / "tiny.json", program, reader
    )
    reader.read_semantic = lambda _semantic_hash: (_ for _ in ()).throw(
        AssertionError("resident execution fell back to the tensor store")
    )
    resident_compiled = execute_tensor_program_mpfr(
        program, reader, domain, CompiledMPFRBackend(Path(library)),
        resident_plan=resident_plan, resident_arrays=resident_arrays,
        sparse_axis0_execution=True, return_runtime_metrics=True,
        return_dispatch_trace=True,
    )
    assert set(reference) == {"PAT_J", "PAT_B", "TAR_J", "TAR_B", "output", "dispatch_trace"}
    assert set(compiled) == {"PAT_J", "PAT_B", "TAR_J", "TAR_B", "output"}
    assert len(reference["dispatch_trace"]["events"]) == 81
    assert (reference["dispatch_trace"]["program_dispatch_signature_sha256"]
            == program.resource_formula["dispatcher_signature_sha256"])
    for name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "output"):
        assert jet_exact_payload(compiled[name]) == jet_exact_payload(reference[name])
        assert jet_exact_payload(resident_compiled[name]) == jet_exact_payload(reference[name])
    metrics = resident_compiled["runtime_metrics"]
    assert metrics["materialized_axis0_row_count"] < metrics["dense_axis0_row_slot_count"]
    assert metrics["static_row_cache_hits_by_kernel"]["layer_norm.v1"] > 0
    assert metrics["static_row_cache_hits_by_kernel"]["pairwise_affine.v1"] > 0
    assert metrics["resident_packed_tensor_binding_reads"] > 0
    assert metrics["tensor_store_fallback_reads"] == 0
    assert metrics["resident_fused_contrast_nodes"] == 4
    assert metrics["resident_gelu_batch_rows"] > 0
    assert resident_compiled["dispatch_trace"] == reference["dispatch_trace"]
