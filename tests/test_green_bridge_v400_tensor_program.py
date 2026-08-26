from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_tensor_program import (
    KERNEL_REGISTRY_HASH, PROGRAM_SCHEMA_VERSION, TensorNode, TensorProgram,
    TensorSpec,
)
from green_bridge_v400_tensor_store import TensorRef, TENSOR_REF_SCHEMA_VERSION


H = "a" * 64


def _ref():
    return TensorRef(TENSOR_REF_SCHEMA_VERSION, "b"*64, "<f4", (2,), "C", 8)


def _node(kernel, parents=(), attrs=None, provenance="fixture"):
    return TensorNode.build(kernel, tuple(parents), (_ref(),), attrs or {},
                            TensorSpec("<f4", (2,)), provenance, "c"*64)


def test_tensor_program_semantic_hash_and_four_branch_roots():
    roots = [_node("affine_scatter.v1", provenance=name)
             for name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B")]
    output = _node("branch_linear_combination.v1",
                   tuple(node.semantic_id for node in roots),
                   {"weights": [1, -1, -1, 1],
                    "order": ["PAT_J", "PAT_B", "TAR_J", "TAR_B"]})
    program = TensorProgram.build(
        H, tuple(roots + [output]),
        {name: node.semantic_id for name, node in zip(
            ("PAT_J", "PAT_B", "TAR_J", "TAR_B"), roots)},
        output.semantic_id,
        {"formula_version": "fixture-v1", "coefficient_terms": 8},
    )
    assert len(program.semantic_hash()) == 64
    assert program.to_dict()["nodes"][-1]["exact_attrs"]["weights"] == [1, -1, -1, 1]
    assert TensorProgram.from_dict(program.to_dict()) == program


def test_tensor_node_rejects_ordinary_json_float_attrs():
    with pytest.raises(ValueError, match="JSON float"):
        _node("layer_norm.v1", attrs={"epsilon": 1e-5})


def test_tensor_node_semantic_id_detects_attr_mutation():
    node = _node("static_view.v1", attrs={"indices": [0, 1]})
    with pytest.raises(ValueError, match="semantic id"):
        TensorNode(node.schema_version, node.semantic_id, node.kernel_id,
                   node.parent_semantic_ids, node.tensor_inputs,
                   {"indices": [1, 0]}, node.output_spec,
                   node.provenance_identity, node.dependency_mask_hash)


def test_tensor_program_rejects_parent_after_child():
    parent = _node("affine_scatter.v1")
    child = _node("static_view.v1", (parent.semantic_id,))
    with pytest.raises(ValueError, match="topological"):
        TensorProgram.build(H, (child, parent), {name: parent.semantic_id for name in
                            ("PAT_J", "PAT_B", "TAR_J", "TAR_B")},
                            child.semantic_id, {"terms": 1})


def test_tensor_program_round_trip_rejects_scalarization_mutation():
    roots = [_node("affine_scatter.v1", provenance=name)
             for name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B")]
    output = _node("branch_linear_combination.v1",
                   tuple(node.semantic_id for node in roots),
                   {"weights": [1, -1, -1, 1]})
    program = TensorProgram.build(
        H, tuple(roots + [output]),
        {name: node.semantic_id for name, node in zip(
            ("PAT_J", "PAT_B", "TAR_J", "TAR_B"), roots)},
        output.semantic_id, {"terms": 8},
    )
    payload = program.to_dict()
    payload["scalarization_merkle_root"] = "0" * 64
    with pytest.raises(ValueError, match="scalarization closure"):
        TensorProgram.from_dict(payload)
