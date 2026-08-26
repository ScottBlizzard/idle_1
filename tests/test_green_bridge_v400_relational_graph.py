from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_interval import Interval
from green_bridge_v400_relational_graph import (
    GraphNode, RelationalGraph, audit_dependency_completeness,
    extract_joint_witness_graph, reduce_exact_shared_graph,
)
from green_bridge_v400_schemas import JointWitnessRowSpec


def _graph(nodes, output):
    return RelationalGraph({node.node_id: node for node in nodes}, output, 256)


def _point(graph, t=0.0):
    return graph.evaluate(Interval.point(t, graph.precision_bits))


def test_identical_branch_exact_cancellation():
    graph = _graph([
        GraphNode("x", "affine_control", params={"base": 1, "direction": 2},
                  provenance="shared", depends_on_t=True),
        GraphNode("out", "sub", ("x", "x"), depends_on_t=True),
    ], "out")
    reduced, proof = reduce_exact_shared_graph(graph)
    assert proof.cancellations[0]["rule"] == "x-x"
    assert _point(reduced).value.contains(0.0)


def test_exp_of_shared_node_minus_itself_is_exact_zero():
    graph = _graph([
        GraphNode("x", "affine_control", params={"base": 0, "direction": 1},
                  provenance="shared", depends_on_t=True),
        GraphNode("e", "exp", ("x",), depends_on_t=True),
        GraphNode("out", "sub", ("e", "e"), depends_on_t=True),
    ], "out")
    reduced, _ = reduce_exact_shared_graph(graph)
    result = reduced.evaluate(Interval.from_bounds(-1, 1, 256))
    assert result.value.lower == result.value.upper == 0


def test_numerically_equal_distinct_provenance_not_cancelled():
    graph = _graph([
        GraphNode("a", "constant", params={"value": 1}, provenance="PAT"),
        GraphNode("b", "constant", params={"value": 1}, provenance="TAR"),
        GraphNode("out", "sub", ("a", "b")),
    ], "out")
    reduced, proof = reduce_exact_shared_graph(graph)
    assert not proof.cancellations
    assert "out" in reduced.nodes


def test_pat_tar_bypass_sign_order():
    nodes = [GraphNode(name, "constant", params={"value": value}, provenance=name)
             for name, value in (("pj", 10), ("pb", 3), ("tj", 8), ("tb", 2))]
    nodes += [GraphNode("pat", "sub", ("pj", "pb")),
              GraphNode("tar", "sub", ("tj", "tb")),
              GraphNode("out", "sub", ("pat", "tar"))]
    assert _point(_graph(nodes, "out")).value.contains(1.0)


def test_affine_collection_preserves_four_branch_scalar():
    nodes = [GraphNode("x", "affine_control", params={"base": 1, "direction": 1},
                       provenance="control", depends_on_t=True),
             GraphNode("two", "constant", params={"value": 2}, provenance="weight"),
             GraphNode("twox", "mul", ("two", "x"), depends_on_t=True),
             GraphNode("out", "sub", ("twox", "x"), depends_on_t=True)]
    graph = _graph(nodes, "out")
    reduced, _ = reduce_exact_shared_graph(graph)
    for t in (-0.5, 0, 0.5):
        assert _point(graph, t).value.intersect(_point(reduced, t).value)


def test_dependency_tags_complete_through_layernorm():
    graph = _graph([
        GraphNode("x", "affine_control", params={"base": 1, "direction": 1},
                  depends_on_t=True),
        GraphNode("ln", "layernorm", ("x",), depends_on_t=True),
    ], "ln")
    assert audit_dependency_completeness(graph).complete


def test_dependency_tags_complete_through_attention():
    graph = _graph([
        GraphNode("x", "affine_control", params={"base": 1, "direction": 1},
                  depends_on_t=True),
        GraphNode("attn", "attention", ("x",), depends_on_t=True),
    ], "attn")
    assert audit_dependency_completeness(graph).complete


def test_static_mask_and_offcone_elimination():
    graph = _graph([
        GraphNode("x", "constant", params={"value": 1}),
        GraphNode("unused_masked", "constant", params={"value": 999}, provenance="masked"),
    ], "x")
    reduced, proof = reduce_exact_shared_graph(graph)
    assert "unused_masked" not in reduced.nodes
    assert proof.reduced_node_count == 1


def test_unsupported_dynamic_gather_rejected():
    graph = _graph([
        GraphNode("x", "constant", params={"value": 1}),
        GraphNode("out", "gather_dynamic", ("x",)),
    ], "out")
    with pytest.raises(RuntimeError, match="UNSUPPORTED_PRIMITIVE"):
        _point(graph)


def test_graph_hash_stable_under_deterministic_serialization():
    nodes = [GraphNode("x", "constant", params={"value": 1}, provenance="x")]
    assert _graph(nodes, "x").semantic_hash() == _graph(list(reversed(nodes)), "x").semantic_hash()


def test_reduced_and_unreduced_mpfr_singletons_agree():
    graph = _graph([
        GraphNode("x", "affine_control", params={"base": 0.5, "direction": 1},
                  depends_on_t=True),
        GraphNode("e1", "exp", ("x",), depends_on_t=True),
        GraphNode("e2", "exp", ("x",), depends_on_t=True),
        GraphNode("out", "sub", ("e1", "e2"), depends_on_t=True),
    ], "out")
    reduced, _ = reduce_exact_shared_graph(graph)
    assert _point(graph, 0.25).value.intersect(_point(reduced, 0.25).value)


def test_tiny_transformer_end_to_end_interval_contains_symbolic_values():
    payload = {
        "precision_bits": 256,
        "output_id": "out",
        "nodes": [
            {"node_id": "x", "op": "affine_control", "params": {"base": 0, "direction": 1},
             "provenance": "hook", "depends_on_t": True},
            {"node_id": "g", "op": "gelu_new", "parents": ["x"],
             "params": {"kappa": 0.7978845608028654, "lambda": 0.044715},
             "depends_on_t": True},
            {"node_id": "res", "op": "add", "parents": ["x", "g"],
             "depends_on_t": True},
            {"node_id": "out", "op": "tanh", "parents": ["res"],
             "depends_on_t": True},
        ],
    }
    row = JointWitnessRowSpec("green-v400-row-v1", "0"*64, "synthetic", "1"*64,
                              "2"*64, "3"*64, "4"*64, "5"*64,
                              ("PAT_J", "PAT_B", "TAR_J", "TAR_B"), payload)
    graph = extract_joint_witness_graph(row)
    result = graph.evaluate(Interval.from_bounds(-0.1, 0.1, 256))
    assert result.value.contains(0.0)
    assert result.first.contains(1.5)
