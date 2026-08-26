from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_interval import Interval
from green_bridge_v400_relational_graph import (
    GraphNode, RelationalGraph, audit_dependency_completeness,
    build_tiny_transformer_fixture_graph,
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
    graph = build_tiny_transformer_fixture_graph(256)
    payload = graph.to_payload()
    row = JointWitnessRowSpec("green-v400-row-v1", "0"*64, "synthetic", "1"*64,
                              "2"*64, "3"*64, "4"*64, "5"*64,
                              ("PAT_J", "PAT_B", "TAR_J", "TAR_B"), payload)
    graph = extract_joint_witness_graph(row)
    operations = {node.op for node in graph.nodes.values()}
    assert {"layernorm", "einsum", "attention", "residual_add",
            "gelu_new", "contrast"} <= operations
    assert len(graph.nodes) >= 35
    assert audit_dependency_completeness(graph).complete

    def layernorm(vector):
        mean = sum(vector) / len(vector)
        centered = [value - mean for value in vector]
        variance = sum(value * value for value in centered) / len(vector)
        scale = 1.0 / (variance + 1e-5) ** 0.5
        return [value * scale for value in centered]

    def reference(t):
        tokens = [[1.0 + t, -1.0], [0.5, -0.5]]
        normalized = [layernorm(token) for token in tokens]
        queries = normalized
        keys = [[0.5 * value for value in token] for token in normalized]
        values = normalized
        attended = []
        for query_index, query in enumerate(queries):
            allowed = range(query_index + 1)
            scores = [sum(a*b for a, b in zip(query, keys[index])) / 2**0.5
                      for index in allowed]
            pivot = scores[0]
            exponentials = [math.exp(score - pivot) for score in scores]
            total = sum(exponentials)
            weights = [value / total for value in exponentials]
            attended.append([sum(weight * values[index][coordinate]
                                 for weight, index in zip(weights, allowed))
                              for coordinate in range(2)])
        resid1 = [[tokens[i][j] + attended[i][j] for j in range(2)] for i in range(2)]
        normalized2 = [layernorm(token) for token in resid1]
        activated = []
        for token in normalized2:
            x = token[0] - token[1]
            activated.append(0.5*x*(1 + math.tanh(0.7978845608028654*(x + 0.044715*x**3))))
        resid2 = [[resid1[i][0] + 0.25*activated[i],
                   resid1[i][1] - 0.25*activated[i]] for i in range(2)]
        final = layernorm(resid2[1])
        return final[0] - final[1]

    result = graph.evaluate(Interval.point(0.0, 256))
    expected = reference(0.0)
    assert abs(float(result.value.midpoint()) - expected) < 1e-12
    step = 1e-5
    numerical_first = (reference(step) - reference(-step)) / (2 * step)
    assert abs(float(result.first.midpoint()) - numerical_first) < 1e-5

    replay = extract_joint_witness_graph(row)
    assert replay.semantic_hash() == graph.semantic_hash()
