"""Exact shared scalar expression DAG for the GREEN v4 Joint Witness."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import (
    Jet2, add_jet, affine_control_jet, compose_jet, constant_jet, mul_jet,
    reciprocal_jet, sub_jet,
)
from green_bridge_v400_schemas import JointWitnessRowSpec, canonical_json
from green_bridge_v400_transformer_ops import (
    erf_primitive, exp_primitive, gelu_erf_jet, gelu_new_jet,
    sigmoid_jet, tanh_primitive,
)


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    op: str
    parents: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    depends_on_t: bool = False

    def semantic_payload(self, parent_hashes: tuple[str, ...]) -> dict:
        return {
            "op": self.op, "parent_hashes": parent_hashes,
            "params": self.params, "provenance": self.provenance,
            "depends_on_t": self.depends_on_t,
        }


@dataclass(frozen=True)
class DependencyAudit:
    complete: bool
    dependent_nodes: tuple[str, ...]
    violations: tuple[str, ...]


@dataclass(frozen=True)
class ReductionProof:
    original_hash: str
    reduced_hash: str
    cancellations: tuple[dict, ...]
    original_node_count: int
    reduced_node_count: int


@dataclass
class RelationalGraph:
    nodes: dict[str, GraphNode]
    output_id: str
    precision_bits: int = 256

    def topological_order(self) -> list[str]:
        temporary, permanent, order = set(), set(), []
        def visit(node_id: str):
            if node_id in permanent:
                return
            if node_id in temporary:
                raise ValueError("graph contains a cycle")
            if node_id not in self.nodes:
                raise KeyError(f"missing graph node {node_id}")
            temporary.add(node_id)
            for parent in self.nodes[node_id].parents:
                visit(parent)
            temporary.remove(node_id)
            permanent.add(node_id)
            order.append(node_id)
        visit(self.output_id)
        return order

    def semantic_hashes(self) -> dict[str, str]:
        hashes = {}
        for node_id in self.topological_order():
            node = self.nodes[node_id]
            payload = node.semantic_payload(tuple(hashes[parent] for parent in node.parents))
            hashes[node_id] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        return hashes

    def semantic_hash(self) -> str:
        return self.semantic_hashes()[self.output_id]

    def evaluate(self, domain: Interval) -> Jet2:
        if domain.precision_bits != self.precision_bits:
            domain = Interval.from_bounds(domain.lower, domain.upper, self.precision_bits)
        values: dict[str, Jet2] = {}
        for node_id in self.topological_order():
            node = self.nodes[node_id]
            parents = [values[parent] for parent in node.parents]
            if node.op == "constant":
                values[node_id] = constant_jet(Interval.point(node.params["value"], self.precision_bits))
            elif node.op == "affine_control":
                values[node_id] = affine_control_jet(
                    Interval.point(node.params["base"], self.precision_bits),
                    Interval.point(node.params["direction"], self.precision_bits), domain,
                )
            elif node.op == "add":
                values[node_id] = add_jet(parents[0], parents[1])
            elif node.op == "sub":
                values[node_id] = sub_jet(parents[0], parents[1])
            elif node.op == "mul":
                values[node_id] = mul_jet(parents[0], parents[1])
            elif node.op == "reciprocal":
                values[node_id] = reciprocal_jet(parents[0])
            elif node.op == "exp":
                values[node_id] = compose_jet(parents[0], exp_primitive())
            elif node.op == "tanh":
                values[node_id] = compose_jet(parents[0], tanh_primitive())
            elif node.op == "erf":
                values[node_id] = compose_jet(parents[0], erf_primitive())
            elif node.op == "sigmoid":
                values[node_id] = sigmoid_jet(parents[0])
            elif node.op == "gelu_new":
                values[node_id] = gelu_new_jet(
                    parents[0], kappa=node.params["kappa"], lam=node.params["lambda"]
                )
            elif node.op == "gelu_erf":
                values[node_id] = gelu_erf_jet(parents[0])
            elif node.op == "identity":
                values[node_id] = parents[0]
            else:
                raise RuntimeError(f"UNSUPPORTED_PRIMITIVE:{node.op}")
        return values[self.output_id]


def extract_joint_witness_graph(row_spec: JointWitnessRowSpec, model=None) -> RelationalGraph:
    if model is not None and hasattr(model, "export_green_v400_graph"):
        payload = model.export_green_v400_graph(row_spec)
    else:
        payload = row_spec.graph_payload
    nodes = {}
    for record in payload.get("nodes", []):
        node = GraphNode(
            node_id=record["node_id"], op=record["op"],
            parents=tuple(record.get("parents", ())),
            params=dict(record.get("params", {})),
            provenance=record.get("provenance", ""),
            depends_on_t=bool(record.get("depends_on_t", False)),
        )
        if node.node_id in nodes:
            raise ValueError("duplicate graph node id")
        nodes[node.node_id] = node
    graph = RelationalGraph(nodes, payload["output_id"], int(payload.get("precision_bits", 256)))
    graph.topological_order()
    return graph


def audit_dependency_completeness(graph: RelationalGraph) -> DependencyAudit:
    dependent, violations = set(), []
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        inferred = node.op == "affine_control" and node.params.get("direction", 0) != 0
        inferred = inferred or any(parent in dependent for parent in node.parents)
        if inferred:
            dependent.add(node_id)
        if bool(node.depends_on_t) != inferred:
            violations.append(node_id)
    return DependencyAudit(not violations, tuple(sorted(dependent)), tuple(violations))


def reduce_exact_shared_graph(graph: RelationalGraph) -> tuple[RelationalGraph, ReductionProof]:
    order = graph.topological_order()
    old_hash = graph.semantic_hash()
    replacements: dict[str, str] = {}
    new_nodes: dict[str, GraphNode] = {}
    new_hashes: dict[str, str] = {}
    semantic_owner: dict[str, str] = {}
    cancellations = []
    for node_id in order:
        node = graph.nodes[node_id]
        parents = tuple(replacements.get(parent, parent) for parent in node.parents)
        if node.op == "sub" and len(parents) == 2 and parents[0] == parents[1]:
            zero_id = f"zero_{node_id}"
            new_nodes[zero_id] = GraphNode(zero_id, "constant", params={"value": 0},
                                           provenance="exact_x_minus_x", depends_on_t=False)
            new_hashes[zero_id] = hashlib.sha256(canonical_json(
                new_nodes[zero_id].semantic_payload(())
            ).encode()).hexdigest()
            replacements[node_id] = zero_id
            cancellations.append({"node_id": node_id, "rule": "x-x", "replacement": zero_id})
            continue
        candidate = GraphNode(node_id, node.op, parents, node.params,
                              node.provenance, node.depends_on_t)
        parent_hashes = tuple(new_hashes[parent] for parent in parents)
        local_key = hashlib.sha256(canonical_json(candidate.semantic_payload(parent_hashes)).encode()).hexdigest()
        if local_key in semantic_owner:
            replacements[node_id] = semantic_owner[local_key]
            cancellations.append({"node_id": node_id, "rule": "hash_cons",
                                  "replacement": semantic_owner[local_key]})
        else:
            new_nodes[node_id] = candidate
            replacements[node_id] = node_id
            semantic_owner[local_key] = node_id
            new_hashes[node_id] = local_key
    output = replacements.get(graph.output_id, graph.output_id)
    reduced = RelationalGraph(new_nodes, output, graph.precision_bits)
    reduced.topological_order()
    proof = ReductionProof(old_hash, reduced.semantic_hash(), tuple(cancellations),
                           len(graph.nodes), len(reduced.nodes))
    return reduced, proof
