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
    affine_map_jets, attention_head_jets, contrast_jet, erf_primitive,
    exp_primitive, gelu_erf_jet, gelu_new_jet, inv_sqrt_primitive,
    layernorm_jets, sigmoid_jet, softmax_jets, sqrt_primitive, tanh_primitive,
)


EXECUTABLE_OPERATIONS = frozenset({
    "constant", "affine_control", "add", "sub", "mul", "reciprocal",
    "exp", "sqrt", "inv_sqrt", "tanh", "erf", "sigmoid", "gelu_new",
    "gelu_erf", "affine_component", "einsum", "layernorm_component",
    "layernorm", "softmax_component", "softmax", "attention_component",
    "attention", "contrast", "residual_add", "reshape", "transpose",
    "slice", "gather_static", "identity",
})


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

    def to_payload(self) -> dict:
        """Canonical replay payload; unlike a manifest, this is executable."""
        return {
            "schema_version": "green-v400-relational-graph-v1",
            "precision_bits": self.precision_bits,
            "output_id": self.output_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "op": node.op,
                    "parents": list(node.parents),
                    "params": node.params,
                    "provenance": node.provenance,
                    "depends_on_t": node.depends_on_t,
                }
                for node_id in self.topological_order()
                for node in (self.nodes[node_id],)
            ],
        }

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
            elif node.op == "sqrt":
                values[node_id] = compose_jet(parents[0], sqrt_primitive())
            elif node.op == "inv_sqrt":
                values[node_id] = compose_jet(parents[0], inv_sqrt_primitive())
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
            elif node.op in {"affine_component", "einsum"}:
                weights = node.params["weights"]
                bias = node.params.get("bias", 0)
                values[node_id] = affine_map_jets([weights], parents, [bias])[0]
            elif node.op in {"layernorm_component", "layernorm"}:
                normalized = layernorm_jets(
                    parents, epsilon=node.params["epsilon"],
                    gamma=node.params.get("gamma"), beta=node.params.get("beta"),
                )
                values[node_id] = normalized[int(node.params["index"])]
            elif node.op in {"softmax_component", "softmax"}:
                weights = softmax_jets(parents, pivot=int(node.params.get("pivot", 0)))
                values[node_id] = weights[int(node.params["index"])]
            elif node.op in {"attention_component", "attention"}:
                token_count = int(node.params["token_count"])
                head_dim = int(node.params["head_dim"])
                value_dim = int(node.params["value_dim"])
                q_size = token_count * head_dim
                k_size = token_count * head_dim
                expected = q_size + k_size + token_count * value_dim
                if len(parents) != expected:
                    raise ValueError("attention_component parent shape mismatch")
                offset = 0
                queries = [parents[offset + i*head_dim:offset + (i+1)*head_dim]
                           for i in range(token_count)]
                offset += q_size
                keys = [parents[offset + i*head_dim:offset + (i+1)*head_dim]
                        for i in range(token_count)]
                offset += k_size
                vector_values = [parents[offset + i*value_dim:offset + (i+1)*value_dim]
                                 for i in range(token_count)]
                attended = attention_head_jets(
                    queries, keys, vector_values,
                    causal=bool(node.params.get("causal", True)),
                )
                values[node_id] = attended[int(node.params["query_index"])][int(node.params["coordinate"])]
            elif node.op == "contrast":
                values[node_id] = contrast_jet(parents, node.params["weights"])
            elif node.op == "residual_add":
                values[node_id] = add_jet(parents[0], parents[1])
            elif node.op in {"identity", "reshape", "transpose", "slice", "gather_static"}:
                if len(parents) != 1:
                    raise ValueError(f"{node.op} scalar view needs exactly one parent")
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


def build_tiny_transformer_fixture_graph(precision_bits: int = 256) -> RelationalGraph:
    """A real 2-token/1-head/1-block pre-LN Transformer scalar fixture.

    It contains LN, Q/K/V projections, causal softmax attention, residuals,
    an MLP with GELU, final LN, and an unembedding contrast.  All constants are
    serialized in the returned executable DAG.
    """
    nodes: dict[str, GraphNode] = {}

    def put(node: GraphNode) -> str:
        if node.node_id in nodes:
            raise ValueError(f"duplicate tiny graph node {node.node_id}")
        nodes[node.node_id] = node
        return node.node_id

    inputs = [
        put(GraphNode("x00", "affine_control", params={"base": 1.0, "direction": 1.0},
                      provenance="token0.coordinate0", depends_on_t=True)),
        put(GraphNode("x01", "constant", params={"value": -1.0},
                      provenance="token0.coordinate1")),
        put(GraphNode("x10", "constant", params={"value": 0.5},
                      provenance="token1.coordinate0")),
        put(GraphNode("x11", "constant", params={"value": -0.5},
                      provenance="token1.coordinate1")),
    ]

    ln1: list[str] = []
    for token in range(2):
        parents = tuple(inputs[token*2:(token+1)*2])
        for coordinate in range(2):
            ln1.append(put(GraphNode(
                f"ln1_{token}_{coordinate}", "layernorm", parents,
                {"epsilon": 1e-5, "index": coordinate},
                provenance=f"block0.ln1.token{token}", depends_on_t=(token == 0),
            )))

    projections: dict[str, list[str]] = {"q": [], "k": [], "v": []}
    matrices = {
        "q": ((1.0, 0.0), (0.0, 1.0)),
        "k": ((0.5, 0.0), (0.0, 0.5)),
        "v": ((1.0, 0.0), (0.0, 1.0)),
    }
    for name, matrix in matrices.items():
        for token in range(2):
            parents = tuple(ln1[token*2:(token+1)*2])
            for coordinate in range(2):
                projections[name].append(put(GraphNode(
                    f"{name}_{token}_{coordinate}", "einsum", parents,
                    {"weights": list(matrix[coordinate]), "bias": 0.0},
                    provenance=f"block0.attn.{name}", depends_on_t=(token == 0),
                )))

    attention_parents = tuple(projections["q"] + projections["k"] + projections["v"])
    attended: list[str] = []
    for token in range(2):
        for coordinate in range(2):
            attended.append(put(GraphNode(
                f"attn_{token}_{coordinate}", "attention", attention_parents,
                {"token_count": 2, "head_dim": 2, "value_dim": 2,
                 "query_index": token, "coordinate": coordinate, "causal": True},
                provenance="block0.causal_attention", depends_on_t=True,
            )))

    resid1 = [put(GraphNode(f"resid1_{token}_{coordinate}", "residual_add",
                            (inputs[token*2 + coordinate], attended[token*2 + coordinate]),
                            provenance="block0.attn_residual", depends_on_t=True))
              for token in range(2) for coordinate in range(2)]

    ln2: list[str] = []
    for token in range(2):
        parents = tuple(resid1[token*2:(token+1)*2])
        for coordinate in range(2):
            ln2.append(put(GraphNode(
                f"ln2_{token}_{coordinate}", "layernorm", parents,
                {"epsilon": 1e-5, "index": coordinate},
                provenance=f"block0.ln2.token{token}", depends_on_t=True,
            )))
    hidden, activated = [], []
    for token in range(2):
        hidden.append(put(GraphNode(
            f"mlp_pre_{token}", "einsum", tuple(ln2[token*2:(token+1)*2]),
            {"weights": [1.0, -1.0], "bias": 0.0},
            provenance="block0.mlp.in", depends_on_t=True,
        )))
        activated.append(put(GraphNode(
            f"mlp_gelu_{token}", "gelu_new", (hidden[-1],),
            {"kappa": 0.7978845608028654, "lambda": 0.044715},
            provenance="block0.mlp.gelu", depends_on_t=True,
        )))
    projected: list[str] = []
    for token in range(2):
        for coordinate, weight in enumerate((0.25, -0.25)):
            projected.append(put(GraphNode(
                f"mlp_out_{token}_{coordinate}", "einsum", (activated[token],),
                {"weights": [weight], "bias": 0.0},
                provenance="block0.mlp.out", depends_on_t=True,
            )))
    resid2 = [put(GraphNode(f"resid2_{token}_{coordinate}", "residual_add",
                            (resid1[token*2 + coordinate], projected[token*2 + coordinate]),
                            provenance="block0.mlp_residual", depends_on_t=True))
              for token in range(2) for coordinate in range(2)]

    final = [put(GraphNode(
        f"final_ln_1_{coordinate}", "layernorm", tuple(resid2[2:4]),
        {"epsilon": 1e-5, "index": coordinate},
        provenance="final_ln.token1", depends_on_t=True,
    )) for coordinate in range(2)]
    output = put(GraphNode("unembed_contrast", "contrast", tuple(final),
                           {"weights": [1.0, -1.0]},
                           provenance="unembedding.contrast", depends_on_t=True))
    graph = RelationalGraph(nodes, output, precision_bits)
    graph.topological_order()
    if not audit_dependency_completeness(graph).complete:
        raise RuntimeError("tiny Transformer dependency annotation failure")
    return graph
