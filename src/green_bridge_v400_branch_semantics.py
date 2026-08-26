"""Binding PAT/TAR × joint/bypass semantics for the GREEN v4 witness."""
from __future__ import annotations

from dataclasses import dataclass

from green_bridge_v400_relational_graph import GraphNode, RelationalGraph


BRANCH_ORDER = ("PAT_J", "PAT_B", "TAR_J", "TAR_B")
BRANCH_WEIGHTS = (1, -1, -1, 1)


@dataclass(frozen=True)
class BranchSemantics:
    condition: str
    branch: str
    anchor: str
    residual_control: str
    selected_gate_post_policy: str
    residual_bypass_kept: bool

    def __post_init__(self):
        if self.condition not in {"PAT", "TAR"} or self.branch not in {"J", "B"}:
            raise ValueError("invalid Joint Witness branch identity")
        expected = "live" if self.branch == "J" else "frozen_to_anchor"
        if self.selected_gate_post_policy != expected or not self.residual_bypass_kept:
            raise ValueError("branch does not implement binding matched-bypass semantics")


def binding_branch_records() -> tuple[BranchSemantics, ...]:
    return (
        BranchSemantics("PAT", "J", "corrupt_with_TAR_block8_mlp_out_patch",
                        "same_t_times_physical_direction", "live", True),
        BranchSemantics("PAT", "B", "corrupt_with_TAR_block8_mlp_out_patch",
                        "same_t_times_physical_direction", "frozen_to_anchor", True),
        BranchSemantics("TAR", "J", "clean",
                        "same_t_times_physical_direction", "live", True),
        BranchSemantics("TAR", "B", "clean",
                        "same_t_times_physical_direction", "frozen_to_anchor", True),
    )


def binding_control_ast() -> dict:
    return {
        "operation": "affine_control",
        "form": "A0_plus_t_times_D",
        "dynamic_hook_selection": False,
        "conditions": {
            "PAT": "corrupt_anchor_with_TAR_block8_mlp_out_patch",
            "TAR": "clean_anchor",
        },
        "branches": {
            "J": {"selected_gate_posts": "live", "residual_bypass_kept": True},
            "B": {"selected_gate_posts": "frozen_to_anchor", "residual_bypass_kept": True},
        },
        "shared_control": "same_physical_direction_and_t_in_all_four_branches",
        "contrast_order": list(BRANCH_ORDER),
        "contrast_weights": list(BRANCH_WEIGHTS),
        "internal_residual_subtraction_is_official_curve": False,
    }


def compose_four_branch_graph(branches: dict[str, RelationalGraph],
                              precision_bits: int | None = None) -> RelationalGraph:
    """Merge exact branch DAGs and append the binding signed scalar contrast."""
    if set(branches) != set(BRANCH_ORDER):
        raise ValueError("four-branch graph keys do not match binding order")
    precisions = {graph.precision_bits for graph in branches.values()}
    if len(precisions) != 1:
        raise ValueError("branch graph precision mismatch")
    precision = precisions.pop() if precision_bits is None else int(precision_bits)
    nodes: dict[str, GraphNode] = {}
    outputs = {}
    for name in BRANCH_ORDER:
        graph = branches[name]
        graph.topological_order()
        for node_id, node in graph.nodes.items():
            if node_id in nodes and nodes[node_id] != node:
                raise ValueError(f"conflicting shared branch node {node_id}")
            nodes[node_id] = node
        outputs[name] = graph.output_id
    nodes["joint_witness_pat_effect"] = GraphNode(
        "joint_witness_pat_effect", "sub", (outputs["PAT_J"], outputs["PAT_B"]),
        provenance="binding_PAT_J_minus_PAT_B", depends_on_t=True,
    )
    nodes["joint_witness_tar_effect"] = GraphNode(
        "joint_witness_tar_effect", "sub", (outputs["TAR_J"], outputs["TAR_B"]),
        provenance="binding_TAR_J_minus_TAR_B", depends_on_t=True,
    )
    nodes["joint_witness_psi"] = GraphNode(
        "joint_witness_psi", "sub",
        ("joint_witness_pat_effect", "joint_witness_tar_effect"),
        provenance="binding_PAT_minus_TAR", depends_on_t=True,
    )
    result = RelationalGraph(nodes, "joint_witness_psi", precision)
    result.topological_order()
    return result

