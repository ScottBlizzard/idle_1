from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_branch_semantics import (
    BRANCH_ORDER, binding_branch_records, binding_control_ast,
    compose_four_branch_graph,
)
from green_bridge_v400_interval import Interval
from green_bridge_v400_relational_graph import GraphNode, RelationalGraph


P = 256
KAPPA = 0.7978845608028654
LAMBDA = 0.044715


def _condition_graphs(condition: str, base: float, direction: float,
                      gate_scale: float, tail_scale: float):
    prefix = condition.lower()
    shared = {
        f"{prefix}_x": GraphNode(
            f"{prefix}_x", "affine_control",
            params={"base": base, "direction": direction},
            provenance=f"{condition}.same_residual_control", depends_on_t=True,
        ),
        f"{prefix}_x0": GraphNode(
            f"{prefix}_x0", "constant", params={"value": base},
            provenance=f"{condition}.anchor",
        ),
        f"{prefix}_gate": GraphNode(
            f"{prefix}_gate", "gelu_new", (f"{prefix}_x",),
            {"kappa": KAPPA, "lambda": LAMBDA},
            provenance=f"{condition}.live_selected_gate", depends_on_t=True,
        ),
        f"{prefix}_gate0": GraphNode(
            f"{prefix}_gate0", "gelu_new", (f"{prefix}_x0",),
            {"kappa": KAPPA, "lambda": LAMBDA},
            provenance=f"{condition}.frozen_selected_gate",
        ),
        f"{prefix}_gate_delta": GraphNode(
            f"{prefix}_gate_delta", "sub",
            (f"{prefix}_gate", f"{prefix}_gate0"),
            provenance=f"{condition}.selected_gate_delta", depends_on_t=True,
        ),
        f"{prefix}_gate_weight": GraphNode(
            f"{prefix}_gate_weight", "constant", params={"value": gate_scale},
            provenance=f"{condition}.W_out",
        ),
        f"{prefix}_path": GraphNode(
            f"{prefix}_path", "mul",
            (f"{prefix}_gate_weight", f"{prefix}_gate_delta"),
            provenance=f"{condition}.path_contribution", depends_on_t=True,
        ),
        f"{prefix}_joint_input": GraphNode(
            f"{prefix}_joint_input", "add", (f"{prefix}_x", f"{prefix}_path"),
            provenance=f"{condition}.J_keeps_residual_bypass", depends_on_t=True,
        ),
        f"{prefix}_joint_tail": GraphNode(
            f"{prefix}_joint_tail", "tanh", (f"{prefix}_joint_input",),
            provenance=f"{condition}.J_tail", depends_on_t=True,
        ),
        f"{prefix}_bypass_tail": GraphNode(
            f"{prefix}_bypass_tail", "tanh", (f"{prefix}_x",),
            provenance=f"{condition}.B_tail", depends_on_t=True,
        ),
    }
    # Tail scale is a final exact contrast coefficient and is shared by J/B.
    shared[f"{prefix}_tail_scale"] = GraphNode(
        f"{prefix}_tail_scale", "constant", params={"value": tail_scale},
        provenance=f"{condition}.contrast",
    )
    shared[f"{prefix}_j"] = GraphNode(
        f"{prefix}_j", "mul", (f"{prefix}_tail_scale", f"{prefix}_joint_tail"),
        provenance=f"{condition}.J_scalar", depends_on_t=True,
    )
    shared[f"{prefix}_b"] = GraphNode(
        f"{prefix}_b", "mul", (f"{prefix}_tail_scale", f"{prefix}_bypass_tail"),
        provenance=f"{condition}.B_scalar", depends_on_t=True,
    )
    return (RelationalGraph(shared, f"{prefix}_j", P),
            RelationalGraph(shared, f"{prefix}_b", P))


def _gelu_derivative(x: float) -> float:
    u = KAPPA * (x + LAMBDA*x**3)
    return 0.5*(1 + math.tanh(u)) + 0.5*x*(1-math.tanh(u)**2)*KAPPA*(1+3*LAMBDA*x*x)


def test_binding_branch_records_are_live_vs_frozen_with_bypass_kept():
    records = binding_branch_records()
    assert tuple(f"{row.condition}_{row.branch}" for row in records) == BRANCH_ORDER
    assert all(row.residual_bypass_kept for row in records)
    assert [row.selected_gate_post_policy for row in records] == [
        "live", "frozen_to_anchor", "live", "frozen_to_anchor"
    ]


def test_control_ast_forbids_internal_residual_subtraction_as_official_curve():
    ast = binding_control_ast()
    assert ast["contrast_order"] == list(BRANCH_ORDER)
    assert ast["contrast_weights"] == [1, -1, -1, 1]
    assert ast["internal_residual_subtraction_is_official_curve"] is False


def test_four_branch_graph_derivative_equals_compositional_theta():
    pat_j, pat_b = _condition_graphs("PAT", 0.3, 0.7, 0.4, 1.2)
    tar_j, tar_b = _condition_graphs("TAR", -0.2, 0.7, -0.25, 0.9)
    graph = compose_four_branch_graph({
        "PAT_J": pat_j, "PAT_B": pat_b, "TAR_J": tar_j, "TAR_B": tar_b,
    })
    result = graph.evaluate(Interval.point(0.0, P))
    pat_theta = 1.2*(1-math.tanh(0.3)**2)*0.4*_gelu_derivative(0.3)*0.7
    tar_theta = 0.9*(1-math.tanh(-0.2)**2)*(-0.25)*_gelu_derivative(-0.2)*0.7
    theta = pat_theta - tar_theta
    assert abs(float(result.first.midpoint()) - theta) < 1e-14


def test_internal_subtraction_curve_matches_first_order_but_not_curvature():
    joint, bypass = _condition_graphs("PAT", 0.3, 0.7, 0.4, 1.2)
    official = compose_four_branch_graph({
        "PAT_J": joint, "PAT_B": bypass,
        "TAR_J": bypass, "TAR_B": bypass,
    })
    nodes = dict(joint.nodes)
    nodes["pat_internal_input"] = GraphNode(
        "pat_internal_input", "add", ("pat_x0", "pat_path"),
        provenance="legacy_direct_residual_removed_before_tail", depends_on_t=True,
    )
    nodes["pat_internal_subtracted"] = GraphNode(
        "pat_internal_subtracted", "tanh", ("pat_internal_input",),
        provenance="legacy_internal_residual_subtraction", depends_on_t=True,
    )
    nodes["pat_internal_scaled"] = GraphNode(
        "pat_internal_scaled", "mul", ("pat_tail_scale", "pat_internal_subtracted"),
        provenance="legacy_target_scalar", depends_on_t=True,
    )
    internal = RelationalGraph(nodes, "pat_internal_scaled", P)
    at_zero_official = official.evaluate(Interval.point(0.0, P))
    at_zero_internal = internal.evaluate(Interval.point(0.0, P))
    assert abs(float(at_zero_official.first.midpoint()) - float(at_zero_internal.first.midpoint())) < 1e-14
    assert abs(float(at_zero_official.second.midpoint()) - float(at_zero_internal.second.midpoint())) > 1e-5
