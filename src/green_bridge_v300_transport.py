"""Exact and finite held-out transport evaluators for GREEN v3.0.0."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from green_bridge_spec import AD_ROUTE_GAMMA, SELECTED_GATES
from green_bridge_numerics import norm_up, round_up


def _torch():
    import torch
    return torch


def _batch_addmm(bias, weight, value):
    from transformer_lens.utilities.addmm import batch_addmm
    return batch_addmm(bias, weight, value)


def _route_array(left, right) -> dict:
    left = np.asarray(left, dtype=np.float64); right = np.asarray(right, dtype=np.float64)
    difference = norm_up(left - right)
    scale = max(1.0, norm_up(left), norm_up(right))
    guard = round_up(2.0 * AD_ROUTE_GAMMA * scale)
    radius = round_up(difference / 2.0 + AD_ROUTE_GAMMA * scale)
    return {"difference": difference, "radius": radius, "passed": bool(difference <= guard),
            "midpoint": 0.5 * (left + right)}


def _ad_anchor(ad_tail, anchor):
    torch = _torch()
    residual = anchor.resid_mid.detach().double()
    with torch.no_grad():
        normalized = ad_tail.block10.ln2(residual)
        pre = _batch_addmm(ad_tail.block10.mlp.b_in, ad_tail.block10.mlp.W_in, normalized)
        post = ad_tail.block10.mlp.act_fn(pre)
    return residual, pre, post


def _tail_logits(ad_tail, resid_post, suffix, position):
    from transformer_lens.utilities import apply_softcap
    resid = ad_tail.block11(resid_post)
    normalized = ad_tail.ln_final(resid)
    logits = ad_tail.unembed(normalized)
    logits = apply_softcap(logits, ad_tail.cfg.output_logits_soft_cap)
    return logits[0, position].index_select(0, suffix)


def build_physical_path_control_v300(ad_tail, anchor, directions, suffix_ids, gate_index: int):
    """Return path/control maps over a frozen physical direction matrix."""
    torch = _torch()
    residual, anchored_pre, anchored_post = _ad_anchor(ad_tail, anchor)
    directions = torch.as_tensor(directions, dtype=torch.float64, device=residual.device)
    if directions.ndim != 2 or directions.shape[0] != 768:
        raise ValueError("directions must have shape [768,k]")
    suffix = torch.as_tensor(suffix_ids, dtype=torch.long, device=residual.device)
    position = int(anchor.final_positions[0].item())
    sequence_mask = torch.nn.functional.one_hot(
        torch.tensor(position, device=residual.device), num_classes=residual.shape[1]
    ).to(torch.float64)
    gate_mask = torch.nn.functional.one_hot(
        torch.tensor(int(gate_index), device=residual.device), num_classes=ad_tail.cfg.d_mlp
    ).to(torch.float64)

    def evaluate(coefficients, z, mode: str):
        physical = directions @ coefficients
        resid_mid = residual + sequence_mask[None, :, None] * physical[None, None, :]
        normalized = ad_tail.block10.ln2(resid_mid)
        pre = _batch_addmm(ad_tail.block10.mlp.b_in, ad_tail.block10.mlp.W_in, normalized)
        selected_pre = (pre[0, position, gate_index] if mode == "path"
                        else anchored_pre[0, position, gate_index]) + z
        selected_post = ad_tail.block10.mlp.act_fn(selected_pre)
        anchored_value = anchored_post[0, position, gate_index]
        post = anchored_post + sequence_mask[None, :, None] * gate_mask[None, None, :] * (
            selected_post - anchored_value
        )
        mlp_out = _batch_addmm(ad_tail.block10.mlp.b_out, ad_tail.block10.mlp.W_out, post)
        return _tail_logits(ad_tail, resid_mid + mlp_out, suffix, position)

    return (lambda x, z: evaluate(x, z, "path")), (lambda x, z: evaluate(x, z, "control"))


def direct_path_control_ad_v300(ad_tail, anchor, directions, suffix_ids, gate_index: int) -> dict:
    torch = _torch()
    path, control = build_physical_path_control_v300(ad_tail, anchor, directions, suffix_ids, gate_index)
    k = int(np.asarray(directions).shape[1])
    device = anchor.resid_mid.device
    x = torch.zeros(k, dtype=torch.float64, device=device)
    z = torch.zeros((), dtype=torch.float64, device=device)
    forward_path = torch.func.jacfwd(path, argnums=0)(x, z)
    reverse_path = torch.func.jacrev(path, argnums=0)(x, z)
    forward_control = torch.func.jacfwd(control, argnums=0)(x, z)
    reverse_control = torch.func.jacrev(control, argnums=0)(x, z)
    forward_g = torch.func.jacfwd(path, argnums=1)(x, z)
    reverse_g = torch.func.jacrev(path, argnums=1)(x, z)
    jp = _route_array(forward_path.detach().cpu().numpy(), reverse_path.detach().cpu().numpy())
    jc = _route_array(forward_control.detach().cpu().numpy(), reverse_control.detach().cpu().numpy())
    gate_response = _route_array(forward_g.detach().cpu().numpy(), reverse_g.detach().cpu().numpy())
    direct = np.asarray(jp["midpoint"] - jc["midpoint"], dtype=np.float64).T
    return {"direct": direct, "J_path": jp, "J_control": jc, "G": gate_response,
            "route_passed": bool(jp["passed"] and jc["passed"] and gate_response["passed"])}


def response_operator_v300(gate_response, gradient) -> np.ndarray:
    return np.outer(np.asarray(gate_response, dtype=np.float64),
                    np.asarray(gradient, dtype=np.float64))


def heldout_transport_prediction_v300(gate_response, gradient, directions) -> np.ndarray:
    g = np.asarray(gradient, dtype=np.float64)
    G = np.asarray(gate_response, dtype=np.float64)
    U = np.asarray(directions, dtype=np.float64)
    return np.outer(U.T @ g, G)


def direct_path_control_finite_v300(path_evaluate, control_evaluate, *, radius: float) -> dict:
    """Central finite direct derivatives for a coefficient panel.

    This generic helper is used with one direction coefficient at a time by
    the v3 scientific runner.  It cannot consume held-out targets during
    response identification.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    torch = _torch()
    device = getattr(path_evaluate, "_green_device", None)
    x0 = torch.zeros(1, dtype=torch.float64, device=device)
    z = torch.zeros((), dtype=torch.float64, device=device)
    plus = torch.tensor([radius], dtype=torch.float64, device=device)
    minus = -plus
    return {
        "path": ((path_evaluate(plus, z) - path_evaluate(minus, z)) / (2 * radius)).detach(),
        "control": ((control_evaluate(plus, z) - control_evaluate(minus, z)) / (2 * radius)).detach(),
    }


def _build_joint_curve(ad_tail, anchor, suffix_ids, direction):
    torch = _torch()
    residual, anchored_pre, anchored_post = _ad_anchor(ad_tail, anchor)
    suffix = torch.as_tensor(suffix_ids, dtype=torch.long, device=residual.device)
    direction = torch.as_tensor(direction, dtype=torch.float64, device=residual.device)
    position = int(anchor.final_positions[0].item())
    sequence_mask = torch.nn.functional.one_hot(
        torch.tensor(position, device=residual.device), num_classes=residual.shape[1]
    ).to(torch.float64)
    gate_ids = torch.tensor(SELECTED_GATES, dtype=torch.long, device=residual.device)

    def curve(t):
        physical = t * direction
        resid_mid = residual + sequence_mask[None, :, None] * physical[None, None, :]
        normalized = ad_tail.block10.ln2(resid_mid)
        pre = _batch_addmm(ad_tail.block10.mlp.b_in, ad_tail.block10.mlp.W_in, normalized)
        live_post = ad_tail.block10.mlp.act_fn(pre)
        post = anchored_post.clone()
        post[0, position, gate_ids] = live_post[0, position, gate_ids]
        mlp_out = _batch_addmm(ad_tail.block10.mlp.b_out, ad_tail.block10.mlp.W_out, post)
        resid_post = resid_mid + mlp_out - sequence_mask[None, :, None] * physical[None, None, :]
        return _tail_logits(ad_tail, resid_post, suffix, position)
    return curve


def joint_target_ad_v300(ad_tail, anchor, suffix_ids, direction, contrast) -> dict:
    torch = _torch()
    curve = _build_joint_curve(ad_tail, anchor, suffix_ids, direction)
    contrast = torch.as_tensor(contrast, dtype=torch.float64, device=anchor.resid_mid.device)
    scalar = lambda t: torch.dot(curve(t), contrast)
    zero = torch.zeros((), dtype=torch.float64, device=anchor.resid_mid.device)
    forward = torch.func.jacfwd(scalar)(zero)
    reverse = torch.func.jacrev(scalar)(zero)
    route = _route_array(np.asarray(float(forward.item())), np.asarray(float(reverse.item())))
    return {"midpoint": float(np.asarray(route["midpoint"])), "difference": route["difference"],
            "radius": route["radius"], "passed": route["passed"]}


def joint_operator_prediction_v300(gate_responses, gradients, direction, contrast) -> float:
    direction = np.asarray(direction, dtype=np.float64)
    contrast = np.asarray(contrast, dtype=np.float64)
    return float(math.fsum(float(contrast @ np.asarray(G)) * float(np.asarray(g) @ direction)
                           for G, g in zip(gate_responses, gradients)))
