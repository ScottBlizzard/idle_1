"""Code-isolated finite path target for the green-bridge experiment.

This module intentionally does not import predictor, matched-bypass, tail, or
baseline code.  It implements the joint ten-gate intervention directly from the
frozen model tensors and subtracts the residual bypass at block-10 resid_post.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from green_bridge_spec import SELECTED_GATES


@dataclass(frozen=True)
class TargetAnchor:
    resid_mid: Any
    pre: Any
    post: Any
    final_positions: Any
    system: str


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("path target requires the pinned torch environment") from exc
    return torch


def _batch_addmm(bias, weight, value):
    try:
        from transformer_lens.utilities.addmm import batch_addmm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("path target requires pinned TransformerLens") from exc
    return batch_addmm(bias, weight, value)


def logit_contrast(clean_suffix: int, *, device=None, dtype=None):
    torch = _torch()
    if not 5 <= int(clean_suffix) <= 94:
        raise ValueError("clean suffix must lie in [05,94]")
    value = torch.empty(100, device=device, dtype=dtype or torch.float64)
    value[: clean_suffix + 1] = -1.0 / (clean_suffix + 1)
    value[clean_suffix + 1 :] = 1.0 / (99 - clean_suffix)
    return value


def evaluate_joint_target(
    model,
    anchor: TargetAnchor,
    residual_basis,
    suffix_token_ids,
    x,
    *,
    selected_gates=SELECTED_GATES,
):
    """Evaluate the joint selected-gate curve with the residual bypass removed."""
    torch = _torch()
    batch = anchor.resid_mid.shape[0]
    if tuple(x.shape) != (batch, 4):
        raise ValueError(f"x must have shape [{batch},4]")
    gate_ids = torch.tensor(selected_gates, dtype=torch.long, device=x.device)
    rows = torch.arange(batch, device=x.device)
    positions = anchor.final_positions
    residual_delta = (x @ residual_basis.T).to(anchor.resid_mid.dtype)
    resid_mid = anchor.resid_mid.clone()
    resid_mid[rows, positions, :] += residual_delta

    block10 = model.blocks[10]
    normalized = block10.ln2(resid_mid)
    pre = _batch_addmm(block10.mlp.b_in, block10.mlp.W_in, normalized)
    live_post = block10.mlp.act_fn(pre)
    post = live_post.clone()
    post[rows, positions, :] = anchor.post[rows, positions, :]
    post[rows[:, None], positions[:, None], gate_ids[None, :]] = live_post[
        rows[:, None], positions[:, None], gate_ids[None, :]
    ]
    mlp_out = _batch_addmm(block10.mlp.b_out, block10.mlp.W_out, post)
    resid_post = resid_mid + mlp_out
    resid_post[rows, positions, :] -= residual_delta

    resid = model.blocks[11](resid_post)
    normalized_final = model.ln_final(resid)
    final = normalized_final[rows, positions, :]
    W_selected = model.W_U.index_select(1, suffix_token_ids)
    logits = final @ W_selected
    if getattr(model, "b_U", None) is not None:
        logits = logits + model.b_U.index_select(0, suffix_token_ids)
    return logits


def finite_path_effect(
    model,
    anchor: TargetAnchor,
    residual_basis,
    suffix_token_ids,
    direction,
    clean_suffixes,
    *,
    rho: float,
):
    """Return per-item signed central finite path effects in logit-margin units."""
    torch = _torch()
    if rho <= 0:
        raise ValueError("rho must be positive")
    plus = evaluate_joint_target(
        model, anchor, residual_basis, suffix_token_ids, rho * direction
    ).double()
    minus = evaluate_joint_target(
        model, anchor, residual_basis, suffix_token_ids, -rho * direction
    ).double()
    response = (plus - minus) / (2.0 * rho)
    contrast = torch.stack(
        [logit_contrast(int(y), device=response.device) for y in clean_suffixes], dim=0
    )
    return torch.sum(response * contrast, dim=1)


def target_richardson(full, half):
    return (4.0 * half - full) / 3.0


def target_jvp(
    model,
    anchor: TargetAnchor,
    residual_basis,
    suffix_token_ids,
    direction,
    clean_suffixes,
):
    """Independent zero-radius JVP audit; never used as the primary target."""
    torch = _torch()
    zero = torch.zeros_like(direction, requires_grad=True)

    def curve(x):
        logits = evaluate_joint_target(
            model, anchor, residual_basis, suffix_token_ids, x
        ).double()
        contrasts = torch.stack(
            [logit_contrast(int(y), device=logits.device) for y in clean_suffixes], dim=0
        )
        return torch.sum(logits * contrasts, dim=1)

    _, tangent = torch.func.jvp(curve, (zero,), (direction,))
    return tangent
