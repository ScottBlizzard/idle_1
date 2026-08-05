"""Audit-only LayerNorm gate cotangents for green-bridge v1.3.

Response-derived matched-bypass estimates must never use values from this
module.  These functions only verify the architecture-derived envelope.
"""
from __future__ import annotations

import numpy as np


def _vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    return result


def layernorm_gate_gradient_formula(
    residual, ln_scale, mlp_input_weight, *, eps: float
) -> np.ndarray:
    r = _vector(residual, "residual")
    gamma = _vector(ln_scale, "ln_scale")
    weight = _vector(mlp_input_weight, "mlp_input_weight")
    if r.shape != gamma.shape or gamma.shape != weight.shape:
        raise ValueError("LayerNorm gradient inputs must have equal shape")
    if eps < 0.0:
        raise ValueError("LayerNorm epsilon must be nonnegative")
    dimension = r.size
    centered = r - np.mean(r, dtype=np.float64)
    tau = float(np.sqrt(np.mean(centered * centered, dtype=np.float64) + eps))
    if tau == 0.0:
        raise ValueError("LayerNorm scale denominator is zero")
    atom = gamma * weight
    return (
        atom
        - np.mean(atom, dtype=np.float64)
        - centered * float(centered @ atom) / (dimension * tau * tau)
    ) / tau


def layernorm_gate_gradient_autograd(
    residual,
    ln_scale,
    mlp_input_weight,
    *,
    eps: float,
    ln_bias=None,
    mlp_input_bias: float = 0.0,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("float64 PyTorch is required for the autograd audit") from exc
    r_np = _vector(residual, "residual")
    gamma_np = _vector(ln_scale, "ln_scale")
    weight_np = _vector(mlp_input_weight, "mlp_input_weight")
    if r_np.shape != gamma_np.shape or gamma_np.shape != weight_np.shape:
        raise ValueError("LayerNorm gradient inputs must have equal shape")
    beta_np = np.zeros_like(r_np) if ln_bias is None else _vector(ln_bias, "ln_bias")
    if beta_np.shape != r_np.shape:
        raise ValueError("LayerNorm bias shape disagrees")
    r = torch.tensor(r_np, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(gamma_np, dtype=torch.float64)
    beta = torch.tensor(beta_np, dtype=torch.float64)
    weight = torch.tensor(weight_np, dtype=torch.float64)
    centered = r - r.mean()
    normalized = centered / torch.sqrt(torch.mean(centered * centered) + float(eps))
    gate = torch.dot(normalized * gamma + beta, weight) + float(mlp_input_bias)
    (gradient,) = torch.autograd.grad(gate, r, create_graph=False)
    return gradient.detach().cpu().numpy()


def gradient_envelope_residual(frame, gradient) -> dict[str, float]:
    q = np.asarray(frame, dtype=np.float64)
    g = _vector(gradient, "gradient")
    if q.ndim != 2 or q.shape[0] != g.size:
        raise ValueError("frame and gradient dimensions disagree")
    residual = g - q @ (q.T @ g)
    return {
        "absolute": float(np.linalg.norm(residual)),
        "relative": float(np.linalg.norm(residual) / max(float(np.linalg.norm(g)), 1e-12)),
    }


def shift_null_metric(gradient) -> float:
    g = _vector(gradient, "gradient")
    return float(
        abs(np.sum(g, dtype=np.float64))
        / (np.sqrt(float(g.size)) * max(float(np.linalg.norm(g)), 1.0))
    )


def whitebox_A_coordinates(frame, gradient) -> np.ndarray:
    q = np.asarray(frame, dtype=np.float64)
    g = _vector(gradient, "gradient")
    if q.ndim != 2 or q.shape[0] != g.size:
        raise ValueError("frame and gradient dimensions disagree")
    return q.T @ g
