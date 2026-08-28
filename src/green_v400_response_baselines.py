"""Shared response-field baselines for the GREEN silent-failure challenge.

Every method receives the same scalar response function, activation center, and
direction panel. The module is task-agnostic and does not select records or open
held-out endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch


ScalarResponse = Callable[[torch.Tensor], torch.Tensor]
BatchedScalarResponse = Callable[[torch.Tensor], torch.Tensor]
BaselineMethod = Literal["exact", "first_order", "integrated_gradients", "hvp"]


@dataclass(frozen=True)
class ResponseFieldComparison:
    method: str
    target_effects: torch.Tensor
    patched_effects: torch.Tensor
    discrepancies: torch.Tensor
    rmse: float
    normalized_rmse: float


def _validate_inputs(center: torch.Tensor, directions: torch.Tensor) -> None:
    if center.ndim != 1:
        raise ValueError("center must be a one-dimensional activation vector")
    if directions.ndim != 2 or directions.shape[1] != center.numel():
        raise ValueError("directions must have shape [n_direction, center_width]")
    if directions.shape[0] == 0:
        raise ValueError("direction panel must be nonempty")
    if not center.is_floating_point() or not directions.is_floating_point():
        raise ValueError("center and directions must be floating point")
    if not torch.isfinite(center).all() or not torch.isfinite(directions).all():
        raise ValueError("center and directions must be finite")


def _scalar(response: ScalarResponse, value: torch.Tensor) -> torch.Tensor:
    output = response(value)
    if not isinstance(output, torch.Tensor) or output.numel() != 1:
        raise ValueError("response function must return one scalar tensor")
    result = output.reshape(())
    if not torch.isfinite(result):
        raise ValueError("response function returned a non-finite scalar")
    return result


def _vector(
    response: BatchedScalarResponse, values: torch.Tensor
) -> torch.Tensor:
    output = response(values)
    if not isinstance(output, torch.Tensor) or output.ndim != 1:
        raise ValueError("batched response must return one vector")
    if output.shape[0] != values.shape[0]:
        raise ValueError("batched response output length must equal input batch")
    if not torch.isfinite(output).all():
        raise ValueError("batched response returned non-finite values")
    return output


def exact_finite_response(
    response: ScalarResponse, center: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Evaluate exact finite responses f(center + d) - f(center)."""

    _validate_inputs(center, directions)
    base = _scalar(response, center)
    values = [_scalar(response, center + direction) - base for direction in directions]
    return torch.stack(values)


def first_order_response(
    response: ScalarResponse, center: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Evaluate the attribution-patching/JVP baseline grad f(center) dot d."""

    _validate_inputs(center, directions)
    point = center.detach().clone().requires_grad_(True)
    value = _scalar(response, point)
    gradient = torch.autograd.grad(value, point, create_graph=False)[0]
    return directions @ gradient.detach()


def integrated_gradients_response(
    response: ScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    steps: int = 65,
) -> torch.Tensor:
    """Integrate directional gradients on each center-to-endpoint segment.

    A fixed trapezoidal grid includes both endpoints. The result is a numerical
    baseline, not a certificate.
    """

    _validate_inputs(center, directions)
    if steps < 2:
        raise ValueError("integrated gradients requires at least two steps")
    alphas = torch.linspace(0.0, 1.0, steps, dtype=center.dtype, device=center.device)
    effects = []
    for direction in directions:
        derivatives = []
        for alpha in alphas:
            point = (center + alpha * direction).detach().clone().requires_grad_(True)
            value = _scalar(response, point)
            gradient = torch.autograd.grad(value, point, create_graph=False)[0]
            derivatives.append(torch.dot(gradient, direction))
        effects.append(torch.trapz(torch.stack(derivatives), alphas))
    return torch.stack(effects)


def hvp_second_order_response(
    response: ScalarResponse, center: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Evaluate grad f dot d + 1/2 d^T Hessian(f) d at the center."""

    _validate_inputs(center, directions)
    effects = []
    for direction in directions:
        point = center.detach().clone().requires_grad_(True)
        value = _scalar(response, point)
        gradient = torch.autograd.grad(value, point, create_graph=True)[0]
        first = torch.dot(gradient, direction)
        if first.requires_grad:
            hvp = torch.autograd.grad(
                first, point, create_graph=False, allow_unused=True
            )[0]
            if hvp is None:
                hvp = torch.zeros_like(point)
        else:
            hvp = torch.zeros_like(point)
        effects.append(first.detach() + 0.5 * torch.dot(direction, hvp.detach()))
    return torch.stack(effects)


def response_effects(
    method: BaselineMethod,
    response: ScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    integrated_gradients_steps: int = 65,
) -> torch.Tensor:
    if method == "exact":
        return exact_finite_response(response, center, directions)
    if method == "first_order":
        return first_order_response(response, center, directions)
    if method == "integrated_gradients":
        return integrated_gradients_response(
            response, center, directions, steps=integrated_gradients_steps
        )
    if method == "hvp":
        return hvp_second_order_response(response, center, directions)
    raise ValueError(f"unsupported baseline method: {method}")


def batched_response_effects(
    method: BaselineMethod,
    response: BatchedScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    integrated_gradients_steps: int = 65,
) -> torch.Tensor:
    """Vectorized equivalents for batch-independent model response maps."""

    _validate_inputs(center, directions)
    count, width = directions.shape
    if method == "exact":
        values = torch.cat([center[None, :], center[None, :] + directions], dim=0)
        output = _vector(response, values)
        return output[1:] - output[0]
    if method == "first_order":
        point = center.detach().clone()[None, :].requires_grad_(True)
        output = _vector(response, point)
        gradient = torch.autograd.grad(output.sum(), point, create_graph=False)[0][0]
        return directions @ gradient.detach()
    if method == "integrated_gradients":
        if integrated_gradients_steps < 2:
            raise ValueError("integrated gradients requires at least two steps")
        alphas = torch.linspace(
            0.0,
            1.0,
            integrated_gradients_steps,
            dtype=center.dtype,
            device=center.device,
        )
        points = (
            center[None, None, :]
            + alphas[None, :, None] * directions[:, None, :]
        ).reshape(count * integrated_gradients_steps, width)
        points = points.detach().clone().requires_grad_(True)
        output = _vector(response, points)
        gradients = torch.autograd.grad(output.sum(), points, create_graph=False)[0]
        gradients = gradients.reshape(count, integrated_gradients_steps, width)
        derivatives = torch.einsum("nsd,nd->ns", gradients, directions)
        return torch.trapz(derivatives, alphas, dim=1)
    if method == "hvp":
        points = center[None, :].expand(count, -1).detach().clone().requires_grad_(True)
        output = _vector(response, points)
        gradient = torch.autograd.grad(output.sum(), points, create_graph=True)[0]
        first = torch.einsum("nd,nd->n", gradient, directions)
        if first.requires_grad:
            hvp = torch.autograd.grad(
                first.sum(), points, create_graph=False, allow_unused=True
            )[0]
            if hvp is None:
                hvp = torch.zeros_like(points)
        else:
            hvp = torch.zeros_like(points)
        return first.detach() + 0.5 * torch.einsum(
            "nd,nd->n", directions, hvp.detach()
        )
    raise ValueError(f"unsupported baseline method: {method}")


def compare_response_fields(
    method: BaselineMethod,
    target_response: ScalarResponse,
    patched_response: ScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    integrated_gradients_steps: int = 65,
    normalization_floor: float = 1e-12,
) -> ResponseFieldComparison:
    """Compare target and patched response fields on one non-held-out panel."""

    if normalization_floor <= 0:
        raise ValueError("normalization_floor must be positive")
    target = response_effects(
        method,
        target_response,
        center,
        directions,
        integrated_gradients_steps=integrated_gradients_steps,
    )
    patched = response_effects(
        method,
        patched_response,
        center,
        directions,
        integrated_gradients_steps=integrated_gradients_steps,
    )
    discrepancy = patched - target
    rmse_tensor = torch.sqrt(torch.mean(discrepancy.square()))
    scale = torch.sqrt(torch.mean(target.square()))
    normalized = rmse_tensor / torch.clamp(scale, min=normalization_floor)
    return ResponseFieldComparison(
        method=method,
        target_effects=target.detach(),
        patched_effects=patched.detach(),
        discrepancies=discrepancy.detach(),
        rmse=float(rmse_tensor.detach().cpu()),
        normalized_rmse=float(normalized.detach().cpu()),
    )


def compare_batched_response_fields(
    method: BaselineMethod,
    target_response: BatchedScalarResponse,
    patched_response: BatchedScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    integrated_gradients_steps: int = 65,
    normalization_floor: float = 1e-12,
) -> ResponseFieldComparison:
    if normalization_floor <= 0:
        raise ValueError("normalization_floor must be positive")
    target = batched_response_effects(
        method,
        target_response,
        center,
        directions,
        integrated_gradients_steps=integrated_gradients_steps,
    )
    patched = batched_response_effects(
        method,
        patched_response,
        center,
        directions,
        integrated_gradients_steps=integrated_gradients_steps,
    )
    discrepancy = patched - target
    rmse_tensor = torch.sqrt(torch.mean(discrepancy.square()))
    scale = torch.sqrt(torch.mean(target.square()))
    normalized = rmse_tensor / torch.clamp(scale, min=normalization_floor)
    return ResponseFieldComparison(
        method=method,
        target_effects=target.detach(),
        patched_effects=patched.detach(),
        discrepancies=discrepancy.detach(),
        rmse=float(rmse_tensor.detach().cpu()),
        normalized_rmse=float(normalized.detach().cpu()),
    )
