"""Shared response-field baselines for the GREEN silent-failure challenge.

Every method receives the same scalar response function, activation center, and
direction panel. The module is task-agnostic and does not select records or open
held-out endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch


ScalarResponse = Callable[[torch.Tensor], torch.Tensor]
BatchedScalarResponse = Callable[[torch.Tensor], torch.Tensor]
BaselineMethod = Literal[
    "exact", "first_order", "integrated_gradients", "hvp", "ms_hvp"
]


@dataclass(frozen=True)
class ResponseFieldComparison:
    method: str
    target_effects: torch.Tensor
    patched_effects: torch.Tensor
    discrepancies: torch.Tensor
    rmse: float
    normalized_rmse: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegratedGradientsCalibration:
    selected_steps: int
    grid: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    absolute_tolerance: float
    relative_tolerance: float
    converged: bool
    selection_rule: str


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


def ms_hvp_response(
    response: ScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    segments: int = 8,
) -> torch.Tensor:
    """Multi-step HVP approximation along each finite intervention path.

    Each segment applies a second-order Taylor step at its left endpoint.  This
    is strictly stronger than the single-center HVP ablation while remaining a
    derivative approximation rather than direct finite evaluation.
    """

    _validate_inputs(center, directions)
    if segments < 2:
        raise ValueError("MS-HVP requires at least two path segments")
    effects = []
    for direction in directions:
        delta = direction / segments
        total = torch.zeros((), dtype=center.dtype, device=center.device)
        for index in range(segments):
            point = (center + (index / segments) * direction).detach().clone()
            point.requires_grad_(True)
            value = _scalar(response, point)
            gradient = torch.autograd.grad(value, point, create_graph=True)[0]
            first = torch.dot(gradient, delta)
            if first.requires_grad:
                hvp = torch.autograd.grad(
                    first, point, create_graph=False, allow_unused=True
                )[0]
                if hvp is None:
                    raise RuntimeError("MS-HVP returned an unexpected unused Hessian path")
                second = 0.5 * torch.dot(delta, hvp)
            else:
                # A structurally linear response has an exact zero Hessian.
                second = torch.zeros_like(first)
            total = total + first.detach() + second.detach()
        effects.append(total)
    return torch.stack(effects)


def response_effects(
    method: BaselineMethod,
    response: ScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    integrated_gradients_steps: int = 65,
    ms_hvp_segments: int = 8,
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
    if method == "ms_hvp":
        return ms_hvp_response(
            response, center, directions, segments=ms_hvp_segments
        )
    raise ValueError(f"unsupported baseline method: {method}")


def batched_response_effects(
    method: BaselineMethod,
    response: BatchedScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    integrated_gradients_steps: int = 65,
    ms_hvp_segments: int = 8,
    batch_chunk_size: int = 32,
) -> torch.Tensor:
    """Vectorized equivalents for batch-independent model response maps."""

    _validate_inputs(center, directions)
    if batch_chunk_size <= 0:
        raise ValueError("batch chunk size must be positive")
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
        gradient_chunks = []
        for raw_chunk in points.split(batch_chunk_size, dim=0):
            chunk = raw_chunk.detach().clone().requires_grad_(True)
            output = _vector(response, chunk)
            gradient_chunks.append(
                torch.autograd.grad(output.sum(), chunk, create_graph=False)[0]
            )
        gradients = torch.cat(gradient_chunks, dim=0)
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
    if method == "ms_hvp":
        if ms_hvp_segments < 2:
            raise ValueError("MS-HVP requires at least two path segments")
        delta = directions / ms_hvp_segments
        total = torch.zeros(count, dtype=center.dtype, device=center.device)
        for index in range(ms_hvp_segments):
            points = (
                center[None, :] + (index / ms_hvp_segments) * directions
            ).detach().clone().requires_grad_(True)
            output = _vector(response, points)
            gradient = torch.autograd.grad(output.sum(), points, create_graph=True)[0]
            first = torch.einsum("nd,nd->n", gradient, delta)
            if first.requires_grad:
                hvp = torch.autograd.grad(
                    first.sum(), points, create_graph=False, allow_unused=True
                )[0]
                if hvp is None:
                    raise RuntimeError(
                        "batched MS-HVP returned an unexpected unused Hessian path"
                    )
                second = 0.5 * torch.einsum("nd,nd->n", delta, hvp)
            else:
                second = torch.zeros_like(first)
            total = total + first.detach() + second.detach()
        return total
    raise ValueError(f"unsupported baseline method: {method}")


def compare_response_fields(
    method: BaselineMethod,
    target_response: ScalarResponse,
    patched_response: ScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    integrated_gradients_steps: int = 65,
    ms_hvp_segments: int = 8,
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
        ms_hvp_segments=ms_hvp_segments,
    )
    patched = response_effects(
        method,
        patched_response,
        center,
        directions,
        integrated_gradients_steps=integrated_gradients_steps,
        ms_hvp_segments=ms_hvp_segments,
    )
    discrepancy = patched - target
    rmse_tensor = torch.sqrt(torch.mean(discrepancy.square()))
    target_scale = torch.sqrt(torch.mean(target.square()))
    patched_scale = torch.sqrt(torch.mean(patched.square()))
    scale = torch.maximum(target_scale, patched_scale)
    normalized = rmse_tensor / torch.clamp(scale, min=normalization_floor)
    return ResponseFieldComparison(
        method=method,
        target_effects=target.detach(),
        patched_effects=patched.detach(),
        discrepancies=discrepancy.detach(),
        rmse=float(rmse_tensor.detach().cpu()),
        normalized_rmse=float(normalized.detach().cpu()),
        diagnostics={
            "ms_hvp_segments": ms_hvp_segments if method == "ms_hvp" else None,
            "single_point_hvp_only": method == "hvp",
        },
    )


def compare_batched_response_fields(
    method: BaselineMethod,
    target_response: BatchedScalarResponse,
    patched_response: BatchedScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    integrated_gradients_steps: int = 65,
    ms_hvp_segments: int = 8,
    batch_chunk_size: int = 32,
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
        ms_hvp_segments=ms_hvp_segments,
        batch_chunk_size=batch_chunk_size,
    )
    patched = batched_response_effects(
        method,
        patched_response,
        center,
        directions,
        integrated_gradients_steps=integrated_gradients_steps,
        ms_hvp_segments=ms_hvp_segments,
        batch_chunk_size=batch_chunk_size,
    )
    discrepancy = patched - target
    rmse_tensor = torch.sqrt(torch.mean(discrepancy.square()))
    target_scale = torch.sqrt(torch.mean(target.square()))
    patched_scale = torch.sqrt(torch.mean(patched.square()))
    scale = torch.maximum(target_scale, patched_scale)
    normalized = rmse_tensor / torch.clamp(scale, min=normalization_floor)
    return ResponseFieldComparison(
        method=method,
        target_effects=target.detach(),
        patched_effects=patched.detach(),
        discrepancies=discrepancy.detach(),
        rmse=float(rmse_tensor.detach().cpu()),
        normalized_rmse=float(normalized.detach().cpu()),
        diagnostics={
            "ms_hvp_segments": ms_hvp_segments if method == "ms_hvp" else None,
            "single_point_hvp_only": method == "hvp",
            "batch_chunk_size": batch_chunk_size,
        },
    )


def calibrate_integrated_gradients_grid(
    target_response: ScalarResponse | BatchedScalarResponse,
    patched_response: ScalarResponse | BatchedScalarResponse,
    center: torch.Tensor,
    directions: torch.Tensor,
    *,
    grid: tuple[int, ...] = (33, 65, 129, 257),
    absolute_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-3,
    batched: bool = False,
    batch_chunk_size: int = 32,
    selection_rule: str = "finite_reference",
) -> IntegratedGradientsCalibration:
    """Outcome-blind numerical calibration of IG quadrature resolution.

    The smallest frozen grid whose target and patched effects agree with direct
    finite evaluation is selected.  This calibrates numerical quadrature only;
    it creates neither a statistical baseline nor a scientific certificate.
    """

    if not grid or tuple(sorted(set(grid))) != grid or min(grid) < 2:
        raise ValueError("IG calibration grid must be unique, increasing, and >= 2")
    if absolute_tolerance <= 0 or relative_tolerance < 0:
        raise ValueError("IG calibration tolerances are invalid")
    if selection_rule not in {"finite_reference", "successive_grid_stability"}:
        raise ValueError("unsupported IG calibration selection rule")
    effect_fn = batched_response_effects if batched else response_effects
    batching_kwargs = {"batch_chunk_size": batch_chunk_size} if batched else {}
    target_exact = effect_fn(
        "exact", target_response, center, directions, **batching_kwargs
    )
    patched_exact = effect_fn(
        "exact", patched_response, center, directions, **batching_kwargs
    )
    reference_scale = float(
        torch.maximum(target_exact.abs().max(), patched_exact.abs().max())
        .double()
        .cpu()
    )
    allowed = absolute_tolerance + relative_tolerance * reference_scale
    records = []
    selected = None
    previous_target = None
    previous_patched = None
    for steps in grid:
        target_ig = effect_fn(
            "integrated_gradients",
            target_response,
            center,
            directions,
            integrated_gradients_steps=steps,
            **batching_kwargs,
        )
        patched_ig = effect_fn(
            "integrated_gradients",
            patched_response,
            center,
            directions,
            integrated_gradients_steps=steps,
            **batching_kwargs,
        )
        max_error = float(
            torch.maximum(
                (target_ig - target_exact).abs().max(),
                (patched_ig - patched_exact).abs().max(),
            )
            .double()
            .cpu()
        )
        successive_error = None
        if previous_target is not None and previous_patched is not None:
            successive_error = float(
                torch.maximum(
                    (target_ig - previous_target).abs().max(),
                    (patched_ig - previous_patched).abs().max(),
                )
                .double()
                .cpu()
            )
        passed = (
            max_error <= allowed
            if selection_rule == "finite_reference"
            else successive_error is not None and successive_error <= allowed
        )
        records.append(
            {
                "steps": steps,
                "max_absolute_gap_to_direct_finite_response": max_error,
                "max_absolute_successive_grid_difference": successive_error,
                "allowed_max_absolute_error": allowed,
                "passed": passed,
            }
        )
        if passed and selected is None:
            selected = steps
        previous_target = target_ig.detach()
        previous_patched = patched_ig.detach()
    return IntegratedGradientsCalibration(
        selected_steps=selected if selected is not None else grid[-1],
        grid=grid,
        records=tuple(records),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        converged=selected is not None,
        selection_rule=selection_rule,
    )
