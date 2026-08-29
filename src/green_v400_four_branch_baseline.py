"""Empirical matched-bypass four-branch comparator for GREEN.

This comparator point-samples the same signed functional used by the formal
certificate.  It is deliberately non-certifying: no finite collection of
samples is represented as an interval proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from green_bridge_v400_branch_semantics import BRANCH_ORDER, BRANCH_WEIGHTS


ScalarResponse = Callable[[torch.Tensor], torch.Tensor]
BatchedScalarResponse = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FourBranchInteractionResult:
    psi_at_center: torch.Tensor
    psi_effects: torch.Tensor
    rms_effect: float
    branch_order: tuple[str, ...] = BRANCH_ORDER
    branch_weights: tuple[int, ...] = BRANCH_WEIGHTS
    point_sampling_only: bool = True
    certificate_claimed: bool = False


def _validate(
    branches: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
    center: torch.Tensor,
    directions: torch.Tensor,
) -> None:
    if set(branches) != set(BRANCH_ORDER):
        raise ValueError("four-branch responses must match the binding branch order")
    if center.ndim != 1 or not center.is_floating_point():
        raise ValueError("center must be one floating-point activation vector")
    if directions.ndim != 2 or directions.shape[1] != center.numel():
        raise ValueError("directions must have shape [n_direction, center_width]")
    if directions.shape[0] == 0:
        raise ValueError("direction panel must be nonempty")
    if not directions.is_floating_point():
        raise ValueError("directions must be floating point")
    if not torch.isfinite(center).all() or not torch.isfinite(directions).all():
        raise ValueError("center and directions must be finite")


def _scalar_psi(
    branches: Mapping[str, ScalarResponse], value: torch.Tensor
) -> torch.Tensor:
    terms = []
    for name, weight in zip(BRANCH_ORDER, BRANCH_WEIGHTS, strict=True):
        output = branches[name](value)
        if not isinstance(output, torch.Tensor) or output.numel() != 1:
            raise ValueError(f"{name} must return one scalar tensor")
        scalar = output.reshape(())
        if not torch.isfinite(scalar):
            raise ValueError(f"{name} returned a non-finite scalar")
        terms.append(weight * scalar)
    return torch.stack(terms).sum()


def empirical_four_branch_interaction_response(
    branches: Mapping[str, ScalarResponse],
    center: torch.Tensor,
    directions: torch.Tensor,
) -> FourBranchInteractionResult:
    """Evaluate Psi(center+d)-Psi(center) by direct four-branch sampling."""

    _validate(branches, center, directions)
    base = _scalar_psi(branches, center)
    effects = torch.stack(
        [_scalar_psi(branches, center + direction) - base for direction in directions]
    )
    rms = torch.sqrt(torch.mean(effects.double().square()))
    return FourBranchInteractionResult(
        psi_at_center=base.detach(),
        psi_effects=effects.detach(),
        rms_effect=float(rms.detach().cpu()),
    )


def empirical_four_branch_interaction_response_batched(
    branches: Mapping[str, BatchedScalarResponse],
    center: torch.Tensor,
    directions: torch.Tensor,
) -> FourBranchInteractionResult:
    """Vectorized point-sampling equivalent for batch-independent branches."""

    _validate(branches, center, directions)
    values = torch.cat([center[None, :], center[None, :] + directions], dim=0)
    signed = []
    for name, weight in zip(BRANCH_ORDER, BRANCH_WEIGHTS, strict=True):
        output = branches[name](values)
        if not isinstance(output, torch.Tensor) or output.ndim != 1:
            raise ValueError(f"batched {name} must return one vector")
        if output.shape[0] != values.shape[0] or not torch.isfinite(output).all():
            raise ValueError(f"batched {name} returned an invalid vector")
        signed.append(weight * output)
    psi = torch.stack(signed).sum(dim=0)
    effects = psi[1:] - psi[0]
    rms = torch.sqrt(torch.mean(effects.double().square()))
    return FourBranchInteractionResult(
        psi_at_center=psi[0].detach(),
        psi_effects=effects.detach(),
        rms_effect=float(rms.detach().cpu()),
    )
