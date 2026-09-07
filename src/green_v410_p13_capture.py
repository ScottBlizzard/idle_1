"""Float64 response capture for the one-shot GREEN v4.1 P13 cohort."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping

import torch

from green_v410_protocol import BRANCH_ORDER, sha256_canonical


CENTRAL_SECANT_RADIUS = Fraction(1, 4)


def _float_pair(value: torch.Tensor) -> list[int]:
    scalar = float(value.detach().cpu().item())
    if not torch.isfinite(torch.tensor(scalar, dtype=torch.float64)):
        raise ValueError("P13 response route produced a nonfinite scalar")
    numerator, denominator = scalar.as_integer_ratio()
    return [numerator, denominator]


def compute_response_only_panel(
    *,
    branches: Mapping[str, Any],
    center: torch.Tensor,
    directions: torch.Tensor,
) -> dict[str, Any]:
    """Compute frozen central-secants and independent AD derivatives.

    Four response maps are evaluated independently.  The eight directions are
    only batched within one branch; no branch output is shared with another.
    """

    if tuple(branches) != BRANCH_ORDER:
        raise ValueError("P13 branch order mismatch")
    if center.dtype != torch.float64 or center.ndim != 1:
        raise ValueError("P13 center must be one float64 vector")
    if directions.dtype != torch.float64 or directions.shape != (8, center.numel()):
        raise ValueError("P13 direction panel must be float64 [8,width]")
    if not torch.isfinite(center).all() or not torch.isfinite(directions).all():
        raise ValueError("P13 response inputs must be finite")

    radius = float(CENTRAL_SECANT_RADIUS)
    injections = torch.cat(
        (center[None, :] + radius * directions,
         center[None, :] - radius * directions),
        dim=0,
    )
    secants: dict[str, list[list[int]]] = {}
    derivatives: dict[str, list[list[int]]] = {}
    for name in BRANCH_ORDER:
        branch = branches[name]
        with torch.no_grad():
            values = branch(injections)
        if values.shape != (16,) or values.dtype != torch.float64:
            raise ValueError("P13 branch central-secant output contract mismatch")
        central = (values[:8] - values[8:]) / (2 * radius)

        point = center.detach().clone().requires_grad_(True)
        point_value = branch(point)
        if point_value.ndim != 0 or point_value.dtype != torch.float64:
            raise ValueError("P13 branch AD output contract mismatch")
        gradient, = torch.autograd.grad(point_value, point, create_graph=False)
        directional = directions @ gradient
        if not torch.isfinite(central).all() or not torch.isfinite(directional).all():
            raise ValueError("P13 response validation produced nonfinite values")
        secants[name] = [_float_pair(value) for value in central]
        derivatives[name] = [_float_pair(value) for value in directional]

    payload = {
        "schema_version": "green-v410-p13-response-only-panel-v1",
        "central_secant_radius": [1, 4],
        "branch_order": list(BRANCH_ORDER),
        "direction_count": 8,
        "central_secants": secants,
        "ad_derivatives": derivatives,
        "response_evaluation_dtype": "float64",
    }
    return payload | {"panel_sha256": sha256_canonical(payload)}


def capture_controlled_hook_t0(
    *,
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    hook_name: str,
    site_position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clean center and PAT/TAR full controlled-hook tensors at t=0."""

    if clean_tokens.shape != corrupt_tokens.shape or clean_tokens.ndim != 2:
        raise ValueError("P13 clean/corrupt token shapes differ")
    if clean_tokens.shape[0] != 1:
        raise ValueError("P13 capture requires singleton prompts")

    def capture(tokens: torch.Tensor) -> torch.Tensor:
        found: list[torch.Tensor] = []

        def hook(activation: torch.Tensor, hook: Any) -> torch.Tensor:
            found.append(activation[0].detach().clone())
            return activation

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook)])
        if len(found) != 1:
            raise RuntimeError("P13 controlled-hook capture did not fire exactly once")
        return found[0]

    clean = capture(clean_tokens)
    corrupt = capture(corrupt_tokens)
    if clean.dtype != torch.float64 or corrupt.dtype != torch.float64:
        raise ValueError("P13 controlled-hook capture must use float64 response model")
    if not 0 <= site_position < clean.shape[0] or clean.shape != corrupt.shape:
        raise ValueError("P13 controlled-hook site is invalid")
    center = clean[site_position].detach().clone()
    pat = corrupt.detach().clone()
    pat[site_position] = center
    tar = clean.detach().clone()
    if not torch.equal(pat[site_position], tar[site_position]):
        raise RuntimeError("P13 PAT/TAR center identity failed")
    return center, pat, tar
