"""IOI response adapters for the prospective silent-failure challenge.

This module defines execution semantics but performs no work at import time. A
caller must separately receive scientific authorization before loading a model or
evaluating an untouched row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class IOIInterventionSite:
    layer: int
    position: int
    io_token_id: int
    s_token_id: int
    hook_family: str = "resid_post"

    def __post_init__(self) -> None:
        if self.layer < 0 or self.position < 0:
            raise ValueError("layer and position must be nonnegative")
        if self.io_token_id < 0 or self.s_token_id < 0:
            raise ValueError("token identifiers must be nonnegative")
        if self.io_token_id == self.s_token_id:
            raise ValueError("IO and S token identifiers must differ")
        if self.hook_family != "resid_post":
            raise ValueError("the frozen IOI challenge supports resid_post only")

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_resid_post"


class IOIScalarResponse:
    """Differentiable scalar IOI logit contrast under one activation injection."""

    supports_batch = True

    def __init__(self, model: Any, tokens: torch.Tensor, site: IOIInterventionSite):
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("IOI scalar response requires tokens with shape [1, seq]")
        if site.position >= tokens.shape[1]:
            raise ValueError("site position lies outside the token sequence")
        self.model = model
        self.tokens = tokens
        self.site = site

    def __call__(self, injection: torch.Tensor) -> torch.Tensor:
        if injection.ndim not in (1, 2):
            raise ValueError("injection must be one vector or a batch of vectors")
        scalar_input = injection.ndim == 1
        values = injection[None, :] if scalar_input else injection
        batch = values.shape[0]

        def patch(activation: torch.Tensor, hook: Any) -> torch.Tensor:
            if activation.ndim != 3 or activation.shape[0] != batch:
                raise ValueError("hook activation batch does not match injections")
            if activation.shape[-1] != values.shape[1]:
                raise ValueError("injection width does not match hook activation")
            result = activation.clone()
            result[:, self.site.position, :] = values
            return result

        tokens = self.tokens if scalar_input else self.tokens.expand(batch, -1)
        logits = self.model.run_with_hooks(
            tokens, fwd_hooks=[(self.site.hook_name, patch)]
        )
        if logits.ndim != 3 or logits.shape[0] != batch:
            raise ValueError("model logits batch does not match injections")
        final = logits[:, -1, :]
        if max(self.site.io_token_id, self.site.s_token_id) >= final.shape[1]:
            raise ValueError("IO or S token identifier lies outside model vocabulary")
        contrast = final[:, self.site.io_token_id] - final[:, self.site.s_token_id]
        return contrast[0] if scalar_input else contrast


def build_target_and_patched_responses(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    site: IOIInterventionSite,
) -> tuple[IOIScalarResponse, IOIScalarResponse]:
    """Return clean-context target and corrupt-context patched response maps."""

    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("clean and corrupt token tensors must have equal shape")
    return (
        IOIScalarResponse(model, clean_tokens, site),
        IOIScalarResponse(model, corrupt_tokens, site),
    )


def capture_resid_post_center(
    model: Any, clean_tokens: torch.Tensor, site: IOIInterventionSite
) -> torch.Tensor:
    """Capture the clean activation center for an authorized prediction worker."""

    if clean_tokens.ndim != 2 or clean_tokens.shape[0] != 1:
        raise ValueError("center capture requires tokens with shape [1, seq]")
    captured: list[torch.Tensor] = []

    def capture(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        captured.append(activation[0, site.position, :].detach().clone())
        return activation

    with torch.no_grad():
        model.run_with_hooks(clean_tokens, fwd_hooks=[(site.hook_name, capture)])
    if len(captured) != 1:
        raise RuntimeError("clean center hook must fire exactly once")
    return captured[0]
