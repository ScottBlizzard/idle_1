"""Differentiable Greater-Than response adapter for the frozen replication.

Importing this module performs no model work. Untouched rows may be evaluated
only by a separately authorized prediction worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


def greater_than_contrast(clean_suffix: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not 0 <= clean_suffix < 99:
        raise ValueError("clean_suffix must lie between 0 and 98")
    value = torch.empty(100, dtype=dtype, device=device)
    value[: clean_suffix + 1] = -1.0 / (clean_suffix + 1)
    value[clean_suffix + 1 :] = 1.0 / (99 - clean_suffix)
    return value


@dataclass(frozen=True)
class GreaterThanInterventionSite:
    layer: int
    position: int
    clean_suffix: int
    suffix_token_ids: tuple[int, ...]
    hook_family: str = "resid_post"

    def __post_init__(self) -> None:
        if self.layer < 0 or self.position < 0:
            raise ValueError("layer and position must be nonnegative")
        if not 0 <= self.clean_suffix < 99:
            raise ValueError("clean_suffix must lie between 0 and 98")
        if len(self.suffix_token_ids) != 100:
            raise ValueError("suffix_token_ids must contain all 100 suffixes")
        if len(set(self.suffix_token_ids)) != 100 or min(self.suffix_token_ids) < 0:
            raise ValueError("suffix token identifiers must be nonnegative and unique")
        if self.hook_family != "resid_post":
            raise ValueError("the frozen replication supports resid_post only")

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_resid_post"


class GreaterThanScalarResponse:
    """Greater-Than mean-logit contrast under one activation injection."""

    supports_batch = True

    def __init__(self, model: Any, tokens: torch.Tensor, site: GreaterThanInterventionSite):
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("Greater-Than response requires tokens with shape [1, seq]")
        if site.position >= tokens.shape[1]:
            raise ValueError("site position lies outside token sequence")
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
        suffix_ids = torch.as_tensor(
            self.site.suffix_token_ids, dtype=torch.long, device=logits.device
        )
        if int(suffix_ids.max()) >= logits.shape[-1]:
            raise ValueError("suffix token identifier lies outside model vocabulary")
        year_logits = logits[:, -1, :].index_select(-1, suffix_ids)
        contrast = greater_than_contrast(
            self.site.clean_suffix, dtype=year_logits.dtype, device=year_logits.device
        )
        result = (year_logits * contrast).sum(dim=-1)
        return result[0] if scalar_input else result


def build_target_and_patched_responses(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    site: GreaterThanInterventionSite,
) -> tuple[GreaterThanScalarResponse, GreaterThanScalarResponse]:
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("clean and corrupt token tensors must have equal shape")
    return (
        GreaterThanScalarResponse(model, clean_tokens, site),
        GreaterThanScalarResponse(model, corrupt_tokens, site),
    )


def capture_resid_post_center(
    model: Any, clean_tokens: torch.Tensor, site: GreaterThanInterventionSite
) -> torch.Tensor:
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
