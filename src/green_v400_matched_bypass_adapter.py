"""Full-model matched-bypass response adapter for the empirical comparator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from green_bridge_v400_branch_semantics import BRANCH_ORDER


LogitScore = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GateBypassBinding:
    anchor_selected_post: torch.Tensor
    selected_gates: tuple[int, ...]
    gate_layer: int = 10

    def __post_init__(self) -> None:
        if self.anchor_selected_post.ndim != 1:
            raise ValueError("gate bypass anchor must be one vector")
        if not self.anchor_selected_post.is_floating_point():
            raise ValueError("gate bypass anchor must be floating point")
        if not torch.isfinite(self.anchor_selected_post).all():
            raise ValueError("gate bypass anchor must be finite")
        if not self.selected_gates or len(set(self.selected_gates)) != len(
            self.selected_gates
        ):
            raise ValueError("selected gates must be nonempty and unique")
        if min(self.selected_gates) < 0:
            raise ValueError("selected gate indices must be nonnegative")
        if len(self.selected_gates) != self.anchor_selected_post.numel():
            raise ValueError("gate anchor width must match selected gates")
        if self.gate_layer < 0:
            raise ValueError("gate layer must be nonnegative")

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.gate_layer}.mlp.hook_post"


class MatchedBypassScalarResponse:
    """Scalar task response with an optional selected-gate bypass freeze."""

    supports_batch = True

    def __init__(
        self,
        *,
        model: Any,
        tokens: torch.Tensor,
        intervention_hook: str,
        intervention_position: int,
        score_logits: LogitScore,
        bypass_binding: GateBypassBinding | None,
    ) -> None:
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("matched-bypass response requires tokens with shape [1, seq]")
        if not 0 <= intervention_position < tokens.shape[1]:
            raise ValueError("intervention position lies outside the token sequence")
        self.model = model
        self.tokens = tokens
        self.intervention_hook = intervention_hook
        self.intervention_position = intervention_position
        self.score_logits = score_logits
        self.bypass_binding = bypass_binding

    def __call__(self, injection: torch.Tensor) -> torch.Tensor:
        if injection.ndim not in (1, 2):
            raise ValueError("injection must be one vector or a batch of vectors")
        scalar_input = injection.ndim == 1
        values = injection[None, :] if scalar_input else injection
        batch = values.shape[0]

        def patch_site(activation: torch.Tensor, hook: Any) -> torch.Tensor:
            if activation.ndim != 3 or activation.shape[0] != batch:
                raise ValueError("intervention activation batch mismatch")
            if activation.shape[-1] != values.shape[-1]:
                raise ValueError("intervention width mismatch")
            result = activation.clone()
            result[:, self.intervention_position, :] = values
            return result

        hooks = [(self.intervention_hook, patch_site)]
        if self.bypass_binding is not None:
            binding = self.bypass_binding

            def freeze_selected_post(
                activation: torch.Tensor, hook: Any
            ) -> torch.Tensor:
                if activation.ndim != 3 or activation.shape[0] != batch:
                    raise ValueError("gate-post activation batch mismatch")
                gates = torch.as_tensor(
                    binding.selected_gates,
                    dtype=torch.long,
                    device=activation.device,
                )
                if int(gates.max()) >= activation.shape[-1]:
                    raise ValueError("selected gate lies outside MLP width")
                anchor = binding.anchor_selected_post.to(
                    dtype=activation.dtype, device=activation.device
                )
                result = activation.clone()
                result[:, -1, gates] = anchor[None, :]
                return result

            hooks.append((binding.hook_name, freeze_selected_post))

        tokens = self.tokens if scalar_input else self.tokens.expand(batch, -1)
        logits = self.model.run_with_hooks(tokens, fwd_hooks=hooks)
        score = self.score_logits(logits)
        if not isinstance(score, torch.Tensor) or score.ndim != 1:
            raise ValueError("matched-bypass logit score must return one vector")
        if score.shape[0] != batch or not torch.isfinite(score).all():
            raise ValueError("matched-bypass logit score returned invalid values")
        return score[0] if scalar_input else score


def capture_selected_gate_anchor(
    *,
    model: Any,
    tokens: torch.Tensor,
    intervention_hook: str,
    intervention_position: int,
    center: torch.Tensor,
    selected_gates: tuple[int, ...],
    gate_layer: int = 10,
) -> GateBypassBinding:
    """Capture the selected gate posts at t=0 under one branch context."""

    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("gate anchor capture requires tokens with shape [1, seq]")
    if center.ndim != 1 or not center.is_floating_point() or not torch.isfinite(center).all():
        raise ValueError("gate anchor center must be one finite floating vector")
    captured: list[torch.Tensor] = []

    def patch_site(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        if activation.shape[-1] != center.numel():
            raise ValueError("gate anchor intervention width mismatch")
        result = activation.clone()
        result[0, intervention_position, :] = center.to(
            dtype=activation.dtype, device=activation.device
        )
        return result

    def capture_gate(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        gates = torch.as_tensor(selected_gates, dtype=torch.long, device=activation.device)
        if gates.numel() == 0 or len(set(selected_gates)) != len(selected_gates):
            raise ValueError("selected gates must be nonempty and unique")
        if int(gates.min()) < 0 or int(gates.max()) >= activation.shape[-1]:
            raise ValueError("selected gate lies outside MLP width")
        captured.append(activation[0, -1, :].index_select(0, gates).detach().clone())
        return activation

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (intervention_hook, patch_site),
                (f"blocks.{gate_layer}.mlp.hook_post", capture_gate),
            ],
        )
    if len(captured) != 1:
        raise RuntimeError("selected gate anchor hook must fire exactly once")
    return GateBypassBinding(captured[0], selected_gates, gate_layer)


def build_matched_bypass_four_branches(
    *,
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    intervention_hook: str,
    intervention_position: int,
    center: torch.Tensor,
    score_logits: LogitScore,
    selected_gates: tuple[int, ...],
    gate_layer: int = 10,
) -> dict[str, MatchedBypassScalarResponse]:
    """Build PAT/TAR x joint/bypass responses with matched t=0 anchors."""

    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("clean and corrupt token tensors must have equal shape")
    clean_binding = capture_selected_gate_anchor(
        model=model,
        tokens=clean_tokens,
        intervention_hook=intervention_hook,
        intervention_position=intervention_position,
        center=center,
        selected_gates=selected_gates,
        gate_layer=gate_layer,
    )
    corrupt_binding = capture_selected_gate_anchor(
        model=model,
        tokens=corrupt_tokens,
        intervention_hook=intervention_hook,
        intervention_position=intervention_position,
        center=center,
        selected_gates=selected_gates,
        gate_layer=gate_layer,
    )
    common = {
        "model": model,
        "intervention_hook": intervention_hook,
        "intervention_position": intervention_position,
        "score_logits": score_logits,
    }
    result = {
        "PAT_J": MatchedBypassScalarResponse(
            **common, tokens=corrupt_tokens, bypass_binding=None
        ),
        "PAT_B": MatchedBypassScalarResponse(
            **common, tokens=corrupt_tokens, bypass_binding=corrupt_binding
        ),
        "TAR_J": MatchedBypassScalarResponse(
            **common, tokens=clean_tokens, bypass_binding=None
        ),
        "TAR_B": MatchedBypassScalarResponse(
            **common, tokens=clean_tokens, bypass_binding=clean_binding
        ),
    }
    if tuple(result) != BRANCH_ORDER:
        raise RuntimeError("constructed branch order differs from binding semantics")
    return result
