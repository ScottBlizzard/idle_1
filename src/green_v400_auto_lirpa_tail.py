"""Generic auto_LiRPA comparator for a GPT-2 residual-stream tail.

The adapter uses the Hugging Face GPT-2 graph directly: a variable activation
vector is inserted into one position of a fixed `resid_post` tensor, remaining
blocks are evaluated, and the final IO-minus-S logit contrast is returned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class LiRPABoundResult:
    method: str
    norm: str
    epsilon: float
    point_value: float
    lower_bound: float
    upper_bound: float

    @property
    def width(self) -> float:
        return self.upper_bound - self.lower_bound

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "width": self.width}


class GPT2EagerAttentionNoDropout(nn.Module):
    """Evaluation-equivalent eager GPT-2 attention without dropout operators."""

    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        if getattr(attention, "reorder_and_upcast_attn", False):
            raise ValueError("upcast/reordered GPT-2 attention is not supported")
        self.c_attn = attention.c_attn
        self.c_proj = attention.c_proj
        self.split_size = int(attention.split_size)
        self.num_heads = int(attention.num_heads)
        self.head_dim = int(attention.head_dim)
        self.scaling = float(attention.scaling)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        query, key, value = self.c_attn(hidden_states).split(
            self.split_size, dim=2
        )
        batch, sequence, _ = query.shape
        query = query.view(batch, sequence, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, sequence, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        weights = torch.softmax(weights + attention_mask, dim=-1)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(
            batch, sequence, self.num_heads * self.head_dim
        )
        return self.c_proj(output)


class GPT2MLPNoDropout(nn.Module):
    """Evaluation-equivalent GPT-2 MLP without its no-op dropout module."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.c_fc = mlp.c_fc
        self.act = mlp.act
        self.c_proj = mlp.c_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(hidden_states)))


class GPT2BlockNoDropout(nn.Module):
    """Evaluation-equivalent GPT-2 block expressed with verifier-friendly ops."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        if hasattr(block, "crossattention"):
            raise ValueError("cross-attention GPT-2 blocks are not supported")
        self.ln_1 = block.ln_1
        self.attention = GPT2EagerAttentionNoDropout(block.attn)
        self.ln_2 = block.ln_2
        self.mlp = GPT2MLPNoDropout(block.mlp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.ln_1(hidden_states), attention_mask
        )
        return hidden_states + self.mlp(self.ln_2(hidden_states))


class GPT2ResidualTail(nn.Module):
    """Traceable GPT-2 tail from one full residual vector to logit contrast."""

    def __init__(
        self,
        *,
        blocks: Iterable[nn.Module],
        final_layer_norm: nn.Module,
        lm_head: nn.Module,
        fixed_hidden: torch.Tensor,
        position: int,
        io_token_id: int,
        s_token_id: int,
    ) -> None:
        super().__init__()
        if fixed_hidden.ndim != 3 or fixed_hidden.shape[0] != 1:
            raise ValueError("fixed_hidden must have shape [1, sequence, width]")
        if not 0 <= position < fixed_hidden.shape[1]:
            raise ValueError("position lies outside fixed_hidden")
        if io_token_id < 0 or s_token_id < 0 or io_token_id == s_token_id:
            raise ValueError("IO and S token identifiers must be distinct and nonnegative")
        block_list = list(blocks)
        if not block_list:
            raise ValueError("tail requires at least one remaining transformer block")
        self.blocks = nn.ModuleList(block_list)
        self.final_layer_norm = final_layer_norm
        self.lm_head = lm_head
        self.position = position
        self.io_token_id = io_token_id
        self.s_token_id = s_token_id
        self.register_buffer("fixed_hidden", fixed_hidden.detach().clone())
        mask = torch.zeros(
            1,
            fixed_hidden.shape[1],
            1,
            dtype=fixed_hidden.dtype,
            device=fixed_hidden.device,
        )
        mask[:, position, :] = 1.0
        self.register_buffer("position_mask", mask)
        sequence = fixed_hidden.shape[1]
        causal_mask = torch.triu(
            torch.full(
                (sequence, sequence),
                torch.finfo(fixed_hidden.dtype).min,
                dtype=fixed_hidden.dtype,
                device=fixed_hidden.device,
            ),
            diagonal=1,
        )[None, None, :, :]
        self.register_buffer("causal_mask", causal_mask)

    @property
    def width(self) -> int:
        return int(self.fixed_hidden.shape[-1])

    def forward(self, injection: torch.Tensor) -> torch.Tensor:
        if injection.ndim != 2 or injection.shape[1] != self.width:
            raise ValueError("injection must have shape [batch, width]")
        batch = injection.shape[0]
        fixed = self.fixed_hidden.expand(batch, -1, -1)
        mask = self.position_mask.expand(batch, -1, -1)
        hidden = fixed * (1.0 - mask) + injection[:, None, :] * mask
        attention_mask = self.causal_mask.expand(batch, -1, -1, -1)
        for block in self.blocks:
            block_output = block(
                hidden_states=hidden,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )
            # Transformers versions before the 2025 modeling-layer refactor
            # returned tuples; the pinned server version returns the tensor.
            hidden = block_output[0] if isinstance(block_output, tuple) else block_output
        hidden = self.final_layer_norm(hidden)
        logits = self.lm_head(hidden[:, -1, :])
        return (
            logits[:, self.io_token_id] - logits[:, self.s_token_id]
        ).unsqueeze(-1)


def capture_hf_resid_post(
    hf_model: Any, tokens: torch.Tensor, *, layer: int
) -> torch.Tensor:
    """Capture the Hugging Face hidden state after zero-indexed block `layer`."""

    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("tokens must have shape [1, sequence]")
    n_layers = len(hf_model.transformer.h)
    if not 0 <= layer < n_layers - 1:
        raise ValueError("layer must leave at least one downstream block")
    with torch.no_grad():
        output = hf_model.transformer(
            input_ids=tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) != n_layers + 1:
        raise RuntimeError("Hugging Face model did not return the expected hidden states")
    return hidden_states[layer + 1].detach()


def build_hf_gpt2_residual_tail(
    hf_model: Any,
    fixed_hidden: torch.Tensor,
    *,
    layer: int,
    position: int,
    io_token_id: int,
    s_token_id: int,
) -> GPT2ResidualTail:
    n_layers = len(hf_model.transformer.h)
    if not 0 <= layer < n_layers - 1:
        raise ValueError("layer must leave at least one downstream block")
    return GPT2ResidualTail(
        blocks=[
            GPT2BlockNoDropout(block)
            for block in hf_model.transformer.h[layer + 1 :]
        ],
        final_layer_norm=hf_model.transformer.ln_f,
        lm_head=hf_model.lm_head,
        fixed_hidden=fixed_hidden,
        position=position,
        io_token_id=io_token_id,
        s_token_id=s_token_id,
    ).eval()


def auto_lirpa_linf_bounds(
    tail: nn.Module,
    center: torch.Tensor,
    *,
    epsilon: float,
    method: str = "backward",
) -> LiRPABoundResult:
    """Compute a generic certified L-infinity bound with auto_LiRPA."""

    if center.ndim != 2 or center.shape[0] != 1:
        raise ValueError("center must have shape [1, width]")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
    except ImportError as error:  # pragma: no cover - isolated runtime path
        raise RuntimeError("auto_LiRPA is unavailable in this runtime") from error

    # PyTorch 2.7 otherwise exports LayerNorm as a long primitive chain under
    # the legacy opset.  auto_LiRPA 0.7.2 contains a dedicated
    # BoundLayerNormalization operator, exposed by ONNX opset 17.
    onnx_globals = None
    previous_opset = None
    try:
        from torch.onnx._globals import GLOBALS as onnx_globals

        previous_opset = onnx_globals.export_onnx_opset_version
        onnx_globals.export_onnx_opset_version = 17
    except (ImportError, AttributeError):  # pragma: no cover - version-specific
        pass
    try:
        bounded_module = BoundedModule(tail, center, device=center.device)
    finally:
        if onnx_globals is not None and previous_opset is not None:
            onnx_globals.export_onnx_opset_version = previous_opset
    perturbation = PerturbationLpNorm(norm=float("inf"), eps=epsilon)
    bounded_center = BoundedTensor(center, perturbation)
    point = bounded_module(center)
    lower, upper = bounded_module.compute_bounds(
        x=(bounded_center,), method=method
    )
    values = [point.reshape(()), lower.reshape(()), upper.reshape(())]
    if not all(torch.isfinite(value) for value in values):
        raise ValueError("auto_LiRPA returned non-finite output")
    point_value, lower_value, upper_value = [
        float(value.detach().cpu()) for value in values
    ]
    tolerance = 1e-5 * max(1.0, abs(point_value))
    if lower_value > point_value + tolerance or upper_value < point_value - tolerance:
        raise RuntimeError("certified interval does not contain the center evaluation")
    return LiRPABoundResult(
        method=method,
        norm="linf",
        epsilon=epsilon,
        point_value=point_value,
        lower_bound=lower_value,
        upper_bound=upper_value,
    )
