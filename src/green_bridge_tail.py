"""Exact block-10-to-year-logit tail for matched-bypass predictor endpoints.

Torch and TransformerLens are imported lazily so CPU-only contract tests can run
without the server environment.  Every manual-tail result must pass the frozen
full-hook equivalence audit before scientific scores are produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from green_bridge_spec import PROBE_FRAME_DIM, SELECTED_GATES


TailMode = Literal["path", "control", "joint"]


@dataclass(frozen=True)
class TailAnchor:
    resid_mid: Any
    pre: Any
    post: Any
    resid_post: Any
    year_logits: Any
    final_positions: Any
    system: str
    mlp8_out: Any | None = None


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on the server
        raise RuntimeError("green-bridge tail requires the pinned torch environment") from exc
    return torch


def _batch_addmm(bias, weight, value):
    try:
        from transformer_lens.utilities.addmm import batch_addmm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pinned TransformerLens is required") from exc
    return batch_addmm(bias, weight, value)


def gather_year_logits(model, logits, final_positions, suffix_token_ids):
    torch = _torch()
    rows = torch.arange(logits.shape[0], device=logits.device)
    final = logits[rows, final_positions]
    return final.index_select(-1, suffix_token_ids)


def capture_tail_anchor(
    model,
    tokens,
    suffix_token_ids,
    *,
    system: str,
    block8_patch=None,
) -> TailAnchor:
    """Capture the exact anchors required by the manual tail.

    ``block8_patch`` has shape ``[batch, d_model]`` and is applied only at each
    example's final token position.
    """
    torch = _torch()
    final_positions = torch.full(
        (tokens.shape[0],), tokens.shape[1] - 1, dtype=torch.long, device=tokens.device
    )
    hooks = []
    if block8_patch is not None:
        def patch_mlp8(value, hook):
            result = value.clone()
            rows = torch.arange(len(result), device=result.device)
            result[rows, final_positions, :] = block8_patch
            return result
        hooks.append(("blocks.8.hook_mlp_out", patch_mlp8))
    names = {
        "blocks.8.hook_mlp_out",
        "blocks.10.hook_resid_mid",
        "blocks.10.mlp.hook_pre",
        "blocks.10.mlp.hook_post",
        "blocks.10.hook_resid_post",
    }
    try:
        # ``run_with_cache`` forwards unknown keywords to ``model.forward``;
        # interventions therefore belong in the outer hook context, not in a
        # fictitious ``fwd_hooks`` keyword argument.
        with model.hooks(fwd_hooks=hooks):
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in names,
            )
    finally:
        model.reset_hooks()
    return TailAnchor(
        resid_mid=cache["blocks.10.hook_resid_mid"].detach(),
        pre=cache["blocks.10.mlp.hook_pre"].detach(),
        post=cache["blocks.10.mlp.hook_post"].detach(),
        resid_post=cache["blocks.10.hook_resid_post"].detach(),
        year_logits=gather_year_logits(model, logits, final_positions, suffix_token_ids).detach(),
        final_positions=final_positions.detach(),
        system=system,
        mlp8_out=cache["blocks.8.hook_mlp_out"][
            torch.arange(tokens.shape[0], device=tokens.device), final_positions
        ].detach(),
    )


class GreenBridgeTail:
    """Continue exactly from cached block-10 ``resid_mid`` anchors."""

    def __init__(self, model, residual_basis, suffix_token_ids, selected_gates=SELECTED_GATES):
        torch = _torch()
        self.model = model
        self.U = residual_basis
        self.suffix_ids = suffix_token_ids
        self.gates = tuple(int(gate) for gate in selected_gates)
        if tuple(self.U.shape) != (model.cfg.d_model, PROBE_FRAME_DIM):
            raise ValueError(
                f"legacy coordinate frame must have shape [768,{PROBE_FRAME_DIM}], got {self.U.shape}"
            )
        if len(self.gates) != 10 or len(set(self.gates)) != 10:
            raise ValueError("exactly ten unique selected gates are required")
        if min(self.gates) < 0 or max(self.gates) >= model.cfg.d_mlp:
            raise ValueError("selected gate index is outside the MLP width")
        if tuple(self.suffix_ids.shape) != (100,):
            raise ValueError("suffix_token_ids must have shape [100]")
        self.gate_index = torch.tensor(self.gates, dtype=torch.long, device=self.U.device)

    def _project_x(self, x):
        return x @ self.U.T

    def evaluate(
        self,
        anchor: TailAnchor,
        x,
        z,
        *,
        mode: TailMode,
        gate_slot: int | None = None,
        subtract_residual_bypass: bool = False,
    ):
        torch = _torch()
        if mode not in {"path", "control", "joint"}:
            raise ValueError(f"unknown tail mode {mode}")
        batch = anchor.resid_mid.shape[0]
        if tuple(x.shape) != (batch, PROBE_FRAME_DIM):
            raise ValueError(f"x must have shape [{batch},{PROBE_FRAME_DIM}]")
        return self.evaluate_physical(
            anchor,
            self._project_x(x),
            z,
            mode=mode,
            gate_slot=gate_slot,
            subtract_residual_bypass=subtract_residual_bypass,
        )

    def evaluate_physical(
        self,
        anchor: TailAnchor,
        residual_delta,
        z,
        *,
        mode: TailMode,
        gate_slot: int | None = None,
        subtract_residual_bypass: bool = False,
    ):
        """Evaluate an intervention expressed directly in residual coordinates."""
        torch = _torch()
        if mode not in {"path", "control", "joint"}:
            raise ValueError(f"unknown tail mode {mode}")
        batch = anchor.resid_mid.shape[0]
        if tuple(residual_delta.shape) != (batch, self.model.cfg.d_model):
            raise ValueError(
                f"residual_delta must have shape [{batch},{self.model.cfg.d_model}]"
            )
        if mode == "joint":
            if tuple(z.shape) != (batch, 10):
                raise ValueError(f"joint z must have shape [{batch},10]")
        else:
            if gate_slot is None or not 0 <= gate_slot < 10:
                raise ValueError("one-gate modes require gate_slot in [0,10)")
            if tuple(z.shape) not in {(batch,), (batch, 1)}:
                raise ValueError(f"one-gate z must have shape [{batch}] or [{batch},1]")
            z = z.reshape(batch)

        rows = torch.arange(batch, device=anchor.resid_mid.device)
        positions = anchor.final_positions
        residual_delta = residual_delta.to(anchor.resid_mid.dtype)
        resid_mid = anchor.resid_mid.clone()
        resid_mid[rows, positions, :] += residual_delta
        block10 = self.model.blocks[10]
        normalized = block10.ln2(resid_mid)
        pre = _batch_addmm(block10.mlp.b_in, block10.mlp.W_in, normalized)
        if mode == "joint":
            pre[rows[:, None], positions[:, None], self.gate_index[None, :]] += z.to(pre.dtype)
        elif mode == "path":
            gate = self.gates[gate_slot]
            pre[rows, positions, gate] += z.to(pre.dtype)

        live_post = block10.mlp.act_fn(pre)
        post = live_post.clone()
        post[rows, positions, :] = anchor.post[rows, positions, :]
        if mode == "joint":
            post[rows[:, None], positions[:, None], self.gate_index[None, :]] = live_post[
                rows[:, None], positions[:, None], self.gate_index[None, :]
            ]
        elif mode == "path":
            gate = self.gates[gate_slot]
            post[rows, positions, gate] = live_post[rows, positions, gate]
        else:
            gate = self.gates[gate_slot]
            controlled_pre = anchor.pre[rows, positions, gate] + z.to(anchor.pre.dtype)
            post[rows, positions, gate] = block10.mlp.act_fn(controlled_pre)

        mlp_out = _batch_addmm(block10.mlp.b_out, block10.mlp.W_out, post)
        resid_post = resid_mid + mlp_out
        if subtract_residual_bypass:
            resid_post[rows, positions, :] -= residual_delta

        # Block 11 is the complete arbitrary downstream transformer tail.
        resid = self.model.blocks[11](resid_post)
        normalized_final = self.model.ln_final(resid)
        final = normalized_final[rows, positions, :]
        W_selected = self.model.W_U.index_select(1, self.suffix_ids)
        logits = final @ W_selected
        if getattr(self.model, "b_U", None) is not None:
            logits = logits + self.model.b_U.index_select(0, self.suffix_ids)
        return logits


def max_abs_and_rms(actual, expected) -> dict[str, float]:
    torch = _torch()
    difference = (actual.double() - expected.double()).reshape(-1)
    return {
        "max_abs": float(difference.abs().max().item()),
        "rms": float(torch.sqrt(torch.mean(difference**2)).item()),
    }
