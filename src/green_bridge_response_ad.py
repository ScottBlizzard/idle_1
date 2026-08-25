"""Prepare-only float64 derivative enclosure audit for GREEN v2.0.0.

This module is intentionally isolated from cell scoring and Parquet analysis.
It accepts differentiable local response functions and metadata-only records.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from matched_bypass_gate import GateJet


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("PyTorch 2.7.1 is required for the v2.0.0 AD audit") from exc
    return torch


def build_ad_response_functions_v200(model, anchor, frame, suffix_ids, gate_index: int):
    """Build isolated float64 path/control maps for one anchor and gate.

    The caller must place the model in float64 for the duration of the audit.
    No tensor returned by these functions is used by the scientific estimator.
    """
    torch = _torch()
    try:
        from transformer_lens.utilities import apply_softcap
        from transformer_lens.utilities.addmm import batch_addmm
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("pinned TransformerLens is required for the AD audit") from exc
    residual = anchor.resid_mid.detach().double()
    anchored_pre = anchor.pre.detach().double()
    anchored_post = anchor.post.detach().double()
    frame64 = torch.as_tensor(frame, dtype=torch.float64, device=residual.device)
    suffix = torch.as_tensor(suffix_ids, dtype=torch.long, device=residual.device)
    position = int(anchor.final_positions[0].item())
    sequence_mask = torch.nn.functional.one_hot(
        torch.tensor(position, device=residual.device), num_classes=residual.shape[1]
    ).to(torch.float64)
    gate_mask = torch.nn.functional.one_hot(
        torch.tensor(int(gate_index), device=residual.device), num_classes=model.cfg.d_mlp
    ).to(torch.float64)

    def evaluate(x, z, *, mode: str):
        physical = frame64 @ x
        resid_mid = residual + sequence_mask[None, :, None] * physical[None, None, :]
        block10 = model.blocks[10]
        normalized = block10.ln2(resid_mid)
        pre = batch_addmm(block10.mlp.b_in, block10.mlp.W_in, normalized)
        if mode == "path":
            selected_pre = pre[0, position, gate_index] + z
        elif mode == "control":
            selected_pre = anchored_pre[0, position, gate_index] + z
        else:
            raise ValueError(mode)
        selected_post = block10.mlp.act_fn(selected_pre)
        anchored_value = anchored_post[0, position, gate_index]
        post = anchored_post + (
            sequence_mask[None, :, None]
            * gate_mask[None, None, :]
            * (selected_post - anchored_value)
        )
        mlp_out = batch_addmm(block10.mlp.b_out, block10.mlp.W_out, post)
        resid_post = resid_mid + mlp_out
        resid = model.blocks[11](resid_post)
        normalized_final = model.ln_final(resid)
        logits = model.unembed(normalized_final)
        logits = apply_softcap(logits, model.cfg.output_logits_soft_cap)
        return logits[0, position].index_select(0, suffix)

    def path_evaluate(x, z):
        return evaluate(x, z, mode="path")

    def control_evaluate(x, z):
        return evaluate(x, z, mode="control")

    # The derivative entry points use this prepare-only attribute to allocate
    # their independent variables on the same device as the frozen anchor.
    path_evaluate._green_device = residual.device
    control_evaluate._green_device = residual.device
    return path_evaluate, control_evaluate


def _to_jet(G, C, J, HP, HC) -> GateJet:
    arrays = [value.detach().cpu().double().numpy() for value in (G, C, J, HP, HC)]
    return GateJet(arrays[0], arrays[1], arrays[2].T, arrays[3].T, arrays[4].T)


def response_gate_jet_forward_ad64(path_evaluate, control_evaluate, *, frame_dim: int = 5) -> GateJet:
    """Compute a center GateJet by forward-over-forward automatic differentiation."""
    torch = _torch()
    device = getattr(path_evaluate, "_green_device", None)
    x = torch.zeros(frame_dim, dtype=torch.float64, device=device)
    z = torch.zeros((), dtype=torch.float64, device=device)
    G = torch.func.jacfwd(path_evaluate, argnums=1)(x, z)
    C = torch.func.jacfwd(torch.func.jacfwd(path_evaluate, argnums=1), argnums=1)(x, z)
    J = torch.func.jacfwd(path_evaluate, argnums=0)(x, z)
    HP = torch.func.jacfwd(torch.func.jacfwd(path_evaluate, argnums=1), argnums=0)(x, z)
    HC = torch.func.jacfwd(torch.func.jacfwd(control_evaluate, argnums=1), argnums=0)(x, z)
    return _to_jet(G, C, J, HP, HC)


def response_gate_jet_reverse_ad64(path_evaluate, control_evaluate, *, frame_dim: int = 5) -> GateJet:
    """Compute the same GateJet by reverse-over-forward differentiation."""
    torch = _torch()
    device = getattr(path_evaluate, "_green_device", None)
    x = torch.zeros(frame_dim, dtype=torch.float64, device=device)
    z = torch.zeros((), dtype=torch.float64, device=device)
    G = torch.func.jacrev(path_evaluate, argnums=1)(x, z)
    C = torch.func.jacrev(torch.func.jacfwd(path_evaluate, argnums=1), argnums=1)(x, z)
    J = torch.func.jacrev(path_evaluate, argnums=0)(x, z)
    HP = torch.func.jacrev(torch.func.jacfwd(path_evaluate, argnums=1), argnums=0)(x, z)
    HC = torch.func.jacrev(torch.func.jacfwd(control_evaluate, argnums=1), argnums=0)(x, z)
    return _to_jet(G, C, J, HP, HC)


def select_ad_audit_panel_v200(records) -> list[dict]:
    """Select the fixed 2 systems x 10 gates x 2 distance strata."""
    panel = []
    for system in ("tar", "pat"):
        for gate_slot in range(10):
            for distance_bin in ("near", "far"):
                candidates = [row for row in records if row.distance_bin == distance_bin]
                if not candidates:
                    raise ValueError(f"empty AD stratum {system}/{gate_slot}/{distance_bin}")
                selected = min(
                    candidates,
                    key=lambda row: hashlib.sha256(
                        f"green-v200-ad-audit|{row.pair_digest}|{system}|{gate_slot}|{distance_bin}".encode("utf-8")
                    ).hexdigest(),
                )
                panel.append({
                    "pair_digest": selected.pair_digest,
                    "system": system,
                    "gate_slot": gate_slot,
                    "distance_bin": distance_bin,
                })
    if len(panel) != 40 or len({tuple(sorted(row.items())) for row in panel}) != 40:
        raise AssertionError("AD panel must contain exactly forty unique strata")
    return panel


def _jet_objects(jet: GateJet) -> dict[str, np.ndarray]:
    return {
        "G": np.asarray(jet.G, dtype=np.float64),
        "C": np.asarray(jet.C, dtype=np.float64),
        "J": np.asarray(jet.J_path, dtype=np.float64),
        "delta_H": np.asarray(jet.H_path, dtype=np.float64) - np.asarray(jet.H_control, dtype=np.float64),
    }


def audit_richardson_enclosure_v200(ad_forward, ad_reverse, coarse_jet, fine_jet, coarse_bounds, fine_bounds) -> dict:
    """Falsify the coarse/fine balls against two independent AD routes."""
    forward = _jet_objects(ad_forward)
    reverse = _jet_objects(ad_reverse)
    coarse = _jet_objects(coarse_jet)
    fine = _jet_objects(fine_jet)
    bound_names = {
        "G": "epsilon_G", "C": "epsilon_C", "J": "epsilon_J",
        "delta_H": "epsilon_delta_H",
    }
    rows = {}
    passed = True
    for name in ("G", "C", "J", "delta_H"):
        fwd = forward[name]
        rev = reverse[name]
        route_difference = (
            np.linalg.norm(fwd - rev, axis=1) if name == "delta_H"
            else float(np.linalg.norm(fwd - rev))
        )
        scale = max(1.0, float(np.linalg.norm(fwd)), float(np.linalg.norm(rev)))
        route_bound = math.nextafter(64.0 * np.finfo(np.float64).eps * scale, math.inf)
        route_pass = bool(np.all(np.asarray(route_difference) <= route_bound))
        reference = 0.5 * (fwd + rev)
        coarse_difference = (
            np.linalg.norm(reference - coarse[name], axis=1) if name == "delta_H"
            else float(np.linalg.norm(reference - coarse[name]))
        )
        fine_difference = (
            np.linalg.norm(reference - fine[name], axis=1) if name == "delta_H"
            else float(np.linalg.norm(reference - fine[name]))
        )
        coarse_bound = np.asarray(getattr(coarse_bounds, bound_names[name]))
        fine_bound = np.asarray(getattr(fine_bounds, bound_names[name]))
        coarse_pass = bool(np.all(np.asarray(coarse_difference) <= np.nextafter(coarse_bound, np.inf)))
        fine_pass = bool(np.all(np.asarray(fine_difference) <= np.nextafter(fine_bound, np.inf)))
        object_pass = route_pass and coarse_pass and fine_pass
        passed = passed and object_pass
        rows[name] = {
            "route_difference": np.asarray(route_difference).tolist(),
            "route_bound": route_bound,
            "coarse_difference": np.asarray(coarse_difference).tolist(),
            "coarse_bound": coarse_bound.tolist(),
            "fine_difference": np.asarray(fine_difference).tolist(),
            "fine_bound": fine_bound.tolist(),
            "route_pass": route_pass, "coarse_pass": coarse_pass,
            "fine_pass": fine_pass, "passed": object_pass,
        }
    return {"passed": bool(passed), "objects": rows}
