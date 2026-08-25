"""Outcome-blind numerical-certification module for GREEN v2.0.0.

This module is intentionally isolated from cell scoring and Parquet analysis.
It accepts differentiable local response functions and metadata-only records.
"""
from __future__ import annotations

import hashlib
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math

import numpy as np

from matched_bypass_gate import GateJet
from green_bridge_numerics import ad_route_certificate_v200


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("PyTorch 2.7.1 is required for the v2.0.0 AD audit") from exc
    return torch


def _config_payload(cfg):
    def clean(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
        return str(value)
    return clean(vars(cfg))


def active_model_integrity_hash_v200(model) -> dict:
    """Hash active parameter bytes, buffer bytes, and serialized configuration."""
    def tensor_hash(rows):
        digest = hashlib.sha256()
        for name, value in rows:
            array = value.detach().contiguous().cpu().numpy()
            digest.update(name.encode("utf-8")); digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes()); digest.update(array.tobytes())
        return digest.hexdigest()
    cfg = json.dumps(_config_payload(model.cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "parameter_hash": tensor_hash(model.named_parameters()),
        "buffer_hash": tensor_hash(model.named_buffers()),
        "config_hash": hashlib.sha256(cfg.encode("utf-8")).hexdigest(),
    }


@dataclass
class IsolatedADTailV200:
    block10: object
    block11: object
    ln_final: object
    unembed: object
    cfg: object
    integrity_before: dict
    integrity_after: dict | None = None
    active_model_unchanged: bool | None = None

    def modules(self):
        for root in (self.block10, self.block11, self.ln_final, self.unembed):
            yield from root.modules()


@contextmanager
def isolated_ad_tail_v200(scientific_model, anchor=None):
    """Create and destroy an isolated float64 local tail without mutating the model."""
    torch = _torch()
    before = active_model_integrity_hash_v200(scientific_model)
    cfg = copy.deepcopy(scientific_model.cfg)
    cfg.dtype = torch.float64
    ad_tail = IsolatedADTailV200(
        copy.deepcopy(scientific_model.blocks[10]),
        copy.deepcopy(scientific_model.blocks[11]),
        copy.deepcopy(scientific_model.ln_final),
        copy.deepcopy(scientific_model.unembed),
        cfg,
        before,
    )
    for module in ad_tail.modules():
        if hasattr(module, "cfg"):
            module.cfg = cfg
    for root in (ad_tail.block10, ad_tail.block11, ad_tail.ln_final, ad_tail.unembed):
        root.to(device=next(scientific_model.parameters()).device, dtype=torch.float64)
        root.eval()
    try:
        yield ad_tail
    finally:
        after = active_model_integrity_hash_v200(scientific_model)
        ad_tail.integrity_after = after
        ad_tail.active_model_unchanged = before == after
        for name in ("block10", "block11", "ln_final", "unembed"):
            setattr(ad_tail, name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_ad_response_functions_v200(ad_tail, anchor, frame, suffix_ids, gate_index: int):
    """Build isolated float64 path/control maps for one anchor and gate.

    ``ad_tail`` is an isolated float64 tail. No AD tensor is a point estimator.
    """
    torch = _torch()
    try:
        from transformer_lens.utilities import apply_softcap
        from transformer_lens.utilities.addmm import batch_addmm
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("pinned TransformerLens is required for the AD audit") from exc
    residual = anchor.resid_mid.detach().double()
    # Re-anchor the isolated map in float64. Casting cached float32 pre/post
    # values would put the path and matched control at different local points.
    with torch.no_grad():
        anchored_normalized = ad_tail.block10.ln2(residual)
        anchored_pre = batch_addmm(
            ad_tail.block10.mlp.b_in, ad_tail.block10.mlp.W_in,
            anchored_normalized,
        )
        anchored_post = ad_tail.block10.mlp.act_fn(anchored_pre)
    frame64 = torch.as_tensor(frame, dtype=torch.float64, device=residual.device)
    suffix = torch.as_tensor(suffix_ids, dtype=torch.long, device=residual.device)
    position = int(anchor.final_positions[0].item())
    sequence_mask = torch.nn.functional.one_hot(
        torch.tensor(position, device=residual.device), num_classes=residual.shape[1]
    ).to(torch.float64)
    gate_mask = torch.nn.functional.one_hot(
        torch.tensor(int(gate_index), device=residual.device), num_classes=ad_tail.cfg.d_mlp
    ).to(torch.float64)

    def evaluate(x, z, *, mode: str):
        physical = frame64 @ x
        resid_mid = residual + sequence_mask[None, :, None] * physical[None, None, :]
        block10 = ad_tail.block10
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
        resid = ad_tail.block11(resid_post)
        normalized_final = ad_tail.ln_final(resid)
        logits = ad_tail.unembed(normalized_final)
        logits = apply_softcap(logits, ad_tail.cfg.output_logits_soft_cap)
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
    """Serialize route consistency and coarse/fine diagnostics without a hard overlap gate."""
    certificate = ad_route_certificate_v200(ad_forward, ad_reverse)
    coarse = _jet_objects(coarse_jet)
    fine = _jet_objects(fine_jet)
    reference = _jet_objects(certificate.reference)
    rows = {}
    for name in ("G", "C", "J", "delta_H"):
        ref = reference[name]
        coarse_difference = (
            np.linalg.norm(ref - coarse[name], axis=1) if name == "delta_H"
            else float(np.linalg.norm(ref - coarse[name]))
        )
        fine_difference = (
            np.linalg.norm(ref - fine[name], axis=1) if name == "delta_H"
            else float(np.linalg.norm(ref - fine[name]))
        )
        suffix = "delta_H" if name == "delta_H" else name
        route_difference = getattr(certificate, f"route_difference_{suffix}")
        route_radius = getattr(certificate, f"route_radius_{suffix}")
        route_pass = getattr(certificate, f"route_pass_{suffix}")
        rows[name] = {
            "route_difference": np.asarray(route_difference).tolist(),
            "route_radius": np.asarray(route_radius).tolist(),
            "coarse_difference": np.asarray(coarse_difference).tolist(),
            "fine_difference": np.asarray(fine_difference).tolist(),
            "route_pass": bool(np.all(route_pass)),
            "coarse_fine_diagnostic_only": True,
            "active_admissibility_gate": False,
        }
    return {"passed": certificate.passed, "objects": rows}
