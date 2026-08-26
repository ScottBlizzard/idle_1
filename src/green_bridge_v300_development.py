"""GREEN v3.0.0 development-only scientific evaluation helpers.

The point estimator in this module consumes finite function values from an
isolated float64 copy of the frozen tail.  Automatic derivatives are evaluated
on a second isolated copy and are used only for certification and targets.
Confirmation records are never accepted by this module.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

import exp_green_bridge_gpt2 as legacy
from green_bridge_numerics import (
    ad_certified_enclosure_v200,
    ad_matched_bypass_compatibility_v200,
    ad_route_certificate_v200,
    active_envelope_contraction_bound,
)
from matched_bypass_gate import extrapolate_gate_jet, identify_gate, reconstruct_cotangent
from green_bridge_response_ad import (
    build_ad_response_functions_v200,
    response_gate_jet_forward_ad64,
    response_gate_jet_reverse_ad64,
)
from green_bridge_v300_directions import heldout_direction_panel_v300
from green_bridge_v300_prepare import _finite_gate_stencil_float64_v300
from green_bridge_v300_transport import (
    build_physical_path_control_v300,
    direct_path_control_ad_v300,
    joint_operator_prediction_v300,
    joint_target_ad_v300,
)
from green_bridge_whitebox_audit import (
    gradient_envelope_residual,
    layernorm_gate_gradient_autograd,
    layernorm_gate_gradient_formula,
    whitebox_A_coordinates,
)
from green_bridge_spec import GATE_RADIUS, SELECTED_GATES
from green_bridge_v300_spec import RECOVERABLE_RELATIVE_WIDTH_MAX


DEVELOPMENT_AUTHORIZATION_ID = "CODEX-GREEN-V300-DEVELOPMENT-v1-20260826"
SIGNAL_DIRECTION_CLASSES = ("heldout_in_frame", "heldout_mixed")


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative_width(estimate, radius: float) -> float:
    norm = float(np.linalg.norm(np.asarray(estimate, dtype=np.float64)))
    radius = float(radius)
    if norm == 0.0:
        return 0.0 if radius == 0.0 else math.inf
    return radius / norm


def _margin_vector(clean_suffix: int) -> np.ndarray:
    value = np.empty(100, dtype=np.float64)
    value[: clean_suffix + 1] = -1.0 / (clean_suffix + 1)
    value[clean_suffix + 1:] = 1.0 / (99 - clean_suffix)
    return value


def _finite_direct_panel(path_map, control_map, *, radius: float) -> dict:
    """Fine/coarse Richardson direct derivatives for all map coordinates."""
    torch = legacy.torch_module()
    device = getattr(path_map, "_green_device", None)

    def derivative(scale: float, mapping):
        k = int(getattr(path_map, "_green_input_dim"))
        eye = torch.eye(k, dtype=torch.float64, device=device) * (radius * scale)
        zero = torch.zeros((), dtype=torch.float64, device=device)
        plus = torch.vmap(lambda row: mapping(row, zero))(eye)
        minus = torch.vmap(lambda row: mapping(row, zero))(-eye)
        return (plus - minus) / (2.0 * radius * scale)

    path_base = derivative(1.0, path_map)
    path_half = derivative(0.5, path_map)
    path_quarter = derivative(0.25, path_map)
    control_base = derivative(1.0, control_map)
    control_half = derivative(0.5, control_map)
    control_quarter = derivative(0.25, control_map)

    path_coarse = (4.0 * path_half - path_base) / 3.0
    path_fine = (4.0 * path_quarter - path_half) / 3.0
    control_coarse = (4.0 * control_half - control_base) / 3.0
    control_fine = (4.0 * control_quarter - control_half) / 3.0
    return {
        "path_fine": path_fine.detach().cpu().numpy(),
        "path_coarse": path_coarse.detach().cpu().numpy(),
        "control_fine": control_fine.detach().cpu().numpy(),
        "control_coarse": control_coarse.detach().cpu().numpy(),
        "direct_fine": (path_fine - control_fine).detach().cpu().numpy(),
        "direct_coarse": (path_coarse - control_coarse).detach().cpu().numpy(),
    }


def _attach_map_metadata(path_map, control_map, *, device: str, input_dim: int) -> None:
    for mapping in (path_map, control_map):
        mapping._green_device = device
        mapping._green_input_dim = int(input_dim)


def _finite_response_triplet(finite_tail, anchor, frame, suffix_ids, gate_index: int,
                             base_h_x: float, selected_radius: float):
    path_map, control_map = build_ad_response_functions_v200(
        finite_tail, anchor, frame, suffix_ids, gate_index
    )
    _attach_map_metadata(path_map, control_map, device=str(anchor.resid_mid.device), input_dim=5)
    jets = {}
    controls = {}
    for scale in (selected_radius, selected_radius / 2.0, selected_radius / 4.0):
        jets[scale], controls[scale] = _finite_gate_stencil_float64_v300(
            path_map,
            control_map,
            base_h_x * scale,
            float(GATE_RADIUS) * scale,
        )
    half, quarter = selected_radius / 2.0, selected_radius / 4.0
    fine = extrapolate_gate_jet(jets[half], jets[quarter])
    coarse = extrapolate_gate_jet(jets[selected_radius], jets[half])
    fine_control = (4.0 * controls[quarter] - controls[half]) / 3.0
    coarse_control = (4.0 * controls[half] - controls[selected_radius]) / 3.0
    return {
        "fine": fine,
        "coarse": coarse,
        "fine_control_j": fine_control,
        "coarse_control_j": coarse_control,
    }


def _gate_atom_prediction(fine, identification, frame, directions) -> np.ndarray:
    atom_gradient = float(identification.A[4]) * np.asarray(frame[:, 4], dtype=np.float64)
    return np.outer(np.asarray(directions).T @ atom_gradient, np.asarray(fine.G))


def _unmatched_prediction(fine, frame, directions) -> np.ndarray:
    delta = np.asarray(fine.H_path, dtype=np.float64)
    c = np.asarray(fine.C, dtype=np.float64)
    c2 = float(c @ c)
    if c2 <= 0.0:
        return np.full((directions.shape[1], c.size), np.nan)
    coefficients = delta @ c / c2
    gradient = np.asarray(frame, dtype=np.float64) @ coefficients
    return np.outer(np.asarray(directions).T @ gradient, np.asarray(fine.G))


def evaluate_gate_v300(*, model, finite_tail, ad_tail, anchor, frame, suffix_ids,
                       gate_slot: int, directions: np.ndarray, epsilon_y: float,
                       base_h_x: float, selected_radius: float) -> dict:
    """Evaluate one gate-system without exposing held-out targets to identification."""
    gate_index = int(SELECTED_GATES[gate_slot])
    residual = legacy._selected_numpy(anchor, "resid_mid")
    gamma = model.blocks[10].ln2.w.detach().double().cpu().numpy()
    beta = model.blocks[10].ln2.b.detach().double().cpu().numpy()
    w_in = model.blocks[10].mlp.W_in.detach().double().cpu().numpy()
    b_in = model.blocks[10].mlp.b_in.detach().double().cpu().numpy()
    gradient = layernorm_gate_gradient_formula(
        residual, gamma, w_in[:, gate_index], eps=float(model.cfg.eps)
    )
    gradient_ad = layernorm_gate_gradient_autograd(
        residual,
        gamma,
        w_in[:, gate_index],
        eps=float(model.cfg.eps),
        ln_bias=beta,
        mlp_input_bias=float(b_in[gate_index]),
    )
    gradient_error = math.nextafter(
        1.0e-10 + float(np.linalg.norm(gradient - gradient_ad)), math.inf
    )
    envelope_error = math.nextafter(
        gradient_envelope_residual(frame, gradient)["absolute"] + gradient_error,
        math.inf,
    )
    wb_a = whitebox_A_coordinates(frame, gradient)

    response = _finite_response_triplet(
        finite_tail,
        anchor,
        frame,
        suffix_ids,
        gate_index,
        base_h_x,
        selected_radius,
    )
    ad_path, ad_control = build_ad_response_functions_v200(
        ad_tail, anchor, frame, suffix_ids, gate_index
    )
    certificate = ad_route_certificate_v200(
        response_gate_jet_forward_ad64(ad_path, ad_control),
        response_gate_jet_reverse_ad64(ad_path, ad_control),
    )
    enclosure = ad_certified_enclosure_v200(
        response["fine"],
        certificate,
        epsilon_y=epsilon_y,
        fine_h_x=base_h_x * selected_radius / 2.0,
        fine_h_z=float(GATE_RADIUS) * selected_radius / 2.0,
    )
    factorization = ad_matched_bypass_compatibility_v200(certificate, wb_a)

    direct_ad = direct_path_control_ad_v300(
        ad_tail, anchor, directions, suffix_ids, gate_index
    )
    finite_path, finite_control = build_physical_path_control_v300(
        finite_tail, anchor, directions, suffix_ids, gate_index
    )
    _attach_map_metadata(
        finite_path,
        finite_control,
        device=str(anchor.resid_mid.device),
        input_dim=int(directions.shape[1]),
    )
    direct = _finite_direct_panel(
        finite_path,
        finite_control,
        radius=base_h_x * selected_radius,
    )
    target = np.asarray(direct["direct_fine"], dtype=np.float64)
    target_ad = np.asarray(direct_ad["direct"], dtype=np.float64)
    route_radius = float(direct_ad["J_path"]["radius"] + direct_ad["J_control"]["radius"])
    endpoint_radius = float(
        6.0 * math.sqrt(target.shape[1]) * epsilon_y
        / (base_h_x * selected_radius / 2.0)
    )
    target_bounds = np.asarray([
        math.nextafter(float(np.linalg.norm(left - right)) + route_radius + endpoint_radius, math.inf)
        for left, right in zip(target, target_ad)
    ])

    identification = None
    coarse_identification = None
    if enclosure.inverse_admissible:
        try:
            identification = identify_gate(response["fine"])
            coarse_identification = identify_gate(response["coarse"])
        except ValueError:
            identification = None
            coarse_identification = None

    numerical_valid = bool(
        certificate.passed
        and direct_ad["route_passed"]
        and np.isfinite(target).all()
        and np.isfinite(target_bounds).all()
    )
    structural_valid = bool(factorization["passed"])
    predictions = {}
    coarse_prediction = np.full_like(target, np.nan)
    if identification is not None:
        gradient_hat = reconstruct_cotangent(frame, identification.A)
        predictions["matched"] = np.outer(directions.T @ gradient_hat, response["fine"].G)
        predictions["gate_atom_only"] = _gate_atom_prediction(
            response["fine"], identification, frame, directions
        )
        predictions["unmatched_path_mixed"] = _unmatched_prediction(
            response["fine"], frame, directions
        )
        if coarse_identification is not None:
            coarse_gradient = reconstruct_cotangent(frame, coarse_identification.A)
            coarse_prediction = np.outer(directions.T @ coarse_gradient, response["coarse"].G)
    else:
        for name in ("matched", "gate_atom_only", "unmatched_path_mixed"):
            predictions[name] = np.full_like(target, np.nan)
    predictions["zero"] = np.zeros_like(target)
    predictions["raw_path_jacobian"] = np.asarray(direct["path_fine"], dtype=np.float64)

    total_bounds = []
    if identification is not None:
        for direction, target_bound in zip(directions.T, target_bounds):
            prediction_radius = active_envelope_contraction_bound(
                1.0,
                float(np.linalg.norm(frame.T @ direction)),
                float(np.linalg.norm(direction)),
                enclosure.epsilon_P_F,
                float(np.linalg.norm(response["fine"].G)),
                enclosure.epsilon_G,
                envelope_error,
            )
            total_bounds.append(math.nextafter(prediction_radius + float(target_bound), math.inf))
    else:
        total_bounds = [math.inf] * directions.shape[1]
    total_bounds = np.asarray(total_bounds, dtype=np.float64)

    theorem_pass = bool(
        structural_valid
        and identification is not None
        and np.all(
            np.linalg.norm(predictions["matched"] - target, axis=1)
            <= np.nextafter(total_bounds, np.inf)
        )
    )
    if numerical_valid and identification is not None and not theorem_pass:
        structural_valid = False

    curvature_width = _relative_width(response["fine"].C, enclosure.epsilon_C)
    response_width = _relative_width(response["fine"].G, enclosure.epsilon_G)
    operator_width = (
        _relative_width(identification.P, enclosure.epsilon_P_F)
        if identification is not None else math.inf
    )
    recoverable = bool(
        numerical_valid
        and structural_valid
        and identification is not None
        and curvature_width <= RECOVERABLE_RELATIVE_WIDTH_MAX
        and response_width <= RECOVERABLE_RELATIVE_WIDTH_MAX
        and operator_width <= RECOVERABLE_RELATIVE_WIDTH_MAX
    )
    ad_upper = math.nextafter(
        (float(np.linalg.norm(certificate.reference.G)) + certificate.route_radius_G)
        * (float(np.linalg.norm(gradient)) + gradient_error),
        math.inf,
    )
    direct_floor = float(np.linalg.norm(target_bounds))
    certified_null = bool(
        numerical_valid and structural_valid and not recoverable and ad_upper <= direct_floor
    )
    if not numerical_valid:
        gate_class = "numerical-invalid"
    elif not structural_valid:
        gate_class = "structural-contradiction"
    elif recoverable:
        gate_class = "recoverable"
    elif certified_null:
        gate_class = "certified-numerical-null"
    else:
        gate_class = "unresolved"

    signal = slice(0, 8)
    null = slice(8, 10)
    signal_target_norm = float(np.linalg.norm(target[signal]))
    signal_bound = float(np.linalg.norm(target_bounds[signal]))
    signal_numerator = (
        float(np.linalg.norm(predictions["matched"][signal] - target[signal]))
        if np.isfinite(predictions["matched"]).all() else math.inf
    )
    signal_denominator = max(signal_target_norm, float(np.linalg.norm(total_bounds[signal])))
    direct_error = (
        0.0 if signal_denominator == 0.0 and signal_numerator == 0.0
        else (math.inf if signal_denominator == 0.0 else signal_numerator / signal_denominator)
    )
    null_denominator = max(signal_target_norm, signal_bound)
    null_norm = float(np.linalg.norm(target[null]))
    null_leakage = (
        0.0 if null_denominator == 0.0 and null_norm == 0.0
        else (math.inf if null_denominator == 0.0 else null_norm / null_denominator)
    )
    nonnull = bool(signal_bound == 0.0 and signal_target_norm > 0.0
                   or signal_bound > 0.0 and signal_target_norm / signal_bound >= 4.0)

    baseline_errors = {}
    for name, prediction in predictions.items():
        numerator = float(np.linalg.norm(prediction[signal] - target[signal]))
        denominator = max(signal_target_norm, signal_bound)
        baseline_errors[name] = (
            0.0 if denominator == 0.0 and numerator == 0.0
            else (math.inf if denominator == 0.0 else numerator / denominator)
        )
    coarse_error = math.inf
    if np.isfinite(coarse_prediction).all():
        denominator = max(signal_target_norm, signal_bound)
        numerator = float(np.linalg.norm(coarse_prediction[signal] - target[signal]))
        coarse_error = 0.0 if denominator == 0.0 and numerator == 0.0 else (
            math.inf if denominator == 0.0 else numerator / denominator
        )

    return {
        "gate_slot": gate_slot,
        "gate_index": gate_index,
        "gate_class": gate_class,
        "numerical_valid": numerical_valid,
        "structural_valid": structural_valid,
        "ad_route_passed": bool(certificate.passed and direct_ad["route_passed"]),
        "factorization_passed": bool(factorization["passed"]),
        "direct_theorem_passed": theorem_pass,
        "curvature_norm": float(np.linalg.norm(response["fine"].C)),
        "epsilon_C": enclosure.epsilon_C,
        "response_norm": float(np.linalg.norm(response["fine"].G)),
        "epsilon_G": enclosure.epsilon_G,
        "operator_norm": (
            float(np.linalg.norm(identification.P)) if identification is not None else 0.0
        ),
        "epsilon_P_F": enclosure.epsilon_P_F,
        "curvature_relative_width": curvature_width,
        "response_relative_width": response_width,
        "operator_relative_width": operator_width,
        "exact_operator_upper": ad_upper,
        "direct_numerical_floor": direct_floor,
        "direct_error": direct_error,
        "coarse_direct_error": coarse_error,
        "null_leakage": null_leakage,
        "nonnull": nonnull,
        "unresolved_bound": ad_upper if gate_class == "unresolved" else 0.0,
        "target_signal_norm": signal_target_norm,
        "target_signal_bound": signal_bound,
        "target_bounds": target_bounds,
        "total_bounds": total_bounds,
        "target": target,
        "predictions": predictions,
        "gradient_hat": (
            reconstruct_cotangent(frame, identification.A) if identification is not None else None
        ),
        "coarse_gradient_hat": (
            reconstruct_cotangent(frame, coarse_identification.A)
            if coarse_identification is not None else None
        ),
        "gate_response": np.asarray(response["fine"].G, dtype=np.float64),
        "coarse_gate_response": np.asarray(response["coarse"].G, dtype=np.float64),
        "whitebox_gradient": gradient,
        "baseline_errors": baseline_errors,
    }


def heldout_directions_v300(frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
    panel = heldout_direction_panel_v300(frame)
    directions = np.concatenate((panel["in_frame"], panel["mixed"], panel["null"]), axis=1)
    classes = (["heldout_in_frame"] * 4 + ["heldout_mixed"] * 4
               + ["heldout_complement_null"] * 2)
    return directions, classes


def joint_scalar_v300(gates: Iterable[dict], *, direction: np.ndarray,
                      contrast: np.ndarray, ad_tail, anchor, suffix_ids) -> dict:
    gate_rows = list(gates)
    center_terms = []
    coarse_terms = []
    bounds = []
    unresolved_bound = 0.0
    for row in gate_rows:
        scale = float(np.linalg.norm(contrast)) * float(np.linalg.norm(direction))
        if row["gate_class"] == "recoverable":
            center_terms.append(
                float(contrast @ row["gate_response"])
                * float(np.asarray(row["gradient_hat"]) @ direction)
            )
            if row["coarse_gradient_hat"] is not None:
                coarse_terms.append(
                    float(contrast @ row["coarse_gate_response"])
                    * float(np.asarray(row["coarse_gradient_hat"]) @ direction)
                )
            else:
                coarse_terms.append(center_terms[-1])
            bounds.append(scale * float(row["epsilon_P_F"]))
        else:
            center_terms.append(0.0)
            coarse_terms.append(0.0)
            bound = scale * float(row["exact_operator_upper"])
            bounds.append(bound)
            if row["gate_class"] == "unresolved":
                unresolved_bound += bound
    target = joint_target_ad_v300(ad_tail, anchor, suffix_ids, direction, contrast)
    center = float(math.fsum(center_terms))
    coarse = float(math.fsum(coarse_terms))
    bound = math.nextafter(math.fsum(bounds) + float(target["radius"]), math.inf)
    denominator = max(abs(float(target["midpoint"])), bound)
    numerator = abs(center - float(target["midpoint"]))
    error = 0.0 if denominator == 0.0 and numerator == 0.0 else (
        math.inf if denominator == 0.0 else numerator / denominator
    )
    return {
        "center": center,
        "coarse_center": coarse,
        "bound": bound,
        "unresolved_bound": unresolved_bound,
        "target": float(target["midpoint"]),
        "target_bound": float(target["radius"]),
        "target_certified": bool(target["passed"]),
        "error": error,
    }
