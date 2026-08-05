"""Run the CPU-only ASG-RDAG structural-identification verification.

The runner is deliberately independent of transformer libraries and GPUs.  It
produces a machine-readable artifact proving that the implemented estimator
matches the analytic restricted-class inverse before any model experiment is
considered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mixed_path_identification import (
    active_edge_inverse,
    evaluate_cartesian_corners,
    factorization_residuals,
    fit_cartesian_structural_tensors,
    structural_response_energy,
)


def quadratic_asg(A: np.ndarray, C: np.ndarray, D: np.ndarray, y0: np.ndarray):
    def evaluate(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = A @ u + v
        return y0 + D @ u + C @ (q + 0.5 * q**2)

    return evaluate


def structural_truth(A: np.ndarray, C: np.ndarray, D: np.ndarray) -> dict[str, np.ndarray]:
    P = np.stack([np.outer(C[:, j], A[j]) for j in range(A.shape[0])], axis=2)
    return {
        "J1": D + P.sum(axis=2),
        "J2": C.copy(),
        "H": P.copy(),
        "P": P,
        "D": D.copy(),
        "G": C.copy(),
    }


def run(seed: int) -> dict:
    rng = np.random.RandomState(seed)
    A = np.array([[1.0, -2.0], [0.5, 1.0]])
    C = np.array([[2.0, -1.0], [1.0, 3.0]])
    D = np.array([[0.25, -0.5], [1.0, 0.75]])
    y0 = np.array([0.1, -0.2])
    z = 1.0 / np.sqrt(2.0)
    U = np.array([[1.0, 0.0], [0.0, 1.0], [z, z], [z, -z]])
    V = U.copy()
    radius1 = rng.uniform(0.05, 0.4, size=len(U))
    radius2 = rng.uniform(0.05, 0.4, size=len(V))
    Y1, Y2, Y12 = evaluate_cartesian_corners(
        quadratic_asg(A, C, D, y0), U, V, radius1, radius2
    )
    estimate = fit_cartesian_structural_tensors(U, V, Y1, Y2, Y12, np.ones(2))
    truth = structural_truth(A, C, D)
    tensor_errors = {
        name: float(np.max(np.abs(getattr(estimate, name) - expected)))
        for name, expected in truth.items()
    }
    A_hat, C_hat = active_edge_inverse(estimate.G, estimate.P, np.ones(2))

    target_D = D + np.array([[0.1, -0.2], [0.3, 0.05]])
    target_A = A + np.array([[0.2, 0.1], [-0.1, 0.25]])
    target_C = C + np.array([[0.05, -0.15], [0.2, 0.1]])
    target = structural_truth(target_A, target_C, target_D)
    delta_J1 = estimate.J1 - target["J1"]
    delta_J2 = estimate.J2 - target["J2"]
    delta_P = estimate.P - target["P"]
    signs = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    ) / np.sqrt(2.0)
    enumerated_energy, _ = structural_response_energy(
        delta_J1, delta_J2, delta_P, signs, signs, cartesian=True
    )
    analytic_energy = float(
        0.5 * np.linalg.norm(delta_J1) ** 2
        + 0.5 * np.linalg.norm(delta_J2) ** 2
        + 0.25 * np.linalg.norm(delta_P) ** 2
    )

    factorization = factorization_residuals(estimate.G, estimate.P)
    max_error = max(tensor_errors.values())
    status = "PASS" if max_error < 1e-12 else "FAIL"
    return {
        "schema_version": 1,
        "experiment": "cpu_asg_rdag_mixed_path_identification",
        "seed": seed,
        "theory_scope": (
            "ASG-RDAG reduced local path-gain equivalence class only; "
            "not arbitrary transformer circuit identification"
        ),
        "status": status,
        "dimensions": {"d1": 2, "d2": 2, "output": 2, "m1": 4, "m2": 4},
        "tensor_max_abs_error": tensor_errors,
        "overall_max_abs_error": max_error,
        "active_edge_max_abs_error": {
            "A": float(np.max(np.abs(A_hat - A))),
            "C": float(np.max(np.abs(C_hat - C))),
        },
        "probe_design": {
            "U_rank": estimate.U_diagnostics.rank,
            "V_rank": estimate.V_diagnostics.rank,
            "W_rank": estimate.W_diagnostics.rank,
            "W_required_rank": estimate.W_diagnostics.required_rank,
            "U_lambda_min": estimate.U_diagnostics.lambda_min,
            "V_lambda_min": estimate.V_diagnostics.lambda_min,
            "U_condition_number": estimate.U_diagnostics.condition_number,
            "V_condition_number": estimate.V_diagnostics.condition_number,
        },
        "factorization_residual": factorization.tolist(),
        "energy_identity": {
            "enumerated": enumerated_energy,
            "analytic": analytic_energy,
            "absolute_error": abs(enumerated_energy - analytic_energy),
        },
        "radii": {"M1": radius1.tolist(), "M2": radius2.tolist()},
        "gpu_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mixed_path_synthetic_seed0.json"),
    )
    args = parser.parse_args()
    result = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
