"""Known-gradient four-quadrant validation for IRS and conformal support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interventional_response import compare_signatures, isotropic_probes, symmetric_signature
from validity_crossfit import CrossFitSiteReference


def target_function(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 0.25 * x[:, 1] ** 2


def evaluate_condition(name: str, centers: np.ndarray, probes: np.ndarray, beta: float) -> dict:
    def patched_function(x: np.ndarray) -> np.ndarray:
        base = target_function(x)
        if name == "true_restoration":
            return base
        if name == "on_support_lie":
            return base + beta * x[:, -1]
        if name == "off_support_shortcut":
            return base + beta * (x[:, -1] - 5.0)
        if name == "natural_ineffective":
            return base + 2.0
        raise ValueError(name)

    target_center = target_function(centers)
    patch_center = patched_function(centers)
    restoration = np.exp(-np.abs(patch_center - target_center))
    plus = centers[:, None, :] + probes
    minus = centers[:, None, :] - probes
    flat_plus = plus.reshape(-1, centers.shape[1])
    flat_minus = minus.reshape(-1, centers.shape[1])
    shape = plus.shape[:2]
    target_signature = symmetric_signature(
        target_function(flat_plus).reshape(shape),
        target_function(flat_minus).reshape(shape),
        probes,
    )
    patch_signature = symmetric_signature(
        patched_function(flat_plus).reshape(shape),
        patched_function(flat_minus).reshape(shape),
        probes,
    )
    comparison = compare_signatures(patch_signature, target_signature)
    # A fair single-direction witness can be exactly blind: the mechanism gap in
    # the lie conditions is along the last coordinate, while this fixed contrast
    # moves only along the first coordinate.
    blind_delta = np.zeros((len(centers), 1, centers.shape[1]), dtype=np.float64)
    blind_delta[:, 0, 0] = np.linalg.norm(probes[0, 0])
    blind_plus = centers[:, None, :] + blind_delta
    blind_minus = centers[:, None, :] - blind_delta
    blind_shape = blind_plus.shape[:2]
    blind_target = symmetric_signature(
        target_function(blind_plus.reshape(-1, centers.shape[1])).reshape(blind_shape),
        target_function(blind_minus.reshape(-1, centers.shape[1])).reshape(blind_shape),
        blind_delta,
    )
    blind_patch = symmetric_signature(
        patched_function(blind_plus.reshape(-1, centers.shape[1])).reshape(blind_shape),
        patched_function(blind_minus.reshape(-1, centers.shape[1])).reshape(blind_shape),
        blind_delta,
    )
    blind_comparison = compare_signatures(blind_patch, blind_target)
    known_gradient_gap = beta if name in {"on_support_lie", "off_support_shortcut"} else 0.0
    expected_directional_mse = known_gradient_gap ** 2 / centers.shape[1]
    observed_directional_mse = comparison.rmse ** 2
    return {
        "condition": name,
        "mean_restoration": float(restoration.mean()),
        "irs_rmse": comparison.rmse,
        "irs_normalized_rmse": comparison.normalized_rmse,
        "irs_cosine": comparison.mean_cosine,
        "blind_single_direction_rmse": blind_comparison.rmse,
        "known_gradient_gap": known_gradient_gap,
        "expected_directional_mse": expected_directional_mse,
        "observed_directional_mse": observed_directional_mse,
        "directional_mse_abs_error": abs(observed_directional_mse - expected_directional_mse),
    }


def run(args: argparse.Namespace) -> dict:
    rng = np.random.RandomState(args.seed)
    d = args.dimension
    fit = rng.randn(args.n_fit, d)
    cal = rng.randn(args.n_cal, d)
    ordinary = rng.randn(args.n_eval * 4, d)
    on_support = ordinary[np.argsort(np.abs(ordinary[:, -1]))[: args.n_eval]].copy()
    on_support[:, -1] = 0.0
    true_centers = rng.randn(args.n_eval, d)
    ineffective_centers = rng.randn(args.n_eval, d)
    off_support = rng.randn(args.n_eval, d)
    off_support[:, -1] = 5.0

    scorer = CrossFitSiteReference(
        fit,
        cal,
        knn_k=args.knn_k,
        proj_rank=min(args.proj_rank, d),
    )
    conditions = {
        "true_restoration": true_centers,
        "on_support_lie": on_support,
        "off_support_shortcut": off_support,
        "natural_ineffective": ineffective_centers,
    }
    rows = []
    for index, (name, centers) in enumerate(conditions.items()):
        probes = isotropic_probes(
            len(centers), args.n_probes, d, args.radius,
            np.random.RandomState(args.seed + 100 + index),
        )
        row = evaluate_condition(name, centers, probes, args.beta)
        support = scorer.score(centers)["overlap_conformal"]
        row.update({
            "mean_conformal_support": float(support.mean()),
            "support_accept_rate_alpha_0.05": float(np.mean(support > 0.05)),
        })
        rows.append(row)

    by_name = {row["condition"]: row for row in rows}
    verdicts = {
        "true_restore_high_R_high_S_low_IRS": (
            by_name["true_restoration"]["mean_restoration"] > 0.95
            and by_name["true_restoration"]["support_accept_rate_alpha_0.05"] > 0.9
            and by_name["true_restoration"]["irs_rmse"] < 1e-8
        ),
        "on_support_lie_high_R_high_S_high_IRS": (
            by_name["on_support_lie"]["mean_restoration"] > 0.95
            and by_name["on_support_lie"]["support_accept_rate_alpha_0.05"] > 0.9
            and by_name["on_support_lie"]["irs_rmse"] > 0.5
        ),
        "off_support_shortcut_high_R_low_S_high_IRS": (
            by_name["off_support_shortcut"]["mean_restoration"] > 0.95
            and by_name["off_support_shortcut"]["support_accept_rate_alpha_0.05"] < 0.1
            and by_name["off_support_shortcut"]["irs_rmse"] > 0.5
        ),
        "natural_ineffective_low_R_high_S": (
            by_name["natural_ineffective"]["mean_restoration"] < 0.2
            and by_name["natural_ineffective"]["support_accept_rate_alpha_0.05"] > 0.9
        ),
        "known_gradient_mse_recovered": all(
            row["directional_mse_abs_error"] < 0.1 for row in rows
        ),
        "single_direction_blind_but_irs_detects": (
            by_name["on_support_lie"]["blind_single_direction_rmse"] < 1e-8
            and by_name["on_support_lie"]["irs_rmse"] > 0.5
            and by_name["off_support_shortcut"]["blind_single_direction_rmse"] < 1e-8
            and by_name["off_support_shortcut"]["irs_rmse"] > 0.5
        ),
    }
    output = {
        "experiment": "irs_analytic_four_quadrant",
        "seed": args.seed,
        "configuration": vars(args),
        "rows": rows,
        "verdicts": verdicts,
        "all_pass": all(verdicts.values()),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"irs_analytic_synthetic_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved {path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--n-fit", type=int, default=1200)
    parser.add_argument("--n-cal", type=int, default=600)
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--n-probes", type=int, default=128)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--proj-rank", type=int, default=8)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
