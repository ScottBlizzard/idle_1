"""Aggregate multi-seed GPT-2 IRS runs into a claim-facing report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def aggregate(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("no runs")
    layers = sorted({row["layer"] for run in runs for row in run["layer_results"]})
    layer_rows = []
    for layer in layers:
        rows = [
            row for run in runs for row in run["layer_results"]
            if row["layer"] == layer
        ]
        layer_rows.append({
            "layer": layer,
            "n_seeds": len(rows),
            "mean_restoration": float(np.mean([r["mean_restoration"] for r in rows])),
            "min_restoration": float(np.min([r["mean_restoration"] for r in rows])),
            "mean_nmh_recovery": float(np.mean([r["mean_nmh_recovery"] for r in rows])),
            "max_nmh_recovery": float(np.max([r["mean_nmh_recovery"] for r in rows])),
            "mean_irs_normalized_rmse": float(np.mean([
                r["irs_normalized_rmse"] for r in rows
            ])),
            "min_irs_normalized_rmse": float(np.min([
                r["irs_normalized_rmse"] for r in rows
            ])),
            "mean_irs_cosine": float(np.mean([r["irs_mean_cosine"] for r in rows])),
            "mean_center_conformal": float(np.mean([
                r["mean_center_conformal"] for r in rows
            ])),
            "min_center_accept_rate": float(np.min([
                r["center_accept_rate"] for r in rows
            ])),
            "mean_endpoint_accept_rate": float(np.mean([
                r["endpoint_accept_rate"] for r in rows
            ])),
            "min_endpoint_accept_rate": float(np.min([
                r["endpoint_accept_rate"] for r in rows
            ])),
            "max_target_center_replay_error": float(np.max([
                r["target_center_replay_max_abs_error"] for r in rows
            ])),
        })

    layer_seed = [
        {**row, "seed": run["seed"]}
        for run in runs for row in run["layer_results"]
    ]
    irs = np.array([r["irs_normalized_rmse"] for r in layer_seed])
    nmh = np.array([r["mean_nmh_recovery"] for r in layer_seed])
    restoration = np.array([r["mean_restoration"] for r in layer_seed])
    seeds = np.array([r["seed"] for r in layer_seed])
    seed_onehot = OneHotEncoder(sparse_output=False).fit_transform(seeds[:, None])
    controls = np.column_stack([
        seed_onehot,
        StandardScaler().fit_transform(restoration[:, None]),
    ])
    irs_resid = irs - Ridge(alpha=1.0).fit(controls, irs).predict(controls)
    nmh_resid = nmh - Ridge(alpha=1.0).fit(controls, nmh).predict(controls)
    layer_rho, layer_p = spearmanr(irs_resid, nmh_resid)

    prompt_rows = []
    for run in runs:
        for row in run["prompt_rows"]:
            prompt_rows.append({**row, "seed": run["seed"]})
    prompt_irs = np.array([r["irs_normalized_rmse"] for r in prompt_rows])
    prompt_nmh = np.array([r["nmh_recovery"] for r in prompt_rows])
    prompt_r = np.array([r["restoration"] for r in prompt_rows])
    prompt_categories = np.array([
        f"{r['seed']}:{r['layer']}:{r['context_id']}" for r in prompt_rows
    ])
    category_onehot = OneHotEncoder(sparse_output=False).fit_transform(
        prompt_categories[:, None]
    )
    prompt_controls = np.column_stack([
        category_onehot,
        StandardScaler().fit_transform(prompt_r[:, None]),
    ])
    prompt_irs_resid = prompt_irs - Ridge(alpha=1.0).fit(
        prompt_controls, prompt_irs
    ).predict(prompt_controls)
    prompt_nmh_resid = prompt_nmh - Ridge(alpha=1.0).fit(
        prompt_controls, prompt_nmh
    ).predict(prompt_controls)
    prompt_rho, prompt_p = spearmanr(prompt_irs_resid, prompt_nmh_resid)

    stable_divergence_layers = [
        row["layer"] for row in layer_rows
        if row["min_restoration"] > 0.8
        and row["max_nmh_recovery"] < 0.5
        and row["min_endpoint_accept_rate"] > 0.9
    ]
    aligned_layers = [
        row["layer"] for row in layer_rows
        if row["mean_nmh_recovery"] > 0.8
    ]
    divergence_irs = [
        row["mean_irs_normalized_rmse"] for row in layer_rows
        if row["layer"] in stable_divergence_layers
    ]
    aligned_irs = [
        row["mean_irs_normalized_rmse"] for row in layer_rows
        if row["layer"] in aligned_layers
    ]
    return {
        "experiment": "p0_irs_gpt2_aggregate",
        "n_runs": len(runs),
        "seeds": [run["seed"] for run in runs],
        "n_prompt_rows": len(prompt_rows),
        "layer_rows": layer_rows,
        "stable_high_R_low_A_admissible_layers": stable_divergence_layers,
        "aligned_layers_mean_A_gt_0.8": aligned_layers,
        "mean_irs_stable_divergence": finite_float(np.mean(divergence_irs)) if divergence_irs else None,
        "mean_irs_aligned": finite_float(np.mean(aligned_irs)) if aligned_irs else None,
        "layer_seed_fixed_effect_residual_spearman_irs_vs_nmh": {
            "rho": finite_float(layer_rho), "p_value": finite_float(layer_p)
        },
        "prompt_fixed_effect_residual_spearman_irs_vs_nmh": {
            "rho": finite_float(prompt_rho), "p_value": finite_float(prompt_p)
        },
        "interpretation_rule": (
            "A negative IRS-vs-NMH association is supportive only when high-R/low-A "
            "layers retain target-reference endpoint acceptance. IRS remains a local "
            "functional witness, not structural circuit identification."
        ),
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# P0 Interventional Response Signature: GPT-2 aggregate",
        "",
        f"Runs: {result['n_runs']}; prompt-layer rows: {result['n_prompt_rows']}.",
        "",
        "| Layer | Mean R (min) | Mean NMH (max) | IRS normalized RMSE | IRS cosine | Center p | Endpoint accept (min) |",
        "|--:|:--:|:--:|--:|--:|--:|:--:|",
    ]
    for row in result["layer_rows"]:
        lines.append(
            f"| {row['layer']} | {row['mean_restoration']:.3f} ({row['min_restoration']:.3f}) "
            f"| {row['mean_nmh_recovery']:.3f} ({row['max_nmh_recovery']:.3f}) "
            f"| {row['mean_irs_normalized_rmse']:.3f} | {row['mean_irs_cosine']:.3f} "
            f"| {row['mean_center_conformal']:.3f} "
            f"| {row['mean_endpoint_accept_rate']:.3f} ({row['min_endpoint_accept_rate']:.3f}) |"
        )
    lines.extend([
        "",
        "## Decision-facing summary",
        "",
        f"- Stable high-R/low-A/admissible layers: `{result['stable_high_R_low_A_admissible_layers']}`.",
        f"- Mean IRS, stable divergence: {result['mean_irs_stable_divergence']}.",
        f"- Mean IRS, aligned layers: {result['mean_irs_aligned']}.",
        "- Layer/seed fixed-effect residual Spearman IRS vs NMH: "
        f"{result['layer_seed_fixed_effect_residual_spearman_irs_vs_nmh']['rho']:.3f}.",
        "- Prompt fixed-effect residual Spearman IRS vs NMH: "
        f"{result['prompt_fixed_effect_residual_spearman_irs_vs_nmh']['rho']:.3f}.",
        "",
        result["interpretation_rule"],
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()
    result = aggregate(args.paths)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
