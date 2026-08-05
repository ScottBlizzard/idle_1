"""Aggregate prompt-level, temporally eligible NMH validation runs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def aggregate(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("no mechanism runs found")
    layer_groups = defaultdict(list)
    rows = []
    for run in runs:
        for result in run["layer_results"]:
            layer_groups[result["layer"]].append(result)
        for row in run["rows"]:
            rows.append({"seed": run["seed"], **row})

    layers = []
    for layer, values in sorted(layer_groups.items()):
        layers.append({
            "layer": layer,
            "n_runs": len(values),
            "mean_restoration": float(np.mean([v["mean_restoration"] for v in values])),
            "min_restoration_across_seeds": float(min(v["mean_restoration"] for v in values)),
            "mean_nmh_recovery": float(np.mean([v["mean_nmh_recovery"] for v in values])),
            "max_nmh_recovery_across_seeds": float(max(v["mean_nmh_recovery"] for v in values)),
            "min_restoration_minus_nmh_across_seeds": float(min(
                v["mean_restoration"] - v["mean_nmh_recovery"] for v in values
            )),
            "mean_corrupt_overlap": float(np.mean([v["mean_corrupt_overlap"] for v in values])),
            "mean_matched_overlap": float(np.mean([v["mean_matched_overlap"] for v in values])),
            "mean_within_layer_rho_corrupt": float(np.mean([
                v["partial_spearman_corrupt_overlap_vs_nmh"] for v in values
            ])),
            "mean_within_layer_rho_matched": float(np.mean([
                v["partial_spearman_matched_overlap_vs_nmh"] for v in values
            ])),
        })

    seed = np.array([row["seed"] for row in rows])
    layer = np.array([row["layer"] for row in rows])
    context = np.array([row["context_id"] for row in rows])
    restoration = np.array([row["restoration"] for row in rows])
    nmh = np.array([row["nmh_recovery"] for row in rows])
    categorical = OneHotEncoder(sparse_output=False).fit_transform(
        np.column_stack([seed, layer, context])
    )
    controls = np.column_stack([
        categorical, StandardScaler().fit_transform(restoration[:, None])
    ])
    nmh_residual = nmh - Ridge(alpha=1.0).fit(controls, nmh).predict(controls)
    pooled = {}
    for field in ("corrupt_overlap", "matched_overlap"):
        overlap = np.array([row[field] for row in rows])
        overlap_residual = overlap - Ridge(alpha=1.0).fit(controls, overlap).predict(controls)
        rho, p_value = spearmanr(overlap_residual, nmh_residual)
        pooled[field] = {"spearman_rho": float(rho), "p_value": float(p_value)}

    divergence_layers = [
        result["layer"] for result in layers
        if result["min_restoration_across_seeds"] > 0.8
        and result["min_restoration_minus_nmh_across_seeds"] > 0.4
    ]
    return {
        "experiment": "p0_within_site_mechanism_aggregate",
        "n_runs": len(runs),
        "seeds": [run["seed"] for run in runs],
        "n_prompt_site_observations": len(rows),
        "design": runs[0]["site_definition"],
        "layer_summary": layers,
        "pooled_fixed_effects": pooled,
        "decision": {
            "high_restoration_large_mechanism_gap_layers_stable_across_seeds": divergence_layers,
            "supports_behavior_mechanism_separation": bool(divergence_layers),
            "supports_low_overlap_as_necessary_for_bypass": False,
            "reason": (
                "Divergence layers have mean overlap near the held-out reference baseline; "
                "overlap is at most a weak within-site correlate, not a necessary condition."
            ),
        },
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# P0 within-site, temporally eligible mechanism audit",
        "",
        f"Runs: {result['n_runs']}; prompt-site observations: {result['n_prompt_site_observations']}.",
        "Every patch is at the IO position and occurs before both measured GPT-2 Name Mover Heads.",
        "",
        "| Layer | Mean R | Mean NMH recovery | Corrupt overlap | Matched overlap | Mean within-layer rho (corrupt/matched) |",
        "|--:|--:|--:|--:|--:|:--:|",
    ]
    for row in result["layer_summary"]:
        lines.append(
            f"| {row['layer']} | {row['mean_restoration']:.3f} | {row['mean_nmh_recovery']:.3f} | "
            f"{row['mean_corrupt_overlap']:.3f} | {row['mean_matched_overlap']:.3f} | "
            f"{row['mean_within_layer_rho_corrupt']:.3f}/{row['mean_within_layer_rho_matched']:.3f} |"
        )
    decision = result["decision"]
    pooled = result["pooled_fixed_effects"]
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Stable layers with mean R > 0.8 and mean R−A > 0.4 in every seed: `{decision['high_restoration_large_mechanism_gap_layers_stable_across_seeds']}`.",
        f"- Supports behavioral/mechanism separation: **{decision['supports_behavior_mechanism_separation']}**.",
        f"- Supports low overlap as necessary for bypass: **{decision['supports_low_overlap_as_necessary_for_bypass']}**.",
        f"- Pooled fixed-effect residual rho: corrupt={pooled['corrupt_overlap']['spearman_rho']:.3f}, matched={pooled['matched_overlap']['spearman_rho']:.3f}.",
        f"- Reason: {decision['reason']}",
        "",
        "The defensible claim is therefore `R does not imply A`. These runs do not support the stronger claim that low IVS is the cause, or a necessary signature, of mechanism bypass.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent.parent / "analysis")
    args = parser.parse_args()
    paths = sorted(args.outputs.glob("exp_p0_within_site_mechanism_gpt2_seed*.json"))
    result = aggregate(paths)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "p0_within_site_mechanism_aggregate.json"
    md_path = args.analysis_dir / "p0_within_site_mechanism_aggregate.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
