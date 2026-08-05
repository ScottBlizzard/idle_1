"""Compare multi-direction IRS with the clean--corrupt interaction direction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def grouped_rmse(features: np.ndarray, outcome: np.ndarray, groups: np.ndarray) -> float:
    predictions = np.empty(len(outcome), dtype=np.float64)
    for train, test in GroupKFold(n_splits=len(np.unique(groups))).split(
        features, outcome, groups
    ):
        scaler = StandardScaler().fit(features[train])
        model = Ridge(alpha=1.0).fit(scaler.transform(features[train]), outcome[train])
        predictions[test] = model.predict(scaler.transform(features[test]))
    return float(np.sqrt(mean_squared_error(outcome, predictions)))


def analyze(irs_path: Path, single_path: Path) -> dict:
    irs_run = json.loads(irs_path.read_text(encoding="utf-8"))
    single_run = json.loads(single_path.read_text(encoding="utf-8"))
    if irs_run["seed"] != single_run["seed"]:
        raise ValueError("runs must use the same seed")
    irs_layers = {row["layer"]: row for row in irs_run["layer_results"]}
    single_layers = {row["layer"]: row for row in single_run["layer_results"]}
    layers = sorted(irs_layers)
    nmh = np.array([irs_layers[layer]["mean_nmh_recovery"] for layer in layers])
    irs = np.array([irs_layers[layer]["irs_normalized_rmse"] for layer in layers])
    single = np.array([
        single_layers[layer]["single_direction_normalized_rmse"] for layer in layers
    ])
    irs_rho, irs_p = spearmanr(irs, nmh)
    single_rho, single_p = spearmanr(single, nmh)
    aligned = nmh > 0.8
    divergent = (nmh < 0.5) & np.array([
        irs_layers[layer]["mean_restoration"] > 0.8 for layer in layers
    ])

    irs_prompt = {
        (row["layer"], row["context_id"], index % 10): row
        for index, row in enumerate(irs_run["prompt_rows"])
    }
    # Rows are emitted in the same deterministic layer/prompt order.  Use index
    # within each layer as the exact join key rather than relying on prompt text.
    merged = []
    by_layer_counter = {layer: 0 for layer in layers}
    irs_exact = {}
    for row in irs_run["prompt_rows"]:
        index = by_layer_counter[row["layer"]]
        by_layer_counter[row["layer"]] += 1
        irs_exact[(row["layer"], index)] = row
    by_layer_counter = {layer: 0 for layer in layers}
    for row in single_run["prompt_rows"]:
        index = by_layer_counter[row["layer"]]
        by_layer_counter[row["layer"]] += 1
        target = irs_exact[(row["layer"], index)]
        merged.append({
            "layer": row["layer"],
            "context_id": row["context_id"],
            "restoration": row["restoration"],
            "nmh": row["nmh_recovery"],
            "irs": target["irs_normalized_rmse"],
            "single": row["single_direction_normalized_rmse"],
        })

    layer_values = np.array([row["layer"] for row in merged])
    context_values = np.array([row["context_id"] for row in merged])
    restoration = np.array([row["restoration"] for row in merged])[:, None]
    prompt_nmh = np.array([row["nmh"] for row in merged])
    prompt_irs = np.array([row["irs"] for row in merged])[:, None]
    prompt_single = np.array([row["single"] for row in merged])[:, None]
    layer_onehot = OneHotEncoder(sparse_output=False).fit_transform(layer_values[:, None])
    controls = np.column_stack([layer_onehot, restoration])
    rmse = {
        "controls": grouped_rmse(controls, prompt_nmh, context_values),
        "controls_plus_single": grouped_rmse(
            np.column_stack([controls, prompt_single]), prompt_nmh, context_values
        ),
        "controls_plus_irs": grouped_rmse(
            np.column_stack([controls, prompt_irs]), prompt_nmh, context_values
        ),
        "controls_plus_single_plus_irs": grouped_rmse(
            np.column_stack([controls, prompt_single, prompt_irs]),
            prompt_nmh,
            context_values,
        ),
    }
    return {
        "experiment": "irs_vs_single_direction_interaction",
        "seed": irs_run["seed"],
        "layer_level": {
            "irs_spearman_vs_nmh": {"rho": float(irs_rho), "p": float(irs_p)},
            "single_spearman_vs_nmh": {
                "rho": float(single_rho), "p": float(single_p)
            },
            "irs_divergent_to_aligned_ratio": float(irs[divergent].mean() / irs[aligned].mean()),
            "single_divergent_to_aligned_ratio": float(
                single[divergent].mean() / single[aligned].mean()
            ),
            "irs_min_endpoint_accept": float(min(
                irs_layers[layer]["endpoint_accept_rate"] for layer in layers
            )),
            "single_min_endpoint_accept": float(min(
                single_layers[layer]["endpoint_accept_rate"] for layer in layers
            )),
        },
        "leave_one_context_out_prompt_rmse": rmse,
        "decision": {
            "irs_clearly_outperforms_single_direction": bool(
                abs(irs_rho) > abs(single_rho) + 0.1
                and rmse["controls_plus_irs"] < rmse["controls_plus_single"] * 0.95
            ),
            "scope": (
                "A one-seed baseline can falsify an obvious superiority claim but cannot "
                "establish general equivalence. Probe admissibility and cross-task cases "
                "remain separate potential advantages."
            ),
        },
    }


def render(result: dict) -> str:
    layer = result["layer_level"]
    rmse = result["leave_one_context_out_prompt_rmse"]
    return "\n".join([
        "# IRS versus single clean--corrupt interaction direction",
        "",
        f"Seed: {result['seed']}.",
        "",
        f"- Layer Spearman IRS vs NMH: {layer['irs_spearman_vs_nmh']['rho']:.3f}.",
        f"- Layer Spearman single direction vs NMH: {layer['single_spearman_vs_nmh']['rho']:.3f}.",
        f"- Divergent/aligned ratio, IRS: {layer['irs_divergent_to_aligned_ratio']:.3f}.",
        f"- Divergent/aligned ratio, single direction: {layer['single_divergent_to_aligned_ratio']:.3f}.",
        f"- Minimum endpoint acceptance, IRS/single: {layer['irs_min_endpoint_accept']:.3f}/{layer['single_min_endpoint_accept']:.3f}.",
        "",
        "## Leave-one-context-out prompt RMSE",
        "",
        f"- Controls: {rmse['controls']:.4f}.",
        f"- + single direction: {rmse['controls_plus_single']:.4f}.",
        f"- + IRS: {rmse['controls_plus_irs']:.4f}.",
        f"- + both: {rmse['controls_plus_single_plus_irs']:.4f}.",
        "",
        f"Clear IRS superiority: **{result['decision']['irs_clearly_outperforms_single_direction']}**.",
        "",
        result["decision"]["scope"],
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("irs", type=Path)
    parser.add_argument("single", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()
    result = analyze(args.irs, args.single)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.write_text(render(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
