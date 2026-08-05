"""Audit NMH validation after conditioning on intervention timing and site.

This is a diagnostic of the existing experiment, not a replacement experiment.
Residual-stream patches at layer L occur after attention at layer L, so only NMH
heads in strictly later layers can be affected by the patch.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_SPECS = {
    "gpt2": {
        "file": "exp_nmh_ground_truth_gpt2.json",
        "heads": [(9, 9), (10, 0)],
    },
    "gpt2-medium": {
        "file": "exp_nmh_ground_truth_gpt2-medium.json",
        "heads": [(19, 9), (20, 0), (21, 6)],
    },
}


def safe_auc(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = rankdata(scores, method="average")
    rank_sum_positive = float(ranks[labels == 1].sum())
    return (rank_sum_positive - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def analyze(path: Path, heads: list[tuple[int, int]], permutations: int, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["results"]
    ivs = np.array([row["ivs"] for row in rows], dtype=float)
    restoration = np.array([row["restoration"] for row in rows], dtype=float)
    nmh = np.array([row["nmh_recovery"] for row in rows], dtype=float)
    threshold = float(payload["nmh_threshold"])
    bypass = (nmh < threshold).astype(int)
    layers = np.array([row["layer"] for row in rows], dtype=int)
    positions = np.array([row["pos"] for row in rows], dtype=int)
    templates = np.array([row["template_idx"] for row in rows], dtype=int)

    n_downstream = np.array([sum(head_layer > layer for head_layer, _ in heads) for layer in layers])
    categories = np.where(
        n_downstream == len(heads), "all_heads_downstream",
        np.where(n_downstream == 0, "no_heads_downstream", "partial_heads_downstream"),
    )

    category_summary = {}
    for category in ("all_heads_downstream", "partial_heads_downstream", "no_heads_downstream"):
        mask = categories == category
        category_summary[category] = {
            "n": int(mask.sum()),
            "n_bypass": int(bypass[mask].sum()),
            "bypass_rate": mean_or_none(bypass[mask]),
            "mean_ivs": mean_or_none(ivs[mask]),
            "mean_nmh_recovery": mean_or_none(nmh[mask]),
            "mean_restoration": mean_or_none(restoration[mask]),
            "auroc_ivs_for_bypass": safe_auc(bypass[mask], -ivs[mask]),
        }

    # Exact site strata across templates. This asks whether IVS separates mechanism
    # outcomes at the same layer and token position, rather than separating sites.
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(layers, positions)):
        strata[key].append(index)
    informative = []
    weighted_numerator = 0.0
    weighted_denominator = 0
    for (layer, pos), indices_list in sorted(strata.items()):
        indices = np.array(indices_list)
        auc = safe_auc(bypass[indices], -ivs[indices])
        if auc is not None:
            informative.append({
                "layer": int(layer), "pos": int(pos), "n": int(len(indices)), "auroc": auc
            })
            weighted_numerator += auc * len(indices)
            weighted_denominator += len(indices)

    # A stratified randomization test preserves exact layer/position. If the pooled
    # AUC is entirely between sites, shuffling scores across templates within each
    # site leaves the AUC unchanged.
    permutation_strata: dict[tuple, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        permutation_strata[(layers[index], positions[index])].append(index)
    observed_auc = safe_auc(bypass, -ivs)
    rng = np.random.RandomState(seed)
    null_aucs = []
    if observed_auc is not None:
        for _ in range(permutations):
            shuffled = ivs.copy()
            for indices_list in permutation_strata.values():
                indices = np.array(indices_list)
                shuffled[indices] = rng.permutation(shuffled[indices])
            null_aucs.append(safe_auc(bypass, -shuffled))
    randomization_p = (
        float((1 + np.sum(np.asarray(null_aucs) >= observed_auc)) / (1 + len(null_aucs)))
        if null_aucs else None
    )

    # Continuous residual association after a flexible additive adjustment for
    # layer, position, template, timing availability, and restoration.
    categorical = np.column_stack([layers, positions, templates, n_downstream])
    one_hot = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(categorical)
    continuous = StandardScaler().fit_transform(restoration[:, None])
    confounds = np.column_stack([one_hot, continuous])
    ivs_residual = ivs - Ridge(alpha=1.0).fit(confounds, ivs).predict(confounds)
    nmh_residual = nmh - Ridge(alpha=1.0).fit(confounds, nmh).predict(confounds)
    residual_rho, residual_p = spearmanr(ivs_residual, nmh_residual)

    all_downstream = n_downstream == len(heads)
    return {
        "model": payload["model"],
        "source": str(path),
        "n_sites": len(rows),
        "nmh_heads": heads,
        "timing_rule": "resid_post patch at layer L can affect attention heads only at layers > L",
        "pooled_auroc_ivs_for_bypass": observed_auc,
        "all_heads_downstream_auroc": safe_auc(bypass[all_downstream], -ivs[all_downstream]),
        "category_summary": category_summary,
        "exact_layer_position": {
            "n_total_strata": len(strata),
            "n_informative_strata": len(informative),
            "n_sites_in_informative_strata": int(weighted_denominator),
            "weighted_within_stratum_auroc": (
                float(weighted_numerator / weighted_denominator) if weighted_denominator else None
            ),
            "informative_strata": informative,
        },
        "stratified_randomization": {
            "n_permutations": permutations,
            "strata": "exact_layer_position",
            "n_permutable_strata": int(sum(len(v) > 1 for v in permutation_strata.values())),
            "n_sites_in_permutable_strata": int(sum(len(v) for v in permutation_strata.values() if len(v) > 1)),
            "p_value": randomization_p,
            "null_auc_mean": mean_or_none(null_aucs),
            "null_auc_q025": float(np.quantile(null_aucs, 0.025)) if null_aucs else None,
            "null_auc_q975": float(np.quantile(null_aucs, 0.975)) if null_aucs else None,
        },
        "continuous_residual_association": {
            "spearman_rho": float(residual_rho),
            "p_value": float(residual_p),
            "adjusted_for": [
                "layer", "position", "template", "number_of_downstream_nmh_heads", "restoration"
            ],
        },
    }


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# P0 NMH timing and site-confound audit",
        "",
        "This analysis re-evaluates the existing NMH experiment without treating pooled AUROC as independent evidence. A residual-stream `resid_post` patch at layer L occurs after attention at layer L and can affect only NMH heads in strictly later layers.",
        "",
        "| Model | Pooled AUROC | All-heads-downstream AUROC | Informative exact site strata | Weighted within-site AUROC | Stratified permutation p | Adjusted residual rho |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for result in results:
        exact = result["exact_layer_position"]
        perm = result["stratified_randomization"]
        residual = result["continuous_residual_association"]
        fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
        lines.append(
            f"| {result['model']} | {fmt(result['pooled_auroc_ivs_for_bypass'])} | "
            f"{fmt(result['all_heads_downstream_auroc'])} | {exact['n_informative_strata']}/{exact['n_total_strata']} | "
            f"{fmt(exact['weighted_within_stratum_auroc'])} | {fmt(perm['p_value'])} | "
            f"{fmt(residual['spearman_rho'])} |"
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "The pooled AUROC is considered timing/site-confounded unless the association survives both temporally eligible subsets and exact layer-position comparisons. An unavailable AUROC is itself informative when a matched subset contains only one mechanism class.",
        "",
    ])
    for result in results:
        lines.extend([f"## {result['model']}", ""])
        for name, values in result["category_summary"].items():
            lines.append(
                f"- `{name}`: n={values['n']}, bypass={values['n_bypass']}, "
                f"mean IVS={values['mean_ivs']}, mean NMH={values['mean_nmh_recovery']}."
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent.parent / "analysis")
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    results = [
        analyze(args.outputs / spec["file"], spec["heads"], args.permutations, args.seed)
        for spec in MODEL_SPECS.values()
    ]
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "p0_nmh_timing_confound.json"
    md_path = args.analysis_dir / "p0_nmh_timing_confound.md"
    json_path.write_text(json.dumps({"models": results}, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
