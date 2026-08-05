"""Aggregate cross-fitted reference-distribution audit runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REFERENCES = (
    "corrupt_observational",
    "clean_source",
    "mixture",
    "matched_semantic_counterfactual",
)


def q(values, probability):
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def aggregate(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("no runs found")
    per_seed = []
    flat = []
    for run in runs:
        per_seed.append({
            "seed": run["seed"],
            "n_retained_sites": run["n_retained_sites"],
            "prompt_audit": run["prompt_audit"],
            "reference_summary": run["reference_summary"],
        })
        for site in run["sites"]:
            flat.append({"seed": run["seed"], **site})

    reference_summary = {}
    for reference in REFERENCES:
        overlap = [row["references"][reference]["overlap_z"] for row in flat]
        ecdf = [row["references"][reference]["overlap_ecdf"] for row in flat]
        recon_z = [row["references"][reference]["recon_z"] for row in flat]
        floor_spans = []
        for row in flat:
            sensitivity = row["references"][reference]["scale_floor_sensitivity"]
            floor_spans.append(max(sensitivity.values()) - min(sensitivity.values()))
        reference_summary[reference] = {
            "n_sites": len(overlap),
            "mean_overlap_z": float(np.mean(overlap)),
            "median_overlap_z": float(np.median(overlap)),
            "overlap_z_q025": q(overlap, 0.025),
            "overlap_z_q975": q(overlap, 0.975),
            "mean_overlap_ecdf": float(np.mean(ecdf)),
            "n_below_historical_0_3": int(np.sum(np.asarray(overlap) < 0.3)),
            "max_abs_recon_z": float(np.max(np.abs(recon_z))),
            "max_scale_floor_overlap_span": float(max(floor_spans)),
        }

    pairwise = {}
    corrupt = np.array([
        row["references"]["corrupt_observational"]["overlap_z"] for row in flat
    ])
    for reference in REFERENCES[1:]:
        alternative = np.array([row["references"][reference]["overlap_z"] for row in flat])
        rho, p_value = spearmanr(corrupt, alternative)
        pairwise[f"corrupt_vs_{reference}"] = {
            "mean_paired_overlap_difference_alternative_minus_corrupt": float(
                np.mean(alternative - corrupt)
            ),
            "n_historical_labels_changed": int(
                np.sum((corrupt < 0.3) != (alternative < 0.3))
            ),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
        }

    categories = {}
    for category, predicate in {
        "io_position": lambda row: row["pos"] == 3,
        "last_position": lambda row: row["pos"] == 16,
    }.items():
        selected = [row for row in flat if predicate(row)]
        category_result = {"n": len(selected)}
        category_result.update({
            reference: {
                "mean_overlap_z": float(np.mean([
                    row["references"][reference]["overlap_z"] for row in selected
                ])),
                "n_below_historical_0_3": int(np.sum([
                    row["references"][reference]["overlap_z"] < 0.3 for row in selected
                ])),
            }
            for reference in REFERENCES
        })
        categories[category] = category_result

    return {
        "experiment": "p0_reference_crossfit_aggregate",
        "n_runs": len(runs),
        "seeds": [run["seed"] for run in runs],
        "n_site_observations": len(flat),
        "per_seed": per_seed,
        "reference_summary": reference_summary,
        "pairwise_reference_sensitivity": pairwise,
        "site_categories": categories,
        "decision": {
            "million_scale_reconstruction_gap_survives": (
                reference_summary["corrupt_observational"]["max_abs_recon_z"] >= 1e5
            ),
            "historical_low_overlap_labels_invariant_to_reference": all(
                values["n_historical_labels_changed"] == 0 for values in pairwise.values()
            ),
            "safe_current_estimand": "conditional target-reference overlap",
        },
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# P0 reference-distribution and cross-fit audit",
        "",
        f"Runs: {result['n_runs']} seeds; retained site observations: {result['n_site_observations']}.",
        "",
        "| Reference | Mean overlap-z | 2.5%–97.5% | Below historical 0.3 | Max |recon-z| | Max floor span |",
        "|:--|--:|:--:|--:|--:|--:|",
    ]
    for reference, values in result["reference_summary"].items():
        lines.append(
            f"| `{reference}` | {values['mean_overlap_z']:.3f} | "
            f"{values['overlap_z_q025']:.3f}–{values['overlap_z_q975']:.3f} | "
            f"{values['n_below_historical_0_3']}/{values['n_sites']} | "
            f"{values['max_abs_recon_z']:.3g} | {values['max_scale_floor_overlap_span']:.3g} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Million-scale reconstruction gap survives: **{result['decision']['million_scale_reconstruction_gap_survives']}**.",
        f"- Historical low-overlap labels are invariant to reference: **{result['decision']['historical_low_overlap_labels_invariant_to_reference']}**.",
        "- The defensible estimand at this stage is **conditional target-reference overlap**, not general causal validity.",
        "",
        "## Reference sensitivity",
        "",
    ])
    for name, values in result["pairwise_reference_sensitivity"].items():
        lines.append(
            f"- `{name}`: mean paired change={values['mean_paired_overlap_difference_alternative_minus_corrupt']:.3f}; "
            f"historical labels changed={values['n_historical_labels_changed']}; "
            f"Spearman rho={values['spearman_rho']:.3f}."
        )
    lines.extend([
        "",
        "The historical 0.3 cutoff is shown only for continuity. It was not re-tuned on these runs and must not be interpreted as a preregistered causal-validity boundary.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent.parent / "analysis")
    args = parser.parse_args()
    paths = sorted(args.outputs.glob("exp_p0_reference_crossfit_gpt2_seed*.json"))
    result = aggregate(paths)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "p0_reference_crossfit_aggregate.json"
    md_path = args.analysis_dir / "p0_reference_crossfit_aggregate.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
