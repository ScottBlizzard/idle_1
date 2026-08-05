"""Aggregate known-ground-truth gated-task cross-fit runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def aggregate(paths):
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("no phase1 cross-fit runs found")
    per_seed = []
    high_rows = []
    for run in runs:
        checkpoint_name = Path(run.get("checkpoint", "")).name
        independently_trained_checkpoint = bool(
            re.fullmatch(r"model_seed[1-9][0-9]*\.pt", checkpoint_name)
        )
        per_seed.append({
            "seed": run["seed"],
            "trained_in_this_run": (
                run.get("trained_in_this_run", False)
                or independently_trained_checkpoint
            ),
            "acc_off": run["acc_off"],
            "acc_on": run["acc_on"],
            "n_high_clean": run["n_high_clean"],
            "n_high_donor": run["n_high_donor"],
            "auroc": run["prompt_level_auroc_target_off_z_for_donor"],
            "clean_ecdf": run["mean_target_off_ecdf_high_clean"],
            "donor_ecdf": run["mean_target_off_ecdf_high_donor"],
            "conformal_auroc": run.get(
                "prompt_level_auroc_target_off_negative_conformal_for_donor"
            ),
            "clean_conformal": run.get("mean_target_off_conformal_high_clean"),
            "donor_conformal": run.get("mean_target_off_conformal_high_donor"),
        })
        high_rows.extend([
            {"seed": run["seed"], **row}
            for row in run["rows"] if row["restoration"] >= 0.5
        ])

    aucs = np.array([row["auroc"] for row in per_seed], dtype=float)
    clean = np.array([row["clean_ecdf"] for row in per_seed], dtype=float)
    donor = np.array([row["donor_ecdf"] for row in per_seed], dtype=float)
    has_conformal = all(row["conformal_auroc"] is not None for row in per_seed)
    conformal_aucs = np.array([
        row["conformal_auroc"] for row in per_seed
    ], dtype=float) if has_conformal else None
    clean_conformal = np.array([
        row["clean_conformal"] for row in per_seed
    ], dtype=float) if has_conformal else None
    donor_conformal = np.array([
        row["donor_conformal"] for row in per_seed
    ], dtype=float) if has_conformal else None
    max_z = max(
        abs(row["references"]["target_natural_gate_off"]["z_sum_mean"])
        for row in high_rows
    )
    floor_spans = []
    for row in high_rows:
        sensitivity = row["references"]["target_natural_gate_off"]["scale_floor_overlap"]
        floor_spans.append(max(sensitivity.values()) - min(sensitivity.values()))
    return {
        "experiment": "phase1_known_ground_truth_crossfit_aggregate",
        "n_model_seeds": len(runs),
        "seeds": [run["seed"] for run in runs],
        "per_seed": per_seed,
        "summary": {
            "mean_auroc": float(aucs.mean()),
            "min_auroc": float(aucs.min()),
            "mean_clean_ecdf": float(clean.mean()),
            "mean_donor_ecdf": float(donor.mean()),
            "min_clean_minus_donor_ecdf": float(np.min(clean - donor)),
            "max_abs_mean_z": float(max_z),
            "max_scale_floor_overlap_span": float(max(floor_spans)),
            "mean_conformal_auroc": (
                float(conformal_aucs.mean()) if has_conformal else None
            ),
            "min_conformal_auroc": (
                float(conformal_aucs.min()) if has_conformal else None
            ),
            "mean_clean_conformal": (
                float(clean_conformal.mean()) if has_conformal else None
            ),
            "mean_donor_conformal": (
                float(donor_conformal.mean()) if has_conformal else None
            ),
        },
        "decision": {
            "cross_model_separation_robust": bool(
                len(runs) >= 3 and aucs.min() > 0.9 and np.min(clean - donor) > 0.2
            ),
            "composite_conformal_separation_robust": bool(
                has_conformal
                and len(runs) >= 3
                and conformal_aucs.min() > 0.9
                and np.min(clean_conformal - donor_conformal) > 0.2
            ),
            "million_scale_z_is_safe_evidence": False,
            "defensible_claim": (
                "When the target support is fixed by task design, cross-fitted empirical "
                "overlap separates on-support clean donors from off-support gate-on donors."
            ),
        },
    }


def render(result):
    lines = [
        "# Known-ground-truth synthetic cross-fit audit",
        "",
        "| Seed | Trained independently | acc off/on | z AUROC | conformal AUROC | Clean/Donor conformal |",
        "|--:|:--:|:--:|--:|--:|:--:|",
    ]
    for row in result["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['trained_in_this_run']} | "
            f"{row['acc_off']:.3f}/{row['acc_on']:.3f} | {row['auroc']:.4f} | "
            f"{row['conformal_auroc']:.4f} | "
            f"{row['clean_conformal']:.3f}/{row['donor_conformal']:.3f} |"
        )
    summary = result["summary"]
    decision = result["decision"]
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Robust across model seeds: **{decision['cross_model_separation_robust']}**.",
        f"- Mean/min AUROC: {summary['mean_auroc']:.4f}/{summary['min_auroc']:.4f}.",
        f"- Mean clean/donor ECDF overlap: {summary['mean_clean_ecdf']:.3f}/{summary['mean_donor_ecdf']:.3f}.",
        f"- Composite conformal robust: **{decision['composite_conformal_separation_robust']}**.",
        f"- Mean/min conformal AUROC: {summary['mean_conformal_auroc']:.4f}/{summary['min_conformal_auroc']:.4f}.",
        f"- Mean clean/donor conformal overlap: {summary['mean_clean_conformal']:.3f}/{summary['mean_donor_conformal']:.3f}.",
        f"- Million-scale z is safe evidence: **{decision['million_scale_z_is_safe_evidence']}**.",
        f"- Defensible claim: {decision['defensible_claim']}",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent.parent / "analysis")
    args = parser.parse_args()
    result = aggregate(sorted(args.outputs.glob("phase1_crossfit_seed*.json")))
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "phase1_crossfit_aggregate.json"
    md_path = args.analysis_dir / "phase1_crossfit_aggregate.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render(result), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
