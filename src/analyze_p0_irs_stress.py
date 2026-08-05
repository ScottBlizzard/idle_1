"""Summarize frozen IRS probe-robustness and corruption-shift gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict:
    probe = load(args.probe_json)
    shift = load(args.shift_json)
    admissible = [
        row for row in probe["setting_results"]
        if row["n_probes"] >= 4 and row["min_endpoint_accept_rate"] >= 0.90
    ]
    probe_rows = []
    for row in admissible:
        passed = (
            row["spearman_vs_primary_layer_ranking"] >= 0.90
            and row["spearman_vs_nmh"] <= -0.75
        )
        probe_rows.append({**row, "passes_frozen_gate": passed})
    probe_pass = bool(probe_rows) and all(
        row["passes_frozen_gate"] for row in probe_rows
    )

    irs = shift["stability"]["irs_normalized_rmse"]
    single = shift["stability"]["single_direction_normalized_rmse"]
    rank_delta = (
        irs["spearman_across_corruptions"]
        - single["spearman_across_corruptions"]
    )
    relative_change_ratio = (
        irs["relative_l2_change"] / single["relative_l2_change"]
        if single["relative_l2_change"] > 0 else None
    )
    min_irs_support = min(row["irs_endpoint_accept_rate"] for row in shift["rows"])
    min_single_support = min(
        row["single_endpoint_accept_rate"] for row in shift["rows"]
    )
    corruption_noninferior = (
        rank_delta >= -0.05
        and relative_change_ratio is not None
        and relative_change_ratio <= 1.25
    )
    clear_stability_superiority = (
        min_irs_support >= 0.90
        and (
            rank_delta >= 0.10
            or (
                relative_change_ratio is not None
                and relative_change_ratio <= 0.75
            )
        )
    )
    result = {
        "probe_gate": {
            "n_admissible_settings": len(probe_rows),
            "passes": probe_pass,
            "settings": probe_rows,
        },
        "corruption_gate": {
            "irs": irs,
            "single_direction": single,
            "rank_correlation_delta": rank_delta,
            "relative_l2_change_ratio_irs_over_single": relative_change_ratio,
            "minimum_irs_endpoint_acceptance": min_irs_support,
            "minimum_single_endpoint_acceptance": min_single_support,
            "stability_noninferior": corruption_noninferior,
            "clear_stability_superiority": clear_stability_superiority,
        },
        "decision": {
            "probe_robustness_established": probe_pass,
            "clear_irs_stability_advantage_established": clear_stability_superiority,
            "oral_level_method_novelty_established_by_these_tests": (
                probe_pass and clear_stability_superiority
            ),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "p0_irs_stress_summary.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# P0 IRS frozen stress-test summary",
        "",
        f"Probe robustness pass: **{probe_pass}**.",
        "",
        "| Interpolation | Probes | Rank vs primary | Rank vs NMH | Min support | Pass |",
        "|--:|--:|--:|--:|--:|:--:|",
    ]
    for row in probe_rows:
        lines.append(
            f"| {row['interpolation']:.2f} | {row['n_probes']} | "
            f"{row['spearman_vs_primary_layer_ranking']:.3f} | "
            f"{row['spearman_vs_nmh']:.3f} | "
            f"{row['min_endpoint_accept_rate']:.3f} | "
            f"{row['passes_frozen_gate']} |"
        )
    lines += [
        "",
        "## Corruption stability",
        "",
        f"- IRS/single cross-corruption rank: "
        f"{irs['spearman_across_corruptions']:.3f}/"
        f"{single['spearman_across_corruptions']:.3f}.",
        f"- IRS/single relative L2 change: {irs['relative_l2_change']:.3f}/"
        f"{single['relative_l2_change']:.3f}.",
        f"- Relative-change ratio IRS/single: {relative_change_ratio:.3f}.",
        f"- Minimum endpoint acceptance IRS/single: "
        f"{min_irs_support:.3f}/{min_single_support:.3f}.",
        f"- Stability non-inferior: **{corruption_noninferior}**.",
        f"- Clear stability superiority: **{clear_stability_superiority}**.",
        "",
        "## Frozen novelty decision",
        "",
        "Oral-level method novelty established by these two tests: "
        f"**{result['decision']['oral_level_method_novelty_established_by_these_tests']}**.",
    ]
    md_path = out_dir / "p0_irs_stress_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--shift-json", required=True)
    parser.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parent.parent / "analysis"
    ))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
