"""Summarize the fair supervised-baseline audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path: Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    methods = list(run["summary"])
    groups = {}
    for label, pos in (("io_position", 3), ("last_position", 16)):
        sites = [site for site in run["sites"] if site["pos"] == pos]
        groups[label] = {
            "n_sites": len(sites),
            "mean_test_auroc": {
                method: float(np.mean([
                    site["methods"][method]["test_auroc"] for site in sites
                ]))
                for method in methods
            },
        }
    ivs_mean = run["summary"]["crossfit_ivs"]["mean_test_auroc"]
    supervised_best = max(
        (values["mean_test_auroc"], name)
        for name, values in run["summary"].items() if name != "crossfit_ivs"
    )
    return {
        "experiment": "p0_fair_baseline_analysis",
        "source": str(path),
        "label_definition": run["label_definition"],
        "causal_guardrail": run["causal_guardrail"],
        "split": run["split"],
        "overall": run["summary"],
        "site_groups": groups,
        "decision": {
            "crossfit_ivs_mean_auroc": ivs_mean,
            "best_supervised_method": supervised_best[1],
            "best_supervised_mean_auroc": supervised_best[0],
            "ivs_has_clear_advantage": bool(ivs_mean > supervised_best[0] + 0.02),
            "old_mlp_failure_supports_high_dimensional_overfit_claim": False,
            "reason": (
                "With standardized features, identical splits, and equal tuning, "
                "supervised baselines are competitive overall and perfectly separate "
                "the last-position shift in activation space."
            ),
        },
    }


def render(result: dict) -> str:
    methods = list(result["overall"])
    lines = [
        "# P0 fair baseline audit",
        "",
        result["causal_guardrail"],
        "",
        "| Method | Overall AUROC | IO-position AUROC | Last-position AUROC |",
        "|:--|--:|--:|--:|",
    ]
    for method in methods:
        lines.append(
            f"| `{method}` | {result['overall'][method]['mean_test_auroc']:.3f} | "
            f"{result['site_groups']['io_position']['mean_test_auroc'][method]:.3f} | "
            f"{result['site_groups']['last_position']['mean_test_auroc'][method]:.3f} |"
        )
    decision = result["decision"]
    lines.extend([
        "",
        "## Decision",
        "",
        f"- IVS has a clear advantage under the matched protocol: **{decision['ivs_has_clear_advantage']}**.",
        f"- Best supervised method: `{decision['best_supervised_method']}` (mean AUROC {decision['best_supervised_mean_auroc']:.3f}).",
        f"- The old MLP result supports a high-dimensional-overfit claim: **{decision['old_mlp_failure_supports_high_dimensional_overfit_claim']}**.",
        f"- Reason: {decision['reason']}",
        "",
        "The method-level averages mix two regimes: all detectors are near chance at IO sites, while all detect the last-position clean/corrupt shift. This benchmark therefore does not establish causal validity or mechanism bypass.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" /
        "exp_p0_fair_baselines_gpt2_seed20260712.json",
    )
    parser.add_argument(
        "--analysis-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "analysis",
    )
    args = parser.parse_args()
    result = analyze(args.input)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.analysis_dir / "p0_fair_baselines.json"
    md_path = args.analysis_dir / "p0_fair_baselines.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render(result), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
