"""Reproduce the GREEN v1.3.6 terminal admissibility audit from copied artifacts."""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

THRESHOLDS = {
    "curvature_rms_min": 5e-4,
    "curvature_snr_min": 20.0,
    "gate_response_rms_min": 5e-4,
    "gate_response_snr_min": 20.0,
    "factorization_residual_max": 0.15,
    "whitebox_a_relative_max": 0.05,
    "whitebox_a_small_absolute_max": 1e-4,
    "tensor_cosine_min": 0.95,
    "tensor_symmetric_change_max": 0.25,
    "richardson_change_max": 0.25,
    "tensor_snr_min": 20.0,
    "bypass_disagreement_max": 0.15,
    "active_gates_min": 3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not finite.size:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        name: float(np.quantile(finite, q))
        for name, q in (("min", 0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("max", 1))
    }


def active_failure_flags(gate: dict) -> list[str]:
    failures: list[str] = []
    if not gate["inverse_admissible"]:
        failures.append("inverse_not_admissible")
    identified = gate["factorization_residual"] is not None
    if not identified:
        failures.append("identification_missing")
        return failures
    if gate["curvature_norm"] / 10 < THRESHOLDS["curvature_rms_min"]:
        failures.append("curvature_rms")
    if gate["curvature_norm"] < THRESHOLDS["curvature_snr_min"] * gate["epsilon_C"]:
        failures.append("curvature_snr")
    if gate["gate_response_norm"] / 10 < THRESHOLDS["gate_response_rms_min"]:
        failures.append("gate_response_rms")
    if gate["gate_response_norm"] < THRESHOLDS["gate_response_snr_min"] * gate["epsilon_G"]:
        failures.append("gate_response_snr")
    if gate["factorization_residual"] > THRESHOLDS["factorization_residual_max"]:
        failures.append("factorization_residual")
    wb_norm = gate["whitebox_norm"]
    wb_error = gate["whitebox_error"]
    wb_ok = wb_error is not None and wb_error <= THRESHOLDS["whitebox_a_relative_max"] * max(wb_norm, 1e-6)
    if wb_norm < 1e-6:
        wb_ok = wb_error is not None and wb_error <= THRESHOLDS["whitebox_a_small_absolute_max"]
    if not wb_ok:
        failures.append("whitebox_agreement")
    if gate["full_half_cosine"] < THRESHOLDS["tensor_cosine_min"]:
        failures.append("tensor_cosine")
    if gate["full_half_change"] > THRESHOLDS["tensor_symmetric_change_max"]:
        failures.append("tensor_symmetric_change")
    if gate["richardson_change"] > THRESHOLDS["richardson_change_max"]:
        failures.append("richardson_change")
    if not gate.get("shift_pass", False):
        failures.append("shift_coefficient")
    return failures


def main() -> None:
    tensor = pd.read_parquet(ROOT / "dev_tensor_scores.parquet")
    energy = pd.read_parquet(ROOT / "dev_energy_targets.parquet")
    cells = json.loads((ROOT / "dev_cells.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "dev_result.json").read_text(encoding="utf-8"))

    system_counts: dict[str, Counter] = {"tar": Counter(), "pat": Counter()}
    active_hist: dict[str, Counter] = {"tar": Counter(), "pat": Counter()}
    invalid_hist: dict[str, Counter] = {"tar": Counter(), "pat": Counter()}
    label_counts: dict[str, Counter] = {"tar": Counter(), "pat": Counter()}
    failure_counts: dict[str, Counter] = {"tar": Counter(), "pat": Counter()}
    failure_combinations: dict[str, Counter] = {"tar": Counter(), "pat": Counter()}
    invalid_failure_cardinality: Counter = Counter()
    gate_metrics: dict[str, dict[str, dict[str, list[float]]]] = {
        system: defaultdict(lambda: defaultdict(list)) for system in ("tar", "pat")
    }
    system_metrics: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("tar", "pat")
    }
    per_item = []

    for _, row in tensor.iterrows():
        mixed = json.loads(row["mixed_audit"])
        item_summary = {"pair_digest": row["pair_digest"], "cell_id": row["cell_id"]}
        for system in ("tar", "pat"):
            audit = mixed[system]
            gates = audit["gates"]
            counts = Counter(gate["label"] for gate in gates)
            label_counts[system].update(counts)
            active_hist[system][audit["active_gates"]] += 1
            invalid_hist[system][counts["invalid"]] += 1
            for key in ("admissible", "all_valid"):
                system_counts[system][f"{key}_{str(bool(audit[key])).lower()}"] += 1
            system_metrics[system]["bypass_disagreement"].append(audit.get("bypass_disagreement"))
            system_metrics[system]["center_rms"].append(audit.get("center_rms"))
            system_metrics[system]["center_max"].append(audit.get("center_max"))
            for gate in gates:
                label = gate["label"]
                metrics = gate_metrics[system][label]
                metrics["factorization_residual"].append(gate.get("factorization_residual"))
                metrics["whitebox_relative_error"].append(
                    None if gate.get("whitebox_error") is None else
                    gate["whitebox_error"] / max(gate["whitebox_norm"], 1e-6)
                )
                metrics["curvature_snr_ratio"].append(
                    gate["curvature_norm"] / max(THRESHOLDS["curvature_snr_min"] * gate["epsilon_C"], 1e-30)
                )
                metrics["gate_response_snr_ratio"].append(
                    gate["gate_response_norm"] / max(THRESHOLDS["gate_response_snr_min"] * gate["epsilon_G"], 1e-30)
                )
                metrics["full_half_cosine"].append(gate.get("full_half_cosine"))
                metrics["full_half_change"].append(gate.get("full_half_change"))
                metrics["richardson_change"].append(gate.get("richardson_change"))
                if gate["label"] != "invalid":
                    continue
                flags = active_failure_flags(gate)
                if not flags:
                    # P-norm is not serialized. If all persisted criteria pass but
                    # the label is invalid, the only pre-shift active criterion left
                    # is ||P|| >= 20 epsilon_P_F.
                    flags = ["tensor_snr_unserialized_inferred"]
                failure_counts[system].update(flags)
                failure_combinations[system][" + ".join(sorted(flags))] += 1
                invalid_failure_cardinality[len(flags)] += 1
            item_summary[system] = {
                "admissible": audit["admissible"],
                "all_valid": audit["all_valid"],
                "active": audit["active_gates"],
                "invalid": counts["invalid"],
                "null": counts["certified-target-null"],
                "bypass": audit.get("bypass_disagreement"),
            }
        per_item.append(item_summary)

    energy_systems = {name: Counter() for name in ("tar", "pat", "cor")}
    energy_metrics: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("tar", "pat", "cor")
    }
    for _, row in energy.iterrows():
        systems = json.loads(row["systems"])
        for name, values in systems.items():
            energy_systems[name][str(bool(values["admissible"])).lower()] += 1
            for key in ("jvp_absolute_error", "jvp_relative_error", "locality_error"):
                energy_metrics[name][key].append(values[key])

    numeric_cols = [
        "behavioral", "single", "first_order", "pie", "cancellation_dx",
        "cancellation_dz", "theta_tar", "theta_pat", "physical_target_norm",
        "residual_radius",
    ]
    raw_quantiles = {column: quantiles(tensor[column].tolist()) for column in numeric_cols}
    raw_corr = tensor[["behavioral", "single", "first_order", "pie"]].corr(method="spearman").round(6).to_dict()
    raw_cell = tensor.groupby("cell_id")[["behavioral", "single", "first_order", "pie"]].mean()
    raw_cell_corr = raw_cell.corr(method="spearman").round(6).to_dict()

    audit = {
        "schema": "green-v1.3.6-terminal-audit-v1",
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "dev_tensor_scores.parquet", "dev_energy_targets.parquet",
                "dev_cells.json", "dev_result.json", "development_multigpu_merge.json",
            )
        },
        "terminal_result": result,
        "tensor_records": {
            "total": int(len(tensor)),
            "admissible": int(tensor["admissible"].sum()),
            "inadmissible": int((~tensor["admissible"]).sum()),
        },
        "energy_records": {
            "total": int(len(energy)),
            "admissible": int(energy["admissible"].sum()),
            "inadmissible": int((~energy["admissible"]).sum()),
        },
        "cells": {
            "total": len(cells["cells"]),
            "survived": sum(bool(row["survived"]) for row in cells["cells"]),
            "conditioning_dev_sd": cells["conditioning_dev_sd"],
            "n_tensor_histogram": dict(Counter(row["n_tensor"] for row in cells["cells"])),
            "n_energy_histogram": dict(Counter(row["n_energy"] for row in cells["cells"])),
        },
        "mixed_systems": {
            name: {
                "counts": dict(system_counts[name]),
                "gate_labels": dict(label_counts[name]),
                "active_gate_histogram": dict(sorted(active_hist[name].items())),
                "invalid_gate_histogram": dict(sorted(invalid_hist[name].items())),
                "invalid_active_criterion_failures": dict(failure_counts[name].most_common()),
                "invalid_failure_combinations": dict(failure_combinations[name].most_common()),
                "gate_metric_quantiles_by_label": {
                    label: {metric: quantiles(values) for metric, values in metrics.items()}
                    for label, metrics in gate_metrics[name].items()
                },
                "metrics": {key: quantiles(values) for key, values in system_metrics[name].items()},
            }
            for name in ("tar", "pat")
        },
        "invalid_gate_failure_cardinality": dict(sorted(invalid_failure_cardinality.items())),
        "energy_systems": {
            name: {
                "admissibility": dict(energy_systems[name]),
                "metrics": {key: quantiles(values) for key, values in energy_metrics[name].items()},
            }
            for name in ("tar", "pat", "cor")
        },
        "raw_nonconfirmatory_diagnostics": {
            "warning": "All tensor rows are protocol-inadmissible; these summaries cannot support a scientific claim.",
            "quantiles": raw_quantiles,
            "item_spearman": raw_corr,
            "cell_mean_spearman": raw_cell_corr,
        },
        "per_item_system_summary": per_item,
        "thresholds": THRESHOLDS,
    }
    (ROOT / "terminal_admissibility_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    combined = failure_counts["tar"] + failure_counts["pat"]
    combined_combinations = failure_combinations["tar"] + failure_combinations["pat"]
    both_factor_whitebox = sum(
        count for combination, count in combined_combinations.items()
        if "factorization_residual" in combination and "whitebox_agreement" in combination
    )
    lines = [
        "# GREEN v1.3.6 terminal admissibility audit",
        "",
        "## Frozen outcome",
        "",
        f"- Development verdict: `{result['verdict']}`; the outer runner stopped at `12_DEVELOPMENT_SURVIVAL`.",
        f"- Tensor records: {len(tensor)} produced, {int(tensor['admissible'].sum())} admissible.",
        f"- Energy records: {len(energy)} produced, {int(energy['admissible'].sum())} admissible.",
        f"- Cells: {len(cells['cells'])} evaluated, {sum(bool(row['survived']) for row in cells['cells'])} survived.",
        "- Confirmation remained closed.",
        "",
        "## Localization",
        "",
        "The execution pipeline completed. The terminal failure is localized to the mixed-tensor gate certification: every energy row passed, while no tensor row had both tar and pat mixed systems admissible.",
        "",
    ]
    for system in ("tar", "pat"):
        data = audit["mixed_systems"][system]
        lines.extend([
            f"### {system}",
            "",
            f"- System-admissible items: {data['counts'].get('admissible_true', 0)}/{len(tensor)}.",
            f"- All-valid systems: {data['counts'].get('all_valid_true', 0)}/{len(tensor)}.",
            f"- Gate labels across {len(tensor) * 10} gate audits: `{data['gate_labels']}`.",
            f"- Active-gate histogram: `{data['active_gate_histogram']}`.",
            f"- Invalid-gate histogram: `{data['invalid_gate_histogram']}`.",
            "",
        ])
    lines.extend([
        "## Persisted active-criterion failures among invalid gates",
        "",
        "A gate can fail multiple criteria, so counts are not mutually exclusive.",
        "",
    ])
    for name, count in combined.most_common():
        lines.append(f"- `{name}`: {count}")
    lines.extend([
        "",
        "`tensor_snr_unserialized_inferred` is inferred only when every serialized active criterion and the shift check pass; the P-norm itself was not written into the parquet audit.",
        "",
        f"The dominant pair is factorization residual plus white-box agreement ({combined_combinations['factorization_residual + whitebox_agreement']} gates with exactly that pair; {both_factor_whitebox} gates with at least both). Their invalid-gate medians are near but above the frozen cutoffs: factorization is about 0.185–0.190 versus 0.15, and relative white-box error is about 0.065–0.072 versus 0.05. Exact distributions normalized against the thresholds are serialized in `terminal_admissibility_audit.json`.",
        "",
        "## Non-confirmatory signal diagnostic",
        "",
        f"Although inadmissible rows cannot support the registered claim, the raw PIE baseline has item-level Spearman `{raw_corr['pie']['behavioral']:.3f}` and 16-cell mean Spearman `{raw_cell_corr['pie']['behavioral']:.3f}` with the behavioral target. This is evidence that the run contains structured signal, but it is not a valid estimate of confirmatory performance.",
        "",
        "## Interpretation boundary",
        "",
        "This is a scientific/protocol STOP, not a worker crash or missing-data event. Any change to gate thresholds, null certification, completeness, minimum active gates, or tensor SNR would alter the frozen scientific protocol and therefore requires an explicit theory-level decision before another confirmatory run. Raw inadmissible correlations in the JSON are diagnostic only.",
        "",
    ])
    (ROOT / "TERMINAL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
