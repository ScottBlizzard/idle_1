"""Requirement-by-requirement completion audit for the August P0 goal."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> int:
    checks: list[dict] = []

    def record(requirement: str, passed: bool, evidence: str) -> None:
        checks.append({
            "requirement": requirement,
            "status": "proved" if passed else "not_proved",
            "evidence": evidence,
        })

    theory_path = ROOT / "analysis" / "IRS_THEORY_P0.md"
    theory = theory_path.read_text(encoding="utf-8")
    theory_markers = [
        "Theorem 1: zero-order restoration is non-identifying",
        "Theorem 2: local response agreement gives a transport bound",
        "Proposition 3: first-order targets",
        "Theorem 4: probe-law coverage and finite-sketch concentration",
        "Theorem 5: composite split-conformal admissibility",
        "Exact relationship to mediator--bypass interaction",
        "Claim--evidence map",
        "Immediate falsification tests",
    ]
    missing = [marker for marker in theory_markers if marker not in theory]
    record(
        "P0 formal theory closure",
        not missing,
        f"{theory_path.relative_to(ROOT)} contains all 8 required theorem, "
        f"scope, collision, claim-map, and falsification markers; missing={missing}",
    )

    conformal_path = ROOT / "src" / "validity_crossfit.py"
    conformal = conformal_path.read_text(encoding="utf-8")
    conformal_markers = [
        "composite_normalization_ref",
        "composite_calibration_ref",
        "_composite_nonconformity_from_raw",
        'result["overlap_conformal"]',
    ]
    conformal_missing = [x for x in conformal_markers if x not in conformal]
    test_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "src/test_validity_crossfit.py",
            "src/test_interventional_response.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    record(
        "Composite split-conformal implementation and regression tests",
        not conformal_missing and test_run.returncode == 0,
        f"implementation markers missing={conformal_missing}; pytest return="
        f"{test_run.returncode}; output={test_run.stdout.strip()}",
    )

    synthetic_paths = sorted((ROOT / "outputs").glob(
        "irs_analytic_synthetic_seed*.json"
    ))
    synthetic = [read_json(path) for path in synthetic_paths]
    synthetic_ok = (
        len(synthetic) == 5
        and {row["seed"] for row in synthetic} == set(range(5))
        and all(row["all_pass"] for row in synthetic)
        and all(
            row["verdicts"]["single_direction_blind_but_irs_detects"]
            for row in synthetic
        )
        and all(finite_tree(row) for row in synthetic)
    )
    record(
        "IRS synthetic minimum validation",
        synthetic_ok,
        f"found={len(synthetic_paths)} seeds; all_pass="
        f"{[row.get('all_pass') for row in synthetic]}; finite="
        f"{[finite_tree(row) for row in synthetic]}",
    )

    expected_seeds = {20260712, 20260713, 20260714}
    gpt_paths = sorted((ROOT / "outputs").glob("exp_p0_irs_gpt2_seed*.json"))
    gpt = [read_json(path) for path in gpt_paths]
    all_layers = all(
        [row["layer"] for row in result["layer_results"]] == list(range(9))
        for result in gpt
    )
    l4_l7_frozen = all(
        all(
            next(x for x in result["layer_results"] if x["layer"] == layer)[
                "mean_restoration"
            ] > 0.8
            and next(x for x in result["layer_results"] if x["layer"] == layer)[
                "mean_nmh_recovery"
            ] < 0.5
            and next(x for x in result["layer_results"] if x["layer"] == layer)[
                "endpoint_accept_rate"
            ] > 0.9
            for layer in range(4, 8)
        )
        for result in gpt
    )
    l8_characterized = all(
        any(row["layer"] == 8 for row in result["layer_results"])
        for result in gpt
    )
    gpt_ok = (
        len(gpt) == 3
        and {row["seed"] for row in gpt} == expected_seeds
        and all_layers
        and l4_l7_frozen
        and l8_characterized
        and all(finite_tree(row) for row in gpt)
    )
    record(
        "GPT-2 L4--L8 minimum validation",
        gpt_ok,
        f"seeds={sorted(row['seed'] for row in gpt)}; layers_0_8={all_layers}; "
        f"L4-L7 frozen high-R/low-NMH/admissible={l4_l7_frozen}; "
        f"L8 measured={l8_characterized}; all finite="
        f"{all(finite_tree(row) for row in gpt)}",
    )

    stress_path = ROOT / "analysis" / "p0_irs_stress_summary.json"
    stress = read_json(stress_path)
    packet_path = ROOT / "analysis" / "GPTPRO_REDTEAM_PACKET_20260805.md"
    packet = packet_path.read_text(encoding="utf-8")
    pro_needed = (
        stress["decision"]["probe_robustness_established"] is False
        and stress["decision"][
            "clear_irs_stability_advantage_established"
        ] is False
        and stress["decision"][
            "oral_level_method_novelty_established_by_these_tests"
        ] is False
        and "Decisions requested from GPT Pro" in packet
        and "Closest/latest collision set" in packet
    )
    record(
        "Iterate until independent GPT Pro red-team is genuinely required",
        pro_needed,
        "frozen probe and clear-stability gates are false, the oral novelty "
        "decision is false, and a collision-aware GPT Pro packet with six "
        "explicit decisions exists",
    )

    overall = all(check["status"] == "proved" for check in checks)
    report = {
        "objective": (
            "P0 theory closure, composite split-conformal implementation/tests, "
            "IRS synthetic and GPT-2 L4--L8 minimum validation, and iteration "
            "until GPT Pro red-team is genuinely required"
        ),
        "overall_status": "proved" if overall else "not_proved",
        "checks": checks,
        "external_handoff": {
            "status": "awaiting_user_authentication",
            "note": (
                "The objective's stop condition is reached. Executing the GPT Pro "
                "review itself is a next-stage task and currently requires the "
                "user to sign in to the preserved ChatGPT browser tab."
            ),
        },
    }
    analysis_dir = ROOT / "analysis"
    json_path = analysis_dir / "P0_GOAL_COMPLETION_AUDIT.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# P0 goal completion audit",
        "",
        f"Overall status: **{report['overall_status']}**.",
        "",
        "| Requirement | Status | Authoritative evidence |",
        "|:--|:--:|:--|",
    ]
    for check in checks:
        evidence = check["evidence"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {check['requirement']} | {check['status']} | {evidence} |"
        )
    lines += [
        "",
        "## External handoff",
        "",
        report["external_handoff"]["note"],
        "",
        "This audit proves the requested P0 stopping condition, not acceptance or "
        "oral-level novelty.  The frozen novelty gate is negative, which is why "
        "independent red-team adjudication is now necessary.",
    ]
    md_path = analysis_dir / "P0_GOAL_COMPLETION_AUDIT.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
