from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from green_v400_shared_decision_analyzer import (
    analyze_primary_selective_risk,
    matched_acceptance_sets,
)


METHODS = ["finite_activation_patching", "ms_hvp"]


def rows():
    result = []
    for prompt in range(8):
        for layer in range(2):
            endpoint = 0.05 if prompt < 4 else 0.7
            status = "CERTIFIED_POSITIVE" if prompt < 4 else "UNRESOLVED"
            result.append({
                "row_id": f"r-{prompt}-{layer}",
                "prompt_row_id": f"p-{prompt}",
                "endpoint_status": "VALID",
                "heldout_transport_symmetric_normalized_error": endpoint,
                "ordinary_restoration": 0.9,
                "green_status": status,
                "baseline_risk_scores": {
                    "finite_activation_patching": float((prompt + 3) % 8),
                    "ms_hvp": float(7 - prompt),
                },
            })
    return result


def test_matched_coverage_never_uses_endpoint_to_select_baseline_rows():
    payload = rows()
    first = matched_acceptance_sets(payload, METHODS)
    for row in payload:
        row["heldout_transport_symmetric_normalized_error"] = 1000 - float(
            row["heldout_transport_symmetric_normalized_error"]
        )
    second = matched_acceptance_sets(payload, METHODS)
    assert first == second
    assert all(len(first[method]) == len(first["GREEN"]) for method in METHODS)


def test_prompt_cluster_bootstrap_reports_locked_contrasts():
    report = analyze_primary_selective_risk(
        rows(),
        task="ioi",
        methods=METHODS,
        bootstrap_replicates=200,
    )
    assert report["green_coverage"] == pytest.approx(0.5)
    assert report["cluster_unit"] == "prompt_row_id"
    assert report["selection_used_endpoint_values"] is False
    assert set(report["contrasts"]) == set(METHODS)


def test_invalid_endpoint_or_missing_baseline_score_fails_for_every_method():
    payload = rows()
    payload[0]["endpoint_status"] = "INVALID_NUMERICAL_REPLAY"
    with pytest.raises(ValueError, match="same VALID endpoint population"):
        analyze_primary_selective_risk(
            payload, task="ioi", methods=METHODS, bootstrap_replicates=10
        )
    payload = rows()
    del payload[0]["baseline_risk_scores"]["ms_hvp"]
    with pytest.raises(ValueError, match="ms_hvp"):
        analyze_primary_selective_risk(
            payload, task="ioi", methods=METHODS, bootstrap_replicates=10
        )
