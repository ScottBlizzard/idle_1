import copy
import json
from pathlib import Path

from analysis.green_v400_atp_star_applicability import (
    VERDICT,
    audit_atp_star_applicability,
)


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE = json.loads(
    (ROOT / "configs/green_v400_silent_failure_challenge_prepare.json").read_text(
        encoding="utf-8"
    )
)
READINESS = json.loads(
    (ROOT / "configs/green_v400_baseline_readiness.json").read_text(
        encoding="utf-8"
    )
)


def test_frozen_coarse_residual_sweep_scopes_atp_without_supersession_claim():
    audit = audit_atp_star_applicability(CHALLENGE, READINESS)
    assert audit["verdict"] == VERDICT
    assert audit["coarse_site_count_per_prompt"] == 9
    assert audit["comparison_method"] == "finite_activation_patching_response"
    assert audit["atp_star_claimed_as_executed"] is False
    assert audit["supersedes_atp_star"] is False


def test_head_or_neuron_protocol_cannot_reuse_coarse_site_decision():
    changed = copy.deepcopy(CHALLENGE)
    changed["candidate_population"]["hook"] = "attn_head_output"
    changed["candidate_population"]["primary_site_family"] = "all attention heads"
    audit = audit_atp_star_applicability(changed, READINESS)
    assert audit["verdict"] == "BLOCK_ATP_STAR_APPLICABILITY"


def test_large_sweep_reopens_atp_star_requirement():
    changed = copy.deepcopy(CHALLENGE)
    changed["candidate_population"]["layers"] = list(range(33))
    audit = audit_atp_star_applicability(changed, READINESS)
    assert "ceiling" in " ".join(audit["errors"])


def test_missing_finite_patching_baseline_fails_closed():
    changed = copy.deepcopy(READINESS)
    changed["baselines"]["finite_activation_patching_response"]["status"] = "MISSING"
    audit = audit_atp_star_applicability(CHALLENGE, changed)
    assert "finite_activation_patching_response is not READY" in audit["errors"]


def test_registry_cannot_claim_atp_star_supersession():
    changed = copy.deepcopy(READINESS)
    entry = changed["baselines"]["AtP_star_or_closest_exact_attribution"]
    entry["supersedes_atp_star"] = True
    audit = audit_atp_star_applicability(CHALLENGE, changed)
    assert audit["verdict"] == "BLOCK_ATP_STAR_APPLICABILITY"
