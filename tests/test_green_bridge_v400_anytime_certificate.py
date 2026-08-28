from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import green_bridge_v400_certificate as certificate
from green_bridge_v400_certificate import (
    AnytimeEvaluationFailure, CurvatureCertificate, CurvatureComponentAccounting,
    audit_monotone_anytime_frozen_partition,
    audit_monotone_anytime_frozen_partitions,
    advance_monotone_anytime_state, initialize_monotone_anytime_state,
    restore_monotone_anytime_state, serialize_monotone_anytime_state,
    transition_anytime_resource_failure_after_admission,
)
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import Jet2
from green_bridge_v400_schemas import CertificatePlan, Dyadic


ROW_HASH = "a" * 64
EVALUATOR_HASH = "b" * 64
RESOURCE_HASH = "c" * 64
P = 384


def _plan(*, max_depth=8, max_cells=32):
    return CertificatePlan(
        "green-v400-certificate-plan-v1", ROW_HASH, (Dyadic(1, 0),),
        "[-h,0],[0,h]",
        "curvature-weighted width priority dyadic bisection",
        "0x1p-80", "0x1p-40", max_depth, max_cells,
        P, 512, (), False,
    )


def _jet(domain: Interval) -> Jet2:
    broad = Interval.from_bounds(-10, 10, domain.precision_bits)
    return Jet2(broad, broad, broad)


class SyntheticEvaluator:
    contains_scientific_outcome = False
    synthetic_only = True
    certificate_row_hash = ROW_HASH
    evaluator_identity_sha256 = EVALUATOR_HASH

    def __init__(self):
        self.fail_pairs = False

    def evaluate_interval(self, domain):
        return _jet(domain)

    def evaluate_interval_pair(self, domains):
        if self.fail_pairs:
            raise RuntimeError("synthetic sibling failure")
        return tuple(_jet(domain) for domain in domains)


class MetadataEvaluator(SyntheticEvaluator):
    def evaluate_interval_pair_with_metadata(self, domains):
        return (
            tuple(_jet(domain) for domain in domains),
            ({"result_source": "EXACT_CACHE_HIT"},
             {"result_source": "COMPUTED"}),
        )


def _accounting():
    zero = certificate.gmpy2.mpq(0)
    return CurvatureComponentAccounting(*([zero] * 8))


def _curvature(positive, negative, precision=P):
    positive = Interval.from_bounds(*positive, precision)
    negative = Interval.from_bounds(*negative, precision)
    zero = Interval.point(0, precision)
    return CurvatureCertificate(
        positive, negative, positive - negative, zero, _accounting(),
    )


def _nested(inner_payload, outer_payload):
    inner = certificate._interval_from_payload(inner_payload)
    outer = certificate._interval_from_payload(outer_payload)
    return outer.lower <= inner.lower <= inner.upper <= outer.upper


def test_real_interval_integration_initializes_and_refines_synthetic_state():
    evaluator = SyntheticEvaluator()
    plan = _plan()
    initial = initialize_monotone_anytime_state(
        evaluator, Fraction(1), P, plan,
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    refined = advance_monotone_anytime_state(initial, evaluator, plan)
    assert len(initial.leaves) == 2 and len(refined.leaves) == 3
    assert refined.raw_curvature_accounting
    assert _nested(refined.monotone_witness, initial.monotone_witness)
    assert refined.scientific_threshold_applied is False


def test_public_audit_replays_exact_frozen_partition_without_adaptive_queue():
    evaluator = SyntheticEvaluator()
    plan = _plan()
    initial = initialize_monotone_anytime_state(
        evaluator, Fraction(1), P, plan,
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    refined = advance_monotone_anytime_state(initial, evaluator, plan)
    report = audit_monotone_anytime_frozen_partition(refined, evaluator, plan)
    assert report["same_frozen_partition"] is True
    assert report["independent_audit_adaptive_queue"] is False
    assert len(report["cells"]) == len(refined.leaves) == 3
    assert report["accounting"] == {
        "logical_evaluations": 6,
        "admitted_native_dispatches": 6,
        "completed_native_dispatches": 6,
        "exact_cache_hits": 0,
    }
    assert all(
        component["audit_inside_official"] is True
        for row in report["cells"]
        for component in row["components"].values()
    )
    aggregate = audit_monotone_anytime_frozen_partitions(
        (refined,), evaluator, plan,
    )
    assert aggregate["phase_major_all_official_before_audit"] is True
    assert aggregate["accounting"] == report["accounting"]
    assert aggregate["cross_radius_prefix_intersections"][-1][
        "audit_inside_official"
    ] is True


def test_audit_replay_refuses_unbound_or_identity_mismatched_evaluator():
    evaluator = SyntheticEvaluator()
    plan = _plan()
    state = initialize_monotone_anytime_state(
        evaluator, Fraction(1), P, plan,
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    evaluator.synthetic_only = False
    with pytest.raises(RuntimeError, match="REAL_EXECUTION_UNAUTHORIZED"):
        audit_monotone_anytime_frozen_partition(state, evaluator, plan)


def test_anytime_state_round_trip_hash_and_strict_tamper_rejection(monkeypatch):
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature",
        lambda cells, h: _curvature((-2, 1), (-2, 1)),
    )
    state = initialize_monotone_anytime_state(
        SyntheticEvaluator(), Fraction(1), P, _plan(),
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    encoded = serialize_monotone_anytime_state(state)
    restored = restore_monotone_anytime_state(encoded)
    assert restored == state
    assert serialize_monotone_anytime_state(restored) == encoded
    assert state.parent_state_semantic_hash == "0" * 64
    assert len(state.leaves) == 2
    assert state.logical_evaluations == 5
    assert state.admitted_native_dispatches == 5
    assert state.completed_native_dispatches == 5
    assert state.exact_cache_hits == 0
    payload = json.loads(encoded)
    payload["state"]["leaves"][0]["depth"] = 7
    with pytest.raises(ValueError):
        restore_monotone_anytime_state(payload)


def test_nested_in_place_mutation_fails_closed_before_use(monkeypatch):
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature",
        lambda cells, h: _curvature((-2, 1), (-2, 1)),
    )

    def fresh_state():
        return initialize_monotone_anytime_state(
            SyntheticEvaluator(), Fraction(1), P, _plan(),
            resource_lock_semantic_hash=RESOURCE_HASH,
        )

    checks = (
        lambda state: state.semantic_hash(),
        serialize_monotone_anytime_state,
        lambda state: advance_monotone_anytime_state(
            state, SyntheticEvaluator(), _plan(),
        ),
        lambda state: transition_anytime_resource_failure_after_admission(
            state, logical_evaluations=2, admitted_native_dispatches=2,
            completed_native_dispatches=0, exact_cache_hits=0,
            resource_reason="WALL_DEADLINE_REACHED",
        ),
    )
    for check in checks:
        state = fresh_state()
        state.endpoint_payload["center"]["lower"][0] += 1
        with pytest.raises(RuntimeError, match="ANYTIME_STATE_INTEGRITY_INVALID"):
            check(state)

    leaf_mutated = fresh_state()
    leaf_mutated.leaves[0].jet_payload["second"]["upper"][0] += 1
    with pytest.raises(RuntimeError, match="ANYTIME_STATE_INTEGRITY_INVALID"):
        serialize_monotone_anytime_state(leaf_mutated)


def test_raw_non_nesting_is_tightened_monotonically(monkeypatch):
    raw = iter((
        _curvature((-2, 1), (-2, 1)),
        _curvature((-1, 2), (-1, 2)),
    ))
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature", lambda cells, h: next(raw),
    )
    evaluator = SyntheticEvaluator()
    initial = initialize_monotone_anytime_state(
        evaluator, Fraction(1), P, _plan(),
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    refined = advance_monotone_anytime_state(initial, evaluator, _plan())
    raw_positive = certificate._interval_from_payload(refined.raw_curvature_positive)
    tightened_positive = certificate._interval_from_payload(
        refined.monotone_curvature_positive
    )
    assert (raw_positive.lower, raw_positive.upper) == (-1, 2)
    assert (tightened_positive.lower, tightened_positive.upper) == (-1, 1)
    for field in (
        "monotone_curvature_positive", "monotone_curvature_negative",
        "monotone_residual_positive", "monotone_residual_negative",
        "monotone_witness",
    ):
        assert _nested(getattr(refined, field), getattr(initial, field))
    assert refined.raw_curvature_accounting == certificate._curvature_payload(
        _curvature((-1, 2), (-1, 2))
    )["component_accounting"]
    assert refined.parent_state_semantic_hash == initial.semantic_hash()
    assert len(refined.leaves) == 3
    assert refined.logical_evaluations == 7


def test_multi_budget_resume_is_byte_identical_and_never_widens(monkeypatch):
    def curvature_for_partition(cells, h):
        radius = Fraction(5 - len(cells), 2)
        return _curvature((-radius, radius), (-radius, radius))

    monkeypatch.setattr(
        certificate, "integrate_signed_curvature", curvature_for_partition,
    )
    evaluator = SyntheticEvaluator()
    plan = _plan(max_cells=8)
    direct0 = initialize_monotone_anytime_state(
        evaluator, Fraction(1), P, plan,
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    direct1 = advance_monotone_anytime_state(direct0, evaluator, plan)
    direct2 = advance_monotone_anytime_state(direct1, evaluator, plan)

    resumed1 = restore_monotone_anytime_state(
        serialize_monotone_anytime_state(direct1)
    )
    resumed2 = advance_monotone_anytime_state(
        resumed1, SyntheticEvaluator(), plan,
    )
    assert serialize_monotone_anytime_state(resumed2) == (
        serialize_monotone_anytime_state(direct2)
    )
    for older, newer in ((direct0, direct1), (direct1, direct2)):
        for field in (
            "monotone_curvature_positive", "monotone_curvature_negative",
            "monotone_residual_positive", "monotone_residual_negative",
            "monotone_witness",
        ):
            assert _nested(getattr(newer, field), getattr(older, field))
    assert [state.logical_evaluations for state in (direct0, direct1, direct2)] == [
        5, 7, 9,
    ]


def test_empty_budget_intersection_is_hard_invalid_not_resource(monkeypatch):
    raw = iter((
        _curvature((-2, 1), (-2, 1)),
        _curvature((2, 3), (-1, 1)),
    ))
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature", lambda cells, h: next(raw),
    )
    evaluator = SyntheticEvaluator()
    initial = initialize_monotone_anytime_state(
        evaluator, Fraction(1), P, _plan(),
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    with pytest.raises(RuntimeError, match="IMPLEMENTATION_INVALID.*EMPTY_INTERSECTION"):
        advance_monotone_anytime_state(initial, evaluator, _plan())
    assert initial.computation_status == "PROVISIONAL"
    assert initial.resource_reason is None


def test_sibling_failure_is_atomic(monkeypatch):
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature",
        lambda cells, h: _curvature((-2, 2), (-2, 2)),
    )
    evaluator = SyntheticEvaluator()
    initial = initialize_monotone_anytime_state(
        evaluator, Fraction(1), P, _plan(),
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    before = serialize_monotone_anytime_state(initial)
    evaluator.fail_pairs = True
    with pytest.raises(AnytimeEvaluationFailure, match="synthetic sibling failure") as caught:
        advance_monotone_anytime_state(initial, evaluator, _plan())
    assert caught.value.maximum_new_native_admissions == 2
    assert caught.value.prior_state_semantic_hash == initial.semantic_hash()
    assert serialize_monotone_anytime_state(initial) == before
    assert len(initial.leaves) == 2
    assert initial.admitted_native_dispatches == 5

    terminal = transition_anytime_resource_failure_after_admission(
        initial, logical_evaluations=2, admitted_native_dispatches=2,
        completed_native_dispatches=0, exact_cache_hits=0,
        resource_reason="WALL_DEADLINE_REACHED",
    )
    assert terminal.computation_status == "RESOURCE_INCONCLUSIVE"
    assert terminal.resource_reason == "WALL_DEADLINE_REACHED"
    assert terminal.logical_evaluations == 7
    assert terminal.admitted_native_dispatches == 7
    assert terminal.completed_native_dispatches == 5
    assert terminal.exact_cache_hits == 0
    assert terminal.leaves == initial.leaves


def test_external_process_death_transition_charges_without_refund(monkeypatch):
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature",
        lambda cells, h: _curvature((-2, 2), (-2, 2)),
    )
    initial = initialize_monotone_anytime_state(
        SyntheticEvaluator(), Fraction(1), P, _plan(),
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    terminal = transition_anytime_resource_failure_after_admission(
        initial, logical_evaluations=2, admitted_native_dispatches=2,
        completed_native_dispatches=1, exact_cache_hits=0,
        resource_reason="MEMORY_MAX_REACHED",
    )
    assert terminal.parent_state_semantic_hash == initial.semantic_hash()
    assert terminal.admitted_native_dispatches == initial.admitted_native_dispatches + 2
    assert terminal.completed_native_dispatches == initial.completed_native_dispatches + 1
    assert terminal.logical_evaluations == initial.logical_evaluations + 2
    assert terminal.computation_status == "RESOURCE_INCONCLUSIVE"
    assert terminal.resource_reason == "MEMORY_MAX_REACHED"


def test_resource_cap_uses_frozen_reason_and_never_success(monkeypatch):
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature",
        lambda cells, h: _curvature((-2, 2), (-2, 2)),
    )
    evaluator = SyntheticEvaluator()
    plan = _plan(max_cells=2)
    initial = initialize_monotone_anytime_state(
        evaluator, Fraction(1), P, plan,
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    terminal = advance_monotone_anytime_state(initial, evaluator, plan)
    assert terminal.computation_status == "RESOURCE_INCONCLUSIVE"
    assert terminal.resource_reason == "MAX_FINAL_LEAVES_PER_RADIUS_REACHED"
    assert terminal.scientific_threshold_applied is False
    assert terminal.monotone_witness == initial.monotone_witness
    assert terminal.logical_evaluations == initial.logical_evaluations

    cells, reason = certificate._adaptive_cells_with_reason(
        evaluator, Fraction(1), P, plan,
    )
    assert cells is None
    assert reason == "MAX_FINAL_LEAVES_PER_RADIUS_REACHED"


def test_logical_dispatch_and_cache_hit_counts_are_separate(monkeypatch):
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature",
        lambda cells, h: _curvature((-2, 2), (-2, 2)),
    )
    state = initialize_monotone_anytime_state(
        MetadataEvaluator(), Fraction(1), P, _plan(),
        resource_lock_semantic_hash=RESOURCE_HASH,
    )
    assert state.logical_evaluations == 5
    assert state.admitted_native_dispatches == 4
    assert state.completed_native_dispatches == 4
    assert state.exact_cache_hits == 1
    assert [leaf.result_source for leaf in state.leaves] == [
        "EXACT_CACHE_HIT", "COMPUTED",
    ]


def test_anytime_api_refuses_non_synthetic_evaluator(monkeypatch):
    monkeypatch.setattr(
        certificate, "integrate_signed_curvature",
        lambda cells, h: _curvature((-2, 2), (-2, 2)),
    )
    evaluator = SyntheticEvaluator()
    evaluator.synthetic_only = False
    with pytest.raises(RuntimeError, match="REAL_EXECUTION_UNAUTHORIZED"):
        initialize_monotone_anytime_state(
            evaluator, Fraction(1), P, _plan(),
            resource_lock_semantic_hash=RESOURCE_HASH,
        )
