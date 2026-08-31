from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_resource_calibration import synthetic_modifier_jet
from green_v410_resource_schedule import (
    run_frozen_partition_audit,
    run_official_schedule,
)


class _PolynomialEvaluator:
    def evaluate_interval(self, domain):
        return synthetic_modifier_jet(domain, (Fraction(1, 16), Fraction(-1, 64)))


@pytest.mark.parametrize("budget", [4, 8, 16, 32])
def test_resource_schedule_has_exact_dispatch_counts_and_partition(budget):
    pytest.importorskip("gmpy2")
    evaluator = _PolynomialEvaluator()
    official = run_official_schedule(
        evaluator, (Fraction(511, 1024), Fraction(513, 1024)), budget,
    )
    audit = run_frozen_partition_audit(evaluator, official)
    assert official["dispatch_count"] == 2 * budget + 1
    assert len(official["final_cells"]) == budget
    assert audit["dispatch_count"] == budget + 3
    assert audit["all_nested"] is True
    assert audit["independent_audit_split_queue"] is False


def test_resource_schedule_replay_is_byte_identical():
    pytest.importorskip("gmpy2")
    evaluator = _PolynomialEvaluator()
    first = run_official_schedule(evaluator, (Fraction(-1), Fraction(0)), 8)
    second = run_official_schedule(evaluator, (Fraction(-1), Fraction(0)), 8)
    assert first == second
    assert run_frozen_partition_audit(evaluator, first) == run_frozen_partition_audit(
        evaluator, second
    )


def test_audit_rejects_mutated_official_partition():
    pytest.importorskip("gmpy2")
    official = run_official_schedule(_PolynomialEvaluator(), (Fraction(0), Fraction(1)), 4)
    official["final_cells"][0]["depth"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        run_frozen_partition_audit(_PolynomialEvaluator(), official)
