from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import gmpy2


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import Jet2, add_jet, constant_jet, mul_jet, sub_jet
from green_v410_fixed_budget_certificate import certify_five_outputs


class PolynomialFiveOutputEvaluator:
    contains_scientific_outcome = False

    def evaluate(self, domain: Interval):
        zero = Interval.point(0, domain.precision_bits)
        one_interval = Interval.point(1, domain.precision_bits)
        t = Jet2(domain, one_interval, zero)
        one = constant_jet(one_interval)
        half = constant_jet(Interval.point(gmpy2.mpq(1, 2), domain.precision_bits))
        quarter = constant_jet(Interval.point(gmpy2.mpq(1, 4), domain.precision_bits))
        square = mul_jet(t, t)
        pat_j = add_jet(t, mul_jet(one, square))
        pat_b = mul_jet(one, square)
        tar_j = add_jet(mul_jet(t, half), mul_jet(quarter, square))
        tar_b = mul_jet(quarter, square)
        return {
            "PAT_J": pat_j,
            "PAT_B": pat_b,
            "TAR_J": tar_j,
            "TAR_B": tar_b,
            "PSI": add_jet(sub_jet(sub_jet(pat_j, pat_b), tar_j), tar_b),
        }


def _as_fraction(pair):
    return Fraction(*pair)


def test_fixed_budget_certificate_encloses_exact_derivatives():
    result = certify_five_outputs(PolynomialFiveOutputEvaluator(), leaf_budget=4)
    expected = {
        "PAT_J": Fraction(1),
        "PAT_B": Fraction(0),
        "TAR_J": Fraction(1, 2),
        "TAR_B": Fraction(0),
        "PSI": Fraction(1, 2),
    }
    assert result["precision_nested"] is True
    assert result["radii"] == ["1", "1/2", "1/4"]
    assert len(result["radius_reports"]) == 3
    for name, value in expected.items():
        interval = result["official_intervals"][name]
        assert _as_fraction(interval["lower"]) <= value <= _as_fraction(interval["upper"])
        audit = result["audit_intervals"][name]
        assert _as_fraction(interval["lower"]) <= _as_fraction(audit["lower"])
        assert _as_fraction(audit["upper"]) <= _as_fraction(interval["upper"])


def test_fixed_budget_certificate_replays_identical_partitions():
    result = certify_five_outputs(PolynomialFiveOutputEvaluator(), leaf_budget=4)
    for report in result["radius_reports"]:
        official = [(row["lower"], row["upper"], row["depth"])
                    for row in report["official_cells"]]
        audit = [(row["lower"], row["upper"], row["depth"])
                 for row in report["audit_cells"]]
        assert official == audit
        assert len(official) == 4
