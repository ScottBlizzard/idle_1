"""Exercise the actual five-output resident evaluator across different domains."""
import os
from pathlib import Path

import pytest

from test_green_v410_gpt2_program import _case
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr_tensor_executor import execute_tensor_program_mpfr, jet_exact_payload
from green_v410_fixed_budget_certificate import FixedBudgetProgramEvaluator, certify_five_outputs


@pytest.mark.parametrize('layer', [0, 4, 8])
def test_resident_full_cone_complete_certificate(tmp_path, layer):
    path = os.environ.get('GREEN_V400_MPFR_BACKEND')
    if not path:
        pytest.skip('compiled MPFR backend not configured')
    case = _case(tmp_path, layer)
    evaluator = FixedBudgetProgramEvaluator(case[6], case[4], Path(path))
    try:
        certificate = certify_five_outputs(evaluator, leaf_budget=4)
        assert certificate['precision_nested']
        for domain in (Interval.point(1, 384), Interval.from_bounds(-1, 0, 384),
                       Interval.point(0, 384), Interval.from_bounds(0, 1, 384)):
            actual = evaluator.evaluate(domain)
            reference = execute_tensor_program_mpfr(
                case[6], case[4], domain, evaluator.backend,
                sparse_axis0_execution=True, preloaded_tensors=evaluator.preloaded,
            )
            for name in actual:
                key = 'output' if name == 'PSI' else name
                assert jet_exact_payload(actual[name]) == jet_exact_payload(reference[key]), name
    finally:
        evaluator.close()
