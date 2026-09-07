import json
import sys

import pytest

from green_v410_artifacts import atomic_no_clobber_json, configure_exact_integer_io


def test_exact_large_rationals_roundtrip_without_decimal_conversion_loss(tmp_path):
    if not hasattr(sys, 'set_int_max_str_digits'):
        pytest.skip('Python decimal-conversion guard is not present')
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        value = {'lower': [-(1 << 30000), 1], 'upper': [(1 << 30000), 1]}
        with pytest.raises(ValueError, match='limit'):
            json.dumps(value)
        configure_exact_integer_io()
        path = tmp_path / 'exact.json'
        atomic_no_clobber_json(path, value, job_id='large-rational')
        assert json.loads(path.read_text()) == value
        assert atomic_no_clobber_json(path, value, job_id='large-rational')['publication'] == 'EXISTING_IDENTICAL'
    finally:
        sys.set_int_max_str_digits(previous)
