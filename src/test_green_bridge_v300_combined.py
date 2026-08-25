"""Authoritative 220 + 52 GREEN v3 contract harness."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import test_green_bridge_contract
from tests import test_green_bridge_v300_contract


def main() -> None:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite((
        loader.loadTestsFromModule(test_green_bridge_contract),
        loader.loadTestsFromModule(test_green_bridge_v300_contract),
    ))
    if suite.countTestCases() != 272:
        raise SystemExit(f"expected 272 tests, found {suite.countTestCases()}")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)


if __name__ == "__main__":
    main()
