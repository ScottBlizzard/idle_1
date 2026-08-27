from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_compiled_mpfr import ExactDomainJetMemo
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import Jet2


def _identity(row="a" * 64):
    return {
        "schema_version": "green-v400-exact-domain-evaluator-identity-v1",
        "certificate_row_hash": row,
        "native_plan_identity_sha256": "b" * 64,
        "backend_library_sha256": "c" * 64,
        "backend_version": "test",
        "expected_kernel_tags_sha256": "d" * 64,
        "rounding_environment_sha256": "e" * 64,
    }


def _jet(precision, value=1):
    point = Interval.point(value, precision)
    zero = Interval.point(0, precision)
    return Jet2(point, zero, zero)


def test_exact_domain_memo_caches_only_success_and_retries_failure():
    memo = ExactDomainJetMemo(_identity(), max_entries=4)
    domain = Interval.from_bounds(-1, 0, 384)
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic dispatch failure")
        return _jet(384)

    with pytest.raises(RuntimeError, match="synthetic"):
        memo.get_or_compute(domain, fail_once)
    assert memo.metrics()["entry_count"] == 0
    result = memo.get_or_compute(domain, fail_once)
    assert result == _jet(384)
    assert calls == 2
    assert memo.metrics()["entry_count"] == 1


def test_exact_domain_memo_isolates_precision_and_evicts_deterministically():
    memo = ExactDomainJetMemo(_identity(), max_entries=2)
    calls = []
    for precision in (384, 512):
        domain = Interval.from_bounds(0, 1, precision)
        memo.get_or_compute(domain, lambda p=precision: calls.append(p) or _jet(p))
    assert calls == [384, 512]
    memo.get_or_compute(
        Interval.from_bounds(1, 2, 384), lambda: calls.append(385) or _jet(384, 2)
    )
    assert memo.metrics()["entry_count"] == 2
    memo.get_or_compute(
        Interval.from_bounds(0, 1, 384), lambda: calls.append(386) or _jet(384)
    )
    assert calls == [384, 512, 385, 386]


def test_exact_domain_memo_single_flight_computes_once():
    memo = ExactDomainJetMemo(_identity(), max_entries=4)
    domain = Interval.from_bounds(-1, 1, 384)
    calls = 0
    calls_lock = threading.Lock()
    barrier = threading.Barrier(16)

    def compute():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return _jet(384)

    def worker():
        barrier.wait(timeout=2)
        return memo.get_or_compute(domain, compute)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: worker(), range(16)))
    assert calls == 1
    assert all(result == results[0] for result in results)
    metrics = memo.metrics()["by_precision"]["384"]
    assert metrics == {
        "logical_requests": 16, "hits": 0, "misses": 1, "waits": 15,
    }


def test_exact_domain_memo_rejects_invalid_compute_result():
    memo = ExactDomainJetMemo(_identity(), max_entries=1)
    domain = Interval.from_bounds(0, 1, 384)
    with pytest.raises(RuntimeError, match="invalid Jet2"):
        memo.get_or_compute(domain, lambda: _jet(512))
    assert memo.metrics()["entry_count"] == 0
