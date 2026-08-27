from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_resources import ProcessTreeResourceRecorder


def test_process_tree_record_observes_child_and_is_not_claimed_as_bound():
    with ProcessTreeResourceRecorder(sample_interval_seconds=0.002) as recorder:
        child = subprocess.Popen([
            sys.executable,
            "-c",
            "import time; payload=bytearray(8*1024*1024); time.sleep(0.15)",
        ])
        child.wait(timeout=5)
        time.sleep(0.01)
    record = recorder.record
    payload = record.to_dict()
    assert record.descendant_identity_count >= 1
    assert any(item.pid == child.pid for item in record.process_observations)
    assert record.peak_sampled_tree_rss_kib > 0
    assert record.root_peak_rss_kib > 0
    assert payload["measurement_scope"] == "sampled_root_and_observed_descendants"
    assert payload["is_formal_upper_bound"] is False
    assert payload["missed_processes_possible"] is True


def test_process_tree_record_is_unavailable_inside_scope():
    with ProcessTreeResourceRecorder() as recorder:
        try:
            recorder.record
        except RuntimeError as error:
            assert "only after" in str(error)
        else:
            raise AssertionError("unfinished resource record was exposed")
