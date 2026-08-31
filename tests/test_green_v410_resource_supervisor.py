from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_resource_supervisor import build_resource_queue


def test_resource_queue_is_complete_serial_and_phase_major(tmp_path):
    queue = build_resource_queue(tmp_path / "bundles", tmp_path / "raw")
    assert queue["job_count"] == 240
    assert [row["ordinal"] for row in queue["jobs"]] == list(range(240))
    for offset in range(0, 240, 60):
        candidate_jobs = queue["jobs"][offset:offset + 60]
        assert {row["precision_bits"] for row in candidate_jobs[:30]} == {384}
        assert {row["precision_bits"] for row in candidate_jobs[30:]} == {512}
        for official, audit in zip(candidate_jobs[:30], candidate_jobs[30:]):
            assert (official["profile"], official["fixture_kind"]) == (
                audit["profile"], audit["fixture_kind"]
            )
            assert audit["official_artifact_path"] == official["output_path"]
    assert queue["scheduler"] == "strict_serial_one_cold_process_at_a_time"
