from __future__ import annotations

from pathlib import Path

from green_v410_p13_runtime import valid_capture, valid_certificate_record
from green_v410_protocol import sha256_canonical
from green_v410_p13_runtime import require_unstarted_certificate_unit
import pytest


def write_json(path: Path, value: dict) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_capture_progress_check_is_identity_strict(tmp_path):
    payload = {
        "job_id": "job", "queue_manifest_sha256": "a" * 64,
        "contains_endpoint_material": False,
    }
    value = payload | {"manifest_sha256": sha256_canonical(payload)}
    path = tmp_path / "capture.json"
    write_json(path, value)
    assert valid_capture(path, queue_sha256="a" * 64, job_id="job")
    assert not valid_capture(path, queue_sha256="b" * 64, job_id="job")


def test_certificate_progress_check_rejects_tampering(tmp_path):
    payload = {
        "queue_manifest_sha256": "a" * 64, "job_id": "job",
        "site_row_id": "site", "direction_ordinal": 3,
        "contains_endpoint_material": False,
    }
    value = payload | {"record_sha256": sha256_canonical(payload)}
    path = tmp_path / "record.json"
    write_json(path, value)
    assert valid_certificate_record(
        path, queue_sha256="a" * 64, job_id="job",
        site_row_id="site", direction_ordinal=3,
    )
    value["direction_ordinal"] = 4
    write_json(path, value)
    assert not valid_certificate_record(
        path, queue_sha256="a" * 64, job_id="job",
        site_row_id="site", direction_ordinal=3,
    )


@pytest.mark.parametrize('contents', ['', 'Traceback: numerical failure', 'worker started'])
def test_certificate_history_blocks_uncheckpointed_relaunch(tmp_path, contents):
    (tmp_path / 'job_00.attempt_001.log').write_text(contents)
    with pytest.raises(RuntimeError, match='RELAUNCH_FORBIDDEN'):
        require_unstarted_certificate_unit(tmp_path, 'job_00')
    require_unstarted_certificate_unit(tmp_path, 'job_01')


def test_new_certificate_unit_can_launch(tmp_path):
    require_unstarted_certificate_unit(tmp_path / 'not_created', 'job_00')
