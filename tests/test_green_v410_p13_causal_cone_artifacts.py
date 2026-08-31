from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_causal_cone import validate_causal_cone_plan
from green_v410_protocol import sha256_canonical


ARTIFACT_ROOT = ROOT / "analysis" / "GREEN_V410_P13_QUALIFICATION_PREPARE_20260831"


def test_all_real_p13_sites_have_valid_full_cone_bindings():
    for task in ("ioi", "greater_than"):
        path = ARTIFACT_ROOT / f"{task}_p13_causal_cone_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = payload.pop("manifest_sha256")
        assert claimed == sha256_canonical(payload)
        assert payload["record_count"] == 108
        assert len({row["site_row_id"] for row in payload["records"]}) == 108
        for row in payload["records"]:
            assert row["patch_position"] < row["sequence_length"] - 1
            assert row["cone_plan_sha256"] == row["cone_plan"]["plan_sha256"]
            validate_causal_cone_plan(row["cone_plan"])

