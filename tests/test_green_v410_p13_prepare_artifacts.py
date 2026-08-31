from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_directions import tensor_sha256
from green_v410_protocol import sha256_canonical


ARTIFACT_ROOT = ROOT / "analysis" / "GREEN_V410_P13_QUALIFICATION_PREPARE_20260831"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def test_prepare_receipt_is_self_hashed_and_outcome_free():
    receipt = load("prepare_receipt.json")
    claimed = receipt.pop("receipt_sha256")
    assert claimed == sha256_canonical(receipt)
    assert receipt["contains_scientific_outcome"] is False
    assert receipt["model_weights_loaded"] is False


def test_task_manifests_and_direction_registries_are_closed_by_hash():
    receipt = load("prepare_receipt.json")
    by_task = {row["task"]: row for row in receipt["task_results"]}
    for task in ("ioi", "greater_than"):
        manifest = load(f"{task}_p13_qualification_manifest.json")
        registry = load(f"{task}_p13_direction_registry.json")
        claimed_manifest = manifest.pop("manifest_sha256")
        claimed_registry = registry.pop("registry_sha256")
        assert claimed_manifest == sha256_canonical(manifest)
        assert claimed_registry == sha256_canonical(registry)
        assert by_task[task]["qualification_manifest_sha256"] == claimed_manifest
        assert by_task[task]["direction_registry_sha256"] == claimed_registry
        assert manifest["prompt_count"] == 12
        assert manifest["site_count"] == 108
        assert manifest["direction_count"] == 864
        assert registry["shape"] == [108, 8, 768]


def test_every_direction_row_matches_its_frozen_binding():
    for task in ("ioi", "greater_than"):
        registry = load(f"{task}_p13_direction_registry.json")
        directions_path = ARTIFACT_ROOT / registry["payload_filename"]
        assert file_sha256(directions_path) == registry["payload_file_sha256"]
        values = np.load(directions_path, mmap_mode="r", allow_pickle=False)
        assert list(values.shape) == registry["shape"]
        for binding in registry["bindings"]:
            row = np.asarray(values[binding["row_index"]])
            claimed = binding["binding_sha256"]
            unhashed = {key: value for key, value in binding.items()
                        if key not in {"row_index", "binding_sha256"}}
            assert claimed == sha256_canonical(unhashed)
            assert binding["tensor_sha256"] == tensor_sha256(row)


def test_gt_selection_has_exact_twelve_strata():
    registry = load("greater_than_p13_direction_registry.json")
    prompts = registry["selected_prompts"]
    assert len({
        (row["century"], row["distance_bin"], row["orientation"])
        for row in prompts
    }) == 12
    assert {row["noun"] for row in prompts} <= {
        "ceremony", "committee", "federation", "workshop"
    }


def test_ioi_selection_uses_only_reserve_zero():
    registry = load("ioi_p13_direction_registry.json")
    assert len(registry["selected_prompts"]) == 12
    assert {
        (row["role"], row["template_id"]) for row in registry["selected_prompts"]
    } == {("unused_reserve", "reserve_0")}

