from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from green_bridge_v400_resident_plan import (
    RESIDENT_TENSOR_NAMES, build_resident_plan, validate_resident_plan,
    load_resident_plan_arrays,
)
from test_green_bridge_v400_gpt2_program import _fixture


def test_resident_plan_is_aligned_hash_closed_and_excludes_full_unembed(tmp_path):
    fixture = tmp_path / "fixture"; fixture.mkdir()
    reader, _, program = _fixture(fixture)
    output = tmp_path / "resident"; output.mkdir()
    manifest = build_resident_plan(output, "tiny", program, reader)
    replay = validate_resident_plan(output / "tiny.json", program, reader)
    assert replay == manifest
    assert len(manifest["records"]) == len(RESIDENT_TENSOR_NAMES)
    assert all(record["offset"] % 64 == 0 for record in manifest["records"])
    assert "unembed.W_U_full" not in {record["name"] for record in manifest["records"]}
    assert manifest["full_unembedding_excluded_after_exact_fusion"] is True
    assert len(manifest["resident_plan_semantic_hash"]) == 64
    _, arrays = load_resident_plan_arrays(output / "tiny.json", program, reader)
    assert set(arrays) == set(RESIDENT_TENSOR_NAMES)
    assert all(array.ctypes.data % 64 == 0 for array in arrays.values())
    assert manifest["program_unique_tensor_ref_count"] == 36
    assert manifest["native_execution_ready"] is False
    assert manifest["claim_status"] == "PASS_PACKED_RESIDENT_MANIFEST_PREPARE_ONLY"
    assert len(manifest["program_input_binding_table"]) > len(RESIDENT_TENSOR_NAMES)
