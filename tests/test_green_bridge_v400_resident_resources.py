from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_resident_resources import (
    PRIMITIVE_TAXONOMY, gpt2_joint_witness_cell_jet2,
)


def test_full_size_resident_count_is_exact_and_taxonomy_is_narrow():
    assert gpt2_joint_witness_cell_jet2(
        d_model=768, d_mlp=3072, sequence_length=12,
        n_heads=12, d_head=64, selected_gates=10,
    ) == 352_275_450
    assert PRIMITIVE_TAXONOMY["schema_version"] == "green-v400-directed-enclosure-arithmetic-v1"
    assert "mpfr_set/copy" in PRIMITIVE_TAXONOMY["excluded"]
