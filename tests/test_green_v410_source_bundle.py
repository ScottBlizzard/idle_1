from __future__ import annotations

from pathlib import Path

from green_v410_source_bundle import build_source_bundle_manifest


ROOT = Path(__file__).resolve().parent.parent


def test_runtime_source_bundle_is_stable_and_closed():
    seeds = [ROOT / "analysis" / "green_v410_p13_certificate_worker.py"]
    frozen = [ROOT / "configs" / "green_v410_sfc_jwtec_protocol.json"]
    first = build_source_bundle_manifest(root=ROOT, seeds=seeds, frozen_inputs=frozen)
    second = build_source_bundle_manifest(root=ROOT, seeds=seeds, frozen_inputs=frozen)
    assert first == second
    paths = {row["path"] for row in first["files"]}
    assert "analysis/green_v410_p13_certificate_worker.py" in paths
    assert "src/green_v410_fixed_budget_certificate.py" in paths
    assert "src/green_bridge_v400_tensor_program.py" in paths
    assert "configs/green_v410_sfc_jwtec_protocol.json" in paths
