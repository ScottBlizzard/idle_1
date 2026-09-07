from __future__ import annotations

from types import SimpleNamespace

from analysis.green_v410_p13_materialize_worker import _graph_manifest


def test_graph_manifest_uses_tensor_program_output_root():
    nodes = [SimpleNamespace(semantic_id="n", provenance_identity={"role": "test"})]
    program = SimpleNamespace(
        nodes=nodes,
        branch_roots={
            "PAT_J": "pat-j", "PAT_B": "pat-b",
            "TAR_J": "tar-j", "TAR_B": "tar-b",
        },
        output_root="psi-root",
        resource_formula={"dependency_mask_closure_sha256": "a" * 64},
    )
    job = {
        "model_manifest_sha256": "b" * 64,
        "task_metric_spec_sha256": "c" * 64,
        "selected_gate_spec_sha256": "d" * 64,
    }
    direction = {"direction_ordinal": 0, "direction_payload_sha256": "e" * 64}
    site = {"site_identity_sha256": "f" * 64}
    manifest = _graph_manifest(
        job=job, direction=direction, site_identity=site,
        program=program, reader=None,
    )
    assert manifest["output_node_ids"][-1] == "psi-root"
