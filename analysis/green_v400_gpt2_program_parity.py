"""Outcome-blind real-GPT-2 parity for the replayable four-branch TensorProgram."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_gpt2_program import (
    build_gpt2_joint_witness_program, execute_tensor_program_numpy,
    execute_tensor_program_torch,
    materialize_gpt2_joint_witness_store, program_identity_payload,
)
from green_bridge_v400_schemas import sha256_canonical


def _tensor_hash(value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if "/mnt/sdb/" not in output_root.as_posix() or output_root.exists():
        raise RuntimeError("parity output must be a new directory on /mnt/sdb")
    output_root.mkdir(parents=True)

    import torch
    import exp_green_bridge_gpt2 as legacy
    from green_bridge_spec import PROBE_FRAME_DIM, SELECTED_GATES
    from green_bridge_tail import GreenBridgeTail, capture_tail_anchor

    tokenizer, hf_model, model, observed = legacy.load_models(args.device)
    model.eval(); hf_model.eval()
    clean_ids = tokenizer.encode("The cat sat on the mat.", add_special_tokens=False)
    replacement = tokenizer.encode(" dog", add_special_tokens=False)
    if len(clean_ids) < 4 or len(replacement) != 1:
        raise RuntimeError("synthetic token fixture is not stable")
    corrupt_ids = list(clean_ids)
    corrupt_ids[1] = replacement[0]
    clean = torch.tensor([clean_ids], dtype=torch.long, device=args.device)
    corrupt = torch.tensor([corrupt_ids], dtype=torch.long, device=args.device)
    suffix_ids = torch.arange(100, 200, dtype=torch.long, device=args.device)
    coefficients = torch.where(
        torch.arange(100, device=args.device) % 2 == 0,
        torch.ones(100, device=args.device), -torch.ones(100, device=args.device),
    ).to(torch.float64) / 50.0
    raw_direction = torch.sin(
        torch.arange(model.cfg.d_model, dtype=torch.float64, device=args.device) + 1
    )
    direction = (raw_direction / torch.linalg.vector_norm(raw_direction)).to(model.W_U.dtype)

    with torch.inference_mode():
        tar = capture_tail_anchor(model, clean, suffix_ids, system="synthetic_TAR")
        pat = capture_tail_anchor(
            model, corrupt, suffix_ids, system="synthetic_PAT", block8_patch=tar.mlp8_out,
        )
    store_root = output_root / "tensor_store"
    reader, dims = materialize_gpt2_joint_witness_store(
        store_root, "gpt2_synthetic", model, pat, tar, direction,
        suffix_ids, coefficients, SELECTED_GATES,
    )
    model_manifest_hash = sha256_canonical({
        "model_config": observed,
        "state_dict": {name: _tensor_hash(value) for name, value in model.state_dict().items()},
    })
    program = build_gpt2_joint_witness_program(reader, model_manifest_hash, dims)
    (output_root / "tensor_program.json").write_text(
        json.dumps(program.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    residual_basis = torch.zeros(
        (model.cfg.d_model, PROBE_FRAME_DIM), dtype=model.W_U.dtype, device=args.device
    )
    tail = GreenBridgeTail(model, residual_basis, suffix_ids, SELECTED_GATES)
    rows = []
    for numerator, exponent in ((0, 0), (1, -8), (-1, -8), (1, -5)):
        t = numerator * (2.0 ** exponent)
        delta = direction[None, :] * t
        manual = {}
        manual_traces = {}
        with torch.inference_mode():
            for condition, anchor in (("PAT", pat), ("TAR", tar)):
                joint_logits, joint_trace = tail.evaluate_physical_with_trace(
                    anchor, delta, torch.zeros((1, 10), device=args.device),
                    mode="joint", subtract_residual_bypass=False,
                )
                bypass_logits, bypass_trace = tail.evaluate_physical_with_trace(
                    anchor, delta, torch.zeros((1,), device=args.device),
                    mode="control", gate_slot=0, subtract_residual_bypass=False,
                )
                manual[f"{condition}_J"] = float((joint_logits[0].double() * coefficients).sum().item())
                manual[f"{condition}_B"] = float((bypass_logits[0].double() * coefficients).sum().item())
                manual_traces[f"{condition}_J"] = joint_trace
                manual_traces[f"{condition}_B"] = bypass_trace
        manual_output = manual["PAT_J"] - manual["PAT_B"] - manual["TAR_J"] + manual["TAR_B"]
        replay = execute_tensor_program_torch(
            program, reader, t, args.device, return_node_values=True
        )
        numpy_replay = execute_tensor_program_numpy(program, reader, t)
        errors = {name: abs(float(replay[name].item()) - manual[name]) for name in manual}
        errors["output"] = abs(float(replay["output"].item()) - manual_output)
        by_provenance = {
            node.provenance_identity: replay["node_values"][node.semantic_id]
            for node in program.nodes
        }
        stage_errors = {}
        for branch, trace in manual_traces.items():
            condition, mode = branch.split("_")
            base_prefix = f"{condition}.{mode}"
            resid_key = (f"{condition}.J.block10.resid_post" if mode == "J"
                         else f"{condition}.shared_residual_bypass")
            comparisons = {
                "block10_resid_post": (by_provenance[resid_key], trace["resid_post_after_subtraction"][0]),
                "block11_resid_post": (by_provenance[f"{base_prefix}.block11.resid_post"], trace["block11_resid_post"][0]),
                "ln_final": (by_provenance[f"{base_prefix}.ln_final"], trace["ln_final_output"][0]),
            }
            stage_errors[branch] = {
                name: float((actual - expected).abs().max().item())
                for name, (actual, expected) in comparisons.items()
            }
        rows.append({
            "dyadic_t": {"numerator": numerator, "exponent": exponent},
            "manual": manual | {"output": manual_output},
            "replay": {name: float(replay[name].item()) for name in (*manual, "output")},
            "numpy_audit_replay": {name: float(numpy_replay[name]) for name in (*manual, "output")},
            "absolute_error": errors,
            "stage_max_absolute_error": stage_errors,
            "max_absolute_error": max(errors.values()),
        })

    tolerance = 3e-4
    report = {
        "schema_version": "green-v400-gpt2-synthetic-program-parity-v1",
        "status": "PASS" if max(row["max_absolute_error"] for row in rows) <= tolerance else "FAIL",
        "tolerance": tolerance,
        "fixture": "synthetic tokens/direction/contrast; no scientific row or outcome",
        "identity": program_identity_payload(program, dims, reader),
        "rows": rows,
        "contains_scientific_outcome": False,
    }
    (output_root / "parity_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "max_absolute_error": max(row["max_absolute_error"] for row in rows),
        "output_root": str(output_root),
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
