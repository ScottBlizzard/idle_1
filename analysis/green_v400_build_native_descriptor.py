"""Build the prepare-only native descriptor for the approved synthetic packed plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_native_descriptor import build_native_execution_descriptor
from green_bridge_v400_resident_plan import load_resident_plan_arrays
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


APPROVED_SYNTHETIC_PROGRAM_SHA256 = (
    "38f40999524d465b8ee58fcc8d2d1822caf9af6c36897a72bd404a8fff34fe62"
)
APPROVED_SYNTHETIC_RESIDENT_PLAN_SHA256 = (
    "0d5625e2f7af118615497e9642481946aec0a436b900e3c0d1661f90ba6f9acf"
)


def _require_new_sdb_directory(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = Path("/mnt/sdb").resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError("descriptor output must resolve below /mnt/sdb") from error
    if not relative.parts or resolved.exists():
        raise RuntimeError("descriptor output must be a new directory below /mnt/sdb")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--name", default="gpt2_native_execution")
    args = parser.parse_args()
    output_root = _require_new_sdb_directory(Path(args.output_root))
    program = TensorProgram.from_dict(json.loads(
        Path(args.program).read_text(encoding="utf-8")
    ))
    reader = TensorStoreReader(Path(args.tensor_manifest))
    plan, _ = load_resident_plan_arrays(Path(args.plan), program, reader)
    if (program.semantic_hash() != APPROVED_SYNTHETIC_PROGRAM_SHA256
            or plan["resident_plan_semantic_hash"]
                != APPROVED_SYNTHETIC_RESIDENT_PLAN_SHA256):
        raise RuntimeError("OUTCOME_BLIND_SYNTHETIC_FIXTURE_IDENTITY_MISMATCH")
    output_root.mkdir(parents=True)
    descriptor_path = output_root / f"{args.name}.desc"
    result = build_native_execution_descriptor(descriptor_path, program, plan)
    payload = result["payload"]
    report = {
        "schema_version": "green-v400-native-descriptor-build-report-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "status": "PASS_NATIVE_DESCRIPTOR_PREPARE_ONLY",
        "native_execution_ready": False,
        "source_program_semantic_hash": program.semantic_hash(),
        "program_execution_semantic_hash": payload[
            "program_execution_semantic_hash"
        ],
        "resident_plan_semantic_hash": plan["resident_plan_semantic_hash"],
        "blob_sha256": payload["blob_sha256"],
        "descriptor_file_sha256": result["descriptor_file_sha256"],
        "descriptor_payload_sha256": result["descriptor_payload_sha256"],
        "descriptor_nbytes": result["descriptor_nbytes"],
        "record_count": len(payload["records"]),
        "node_count": len(payload["program_execution_identity"]["nodes"]),
        "binding_count": len(payload["program_input_binding_table"]),
        "fusion_weight_count": len(payload["exact_final_contrast_fusion"]["weights"]),
        "liveness_node_count": len(payload["required_axis0_rows"]),
        "descriptor_path": str(descriptor_path),
        "generator_file_sha256": _sha256_file(Path(__file__)),
        "input_program_file_sha256": _sha256_file(Path(args.program)),
        "input_tensor_manifest_file_sha256": _sha256_file(Path(args.tensor_manifest)),
        "input_resident_plan_file_sha256": _sha256_file(Path(args.plan)),
        "argv": sys.argv,
    }
    report_path = output_root / "build_report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
