"""One-way resource rebind from a frozen descriptor into a new packed plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_native_descriptor import (
    load_native_execution_descriptor, program_execution_identity,
)
from green_bridge_v400_resident_plan import (
    build_resident_plan, load_resident_plan_arrays,
)
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


APPROVED_SOURCE_PROGRAM_SHA256 = (
    "38f40999524d465b8ee58fcc8d2d1822caf9af6c36897a72bd404a8fff34fe62"
)
APPROVED_SOURCE_PLAN_SHA256 = (
    "0d5625e2f7af118615497e9642481946aec0a436b900e3c0d1661f90ba6f9acf"
)
APPROVED_DESCRIPTOR_FILE_SHA256 = (
    "bc673467ac237e59e542634d38d02b8eaa12053cbb0abfc39e4dcaa6659ba3ee"
)


def _require_new_sdb_directory(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = Path("/mnt/sdb").resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError("descriptor rebind output must resolve below /mnt/sdb") from error
    if not relative.parts or resolved.exists():
        raise RuntimeError("descriptor rebind output must be new below /mnt/sdb")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-program", required=True)
    parser.add_argument("--tensor-manifest", required=True)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = _require_new_sdb_directory(Path(args.output_root))
    source = TensorProgram.from_dict(json.loads(
        Path(args.source_program).read_text(encoding="utf-8")
    ))
    reader = TensorStoreReader(Path(args.tensor_manifest))
    source_plan, _ = load_resident_plan_arrays(
        Path(args.source_plan), source, reader
    )
    descriptor = load_native_execution_descriptor(
        Path(args.descriptor), source, source_plan
    )
    if (source.semantic_hash() != APPROVED_SOURCE_PROGRAM_SHA256
            or source_plan["resident_plan_semantic_hash"] != APPROVED_SOURCE_PLAN_SHA256
            or descriptor["descriptor_file_sha256"]
                != APPROVED_DESCRIPTOR_FILE_SHA256):
        raise RuntimeError("OUTCOME_BLIND_DESCRIPTOR_REBIND_IDENTITY_MISMATCH")
    descriptor_binding = {
        "schema_version": "green-v400-native-execution-descriptor-binding-v1",
        "descriptor_file_sha256": descriptor["descriptor_file_sha256"],
        "descriptor_payload_sha256": descriptor["descriptor_payload_sha256"],
        "program_execution_semantic_hash": descriptor["payload"][
            "program_execution_semantic_hash"
        ],
        "prebind_source_program_semantic_hash": source.semantic_hash(),
        "native_execution_ready": False,
        "claim_status": "PASS_NATIVE_DESCRIPTOR_PREPARE_ONLY",
    }
    rebound_payload = source.to_dict()
    rebound_payload["resource_formula"] = dict(rebound_payload["resource_formula"])
    rebound_payload["resource_formula"]["native_execution_descriptor"] = (
        descriptor_binding
    )
    rebound = TensorProgram.from_dict(rebound_payload)
    if program_execution_identity(rebound) != program_execution_identity(source):
        raise RuntimeError("descriptor resource rebind changed executable program")
    output_root.mkdir(parents=True)
    program_path = output_root / "tensor_program.json"
    program_path.write_text(
        json.dumps(rebound.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    descriptor_path = output_root / "gpt2_native_execution.desc"
    descriptor_path.write_bytes(Path(args.descriptor).read_bytes())
    resident_root = output_root / "resident_plan"
    resident_root.mkdir()
    build_resident_plan(resident_root, "gpt2_resident", rebound, reader)
    rebound_plan, _ = load_resident_plan_arrays(
        resident_root / "gpt2_resident.json", rebound, reader
    )
    replay = load_native_execution_descriptor(
        descriptor_path, rebound, rebound_plan
    )
    if replay["descriptor_file_sha256"] != descriptor["descriptor_file_sha256"]:
        raise RuntimeError("descriptor bytes changed during resource rebind")
    report = {
        "schema_version": "green-v400-native-descriptor-resource-rebind-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_scientific_outcome": False,
        "status": "PASS_NATIVE_DESCRIPTOR_RESOURCE_REBIND_PREPARE_ONLY",
        "native_execution_ready": False,
        "source_program_semantic_hash": source.semantic_hash(),
        "rebound_program_semantic_hash": rebound.semantic_hash(),
        "program_execution_semantic_hash": sha256_canonical(
            program_execution_identity(rebound)
        ),
        "executable_program_identity_unchanged": True,
        "source_resident_plan_semantic_hash": source_plan[
            "resident_plan_semantic_hash"
        ],
        "rebound_resident_plan_semantic_hash": rebound_plan[
            "resident_plan_semantic_hash"
        ],
        "packed_blob_sha256_unchanged": (
            rebound_plan["blob_sha256"] == source_plan["blob_sha256"]
        ),
        "descriptor_file_sha256": replay["descriptor_file_sha256"],
        "descriptor_payload_sha256": replay["descriptor_payload_sha256"],
        "descriptor_replays_against_rebound_program_and_plan": True,
        "resource_binding": descriptor_binding,
        "generator_file_sha256": _sha256_file(Path(__file__)),
        "input_source_program_file_sha256": _sha256_file(Path(args.source_program)),
        "input_tensor_manifest_file_sha256": _sha256_file(Path(args.tensor_manifest)),
        "input_source_plan_file_sha256": _sha256_file(Path(args.source_plan)),
        "input_descriptor_file_sha256": _sha256_file(Path(args.descriptor)),
        "argv": sys.argv,
    }
    if not report["packed_blob_sha256_unchanged"]:
        raise RuntimeError("resource rebind changed packed blob bytes")
    (output_root / "rebind_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
