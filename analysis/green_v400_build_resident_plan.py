"""Build the outcome-blind packed plan for the actual resident dispatcher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_resident_plan import build_resident_plan
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--tensor-store", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--name", default="gpt2_resident")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if "/mnt/sdb/" not in output.as_posix() or output.exists():
        raise RuntimeError("resident-plan output must be a new directory on /mnt/sdb")
    output.mkdir(parents=True)
    program = TensorProgram.from_dict(json.loads(
        Path(args.program).read_text(encoding="utf-8")
    ))
    reader = TensorStoreReader(Path(args.tensor_store))
    started = time.perf_counter()
    manifest = build_resident_plan(output, args.name, program, reader)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "status": manifest["claim_status"],
        "output_root": str(output),
        "resident_plan_semantic_hash": manifest["resident_plan_semantic_hash"],
        "program_semantic_hash": manifest["program_semantic_hash"],
        "blob_nbytes": manifest["blob_nbytes"],
        "record_count": len(manifest["records"]),
        "build_and_validate_seconds": elapsed,
        "contains_scientific_outcome": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
