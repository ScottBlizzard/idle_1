"""Print outcome-blind P13 progress and frozen-runtime status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from green_v410_p13_runtime import load_object, progress_counts, verify_runtime_closure
from green_v410_artifacts import configure_exact_integer_io


def main() -> None:
    configure_exact_integer_io()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--compiled-backend", type=Path, required=True)
    args = parser.parse_args()
    source = load_object(args.run_root / "queues" / "source_bundle_manifest.json")
    report = {"runtime_closure": "PASS", "tasks": {}}
    for task in ("ioi", "greater_than"):
        queue = load_object(args.run_root / "queues" / f"{task}_p13_queue.json")
        verify_runtime_closure(
            source_root=args.source_root, queue=queue,
            source_manifest=source, compiled_backend=args.compiled_backend,
        )
        report["tasks"][task] = progress_counts(queue=queue, run_root=args.run_root)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
