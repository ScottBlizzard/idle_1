"""Bind the frozen causal-cone semantics to every prepared P13 site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_v410_artifacts import atomic_no_clobber_json
from green_v410_causal_cone import build_causal_cone_plan
from green_v410_protocol import PROTOCOL_ID, sha256_canonical


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bind_task(root: Path, task: str) -> dict:
    qualification = _load(root / f"{task}_p13_qualification_manifest.json")
    registry = _load(root / f"{task}_p13_direction_registry.json")
    prompts = {row["row_id"]: row for row in registry["selected_prompts"]}
    records = []
    for site in qualification["sites"]:
        prompt = prompts[site["prompt_row_id"]]
        if task == "ioi":
            sequence_length = len(prompt["clean_token_ids"])
            patch_position = int(prompt["signature"][1])
        else:
            sequence_length = int(prompt["sequence_length"])
            patch_position = int(prompt["site_position"])
        plan = build_causal_cone_plan(
            site_layer=int(site["layer"]),
            patch_position=patch_position,
            sequence_length=sequence_length,
        )
        records.append({
            "site_row_id": site["row_id"],
            "prompt_row_id": site["prompt_row_id"],
            "layer": site["layer"],
            "patch_position": patch_position,
            "sequence_length": sequence_length,
            "cone_plan_sha256": plan["plan_sha256"],
            "cone_plan": plan,
        })
    records.sort(key=lambda row: (row["prompt_row_id"], row["layer"], row["site_row_id"]))
    if len(records) != 108 or len({row["site_row_id"] for row in records}) != 108:
        raise ValueError("P13 causal-cone binding count mismatch")
    manifest = {
        "schema_version": "green-v410-p13-causal-cone-binding-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": task,
        "qualification_manifest_sha256": qualification["manifest_sha256"],
        "direction_registry_sha256": registry["registry_sha256"],
        "graph_semantics_id": "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1",
        "record_count": 108,
        "records": records,
    }
    manifest["manifest_sha256"] = sha256_canonical(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    for task in ("ioi", "greater_than"):
        manifest = bind_task(args.artifact_root, task)
        atomic_no_clobber_json(
            args.artifact_root / f"{task}_p13_causal_cone_manifest.json",
            manifest,
            job_id=f"p13-{task}-causal-cones",
        )
        print(task, manifest["manifest_sha256"])


if __name__ == "__main__":
    main()

