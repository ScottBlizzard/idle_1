"""Build tokenizer-only P13 qualification manifests and public directions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.green_v400_greater_than_universe_prepare import build_untouched_universe as build_gt
from analysis.green_v400_ioi_universe_prepare import build_untouched_universe as build_ioi
from green_v410_artifacts import atomic_no_clobber_bytes, atomic_no_clobber_json, file_sha256
from green_v410_directions import build_row_binding, generate_site_directions
from green_v410_p13 import select_gt_qualification, select_ioi_qualification
from green_v410_protocol import PROTOCOL_ID, sha256_canonical


class TokenizerAdapter:
    def __init__(self, tokenizer_json: Path):
        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("special tokens are forbidden in the frozen tokenizer adapter")
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_task_payload(
    *, task: str, universe: dict, qualification: dict, output_directory: Path,
) -> dict:
    sites = qualification["sites"]
    array = np.empty((len(sites), 8, 768), dtype="<f4")
    bindings = []
    for index, site in enumerate(sites):
        directions = generate_site_directions(site["row_id"], qualification["direction_domain"])
        array[index] = directions
        binding = build_row_binding(
            task=task,
            site_row_id=site["row_id"],
            direction_domain=qualification["direction_domain"],
            array=directions,
        )
        bindings.append({"row_index": index, **binding})
    payload_path = output_directory / f"{task}_p13_directions.npy"
    import io
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    publication = atomic_no_clobber_bytes(
        payload_path, buffer.getvalue(), job_id=f"p13-{task}-directions"
    )
    selected_ids = set(qualification["selected_prompt_row_ids"])
    selected_prompts = [row for row in universe["rows"] if row["row_id"] in selected_ids]
    if len(selected_prompts) != 12:
        raise ValueError("selected P13 prompt material is incomplete")
    registry = {
        "schema_version": "green-v410-p13-direction-registry-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": task,
        "contains_scientific_outcome": False,
        "model_weights_loaded": False,
        "tokenizer_only_universe": True,
        "parent_universe_rows_sha256": universe["rows_sha256"],
        "parent_universe_config_sha256": universe["config_sha256"],
        "qualification_manifest_sha256": qualification["manifest_sha256"],
        "selected_prompts": selected_prompts,
        "selected_prompts_sha256": sha256_canonical(selected_prompts),
        "payload_filename": payload_path.name,
        "payload_file_sha256": publication["file_sha256"],
        "shape": [108, 8, 768],
        "dtype": "float32-little-endian",
        "bindings": bindings,
        "bindings_sha256": sha256_canonical(bindings),
    }
    registry["registry_sha256"] = sha256_canonical(registry)
    atomic_no_clobber_json(
        output_directory / f"{task}_p13_qualification_manifest.json",
        qualification,
        job_id=f"p13-{task}-manifest",
    )
    atomic_no_clobber_json(
        output_directory / f"{task}_p13_direction_registry.json",
        registry,
        job_id=f"p13-{task}-registry",
    )
    return {
        "task": task,
        "qualification_manifest_sha256": qualification["manifest_sha256"],
        "direction_registry_sha256": registry["registry_sha256"],
        "direction_payload_file_sha256": publication["file_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer = TokenizerAdapter(args.tokenizer_json)
    ioi_config_path = ROOT / "configs" / "green_v400_ioi_untouched_universe.json"
    gt_config_path = ROOT / "configs" / "green_v400_greater_than_untouched_universe.json"
    ioi_universe = build_ioi(tokenizer, _load(ioi_config_path))
    gt_universe = build_gt(tokenizer, _load(gt_config_path))
    results = [
        _build_task_payload(
            task="ioi", universe=ioi_universe,
            qualification=select_ioi_qualification(ioi_universe["rows"]),
            output_directory=args.output_directory,
        ),
        _build_task_payload(
            task="greater_than", universe=gt_universe,
            qualification=select_gt_qualification(gt_universe["rows"]),
            output_directory=args.output_directory,
        ),
    ]
    receipt = {
        "schema_version": "green-v410-p13-qualification-prepare-receipt-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "contains_scientific_outcome": False,
        "model_weights_loaded": False,
        "tokenizer_json_sha256": file_sha256(args.tokenizer_json),
        "ioi_universe_config_sha256": file_sha256(ioi_config_path),
        "gt_universe_config_sha256": file_sha256(gt_config_path),
        "task_results": results,
    }
    receipt["receipt_sha256"] = sha256_canonical(receipt)
    atomic_no_clobber_json(
        args.output_directory / "prepare_receipt.json", receipt,
        job_id="p13-qualification-prepare",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

