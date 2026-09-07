"""Pure, endpoint-free P13 qualification queue construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from green_v410_artifacts import file_sha256
from green_v410_directions import direction_vector_sha256, tensor_sha256
from green_v410_protocol import (
    DECISION_RULE_SHA256,
    PROTOCOL_ID,
    is_lower_sha256,
    load_protocol_config,
    sha256_canonical,
    strict_fields,
)
from green_v410_schemas import SCHEMA_BUNDLE_SHA256, validate_artifact


QUEUE_IDS = {
    "ioi": "GREEN_V410_P13_IOI_CERTIFICATE_QUEUE_A1",
    "greater_than": "GREEN_V410_P13_GT_CERTIFICATE_QUEUE_A1",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P13 artifact is not an object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str, name: str) -> None:
    payload = dict(value)
    claimed = payload.pop(field, None)
    if claimed != sha256_canonical(payload):
        raise ValueError(f"{name} self hash mismatch")


def task_metric_spec(task: str) -> dict[str, Any]:
    if task == "ioi":
        return {
            "schema_version": "green-v410-ioi-task-metric-v1",
            "task": "ioi",
            "formula": "final_logit_io_minus_s",
            "requires_distinct_single_token_names": True,
        }
    if task == "greater_than":
        return {
            "schema_version": "green-v410-greater-than-task-metric-v1",
            "task": "greater_than",
            "formula": "balanced_mean_logit_above_y_minus_at_or_below_y",
            "suffix_count": 100,
            "clean_suffix_domain": [0, 98],
        }
    raise ValueError("unknown P13 task")


def build_p13_queue(
    *,
    task: str,
    prepare_root: Path,
    resource_manifest: Mapping[str, Any],
    confirmation_seal: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    source_bundle_sha256: str,
    compiled_backend_sha256: str,
) -> dict[str, Any]:
    if task not in QUEUE_IDS:
        raise ValueError("unknown P13 task")
    if not is_lower_sha256(compiled_backend_sha256):
        raise ValueError("P13 queue compiled backend hash is invalid")
    validate_artifact(resource_manifest)
    validate_artifact(confirmation_seal)
    if resource_manifest["selected_leaf_budget"] != 4:
        raise ValueError("P13 queue requires the frozen L4 successor resource manifest")
    if confirmation_seal["terminal_state"] != "PASS":
        raise ValueError("P13 queue requires a passing confirmation seal")

    qualification = _load(prepare_root / f"{task}_p13_qualification_manifest.json")
    registry = _load(prepare_root / f"{task}_p13_direction_registry.json")
    causal = _load(prepare_root / f"{task}_p13_causal_cone_manifest.json")
    payload_path = prepare_root / f"{task}_p13_directions.npy"
    _self_hash(qualification, "manifest_sha256", "qualification manifest")
    _self_hash(registry, "registry_sha256", "direction registry")
    _self_hash(causal, "manifest_sha256", "causal-cone manifest")
    if (
        qualification.get("task") != task
        or registry.get("task") != task
        or causal.get("task") != task
        or qualification.get("prompt_count") != 12
        or qualification.get("site_count") != 108
        or qualification.get("direction_count") != 864
        or causal.get("record_count") != 108
        or registry.get("qualification_manifest_sha256")
            != qualification.get("manifest_sha256")
        or causal.get("qualification_manifest_sha256")
            != qualification.get("manifest_sha256")
        or causal.get("direction_registry_sha256") != registry.get("registry_sha256")
        or file_sha256(payload_path) != registry.get("payload_file_sha256")
    ):
        raise ValueError("P13 prepare artifact closure mismatch")

    directions = np.load(payload_path, mmap_mode="r", allow_pickle=False)
    if directions.shape != (108, 8, 768) or directions.dtype.str != "<f4":
        raise ValueError("P13 direction payload shape or dtype mismatch")
    bindings = {row["site_row_id"]: row for row in registry["bindings"]}
    cones = {row["site_row_id"]: row for row in causal["records"]}
    prompts = {row["row_id"]: row for row in registry["selected_prompts"]}
    if len(bindings) != 108 or len(cones) != 108 or len(prompts) != 12:
        raise ValueError("P13 prepare identities are duplicated or incomplete")

    protocol = load_protocol_config()
    model_manifest_sha256 = sha256_canonical(model_manifest)
    tokenizer_manifest_sha256 = model_manifest.get("tokenizer_hash")
    selected_gate_spec_sha256 = sha256_canonical(protocol["gate"])
    task_metric_spec_sha256 = sha256_canonical(task_metric_spec(task))
    jobs = []
    for site in qualification["sites"]:
        site_id = site["row_id"]
        prompt_id = site["prompt_row_id"]
        binding = bindings[site_id]
        cone = cones[site_id]
        prompt = prompts[prompt_id]
        row_index = int(binding["row_index"])
        panel = np.asarray(directions[row_index], dtype="<f4")
        if tensor_sha256(panel) != binding["tensor_sha256"]:
            raise ValueError("P13 site direction binding mismatch")
        position = (
            int(prompt["signature"][1]) if task == "ioi"
            else int(prompt["site_position"])
        )
        common_hashes = {
            "source_bundle_sha256": source_bundle_sha256,
            "compiled_backend_sha256": compiled_backend_sha256,
            "schema_bundle_sha256": SCHEMA_BUNDLE_SHA256,
            "model_manifest_sha256": model_manifest_sha256,
            "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
            "selected_gate_spec_sha256": selected_gate_spec_sha256,
            "task_metric_spec_sha256": task_metric_spec_sha256,
            "resource_manifest_sha256": resource_manifest["manifest_sha256"],
            "confirmation_seal_sha256": confirmation_seal["receipt_sha256"],
            "decision_rule_sha256": DECISION_RULE_SHA256,
        }
        direction_panel = [{
            "direction_ordinal": ordinal,
            "direction_payload_sha256": direction_vector_sha256(panel[ordinal]),
            "site_direction_tensor_sha256": binding["tensor_sha256"],
            "direction_binding_sha256": binding["binding_sha256"],
        } for ordinal in range(8)]
        identity = {
            "queue_id": QUEUE_IDS[task],
            "task": task,
            "phase": "qualification",
            "prompt_row_id": prompt_id,
            "site_row_id": site_id,
            "layer": int(site["layer"]),
            "hook": "resid_post",
            "site_position": position,
            "causal_cone_plan_sha256": cone["cone_plan_sha256"],
            "direction_panel": direction_panel,
            "selected_leaf_budget": 4,
            **common_hashes,
        }
        jobs.append({"job_id": sha256_canonical(identity), **identity})
    jobs.sort(key=lambda row: (
        row["task"], row["phase"], row["prompt_row_id"], row["layer"], row["site_row_id"]
    ))
    if len(jobs) != 108 or len({row["job_id"] for row in jobs}) != 108:
        raise ValueError("P13 queue job count or identity mismatch")
    queue = {
        "schema_version": "green-v410-p13-certificate-queue-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "queue_id": QUEUE_IDS[task],
        "phase": "qualification",
        "task": task,
        "job_count": 108,
        "direction_record_count": 864,
        "sort_key": ["task", "phase", "prompt_row_id", "layer", "site_row_id"],
        "qualification_manifest_sha256": qualification["manifest_sha256"],
        "direction_registry_sha256": registry["registry_sha256"],
        "direction_payload_file_sha256": registry["payload_file_sha256"],
        "causal_cone_manifest_sha256": causal["manifest_sha256"],
        "resource_manifest_sha256": resource_manifest["manifest_sha256"],
        "confirmation_seal_sha256": confirmation_seal["receipt_sha256"],
        "source_bundle_sha256": source_bundle_sha256,
        "compiled_backend_sha256": compiled_backend_sha256,
        "schema_bundle_sha256": SCHEMA_BUNDLE_SHA256,
        "model_manifest_sha256": model_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "selected_gate_spec_sha256": selected_gate_spec_sha256,
        "task_metric_spec_sha256": task_metric_spec_sha256,
        "decision_rule_sha256": DECISION_RULE_SHA256,
        "contains_endpoint_material": False,
        "jobs": jobs,
    }
    return queue | {"queue_manifest_sha256": sha256_canonical(queue)}


def validate_p13_queue(queue: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "protocol_id", "attempt_index", "queue_id", "phase",
        "task", "job_count", "direction_record_count", "sort_key",
        "qualification_manifest_sha256", "direction_registry_sha256",
        "direction_payload_file_sha256", "causal_cone_manifest_sha256",
        "resource_manifest_sha256", "confirmation_seal_sha256",
        "source_bundle_sha256", "compiled_backend_sha256", "schema_bundle_sha256",
        "model_manifest_sha256",
        "tokenizer_manifest_sha256", "selected_gate_spec_sha256",
        "task_metric_spec_sha256", "decision_rule_sha256",
        "contains_endpoint_material", "jobs", "queue_manifest_sha256",
    }
    strict_fields(queue, expected, "P13 queue")
    payload = dict(queue)
    claimed = payload.pop("queue_manifest_sha256")
    if claimed != sha256_canonical(payload):
        raise ValueError("P13 queue self hash mismatch")
    task = queue["task"]
    if (
        task not in QUEUE_IDS or queue["queue_id"] != QUEUE_IDS[task]
        or queue["protocol_id"] != PROTOCOL_ID or queue["attempt_index"] != 1
        or queue["phase"] != "qualification" or queue["job_count"] != 108
        or queue["direction_record_count"] != 864
        or queue["contains_endpoint_material"] is not False
        or not is_lower_sha256(queue["compiled_backend_sha256"])
        or len(queue["jobs"]) != 108
    ):
        raise ValueError("P13 queue contract mismatch")
    if any(len(job.get("direction_panel", [])) != 8 for job in queue["jobs"]):
        raise ValueError("P13 queue direction panel mismatch")
