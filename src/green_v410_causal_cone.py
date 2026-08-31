"""Exact dependency plan for the v4.1 full downstream causal cone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from green_v410_protocol import BRANCH_ORDER, BRANCH_WEIGHTS, PROTOCOL_ID, sha256_canonical


@dataclass(frozen=True)
class ConeStage:
    condition: str
    branch: str
    block: int | None
    operation: str
    dynamic_token_rows: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "branch": self.branch,
            "block": self.block,
            "operation": self.operation,
            "dynamic_token_rows": list(self.dynamic_token_rows),
        }


def _block_stages(
    *, condition: str, branch: str, block: int, rows_before_attention: tuple[int, ...],
    sequence_length: int,
) -> tuple[list[ConeStage], tuple[int, ...]]:
    stages: list[ConeStage] = []
    for operation in ("ln1", "q", "k", "v"):
        stages.append(ConeStage(condition, branch, block, operation, rows_before_attention))
    first_dynamic = min(rows_before_attention)
    rows_after_attention = tuple(range(first_dynamic, sequence_length))
    for operation in ("causal_attention", "attn_out", "resid_mid", "ln2", "mlp_pre"):
        stages.append(ConeStage(condition, branch, block, operation, rows_after_attention))
    if block == 10:
        stages.extend([
            ConeStage(condition, "SHARED", block, "mlp_post_live", rows_after_attention),
            ConeStage(condition, "J", block, "selected_gate_live", (sequence_length - 1,)),
            ConeStage(condition, "B", block, "selected_gate_anchor_t0", ()),
            ConeStage(condition, "B", block, "selected_gate_replace", (sequence_length - 1,)),
        ])
        for gate_branch in ("J", "B"):
            for operation in ("mlp_out", "resid_post"):
                stages.append(ConeStage(
                    condition, gate_branch, block, operation, rows_after_attention
                ))
    else:
        stages.append(ConeStage(condition, branch, block, "mlp_post", rows_after_attention))
        for operation in ("mlp_out", "resid_post"):
            stages.append(ConeStage(condition, branch, block, operation, rows_after_attention))
    return stages, rows_after_attention


def build_causal_cone_plan(
    *, site_layer: int, patch_position: int, sequence_length: int,
) -> dict[str, Any]:
    if type(site_layer) is not int or site_layer not in range(9):
        raise ValueError("v4.1 site layer must be in 0..8")
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence length must be positive")
    if type(patch_position) is not int or patch_position not in range(sequence_length):
        raise ValueError("patch position is outside the sequence")
    all_stages: list[ConeStage] = []
    for condition in ("PAT", "TAR"):
        rows = (patch_position,)
        # Blocks through 9 are shared between the eventual J/B branches.
        for block in range(site_layer + 1, 10):
            stages, rows = _block_stages(
                condition=condition, branch="SHARED", block=block,
                rows_before_attention=rows, sequence_length=sequence_length,
            )
            all_stages.extend(stages)
        # Block 10 creates the matched-bypass branch split.
        stages, rows = _block_stages(
            condition=condition, branch="J_AND_B", block=10,
            rows_before_attention=rows, sequence_length=sequence_length,
        )
        all_stages.extend(stages)
        # Block 11 is independently evaluated for J and B.
        for branch in ("J", "B"):
            stages, final_rows = _block_stages(
                condition=condition, branch=branch, block=11,
                rows_before_attention=rows, sequence_length=sequence_length,
            )
            all_stages.extend(stages)
            all_stages.extend([
                ConeStage(condition, branch, None, "ln_final", final_rows),
                ConeStage(condition, branch, None, "task_scalar", (sequence_length - 1,)),
            ])
    plan = {
        "schema_version": "green-v410-full-downstream-cone-plan-v1",
        "protocol_id": PROTOCOL_ID,
        "graph_semantics_id": "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1",
        "site_layer": site_layer,
        "patch_position": patch_position,
        "sequence_length": sequence_length,
        "gate_layer": 10,
        "gate_position": sequence_length - 1,
        "selected_gate_indices": [2326, 1138, 2287, 606, 2848, 2305, 46, 2659, 946, 1616],
        "branch_order": list(BRANCH_ORDER),
        "branch_weights": list(BRANCH_WEIGHTS),
        "outputs": ["PAT_J", "PAT_B", "TAR_J", "TAR_B", "PSI"],
        "stages": [stage.to_dict() for stage in all_stages],
        "endpoint_material_allowed": False,
    }
    plan["plan_sha256"] = sha256_canonical(plan)
    validate_causal_cone_plan(plan)
    return plan


def validate_causal_cone_plan(plan: dict[str, Any]) -> None:
    claimed = plan.get("plan_sha256")
    unhashed = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if claimed != sha256_canonical(unhashed):
        raise ValueError("causal-cone plan self hash mismatch")
    if plan.get("endpoint_material_allowed") is not False:
        raise ValueError("endpoint material is forbidden in a certificate cone")
    if plan.get("outputs") != ["PAT_J", "PAT_B", "TAR_J", "TAR_B", "PSI"]:
        raise ValueError("causal-cone output panel mismatch")
    patch_position = plan["patch_position"]
    sequence_length = plan["sequence_length"]
    stages = plan["stages"]
    for condition in ("PAT", "TAR"):
        first_attention = next(
            stage for stage in stages
            if stage["condition"] == condition and stage["operation"] == "causal_attention"
        )
        if first_attention["dynamic_token_rows"] != list(range(patch_position, sequence_length)):
            raise ValueError("first attention did not propagate the patch to all later tokens")
        for branch in ("J", "B"):
            scalar = [
                stage for stage in stages
                if stage["condition"] == condition and stage["branch"] == branch
                and stage["operation"] == "task_scalar"
            ]
            if len(scalar) != 1 or scalar[0]["dynamic_token_rows"] != [sequence_length - 1]:
                raise ValueError("task scalar is not bound to the final causal token")
    gate = [stage for stage in stages if stage["block"] == 10]
    operations = {(stage["condition"], stage["branch"], stage["operation"]) for stage in gate}
    for condition in ("PAT", "TAR"):
        required = {
            (condition, "J", "selected_gate_live"),
            (condition, "B", "selected_gate_anchor_t0"),
            (condition, "B", "selected_gate_replace"),
        }
        if not required <= operations:
            raise ValueError("matched-bypass gate branch is incomplete")
