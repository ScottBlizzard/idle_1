"""Mechanical development-only activation for a sealed GREEN prepare plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ACTIVATION_SOURCE_PATH = "analysis/green_v400_development_activation.py"
ALLOWED_CHANGES = (
    "schema_version",
    "parent_plan_sha256",
    "development_authorization_sha256",
    "execution_enabled",
    "real_outcomes_authorized",
    "development_authorized",
    "confirmation_authorized",
    "plan_gate",
    "plan_sha256",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_no_clobber_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_self_hash(payload: dict[str, Any], field: str, label: str) -> str:
    claimed = payload.get(field)
    unhashed = dict(payload)
    unhashed.pop(field, None)
    if not isinstance(claimed, str) or canonical_sha256(unhashed) != claimed:
        raise ValueError(f"{label} self hash mismatch")
    return claimed


def validate_authorization(
    authorization: dict[str, Any],
    *,
    parent_plan: dict[str, Any],
    parent_plan_file_sha256: str | None = None,
) -> str:
    digest = _verify_self_hash(
        authorization, "authorization_sha256", "development authorization"
    )
    if authorization.get("schema_version") != "green-v400-development-authorization-v1":
        raise ValueError("development authorization schema is invalid")
    if authorization.get("contains_scientific_outcome") is not False:
        raise ValueError("development authorization must be outcome-free")
    authority = authorization.get("authority", {})
    directives = authority.get("verbatim_directives")
    if (
        authority.get("kind") != "project_owner_continuous_execution_directive"
        or authority.get("authentication_claim")
        != "operator_attestation_not_external_signature"
        or not isinstance(directives, list)
        or not directives
        or authority.get("verbatim_directives_sha256")
        != canonical_sha256(directives)
    ):
        raise ValueError("development authority attestation is invalid")
    scope = authorization.get("scope", {})
    if scope != {
        "development_authorized": True,
        "confirmation_authorized": False,
        "automatic_confirmation_unlock": False,
        "post_outcome_rule_changes_allowed": False,
    }:
        raise ValueError("development authorization scope is not development-only")
    if tuple(authorization.get("allowed_parent_to_child_changes", [])) != ALLOWED_CHANGES:
        raise ValueError("development authorization allows unexpected plan changes")
    targets = [
        target
        for target in authorization.get("targets", [])
        if target.get("protocol_id") == parent_plan.get("protocol_id")
    ]
    if len(targets) != 1 or targets[0].get("parent_plan_sha256") != parent_plan.get(
        "plan_sha256"
    ):
        raise ValueError("development authorization does not target this parent plan")
    if parent_plan_file_sha256 is not None and targets[0].get(
        "parent_plan_file_sha256"
    ) != parent_plan_file_sha256:
        raise ValueError("development authorization parent plan file hash mismatch")
    return digest


def derive_development_plan(
    *, parent_plan: dict[str, Any], authorization: dict[str, Any]
) -> dict[str, Any]:
    parent_hash = _verify_self_hash(parent_plan, "plan_sha256", "parent plan")
    if (
        parent_plan.get("execution_enabled") is not False
        or parent_plan.get("real_outcomes_authorized") is not False
        or parent_plan.get("plan_gate")
        != "PLAN_COMPILED_AWAITING_SCIENTIFIC_AUTHORIZATION"
        or parent_plan.get("baseline_readiness", {}).get(
            "ready_for_untouched_execution"
        )
        is not True
        or parent_plan.get("baseline_readiness", {}).get("not_ready_required") != []
    ):
        raise ValueError("parent plan is not an eligible sealed prepare plan")
    expected_source = file_sha256(Path(__file__))
    if parent_plan.get("source_file_sha256", {}).get(
        ACTIVATION_SOURCE_PATH
    ) != expected_source:
        raise ValueError("development activation source differs from parent plan")
    authorization_hash = validate_authorization(
        authorization, parent_plan=parent_plan
    )
    child = copy.deepcopy(parent_plan)
    child.pop("plan_sha256", None)
    child.update(
        {
            "schema_version": "green-v400-development-execution-plan-v1",
            "parent_plan_sha256": parent_hash,
            "development_authorization_sha256": authorization_hash,
            "execution_enabled": True,
            "real_outcomes_authorized": True,
            "development_authorized": True,
            "confirmation_authorized": False,
            "plan_gate": "DEVELOPMENT_ONLY_AUTHORIZED",
        }
    )
    child["plan_sha256"] = canonical_sha256(child)
    return child


def validate_activated_plan(
    *,
    parent_plan: dict[str, Any],
    authorization: dict[str, Any],
    activated_plan: dict[str, Any],
    parent_plan_file_sha256: str | None = None,
) -> None:
    validate_authorization(
        authorization,
        parent_plan=parent_plan,
        parent_plan_file_sha256=parent_plan_file_sha256,
    )
    expected = derive_development_plan(
        parent_plan=parent_plan, authorization=authorization
    )
    if activated_plan != expected:
        raise ValueError("activated plan is not the exact mechanical parent derivation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parent = json.loads(args.parent_plan.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    validate_authorization(
        authorization,
        parent_plan=parent,
        parent_plan_file_sha256=file_sha256(args.parent_plan),
    )
    child = derive_development_plan(
        parent_plan=parent, authorization=authorization
    )
    _atomic_no_clobber_json(args.output, child)
    print(
        json.dumps(
            {
                "plan_sha256": child["plan_sha256"],
                "parent_plan_sha256": child["parent_plan_sha256"],
                "plan_gate": child["plan_gate"],
                "confirmation_authorized": child["confirmation_authorized"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
