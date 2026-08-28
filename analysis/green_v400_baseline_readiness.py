"""Fail-closed readiness audit for GREEN silent-failure baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_baseline_readiness(
    payload: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("real_outcomes_authorized") is not False:
        errors.append("readiness file cannot authorize real outcomes")
    ready_status = payload.get("ready_status")
    if ready_status != "READY":
        errors.append("ready_status must equal READY")
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict) or not baselines:
        errors.append("baseline registry must be nonempty")
        baselines = {}

    missing_required: list[str] = []
    missing_evidence: dict[str, list[str]] = {}
    for name, entry in baselines.items():
        if entry.get("required") is True and entry.get("status") != ready_status:
            missing_required.append(name)
        absent = [
            relative
            for relative in entry.get("evidence", [])
            if not (repository_root / relative).is_file()
        ]
        if absent:
            missing_evidence[name] = absent

    ready = not errors and not missing_required and not missing_evidence
    return {
        "schema_version": "green-v400-baseline-readiness-audit-v1",
        "protocol_id": payload.get("protocol_id"),
        "real_outcomes_authorized": False,
        "ready_for_untouched_execution": ready,
        "required_baseline_count": sum(
            entry.get("required") is True for entry in baselines.values()
        ),
        "ready_required_count": sum(
            entry.get("required") is True and entry.get("status") == ready_status
            for entry in baselines.values()
        ),
        "not_ready_required": sorted(missing_required),
        "missing_evidence": missing_evidence,
        "errors": errors,
        "verdict": "PASS_BASELINES_READY" if ready else "BLOCK_BASELINES_NOT_READY",
    }


def assert_baselines_ready(payload: dict[str, Any], repository_root: Path) -> None:
    audit = audit_baseline_readiness(payload, repository_root)
    if not audit["ready_for_untouched_execution"]:
        raise RuntimeError(json.dumps(audit, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            audit_baseline_readiness(payload, args.repository_root),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

