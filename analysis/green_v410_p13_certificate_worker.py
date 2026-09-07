"""One immutable GREEN v4.1 P13 direction-certificate worker."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_resources import ProcessTreeResourceRecorder
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader
from green_v410_artifacts import (
    atomic_no_clobber_json,
    direction_certificate_row_id,
    file_sha256,
    configure_exact_integer_io,
)
from green_v410_fixed_budget_certificate import FixedBudgetProgramEvaluator, certify_five_outputs
from green_v410_p13 import contraction_ratio
from green_v410_p13_queue import validate_p13_queue
from green_v410_protocol import BRANCH_ORDER, DECISION_RULE_SHA256, PROTOCOL_ID, RADIUS_PANEL, sha256_canonical
from green_v410_schemas import validate_artifact, with_artifact_self_hash


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fraction(value: Any, name: str) -> Fraction:
    if (
        not isinstance(value, list) or len(value) != 2
        or any(type(item) is not int for item in value)
        or value[1] <= 0
    ):
        raise ValueError(f"malformed exact rational: {name}")
    result = Fraction(*value)
    if [result.numerator, result.denominator] != value:
        raise ValueError(f"noncanonical exact rational: {name}")
    return result


def _exact_interval(payload: dict[str, Any], name: str) -> tuple[Fraction, Fraction]:
    if set(payload) != {"lower", "upper"}:
        raise ValueError(f"malformed exact interval: {name}")
    lower = _fraction(payload["lower"], f"{name}.lower")
    upper = _fraction(payload["upper"], f"{name}.upper")
    if lower > upper:
        raise ValueError(f"reversed exact interval: {name}")
    return lower, upper


def _direction_job(queue: dict[str, Any], job_id: str, ordinal: int) -> tuple[dict, dict]:
    matches = [job for job in queue["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError("P13 direction job does not resolve exactly once")
    job = matches[0]
    if not 0 <= ordinal < 8:
        raise ValueError("P13 direction ordinal is invalid")
    direction = job["direction_panel"][ordinal]
    if direction["direction_ordinal"] != ordinal:
        raise ValueError("P13 direction panel order mismatch")
    return job, direction


def _build_records(
    *,
    queue: dict[str, Any],
    job: dict[str, Any],
    direction: dict[str, Any],
    graph: dict[str, Any],
    site_identity: dict[str, Any],
    certificate: dict[str, Any],
    capture: dict[str, Any],
    resource_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordinal = direction["direction_ordinal"]
    official = certificate["official_intervals"]
    branch_centers = {
        name: _fraction(
            capture["response_panel"]["central_secants"][name][ordinal],
            f"central_secants.{name}.{ordinal}",
        ) for name in BRANCH_ORDER
    }
    ad_derivatives = {
        name: _fraction(
            capture["response_panel"]["ad_derivatives"][name][ordinal],
            f"ad_derivatives.{name}.{ordinal}",
        ) for name in BRANCH_ORDER
    }
    branch_intervals = {
        name: _exact_interval(official[name], f"official.{name}")
        for name in BRANCH_ORDER
    }
    ad_overlaps = {
        name: branch_intervals[name][0] <= ad_derivatives[name] <= branch_intervals[name][1]
        for name in BRANCH_ORDER
    }
    if not all(ad_overlaps.values()):
        raise RuntimeError("INVALID_P13_AD_DERIVATIVE_OUTSIDE_OFFICIAL_INTERVAL")
    relational = _exact_interval(official["PSI"], "official.PSI")
    ratio = contraction_ratio(
        branch_centers=branch_centers,
        branch_intervals=branch_intervals,
        relational_interval=relational,
    )
    row_id = direction_certificate_row_id(
        task=job["task"],
        phase="qualification",
        parent_site_row_id=job["site_row_id"],
        site_identity_sha256=site_identity["site_identity_sha256"],
        direction_ordinal=ordinal,
        direction_payload_sha256=direction["direction_payload_sha256"],
        graph_manifest_sha256=graph["graph_manifest_sha256"],
        resource_manifest_sha256=job["resource_manifest_sha256"],
    )
    direction_certificate = with_artifact_self_hash({
        "schema_version": "green-v410-sfc-jwtec-direction-certificate-v1",
        "protocol_id": PROTOCOL_ID,
        "certificate_row_id": row_id,
        "parent_site_row_id": job["site_row_id"],
        "site_identity_sha256": site_identity["site_identity_sha256"],
        "direction_ordinal": ordinal,
        "direction_payload_sha256": direction["direction_payload_sha256"],
        "graph_manifest_sha256": graph["graph_manifest_sha256"],
        "resource_manifest_sha256": job["resource_manifest_sha256"],
        "decision_rule_sha256": DECISION_RULE_SHA256,
        "status": "INTERVAL_COMPUTED",
        "reason_code": "COMPLETE_FIXED_BUDGET_INTERVAL",
        "invalid_scope": "NONE",
        "radii": list(RADIUS_PANEL),
        "intervals": {
            "psi": official["PSI"],
            "pat_joint": official["PAT_J"],
            "tar_joint": official["TAR_J"],
        },
        "precision_nested": certificate["precision_nested"],
        "partition_manifest_sha256": certificate["certificate_sha256"],
    })
    validate_artifact(direction_certificate)
    p13 = {
        "schema_version": "green-v410-p13-direction-contraction-v1",
        "protocol_id": PROTOCOL_ID,
        "attempt_index": 1,
        "task": job["task"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "job_id": job["job_id"],
        "site_row_id": job["site_row_id"],
        "direction_ordinal": ordinal,
        "direction_certificate_row_id": row_id,
        "direction_certificate_sha256": direction_certificate["artifact_sha256"],
        "branch_order": list(BRANCH_ORDER),
        "branch_centers": {
            name: [value.numerator, value.denominator]
            for name, value in branch_centers.items()
        },
        "branch_official_intervals": {
            name: {
                "lower": [value[0].numerator, value[0].denominator],
                "upper": [value[1].numerator, value[1].denominator],
            } for name, value in branch_intervals.items()
        },
        "independent_ad_derivatives": {
            name: [value.numerator, value.denominator]
            for name, value in ad_derivatives.items()
        },
        "ad_overlaps_official_intervals": ad_overlaps,
        "relational_official_interval": official["PSI"],
        "contraction_ratio": [ratio.numerator, ratio.denominator],
        "resource_record": resource_record,
        "contains_endpoint_material": False,
    }
    return direction_certificate, p13 | {"record_sha256": sha256_canonical(p13)}


def main() -> None:
    configure_exact_integer_io()
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--direction-ordinal", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    queue = _load(args.queue)
    validate_p13_queue(queue)
    job, direction = _direction_job(queue, args.job_id, args.direction_ordinal)
    graph_dir = args.graph_root / job["job_id"] / f"direction_{args.direction_ordinal:02d}"
    graph = _load(graph_dir / "graph_manifest.json")
    site_identity = _load(args.graph_root / job["job_id"] / "site_identity.json")
    capture = _load(args.capture_root / job["job_id"] / "capture_manifest.json")
    validate_artifact(graph)
    validate_artifact(site_identity)
    if (
        graph["site_identity_sha256"] != site_identity["site_identity_sha256"]
        or graph["direction_payload_sha256"] != direction["direction_payload_sha256"]
        or graph["graph_semantics_id"] != "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1"
        or graph["contains_endpoint_material"] is not False
        or capture.get("queue_manifest_sha256") != queue["queue_manifest_sha256"]
        or capture.get("contains_endpoint_material") is not False
    ):
        raise ValueError("P13 certificate input identity mismatch")
    if not args.backend.is_file():
        raise FileNotFoundError("P13 compiled MPFR backend is missing")
    if file_sha256(args.backend) != queue["compiled_backend_sha256"]:
        raise ValueError("P13 compiled MPFR backend differs from frozen queue")
    program = TensorProgram.from_dict(_load(graph_dir / "tensor_program.json"))
    reader = TensorStoreReader(graph_dir / "tensor_store.json")
    evaluator = FixedBudgetProgramEvaluator(program, reader, args.backend)
    try:
        with ProcessTreeResourceRecorder(sample_interval_seconds=0.01) as recorder:
            certificate = certify_five_outputs(
                evaluator,
                leaf_budget=job["selected_leaf_budget"],
                official_precision_bits=384,
                audit_precision_bits=512,
            )
    finally:
        evaluator.close()
    resource = recorder.record.to_dict()
    if (
        resource["peak_sampled_tree_rss_kib"] * 1024 > 68_719_476_736
        or resource["wall_seconds"] > 85_800
    ):
        raise RuntimeError("P13_DIRECTION_RESOURCE_CEILING_EXCEEDED")
    direction_certificate, p13 = _build_records(
        queue=queue,
        job=job,
        direction=direction,
        graph=graph,
        site_identity=site_identity,
        certificate=certificate,
        capture=capture,
        resource_record=resource,
    )
    final_dir = args.output_root / job["job_id"] / f"direction_{args.direction_ordinal:02d}"
    atomic_no_clobber_json(
        final_dir / "certificate_detail.json", certificate,
        job_id=f"{job['job_id']}-{args.direction_ordinal}-detail",
    )
    atomic_no_clobber_json(
        final_dir / "direction_certificate.json", direction_certificate,
        job_id=f"{job['job_id']}-{args.direction_ordinal}-certificate",
    )
    atomic_no_clobber_json(
        final_dir / "p13_record.json", p13,
        job_id=f"{job['job_id']}-{args.direction_ordinal}-p13",
    )
    print(direction_certificate["certificate_row_id"])


if __name__ == "__main__":
    main()
