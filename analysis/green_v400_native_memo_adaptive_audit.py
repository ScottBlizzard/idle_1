"""Outcome-blind two-radius audit of exact-domain memoization under adaptive policy."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_certificate import certify_adaptive_cells
from green_bridge_v400_compiled_mpfr import (
    CompiledMPFRBackend, CompiledNativeJointWitnessEvaluator, ExactDomainJetMemo,
)
from green_bridge_v400_resources import ProcessTreeResourceRecorder
from green_bridge_v400_schemas import CertificatePlan, Dyadic, sha256_canonical
from green_v400_native_adaptive_policy_audit import (
    AUDIT_ROW_HASH, _TrackingEvaluator, _git, _sha256,
)
from green_v400_native_payload_parser_audit import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
)
from green_v400_native_typed_plan_audit import EXPECTED_KERNEL_TAGS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        relative = output.relative_to(Path("/mnt/sdb").resolve())
    except ValueError as error:
        raise RuntimeError("memo audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("memo audit output must be new below /mnt/sdb")

    library = Path(args.library).resolve()
    descriptor = Path(args.descriptor).resolve()
    blob = Path(args.blob).resolve()
    plan = CertificatePlan(
        "green-v400-certificate-plan-v1", AUDIT_ROW_HASH,
        (Dyadic(1, -14), Dyadic(1, -15)),
        "[-h,0],[0,h]", "curvature-weighted width priority dyadic bisection",
        "0x1p-80", "0x1p-40", 1, 4, 384, 512, (), False,
    )
    with ProcessTreeResourceRecorder(sample_interval_seconds=0.01) as resources:
        backend = CompiledMPFRBackend(library)
        envelope = backend.open_native_plan_envelope(
            descriptor, blob, descriptor_sha256=DESCRIPTOR_SHA,
            program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
            blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
            blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
        )
        context = backend.open_native_precision_context(envelope, 384)
        identity_probe = CompiledNativeJointWitnessEvaluator(
            backend, {384: context}, certificate_row_hash=AUDIT_ROW_HASH,
            expected_kernel_tags=tuple(EXPECTED_KERNEL_TAGS),
        )
        memo = ExactDomainJetMemo(identity_probe.evaluator_identity, max_entries=128)
        evaluator = CompiledNativeJointWitnessEvaluator(
            backend, {384: context}, certificate_row_hash=AUDIT_ROW_HASH,
            expected_kernel_tags=tuple(EXPECTED_KERNEL_TAGS), exact_domain_memo=memo,
        )
        tracking = _TrackingEvaluator(evaluator)
        statuses = []
        for radius in plan.radii:
            cells = certify_adaptive_cells(
                tracking, radius.as_fraction(), 384, plan,
            )
            statuses.append("RESOURCE_INCONCLUSIVE" if cells is None else "PARTITION_ACCEPTED")
        context.close()
        envelope.close()

    metrics = memo.metrics()
    logical = metrics["by_precision"]["384"]["logical_requests"]
    physical = evaluator.dispatch_count_by_precision[384]
    passed = (
        statuses == ["RESOURCE_INCONCLUSIVE", "RESOURCE_INCONCLUSIVE"]
        and logical == 12 and physical == 10
        and metrics["by_precision"]["384"]["hits"] == 2
    )
    report = {
        "schema_version": "green-v400-native-memo-adaptive-audit-v1",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "status": "PASS_EXACT_DOMAIN_MEMO" if passed else "FAIL",
        "radii": [radius.to_dict() for radius in plan.radii],
        "certificate_plan_semantic_hash": sha256_canonical(plan),
        "radius_statuses": statuses,
        "evaluated_domains": tracking.domains,
        "memo_metrics": metrics,
        "logical_native_requests": logical,
        "physical_native_dispatches": physical,
        "saved_native_dispatches": logical - physical,
        "dispatch_count_by_precision": evaluator.dispatch_count_by_precision,
        "dispatch_attempt_count_by_precision": evaluator.dispatch_attempt_count_by_precision,
        "process_tree_resource_record": resources.record.to_dict(),
        "provenance": {
            "repository_commit": _git("rev-parse", "HEAD"),
            "repository_clean": not bool(_git("status", "--porcelain=v1", "--untracked-files=all")),
            "backend_sha256": _sha256(library),
            "descriptor_sha256": _sha256(descriptor),
            "blob_sha256": _sha256(blob),
            "program_execution_sha256": PROGRAM_SHA,
            "dispatch_sha256": DISPATCH_SHA,
            "fusion_sha256": FUSION_SHA,
            "evaluator_identity_sha256": evaluator.evaluator_identity_sha256,
        },
        "retained_numeric_scope": "control domains and cache counters only; no response Jet2 payload retained",
        "claim_scope": (
            "two nested dyadic radii use the unchanged adaptive policy and one identity-closed "
            "successful-Jet2 memo; only exact-domain reuse, resource, and dispatch identity are audited"
        ),
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "logical": logical, "physical": physical,
        "saved": logical - physical,
        "wall_seconds": resources.record.wall_seconds,
        "report_semantic_hash": report["report_semantic_hash"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
