"""Outcome-blind native evaluator wiring audit for the frozen adaptive policy."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_certificate import certify_adaptive_cells
from green_bridge_v400_compiled_mpfr import (
    CompiledMPFRBackend, CompiledNativeJointWitnessEvaluator,
)
from green_bridge_v400_mpfr import rounding_environment_manifest
from green_bridge_v400_resources import ProcessTreeResourceRecorder
from green_bridge_v400_schemas import CertificatePlan, Dyadic, sha256_canonical
from green_v400_native_payload_parser_audit import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
)
from green_v400_native_typed_plan_audit import EXPECTED_KERNEL_TAGS


AUDIT_ROW_HASH = hashlib.sha256(
    b"green-v400-closed-synthetic-native-adaptive-policy-audit-v1"
).hexdigest()


def _fraction_payload(value) -> list[int]:
    rational = Fraction(value)
    return [rational.numerator, rational.denominator]


class _TrackingEvaluator:
    contains_scientific_outcome = False

    def __init__(self, evaluator: CompiledNativeJointWitnessEvaluator):
        self.evaluator = evaluator
        self.certificate_row_hash = evaluator.certificate_row_hash
        self.domains: list[dict] = []

    def evaluate_interval(self, domain):
        self.domains.append({
            "precision_bits": domain.precision_bits,
            "lower": _fraction_payload(domain.lower),
            "upper": _fraction_payload(domain.upper),
        })
        return self.evaluator.evaluate_interval(domain)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise RuntimeError("native adaptive audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("native adaptive audit output must be new below /mnt/sdb")

    library = Path(args.library).resolve()
    descriptor = Path(args.descriptor).resolve()
    blob = Path(args.blob).resolve()
    plan = CertificatePlan(
        "green-v400-certificate-plan-v1", AUDIT_ROW_HASH, (Dyadic(1, -14),),
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
        evaluator = CompiledNativeJointWitnessEvaluator(
            backend, {384: context}, certificate_row_hash=AUDIT_ROW_HASH,
            expected_kernel_tags=tuple(EXPECTED_KERNEL_TAGS),
        )
        tracking = _TrackingEvaluator(evaluator)
        cells = certify_adaptive_cells(
            tracking, Fraction(1, 2**14), 384, plan,
        )
        context.close()
        envelope.close()

    status = "RESOURCE_INCONCLUSIVE" if cells is None else "PARTITION_ACCEPTED"
    report = {
        "schema_version": "green-v400-native-adaptive-policy-audit-v1",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "native_adaptive_policy_wired": True,
        "status": status,
        "expected_status": "RESOURCE_INCONCLUSIVE",
        "status_matches_expected": status == "RESOURCE_INCONCLUSIVE",
        "resource_reason": "MAX_DEPTH_REACHED" if cells is None else None,
        "certificate_plan": plan.to_dict(),
        "certificate_plan_semantic_hash": sha256_canonical(plan),
        "evaluated_domains": tracking.domains,
        "dispatch_count_by_precision": evaluator.dispatch_count_by_precision,
        "event_count_per_dispatch": len(EXPECTED_KERNEL_TAGS),
        "kernel_tags_sha256": sha256_canonical(EXPECTED_KERNEL_TAGS),
        "process_tree_resource_record": resources.record.to_dict(),
        "provenance": {
            "repository_commit": _git("rev-parse", "HEAD"),
            "repository_clean": not bool(_git("status", "--porcelain=v1", "--untracked-files=all")),
            "backend_path": str(library),
            "backend_sha256": _sha256(library),
            "backend_version": backend.version,
            "descriptor_path": str(descriptor),
            "descriptor_sha256": _sha256(descriptor),
            "blob_path": str(blob),
            "blob_sha256": _sha256(blob),
            "program_execution_sha256": PROGRAM_SHA,
            "dispatch_sha256": DISPATCH_SHA,
            "fusion_sha256": FUSION_SHA,
            "rounding_environment": rounding_environment_manifest(),
            "official_precision_bits": 384,
            "audit_precision_bits": 512,
            "audit_precision_executed": False,
            "audit_precision_skip_reason": "official_precision_resource_inconclusive",
            "gpu_used_for_certificate": False,
        },
        "retained_numeric_scope": "control domains only; no response Jet2 payload retained",
        "claim_scope": (
            "the actual-shape native evaluator is called by the frozen adaptive priority, "
            "tolerance, bisection, and resource-inconclusive logic under a deliberately "
            "bounded synthetic plan; timing and memory are observations, not formal bounds"
        ),
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "dispatch_count_by_precision": evaluator.dispatch_count_by_precision,
        "peak_sampled_tree_rss_kib": resources.record.peak_sampled_tree_rss_kib,
        "report_semantic_hash": report["report_semantic_hash"],
    }, sort_keys=True))
    return 0 if report["status_matches_expected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
