"""Closed-synthetic actual-shape probe for the hard single-process envelope."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

import gmpy2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_certificate import certify_adaptive_cells
from green_bridge_v400_compiled_mpfr import (
    CompiledMPFRBackend, CompiledNativeJointWitnessEvaluator,
)
from green_bridge_v400_mpfr import rounding_environment_manifest
from green_bridge_v400_schemas import (
    CertificatePlan, Dyadic, canonical_json, sha256_canonical,
)
from green_v400_native_payload_parser_audit import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
)
from green_v400_native_typed_plan_audit import EXPECTED_KERNEL_TAGS


AUDIT_ROW_HASH = hashlib.sha256(
    b"green-v400-hard-single-process-actual-shape-audit-v1"
).hexdigest()
CAP_SYS_ADMIN = 21
CAP_SYS_RESOURCE = 24


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _status_fields() -> dict[str, str]:
    fields = {}
    with Path("/proc/self/status").open(
            encoding="ascii", errors="strict") as stream:
        for line in stream:
            key, separator, value = line.partition(":")
            if separator:
                fields[key] = value.strip()
    return fields


def _lock_readback() -> dict:
    fields = _status_fields()
    return {
        "real_uid": os.getuid(),
        "effective_uid": os.geteuid(),
        "threads": int(fields["Threads"]),
        "effective_capabilities_hex": fields["CapEff"],
        "cap_sys_admin_effective": bool(
            int(fields["CapEff"], 16) & (1 << CAP_SYS_ADMIN)
        ),
        "cap_sys_resource_effective": bool(
            int(fields["CapEff"], 16) & (1 << CAP_SYS_RESOURCE)
        ),
        "rlimit_nproc": list(resource.getrlimit(resource.RLIMIT_NPROC)),
        "rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)),
        "rlimit_core": list(resource.getrlimit(resource.RLIMIT_CORE)),
    }


def _require_strict_lock(readback: dict) -> None:
    if readback["real_uid"] == 0 or readback["effective_uid"] == 0:
        raise RuntimeError("STRICT_LOCK_ROOT_IDENTITY")
    if readback["threads"] != 1:
        raise RuntimeError("STRICT_LOCK_NOT_SINGLE_TASK")
    if (
        readback["cap_sys_admin_effective"]
        or readback["cap_sys_resource_effective"]
    ):
        raise RuntimeError("STRICT_LOCK_NPROC_BYPASS_CAPABILITY_PRESENT")
    if readback["rlimit_nproc"] != [1, 1]:
        raise RuntimeError("STRICT_LOCK_NPROC_NOT_ONE")
    if readback["rlimit_as"][0] <= 0 or (
            readback["rlimit_as"][0] != readback["rlimit_as"][1]):
        raise RuntimeError("STRICT_LOCK_ADDRESS_SPACE_NOT_HARD")
    if readback["rlimit_core"] != [0, 0]:
        raise RuntimeError("STRICT_LOCK_CORE_NOT_ZERO")


def _fraction_payload(value) -> list[int]:
    rational = gmpy2.mpq(value)
    return [int(rational.numerator), int(rational.denominator)]


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
        raise RuntimeError("strict audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("strict audit output must be new below /mnt/sdb")

    before = _lock_readback()
    _require_strict_lock(before)
    library = Path(args.library).resolve(strict=True)
    descriptor = Path(args.descriptor).resolve(strict=True)
    blob = Path(args.blob).resolve(strict=True)
    plan = CertificatePlan(
        "green-v400-certificate-plan-v1", AUDIT_ROW_HASH, (Dyadic(1, -14),),
        "[-h,0],[0,h]", "curvature-weighted width priority dyadic bisection",
        "0x1p-80", "0x1p-40", 1, 4, 384, 512, (), False,
    )

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
    cells = certify_adaptive_cells(tracking, Fraction(1, 2**14), 384, plan)
    context.close()
    envelope.close()
    after = _lock_readback()
    _require_strict_lock(after)

    status = "RESOURCE_INCONCLUSIVE" if cells is None else "PARTITION_ACCEPTED"
    report = {
        "schema_version": "green-v400-hard-single-process-actual-shape-audit-v1",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "selector_may_read_this_artifact": False,
        "execution_scope": "outcome_blind_closed_synthetic_actual_shape",
        "status": status,
        "expected_status": "RESOURCE_INCONCLUSIVE",
        "status_matches_expected": status == "RESOURCE_INCONCLUSIVE",
        "resource_reason": "MAX_DEPTH_REACHED" if cells is None else None,
        "hard_lock_readback_before": before,
        "hard_lock_readback_after": after,
        "hard_single_process_lock_verified": True,
        "certificate_plan": plan.to_dict(),
        "certificate_plan_semantic_hash": sha256_canonical(plan),
        "evaluated_domains": tracking.domains,
        "dispatch_count_by_precision": evaluator.dispatch_count_by_precision,
        "event_count_per_dispatch": len(EXPECTED_KERNEL_TAGS),
        "kernel_tags_sha256": sha256_canonical(EXPECTED_KERNEL_TAGS),
        "artifacts": {
            "backend_sha256": _sha256_file(library),
            "descriptor_sha256": _sha256_file(descriptor),
            "blob_sha256": _sha256_file(blob),
            "program_execution_sha256": PROGRAM_SHA,
            "dispatch_sha256": DISPATCH_SHA,
            "fusion_sha256": FUSION_SHA,
        },
        "rounding_environment": rounding_environment_manifest(),
        "retained_numeric_scope": (
            "control domains only; no response Jet2 payload retained"
        ),
        "gpu_used": False,
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((canonical_json(report) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "status": status,
        "dispatch_count_by_precision": evaluator.dispatch_count_by_precision,
        "report_semantic_hash": report["report_semantic_hash"],
    }, sort_keys=True))
    return 0 if report["status_matches_expected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
