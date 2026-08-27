"""Outcome-blind actual-shape audit of canonical sibling-pair parallelism."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

import gmpy2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]

from green_bridge_v400_certificate import certify_adaptive_cells
from green_bridge_v400_compiled_mpfr import (
    CompiledMPFRBackend, CompiledNativeJointWitnessEvaluator, ExactDomainJetMemo,
    ParallelNativeSiblingEvaluator,
)
from green_bridge_v400_resources import ProcessTreeResourceRecorder
from green_bridge_v400_schemas import CertificatePlan, Dyadic, sha256_canonical
from green_v400_native_adaptive_policy_audit import AUDIT_ROW_HASH, _git, _sha256
from green_v400_native_payload_parser_audit import (
    BLOB_NBYTES, BLOB_SHA, DESCRIPTOR_SHA, DISPATCH_SHA, FUSION_SHA, PROGRAM_SHA,
)
from green_v400_native_typed_plan_audit import EXPECTED_KERNEL_TAGS


def _endpoint(value) -> list[int]:
    rational = gmpy2.mpq(value)
    return [int(rational.numerator), int(rational.denominator)]


class _TrackingParallelEvaluator:
    contains_scientific_outcome = False

    def __init__(self, pool: ParallelNativeSiblingEvaluator):
        self.pool = pool
        self.certificate_row_hash = pool.certificate_row_hash
        self.pair_rounds: list[list[dict]] = []
        self._pair_domains = []
        self._pair_results = []

    def evaluate_interval(self, domain):
        return self.pool.evaluate_interval(domain)

    def evaluate_interval_pair(self, domains):
        self.pair_rounds.append([{
            "precision_bits": domain.precision_bits,
            "lower": _endpoint(domain.lower),
            "upper": _endpoint(domain.upper),
        } for domain in domains])
        results = self.pool.evaluate_interval_pair(domains)
        self._pair_domains.append(domains)
        self._pair_results.append(results)
        return results


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
        raise RuntimeError("parallel audit output must resolve below /mnt/sdb") from error
    if not relative.parts or output.exists():
        raise RuntimeError("parallel audit output must be new below /mnt/sdb")

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
        mpfr_build_options = backend.mpfr_build_options()
        envelope = backend.open_native_plan_envelope(
            descriptor, blob, descriptor_sha256=DESCRIPTOR_SHA,
            program_execution_sha256=PROGRAM_SHA, dispatch_sha256=DISPATCH_SHA,
            blob_sha256=BLOB_SHA, fusion_sha256=FUSION_SHA,
            blob_nbytes=BLOB_NBYTES, fusion_weight_count=768,
        )
        contexts = tuple(
            backend.open_native_precision_context(envelope, 384) for _ in range(2)
        )
        identity_probe = CompiledNativeJointWitnessEvaluator(
            backend, {384: contexts[0]}, certificate_row_hash=AUDIT_ROW_HASH,
            expected_kernel_tags=tuple(EXPECTED_KERNEL_TAGS),
        )
        memo = ExactDomainJetMemo(identity_probe.evaluator_identity, max_entries=16)
        workers = tuple(
            CompiledNativeJointWitnessEvaluator(
                backend, {384: context}, certificate_row_hash=AUDIT_ROW_HASH,
                expected_kernel_tags=tuple(EXPECTED_KERNEL_TAGS),
                exact_domain_memo=memo,
            )
            for context in contexts
        )
        pool = ParallelNativeSiblingEvaluator(workers)
        tracking = _TrackingParallelEvaluator(pool)
        backend.reset_native_dispatch_concurrency_metrics()
        cells = certify_adaptive_cells(
            tracking, Fraction(1, 2**14), 384, plan,
        )
        concurrency = backend.native_dispatch_concurrency_info()
        context_concurrency = tuple(
            backend.native_precision_context_dispatch_info(context)
            for context in contexts
        )
        dispatch_counts = pool.dispatch_count_by_precision
        attempt_counts = pool.dispatch_attempt_count_by_precision
        pool.close()
        for context in contexts:
            context.close()
        reference_context = backend.open_native_precision_context(envelope, 384)
        reference = CompiledNativeJointWitnessEvaluator(
            backend, {384: reference_context}, certificate_row_hash=AUDIT_ROW_HASH,
            expected_kernel_tags=tuple(EXPECTED_KERNEL_TAGS),
        )
        sequential_equivalence = []
        for domains, concurrent_results in zip(
            tracking._pair_domains, tracking._pair_results
        ):
            sequential_equivalence.extend(
                reference.evaluate_interval(domain) == concurrent
                for domain, concurrent in zip(domains, concurrent_results)
            )
        reference_context.close()
        envelope.close()

    status = "RESOURCE_INCONCLUSIVE" if cells is None else "PARTITION_ACCEPTED"
    metrics = memo.metrics()
    passed = (
        status == "RESOURCE_INCONCLUSIVE"
        and len(tracking.pair_rounds) == 3
        and dispatch_counts == {384: 6}
        and mpfr_build_options["tls_enabled"]
        and concurrency == {
            "dispatch_entry_count": 6,
            "active_dispatch_count": 0,
            "peak_active_dispatch_count": 2,
        }
        and context_concurrency == ({
            "dispatch_entry_count": 3,
            "active_dispatch_count": 0,
            "peak_active_dispatch_count": 1,
        },) * 2
        and sequential_equivalence == [True] * 6
        and metrics["by_precision"]["384"] == {
            "logical_requests": 6, "hits": 0, "misses": 6, "waits": 0,
        }
    )
    report = {
        "schema_version": "green-v400-native-sibling-parallel-audit-v2",
        "contains_scientific_outcome": False,
        "scientific_threshold_applied": False,
        "status": "PASS_PHYSICAL_SIBLING_PARALLEL" if passed else "FAIL",
        "certificate_status": status,
        "resource_reason": "MAX_DEPTH_REACHED" if cells is None else None,
        "certificate_plan_semantic_hash": sha256_canonical(plan),
        "pair_round_count": len(tracking.pair_rounds),
        "canonical_pair_rounds": tracking.pair_rounds,
        "dispatch_count_by_precision": dispatch_counts,
        "dispatch_attempt_count_by_precision": attempt_counts,
        "memo_metrics": metrics,
        "mpfr_build_options": mpfr_build_options,
        "native_dispatch_concurrency": concurrency,
        "per_context_dispatch_concurrency": context_concurrency,
        "sequential_bit_exact_equivalence_count": sum(sequential_equivalence),
        "sequential_bit_exact_comparison_count": len(sequential_equivalence),
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
            "evaluator_identity_sha256": pool.evaluator_identity_sha256,
            "worker_count": 2,
            "independent_native_context_per_worker": True,
        },
        "retained_numeric_scope": "control domains and scheduling/resource counters only; no response Jet2 payload retained",
        "claim_scope": (
            "one frozen heap parent per round; its canonical left/right children execute on "
            "independent native contexts with measured physical overlap, remain bit-exact to "
            "sequential evaluation, and commit in input order regardless of completion order"
        ),
    }
    report["report_semantic_hash"] = sha256_canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "pair_rounds": len(tracking.pair_rounds),
        "dispatches": dispatch_counts, "wall_seconds": resources.record.wall_seconds,
        "peak_sampled_tree_rss_kib": resources.record.peak_sampled_tree_rss_kib,
        "report_semantic_hash": report["report_semantic_hash"],
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
