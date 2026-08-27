"""ctypes bridge and bit-identity checks for the compiled MPFR backend."""
from __future__ import annotations

import ctypes
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import gmpy2

from green_bridge_v400_interval_jet import Jet2
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr import rounding_environment_manifest
from green_bridge_v400_schemas import sha256_canonical


def _bits_f32(value) -> int:
    return int(np.asarray(value, dtype="<f4").view("<u4").reshape(()))


def _bits_f64_array(values) -> np.ndarray:
    return np.asarray(values, dtype="<f8").view("<u8")


def _exact_fraction(payload: dict) -> Fraction:
    raw = str(payload["significand_hex"])
    negative = raw.startswith("-")
    digits = raw[1:] if negative else raw
    significand = int(digits, 16) * (-1 if negative else 1)
    if significand == 0:
        return Fraction(0, 1)
    exponent = int(payload["exponent_2"])
    return (Fraction(significand * (1 << exponent), 1) if exponent >= 0
            else Fraction(significand, 1 << (-exponent)))


def _binary_endpoint(value) -> tuple[bytes, int]:
    rational = gmpy2.mpq(value)
    numerator = int(rational.numerator)
    denominator = int(rational.denominator)
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("MPFR endpoint is not dyadic")
    exponent = -(denominator.bit_length() - 1)
    sign = "-" if numerator < 0 else ""
    return f"{sign}{abs(numerator):x}".encode("ascii"), exponent


class CompiledResidentJetBuffer:
    """Owned opaque native Jet2 vector; intermediate values never cross JSON/FFI."""

    def __init__(self, backend, handle: ctypes.c_void_p, precision_bits: int, width: int):
        if not handle.value or width <= 0:
            raise ValueError("invalid compiled resident Jet2 buffer")
        self.backend = backend
        self.handle = handle
        self.precision_bits = int(precision_bits)
        self.width = int(width)

    def close(self) -> None:
        if self.handle.value:
            self.backend.library.green_v400_resident_jet_buffer_free(self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CompiledNativePlanEnvelope:
    """Generation-handle owner for a native descriptor/blob mapping."""

    def __init__(self, backend, handle: int, info: dict):
        if handle <= 0:
            raise ValueError("invalid native plan envelope handle")
        self.backend = backend
        self.handle = int(handle)
        self.info = info

    def close(self) -> None:
        if self.handle:
            status = self.backend.library.green_v400_native_plan_envelope_close_v1(
                self.handle
            )
            self.handle = 0
            if status != 0:
                raise RuntimeError(f"native plan envelope close failed with status {status}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CompiledNativePrecisionContext:
    """Owned per-precision native resources retaining their typed plan mapping."""

    def __init__(self, backend, handle: int, info: dict):
        if handle <= 0:
            raise ValueError("invalid native precision context handle")
        self.backend = backend
        self.handle = int(handle)
        self.info = info

    def close(self) -> None:
        if self.handle:
            status = self.backend.library.green_v400_native_precision_context_close_v1(
                self.handle
            )
            self.handle = 0
            if status != 0:
                raise RuntimeError(
                    f"native precision context close failed with status {status}"
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class ExactDomainKey:
    evaluator_identity_sha256: str
    precision_bits: int
    lower_numerator: int
    lower_denominator: int
    upper_numerator: int
    upper_denominator: int


class _MemoFlight:
    def __init__(self, condition: threading.Condition):
        self.condition = condition
        self.done = False
        self.value: Jet2 | None = None
        self.error: BaseException | None = None


class ExactDomainJetMemo:
    """Bounded, identity-closed, successful-Jet2-only single-flight memo."""

    def __init__(self, evaluator_identity: dict, *, max_entries: int):
        if max_entries <= 0:
            raise ValueError("exact-domain memo needs a positive entry cap")
        self.evaluator_identity = json.loads(json.dumps(
            evaluator_identity, sort_keys=True, separators=(",", ":")
        ))
        self.evaluator_identity_sha256 = sha256_canonical(self.evaluator_identity)
        self.max_entries = int(max_entries)
        self._lock = threading.RLock()
        self._values: OrderedDict[ExactDomainKey, Jet2] = OrderedDict()
        self._inflight: dict[ExactDomainKey, _MemoFlight] = {}
        self._metrics: dict[int, dict[str, int]] = {}

    def _counters(self, precision_bits: int) -> dict[str, int]:
        return self._metrics.setdefault(int(precision_bits), {
            "logical_requests": 0, "hits": 0, "misses": 0, "waits": 0,
        })

    def key(self, domain: Interval) -> ExactDomainKey:
        lower = gmpy2.mpq(domain.lower)
        upper = gmpy2.mpq(domain.upper)
        return ExactDomainKey(
            self.evaluator_identity_sha256,
            domain.precision_bits,
            int(lower.numerator), int(lower.denominator),
            int(upper.numerator), int(upper.denominator),
        )

    def get_or_compute(self, domain: Interval, compute) -> Jet2:
        key = self.key(domain)
        with self._lock:
            counters = self._counters(domain.precision_bits)
            counters["logical_requests"] += 1
            cached = self._values.get(key)
            if cached is not None:
                counters["hits"] += 1
                self._values.move_to_end(key)
                return cached
            flight = self._inflight.get(key)
            if flight is not None:
                counters["waits"] += 1
                while not flight.done:
                    flight.condition.wait()
                if flight.error is not None:
                    raise flight.error
                assert flight.value is not None
                return flight.value
            counters["misses"] += 1
            flight = _MemoFlight(threading.Condition(self._lock))
            self._inflight[key] = flight
        try:
            value = compute()
            if not isinstance(value, Jet2) or value.precision_bits != domain.precision_bits:
                raise RuntimeError("exact-domain memo compute returned an invalid Jet2")
        except BaseException as error:
            with self._lock:
                flight.error = error
                flight.done = True
                self._inflight.pop(key, None)
                flight.condition.notify_all()
            raise
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
            flight.value = value
            flight.done = True
            self._inflight.pop(key, None)
            flight.condition.notify_all()
        return value

    def clear(self) -> None:
        with self._lock:
            if self._inflight:
                raise RuntimeError("cannot clear exact-domain memo with in-flight work")
            self._values.clear()

    def metrics(self) -> dict:
        with self._lock:
            return {
                "evaluator_identity_sha256": self.evaluator_identity_sha256,
                "max_entries": self.max_entries,
                "entry_count": len(self._values),
                "by_precision": {
                    str(key): dict(value) for key, value in sorted(self._metrics.items())
                },
            }


class CompiledNativeJointWitnessEvaluator:
    """Outcome-blind adapter from retained native contexts to certificate Jet2."""

    contains_scientific_outcome = False

    def __init__(self, backend, contexts: dict[int, CompiledNativePrecisionContext], *,
                 certificate_row_hash: str, expected_kernel_tags: tuple[int, ...],
                 exact_domain_memo: ExactDomainJetMemo | None = None):
        if len(certificate_row_hash) != 64:
            raise ValueError("native certificate row hash must be SHA-256")
        if not contexts:
            raise ValueError("native evaluator requires at least one precision context")
        for precision, context in contexts.items():
            if (context.backend is not backend or context.handle <= 0
                    or context.info["precision_bits"] != int(precision)):
                raise ValueError("native evaluator context identity mismatch")
        if not expected_kernel_tags:
            raise ValueError("native evaluator requires the frozen dispatch trace")
        plan_identities = {
            context.info.get("native_plan_identity_sha256") for context in contexts.values()
        }
        if len(plan_identities) != 1 or None in plan_identities:
            raise ValueError("native evaluator contexts lack one closed plan identity")
        self.backend = backend
        self.contexts = dict(contexts)
        self.certificate_row_hash = certificate_row_hash
        self.expected_kernel_tags = tuple(int(tag) for tag in expected_kernel_tags)
        self.evaluator_identity = {
            "schema_version": "green-v400-exact-domain-evaluator-identity-v1",
            "certificate_row_hash": certificate_row_hash,
            "native_plan_identity_sha256": next(iter(plan_identities)),
            "backend_library_sha256": backend.library_sha256,
            "backend_version": backend.version,
            "expected_kernel_tags_sha256": sha256_canonical(self.expected_kernel_tags),
            "rounding_environment_sha256": sha256_canonical(
                rounding_environment_manifest()
            ),
        }
        self.evaluator_identity_sha256 = sha256_canonical(self.evaluator_identity)
        if (exact_domain_memo is not None
                and exact_domain_memo.evaluator_identity != self.evaluator_identity):
            raise ValueError("exact-domain memo evaluator identity mismatch")
        self.exact_domain_memo = exact_domain_memo
        self.dispatch_count_by_precision = {int(key): 0 for key in contexts}
        self.dispatch_attempt_count_by_precision = {int(key): 0 for key in contexts}
        self._dispatch_locks = {int(key): threading.Lock() for key in contexts}

    @staticmethod
    def _interval(payload: dict, precision_bits: int) -> Interval:
        lower = _exact_fraction(payload["lower"])
        upper = _exact_fraction(payload["upper"])
        return Interval.from_bounds(lower, upper, precision_bits)

    def evaluate_interval(self, domain: Interval) -> Jet2:
        context = self.contexts.get(domain.precision_bits)
        if context is None or context.handle <= 0:
            raise RuntimeError("native evaluator precision context is unavailable")

        def dispatch_and_validate() -> Jet2:
            with self._dispatch_locks[domain.precision_bits]:
                self.dispatch_attempt_count_by_precision[domain.precision_bits] += 1
                payload = self.backend.dispatch_native_precision_context_cell(context, domain)
            if (payload.get("event_count") != len(self.expected_kernel_tags)
                    or tuple(payload.get("kernel_tags", ())) != self.expected_kernel_tags):
                raise RuntimeError("NATIVE_CERTIFICATE_DISPATCH_IDENTITY_INVALID")
            output = payload["output"]
            jet = Jet2(*(
                self._interval(output[component], domain.precision_bits)
                for component in ("value", "first", "second")
            ))
            self.dispatch_count_by_precision[domain.precision_bits] += 1
            return jet

        if self.exact_domain_memo is None:
            return dispatch_and_validate()
        return self.exact_domain_memo.get_or_compute(domain, dispatch_and_validate)


class ParallelNativeSiblingEvaluator:
    """Two independent native contexts with canonical ordered pair commits."""

    contains_scientific_outcome = False

    def __init__(self, workers: tuple[CompiledNativeJointWitnessEvaluator, ...]):
        if len(workers) != 2:
            raise ValueError("parallel sibling evaluator requires exactly two workers")
        if workers[0].evaluator_identity != workers[1].evaluator_identity:
            raise ValueError("parallel sibling worker identity mismatch")
        if (workers[0].exact_domain_memo is None
                or workers[0].exact_domain_memo is not workers[1].exact_domain_memo):
            raise ValueError("parallel sibling workers require one shared exact-domain memo")
        self.workers = workers
        self.certificate_row_hash = workers[0].certificate_row_hash
        self.evaluator_identity = workers[0].evaluator_identity
        self.evaluator_identity_sha256 = workers[0].evaluator_identity_sha256
        self.exact_domain_memo = workers[0].exact_domain_memo
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="green-v400-native-sibling"
        )
        self._closed = False

    def evaluate_interval(self, domain: Interval) -> Jet2:
        if self._closed:
            raise RuntimeError("parallel sibling evaluator is closed")
        return self.workers[0].evaluate_interval(domain)

    def evaluate_interval_pair(self, domains: tuple[Interval, Interval]) -> tuple[Jet2, Jet2]:
        if self._closed:
            raise RuntimeError("parallel sibling evaluator is closed")
        if len(domains) != 2 or domains[0].precision_bits != domains[1].precision_bits:
            raise ValueError("parallel sibling domains must be one precision pair")
        futures = tuple(
            self._executor.submit(worker.evaluate_interval, domain)
            for worker, domain in zip(self.workers, domains)
        )
        # Result order is canonical input order, independent of physical completion.
        results: list[Jet2 | None] = [None, None]
        errors: list[BaseException] = []
        for index, future in enumerate(futures):
            try:
                results[index] = future.result()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        assert results[0] is not None and results[1] is not None
        return results[0], results[1]

    @property
    def dispatch_count_by_precision(self) -> dict[int, int]:
        precisions = set().union(*(
            worker.dispatch_count_by_precision for worker in self.workers
        ))
        return {
            precision: sum(worker.dispatch_count_by_precision.get(precision, 0)
                           for worker in self.workers)
            for precision in precisions
        }

    @property
    def dispatch_attempt_count_by_precision(self) -> dict[int, int]:
        precisions = set().union(*(
            worker.dispatch_attempt_count_by_precision for worker in self.workers
        ))
        return {
            precision: sum(worker.dispatch_attempt_count_by_precision.get(precision, 0)
                           for worker in self.workers)
            for precision in precisions
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class CompiledMPFRBackend:
    def __init__(self, library_path: Path):
        self.library_path = Path(library_path).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(self.library_path)
        self.library_sha256 = hashlib.sha256(self.library_path.read_bytes()).hexdigest()
        self.library = ctypes.CDLL(str(self.library_path))
        self.library.green_v400_mpfr_backend_version.restype = ctypes.c_char_p
        self.version = self.library.green_v400_mpfr_backend_version().decode("ascii")
        if self.version != "green-v400-compiled-mpfr-v2":
            raise ValueError("compiled MPFR backend version mismatch")
        function = self.library.green_v400_affine_jet2_f32
        function.argtypes = [
            ctypes.c_uint32, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            *([ctypes.POINTER(ctypes.c_uint64)] * 6),
            ctypes.c_char_p, ctypes.c_uint64,
        ]
        function.restype = ctypes.c_int
        exact = self.library.green_v400_affine_jet2_exact
        exact.argtypes = [
            ctypes.c_uint32, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            *sum(([ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64)]
                  for _ in range(6)), []),
            ctypes.c_char_p, ctypes.c_uint64,
        ]
        exact.restype = ctypes.c_int
        packed_affine = self.library.green_v400_packed_affine_layer_jet2_exact
        packed_affine.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_char_p, ctypes.c_uint64,
        ]
        packed_affine.restype = ctypes.c_int
        benchmark = self.library.green_v400_benchmark_affine_jet2_layer
        benchmark.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
        ]
        benchmark.restype = ctypes.c_int
        benchmark_gelu = self.library.green_v400_benchmark_gelu_jet2
        benchmark_gelu.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
        ]
        benchmark_gelu.restype = ctypes.c_int
        benchmark_layer_norm = self.library.green_v400_benchmark_layer_norm_jet2
        benchmark_layer_norm.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
        ]
        benchmark_layer_norm.restype = ctypes.c_int
        benchmark_attention = self.library.green_v400_benchmark_causal_attention_jet2
        benchmark_attention.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        benchmark_attention.restype = ctypes.c_int
        benchmark_joint = self.library.green_v400_benchmark_gpt2_joint_witness_cell
        benchmark_joint.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
        ]
        benchmark_joint.restype = ctypes.c_int
        primitive = self.library.green_v400_interval_primitive_exact
        primitive.argtypes = [
            ctypes.c_char_p, ctypes.c_uint32,
            ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64,
            ctypes.c_char_p, ctypes.c_uint64,
        ]
        primitive.restype = ctypes.c_int
        gelu = self.library.green_v400_gelu_new_jet2_exact
        gelu.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_char_p, ctypes.c_uint64,
        ]
        gelu.restype = ctypes.c_int
        gelu_layer = self.library.green_v400_gelu_new_layer_jet2_exact
        gelu_layer.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint64,
        ]
        gelu_layer.restype = ctypes.c_int
        layer_norm = self.library.green_v400_layer_norm_jet2_exact
        layer_norm.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p, ctypes.c_uint64,
        ]
        layer_norm.restype = ctypes.c_int
        attention = self.library.green_v400_causal_attention_final_head_jet2_exact
        attention.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_char_p, ctypes.c_uint64,
        ]
        attention.restype = ctypes.c_int
        attention_all_heads = self.library.green_v400_causal_attention_final_all_heads_jet2_exact
        attention_all_heads.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int64), ctypes.c_char_p, ctypes.c_uint64,
        ]
        attention_all_heads.restype = ctypes.c_int
        contrast = self.library.green_v400_final_contrast_jet2_exact
        contrast.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_char_p, ctypes.c_uint64,
        ]
        contrast.restype = ctypes.c_int
        fused_contrast = self.library.green_v400_fused_contrast_jet2_exact
        fused_contrast.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_uint64,
        ]
        fused_contrast.restype = ctypes.c_int
        resident_import = self.library.green_v400_resident_jet_buffer_import_exact
        resident_import.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_import.restype = ctypes.c_int
        resident_import_f32 = (
            self.library.green_v400_resident_jet_buffer_import_f32_constants
        )
        resident_import_f32.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_import_f32.restype = ctypes.c_int
        resident_affine = self.library.green_v400_resident_jet_buffer_packed_affine
        resident_affine.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_affine.restype = ctypes.c_int
        resident_gelu = self.library.green_v400_resident_jet_buffer_gelu_new
        resident_gelu.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_gelu.restype = ctypes.c_int
        resident_layer_norm = self.library.green_v400_resident_jet_buffer_layer_norm
        resident_layer_norm.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_layer_norm.restype = ctypes.c_int
        resident_add = self.library.green_v400_resident_jet_buffer_add
        resident_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_add.restype = ctypes.c_int
        resident_sub = self.library.green_v400_resident_jet_buffer_sub
        resident_sub.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_sub.restype = ctypes.c_int
        resident_concat = self.library.green_v400_resident_jet_buffer_concat
        resident_concat.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_concat.restype = ctypes.c_int
        resident_attention = (
            self.library.green_v400_resident_jet_buffer_causal_attention_all_heads
        )
        resident_attention.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_attention.restype = ctypes.c_int
        resident_contrast = (
            self.library.green_v400_resident_jet_buffer_fused_contrast_exact
        )
        resident_contrast.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int64), ctypes.c_char_p, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        resident_contrast.restype = ctypes.c_int
        resident_export = self.library.green_v400_resident_jet_buffer_export_json
        resident_export.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64,
        ]
        resident_export.restype = ctypes.c_int
        resident_width = self.library.green_v400_resident_jet_buffer_width
        resident_width.argtypes = [ctypes.c_void_p]
        resident_width.restype = ctypes.c_uint32
        resident_free = self.library.green_v400_resident_jet_buffer_free
        resident_free.argtypes = [ctypes.c_void_p]
        resident_free.restype = None
        native_open = self.library.green_v400_native_plan_envelope_open_v1
        native_open.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        native_open.restype = ctypes.c_int
        native_info = self.library.green_v400_native_plan_envelope_info_v1
        native_info.argtypes = [ctypes.c_uint64] + [
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ]
        native_info.restype = ctypes.c_int
        native_close = self.library.green_v400_native_plan_envelope_close_v1
        native_close.argtypes = [ctypes.c_uint64]
        native_close.restype = ctypes.c_int
        native_payload_validated = (
            self.library.green_v400_native_plan_payload_validated_v1
        )
        native_payload_validated.argtypes = [ctypes.c_uint64]
        native_payload_validated.restype = ctypes.c_int
        native_typed_info = self.library.green_v400_native_plan_typed_info_v1
        native_typed_info.argtypes = [ctypes.c_uint64] + [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ]
        native_typed_info.restype = ctypes.c_int
        native_typed_trace = self.library.green_v400_native_plan_typed_trace_v1
        native_typed_trace.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        native_typed_trace.restype = ctypes.c_int
        native_context_open = self.library.green_v400_native_precision_context_open_v1
        native_context_open.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
        ]
        native_context_open.restype = ctypes.c_int
        native_context_info = self.library.green_v400_native_precision_context_info_v1
        native_context_info.argtypes = [ctypes.c_uint64] + [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        native_context_info.restype = ctypes.c_int
        native_projection_info = (
            self.library.green_v400_native_precision_context_projection_info_v1
        )
        native_projection_info.argtypes = [ctypes.c_uint64] + [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ]
        native_projection_info.restype = ctypes.c_int
        native_projection_export = (
            self.library.green_v400_native_precision_context_projection_export_json_v1
        )
        native_projection_export.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint64,
        ]
        native_projection_export.restype = ctypes.c_int
        native_dispatch = self.library.green_v400_native_precision_context_dispatch_cell_v1
        native_dispatch.argtypes = [
            ctypes.c_uint64, ctypes.c_char_p, ctypes.c_int64,
            ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_uint64,
        ]
        native_dispatch.restype = ctypes.c_int
        native_context_close = self.library.green_v400_native_precision_context_close_v1
        native_context_close.argtypes = [ctypes.c_uint64]
        native_context_close.restype = ctypes.c_int

    def open_native_plan_envelope(
        self, descriptor_path: Path, blob_path: Path, *, descriptor_sha256: str,
        program_execution_sha256: str, dispatch_sha256: str, blob_sha256: str,
        fusion_sha256: str, blob_nbytes: int, fusion_weight_count: int,
    ) -> CompiledNativePlanEnvelope:
        handle = ctypes.c_uint64()
        status = self.library.green_v400_native_plan_envelope_open_v1(
            str(Path(descriptor_path).resolve()).encode(),
            str(Path(blob_path).resolve()).encode(), descriptor_sha256.encode(),
            program_execution_sha256.encode(), dispatch_sha256.encode(),
            blob_sha256.encode(), fusion_sha256.encode(), int(blob_nbytes),
            int(fusion_weight_count),
            ctypes.byref(handle),
        )
        if status != 0:
            raise RuntimeError(f"native plan envelope open failed with status {status}")
        values64 = [ctypes.c_uint64(), ctypes.c_uint64()]
        values32 = [ctypes.c_uint32() for _ in range(4)]
        status = self.library.green_v400_native_plan_envelope_info_v1(
            handle.value, *(ctypes.byref(value) for value in [*values64, *values32])
        )
        if status != 0:
            self.library.green_v400_native_plan_envelope_close_v1(handle.value)
            raise RuntimeError(f"native plan envelope info failed with status {status}")
        if self.library.green_v400_native_plan_payload_validated_v1(handle.value) != 1:
            self.library.green_v400_native_plan_envelope_close_v1(handle.value)
            raise RuntimeError("native plan envelope payload-table validation was not retained")
        typed32 = [ctypes.c_uint32() for _ in range(5)]
        liveness_rows = ctypes.c_uint64()
        status = self.library.green_v400_native_plan_typed_info_v1(
            handle.value, *(ctypes.byref(value) for value in typed32[:4]),
            ctypes.byref(liveness_rows), ctypes.byref(typed32[4]),
        )
        if (status != 0 or [value.value for value in typed32[:4]]
                != [value.value for value in values32] or typed32[4].value != 4):
            self.library.green_v400_native_plan_envelope_close_v1(handle.value)
            raise RuntimeError("native typed plan materialization is inconsistent")
        native_plan_identity = {
            "schema_version": "green-v400-native-plan-identity-v1",
            "descriptor_file_sha256": descriptor_sha256,
            "program_execution_semantic_hash": program_execution_sha256,
            "dispatch_signature_sha256": dispatch_sha256,
            "blob_sha256": blob_sha256,
            "fusion_sha256": fusion_sha256,
            "blob_nbytes": int(blob_nbytes),
            "record_count": values32[0].value,
            "node_count": values32[1].value,
            "binding_count": values32[2].value,
            "fusion_weight_count": values32[3].value,
        }
        return CompiledNativePlanEnvelope(self, handle.value, {
            "descriptor_nbytes": values64[0].value, "blob_nbytes": values64[1].value,
            "record_count": values32[0].value, "node_count": values32[1].value,
            "binding_count": values32[2].value, "fusion_weight_count": values32[3].value,
            "payload_tables_validated": True,
            "typed_plan_materialized": True,
            "liveness_row_count": liveness_rows.value,
            "branch_root_count": typed32[4].value,
            "native_plan_identity": native_plan_identity,
            "native_plan_identity_sha256": sha256_canonical(native_plan_identity),
        })

    def native_plan_typed_trace(self, envelope: CompiledNativePlanEnvelope) -> dict:
        if envelope.backend is not self or envelope.handle <= 0:
            raise ValueError("native plan envelope is closed or belongs to another backend")
        node_count = int(envelope.info["node_count"])
        root_count = int(envelope.info["branch_root_count"])
        kernels = (ctypes.c_uint32 * node_count)()
        liveness = (ctypes.c_uint32 * node_count)()
        roots = (ctypes.c_uint32 * root_count)()
        output_root = ctypes.c_uint32()
        status = self.library.green_v400_native_plan_typed_trace_v1(
            envelope.handle, kernels, liveness, node_count, roots, root_count,
            ctypes.byref(output_root),
        )
        if status != 0:
            raise RuntimeError(f"native typed plan trace failed with status {status}")
        return {
            "kernel_tags": list(kernels), "liveness_counts": list(liveness),
            "branch_root_indices": list(roots), "output_root_index": output_root.value,
        }

    def open_native_precision_context(
        self, envelope: CompiledNativePlanEnvelope, precision_bits: int,
    ) -> CompiledNativePrecisionContext:
        if envelope.backend is not self or envelope.handle <= 0:
            raise ValueError("native plan envelope is closed or belongs to another backend")
        handle = ctypes.c_uint64()
        status = self.library.green_v400_native_precision_context_open_v1(
            envelope.handle, int(precision_bits), ctypes.byref(handle)
        )
        if status != 0:
            raise RuntimeError(f"native precision context open failed with status {status}")
        precision = ctypes.c_uint32()
        static_buffers = ctypes.c_uint32()
        static_jets = ctypes.c_uint64()
        nodes = ctypes.c_uint32()
        bindings = ctypes.c_uint32()
        status = self.library.green_v400_native_precision_context_info_v1(
            handle.value, ctypes.byref(precision), ctypes.byref(static_buffers),
            ctypes.byref(static_jets), ctypes.byref(nodes), ctypes.byref(bindings),
        )
        if status != 0:
            self.library.green_v400_native_precision_context_close_v1(handle.value)
            raise RuntimeError(f"native precision context info failed with status {status}")
        return CompiledNativePrecisionContext(self, handle.value, {
            "precision_bits": precision.value,
            "static_buffer_count": static_buffers.value,
            "static_jet_count": static_jets.value,
            "node_count": nodes.value, "binding_count": bindings.value,
            "plan_retained": True,
            "native_plan_identity": envelope.info["native_plan_identity"],
            "native_plan_identity_sha256": envelope.info[
                "native_plan_identity_sha256"
            ],
        })

    def native_precision_context_projection_info(
        self, context: CompiledNativePrecisionContext,
    ) -> dict:
        if context.backend is not self or context.handle <= 0:
            raise ValueError("native precision context is closed or belongs to another backend")
        buffers = ctypes.c_uint32()
        jets = ctypes.c_uint64()
        rows = ctypes.c_uint32()
        branches = ctypes.c_uint32()
        status = self.library.green_v400_native_precision_context_projection_info_v1(
            context.handle, ctypes.byref(buffers), ctypes.byref(jets),
            ctypes.byref(rows), ctypes.byref(branches),
        )
        if status != 0:
            raise RuntimeError(f"native projection info failed with status {status}")
        return {
            "projection_buffer_count": buffers.value,
            "projection_jet_count": jets.value,
            "historical_row_count": rows.value, "branch_count": branches.value,
        }

    def export_native_precision_context_projection(
        self, context: CompiledNativePrecisionContext, index: int,
    ) -> dict:
        info = self.native_precision_context_projection_info(context)
        if not 0 <= int(index) < info["projection_buffer_count"]:
            raise ValueError("native projection index is outside the context")
        width = info["projection_jet_count"] // info["projection_buffer_count"]
        output = ctypes.create_string_buffer(max(8192, 4096 * width))
        status = (
            self.library.green_v400_native_precision_context_projection_export_json_v1(
                context.handle, int(index), output, len(output)
            )
        )
        if status != 0:
            raise RuntimeError(f"native projection export failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def dispatch_native_precision_context_cell(
        self, context: CompiledNativePrecisionContext, domain: Interval,
    ) -> dict:
        if context.backend is not self or context.handle <= 0:
            raise ValueError("native precision context is closed or belongs to another backend")
        if domain.precision_bits != context.info["precision_bits"]:
            raise ValueError("native context/domain precision mismatch")
        lower = _binary_endpoint(domain.lower)
        upper = _binary_endpoint(domain.upper)
        output = ctypes.create_string_buffer(256 * 1024)
        status = self.library.green_v400_native_precision_context_dispatch_cell_v1(
            context.handle, lower[0], lower[1], upper[0], upper[1], output, len(output)
        )
        if status != 0:
            raise RuntimeError(f"native cell dispatch failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def affine_jet2(self, weights, bias, values: list[Jet2], precision_bits: int) -> dict:
        weights = np.ascontiguousarray(np.asarray(weights, dtype="<f4").reshape(-1))
        if len(values) != weights.size or not values:
            raise ValueError("compiled affine width mismatch")
        if any(value.precision_bits != precision_bits for value in values):
            raise ValueError("compiled affine precision mismatch")
        weight_bits = weights.view("<u4")
        exact_arrays = []
        for component in ("value", "first", "second"):
            for endpoint in ("lower", "upper"):
                encoded = [_binary_endpoint(getattr(getattr(value, component), endpoint))
                           for value in values]
                strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
                exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
                exact_arrays.append((strings, exponents))
        output = ctypes.create_string_buffer(8192)
        arguments = []
        for strings, exponents in exact_arrays:
            arguments.extend((
                strings,
                exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ))
        status = self.library.green_v400_affine_jet2_exact(
            precision_bits, weights.size,
            weight_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), _bits_f32(bias),
            *arguments,
            output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled affine backend failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    @staticmethod
    def exact_fraction(payload: dict) -> Fraction:
        return _exact_fraction(payload)

    def benchmark_affine_layer(self, precision_bits: int, input_width: int,
                               output_width: int, repeats: int = 1) -> dict:
        elapsed = ctypes.c_double()
        checksum = ctypes.c_uint64()
        status = self.library.green_v400_benchmark_affine_jet2_layer(
            precision_bits, input_width, output_width, repeats,
            ctypes.byref(elapsed), ctypes.byref(checksum),
        )
        if status != 0:
            raise RuntimeError(f"compiled affine benchmark failed with status {status}")
        coefficient_terms = int(input_width) * int(output_width) * int(repeats)
        directed_mpfr_primitives = coefficient_terms * 12
        return {
            "precision_bits": int(precision_bits),
            "input_width": int(input_width),
            "output_width": int(output_width),
            "repeats": int(repeats),
            "coefficient_terms": coefficient_terms,
            "directed_mpfr_primitives": directed_mpfr_primitives,
            "elapsed_seconds": elapsed.value,
            "directed_mpfr_primitives_per_second": directed_mpfr_primitives / elapsed.value,
            "checksum": f"{checksum.value:016x}",
        }

    def benchmark_gelu(self, precision_bits: int, count: int, repeats: int = 1) -> dict:
        elapsed = ctypes.c_double()
        checksum = ctypes.c_uint64()
        status = self.library.green_v400_benchmark_gelu_jet2(
            precision_bits, count, repeats, ctypes.byref(elapsed), ctypes.byref(checksum)
        )
        if status != 0:
            raise RuntimeError(f"compiled GELU benchmark failed with status {status}")
        return {
            "precision_bits": int(precision_bits), "count": int(count),
            "repeats": int(repeats), "elapsed_seconds": elapsed.value,
            "jets_per_second": int(count) * int(repeats) / elapsed.value,
            "checksum": f"{checksum.value:016x}",
        }

    def benchmark_layer_norm(self, precision_bits: int, width: int,
                             vector_count: int, repeats: int = 1) -> dict:
        elapsed = ctypes.c_double()
        checksum = ctypes.c_uint64()
        status = self.library.green_v400_benchmark_layer_norm_jet2(
            precision_bits, width, vector_count, repeats,
            ctypes.byref(elapsed), ctypes.byref(checksum),
        )
        if status != 0:
            raise RuntimeError(f"compiled LayerNorm benchmark failed with status {status}")
        return {
            "precision_bits": int(precision_bits), "width": int(width),
            "vector_count": int(vector_count), "repeats": int(repeats),
            "elapsed_seconds": elapsed.value,
            "vectors_per_second": int(vector_count) * int(repeats) / elapsed.value,
            "checksum": f"{checksum.value:016x}",
        }

    def benchmark_causal_attention(self, precision_bits: int, sequence_length: int,
                                   n_heads: int, head_dim: int,
                                   branch_count: int = 1, repeats: int = 1) -> dict:
        elapsed = ctypes.c_double()
        checksum = ctypes.c_uint64()
        status = self.library.green_v400_benchmark_causal_attention_jet2(
            precision_bits, sequence_length, n_heads, head_dim, branch_count, repeats,
            ctypes.byref(elapsed), ctypes.byref(checksum),
        )
        if status != 0:
            raise RuntimeError(f"compiled attention benchmark failed with status {status}")
        evaluations = int(n_heads) * int(branch_count) * int(repeats)
        return {
            "precision_bits": int(precision_bits),
            "sequence_length": int(sequence_length), "n_heads": int(n_heads),
            "head_dim": int(head_dim), "branch_count": int(branch_count),
            "repeats": int(repeats), "head_evaluations": evaluations,
            "elapsed_seconds": elapsed.value,
            "head_evaluations_per_second": evaluations / elapsed.value,
            "checksum": f"{checksum.value:016x}",
        }

    def packed_affine_layer_jet2(self, weight, bias, values: list[Jet2]) -> dict:
        if not values:
            raise ValueError("packed affine layer requires inputs")
        precision = values[0].precision_bits
        if any(value.precision_bits != precision for value in values):
            raise ValueError("packed affine precision mismatch")
        weight = np.asarray(weight, dtype="<f4", order="C")
        bias = np.asarray(bias, dtype="<f4").reshape(-1)
        if weight.ndim != 2 or weight.shape != (len(values), bias.size):
            raise ValueError("packed affine shape mismatch")
        encoded = []
        for value in values:
            for component in (value.value, value.first, value.second):
                encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        output = ctypes.create_string_buffer(max(8192, 4096 * bias.size))
        status = self.library.green_v400_packed_affine_layer_jet2_exact(
            precision, len(values), bias.size, strings,
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            weight.view("<u4").ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            bias.view("<u4").ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled packed affine layer failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def resident_jet_buffer(self, values: list[Jet2]) -> CompiledResidentJetBuffer:
        if not values:
            raise ValueError("resident Jet2 buffer requires values")
        precision = values[0].precision_bits
        if any(value.precision_bits != precision for value in values):
            raise ValueError("resident Jet2 buffer precision mismatch")
        encoded = []
        for value in values:
            for component in (value.value, value.first, value.second):
                encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        handle = ctypes.c_void_p()
        status = self.library.green_v400_resident_jet_buffer_import_exact(
            precision, len(values), strings,
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.byref(handle),
        )
        if status != 0:
            raise RuntimeError(f"compiled resident Jet2 import failed with status {status}")
        return CompiledResidentJetBuffer(self, handle, precision, len(values))

    def resident_f32_constant_buffer(
        self, values, precision_bits: int,
    ) -> CompiledResidentJetBuffer:
        array = np.ascontiguousarray(np.asarray(values, dtype="<f4").reshape(-1))
        if array.size == 0:
            raise ValueError("resident f32 constant buffer requires values")
        handle = ctypes.c_void_p()
        status = self.library.green_v400_resident_jet_buffer_import_f32_constants(
            int(precision_bits), array.size,
            array.view("<u4").ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.byref(handle),
        )
        if status != 0:
            raise RuntimeError(f"compiled resident f32 import failed with status {status}")
        return CompiledResidentJetBuffer(
            self, handle, int(precision_bits), int(array.size)
        )

    def resident_packed_affine_layer_jet2(
        self, values: CompiledResidentJetBuffer, weight, bias,
    ) -> CompiledResidentJetBuffer:
        weight = np.asarray(weight, dtype="<f4", order="C")
        bias = np.asarray(bias, dtype="<f4").reshape(-1)
        if (not values.handle.value or weight.ndim != 2
                or weight.shape != (values.width, bias.size)):
            raise ValueError("resident packed affine shape mismatch")
        handle = ctypes.c_void_p()
        status = self.library.green_v400_resident_jet_buffer_packed_affine(
            values.handle, bias.size,
            weight.view("<u4").ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            bias.view("<u4").ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.byref(handle),
        )
        if status != 0:
            raise RuntimeError(f"resident packed affine failed with status {status}")
        return CompiledResidentJetBuffer(self, handle, values.precision_bits, bias.size)

    def resident_gelu_new_layer_jet2(
        self, values: CompiledResidentJetBuffer, kappa, lam,
    ) -> CompiledResidentJetBuffer:
        if not values.handle.value:
            raise ValueError("resident GELU input buffer is closed")
        handle = ctypes.c_void_p()
        status = self.library.green_v400_resident_jet_buffer_gelu_new(
            values.handle, _bits_f32(kappa), _bits_f32(lam), ctypes.byref(handle)
        )
        if status != 0:
            raise RuntimeError(f"resident GELU failed with status {status}")
        return CompiledResidentJetBuffer(
            self, handle, values.precision_bits, values.width
        )

    def resident_layer_norm_jet2(
        self, values: CompiledResidentJetBuffer, epsilon, gamma, beta,
    ) -> CompiledResidentJetBuffer:
        gamma_bits = np.asarray(gamma, dtype="<f4").reshape(-1).view("<u4")
        beta_bits = np.asarray(beta, dtype="<f4").reshape(-1).view("<u4")
        if (not values.handle.value or gamma_bits.size != values.width
                or beta_bits.size != values.width):
            raise ValueError("resident LayerNorm affine width mismatch or closed buffer")
        handle = ctypes.c_void_p()
        status = self.library.green_v400_resident_jet_buffer_layer_norm(
            values.handle, _bits_f32(epsilon),
            gamma_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            beta_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.byref(handle),
        )
        if status != 0:
            raise RuntimeError(f"resident LayerNorm failed with status {status}")
        return CompiledResidentJetBuffer(
            self, handle, values.precision_bits, values.width
        )

    def _resident_binary_jet2(
        self, operation: str, left: CompiledResidentJetBuffer,
        right: CompiledResidentJetBuffer,
    ) -> CompiledResidentJetBuffer:
        if (not left.handle.value or not right.handle.value
                or left.backend is not self or right.backend is not self
                or left.precision_bits != right.precision_bits
                or left.width != right.width):
            raise ValueError(f"resident {operation} buffer mismatch")
        handle = ctypes.c_void_p()
        function = getattr(
            self.library, f"green_v400_resident_jet_buffer_{operation}"
        )
        status = function(left.handle, right.handle, ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"resident {operation} failed with status {status}")
        return CompiledResidentJetBuffer(
            self, handle, left.precision_bits, left.width
        )

    def resident_add_jet2(
        self, left: CompiledResidentJetBuffer, right: CompiledResidentJetBuffer,
    ) -> CompiledResidentJetBuffer:
        return self._resident_binary_jet2("add", left, right)

    def resident_sub_jet2(
        self, left: CompiledResidentJetBuffer, right: CompiledResidentJetBuffer,
    ) -> CompiledResidentJetBuffer:
        return self._resident_binary_jet2("sub", left, right)

    def resident_concat_jet2(
        self, values: list[CompiledResidentJetBuffer],
    ) -> CompiledResidentJetBuffer:
        if (not values or any(
            not value.handle.value or value.backend is not self
            or value.precision_bits != values[0].precision_bits for value in values
        )):
            raise ValueError("resident concat buffer mismatch")
        handles = (ctypes.c_void_p * len(values))(
            *(value.handle.value for value in values)
        )
        handle = ctypes.c_void_p()
        status = self.library.green_v400_resident_jet_buffer_concat(
            handles, len(values), ctypes.byref(handle)
        )
        if status != 0:
            raise RuntimeError(f"resident concat failed with status {status}")
        return CompiledResidentJetBuffer(
            self, handle, values[0].precision_bits,
            sum(value.width for value in values),
        )

    def resident_causal_attention_all_heads_jet2(
        self, query: CompiledResidentJetBuffer,
        keys: CompiledResidentJetBuffer, values: CompiledResidentJetBuffer,
        sequence_length: int, n_heads: int, head_dim: int, pivot: int = 0,
    ) -> CompiledResidentJetBuffer:
        d_model = int(n_heads) * int(head_dim)
        if (any(not item.handle.value or item.backend is not self
                for item in (query, keys, values))
                or query.precision_bits != keys.precision_bits
                or query.precision_bits != values.precision_bits
                or query.width != d_model
                or keys.width != int(sequence_length) * d_model
                or values.width != int(sequence_length) * d_model):
            raise ValueError("resident attention buffer shape mismatch")
        handle = ctypes.c_void_p()
        status = (
            self.library.green_v400_resident_jet_buffer_causal_attention_all_heads(
                query.handle, keys.handle, values.handle, sequence_length,
                n_heads, head_dim, pivot, ctypes.byref(handle),
            )
        )
        if status != 0:
            raise RuntimeError(f"resident attention failed with status {status}")
        return CompiledResidentJetBuffer(
            self, handle, query.precision_bits, d_model
        )

    def resident_fused_contrast_jet2(
        self, values: CompiledResidentJetBuffer, fusion_payload: dict,
    ) -> CompiledResidentJetBuffer:
        weights = fusion_payload.get("weights", [])
        if (not values.handle.value or values.backend is not self
                or fusion_payload.get("d_model") != values.width
                or len(weights) != values.width):
            raise ValueError("resident fused contrast shape mismatch")

        def hexadecimal_significand(item: dict) -> bytes:
            value = int(item["significand"])
            return (("-" if value < 0 else "") + format(abs(value), "x")).encode("ascii")

        weight_strings = (ctypes.c_char_p * len(weights))(
            *(hexadecimal_significand(item) for item in weights)
        )
        weight_exponents = np.asarray(
            [item["exponent_2"] for item in weights], dtype="<i8"
        )
        bias = fusion_payload["bias"]
        handle = ctypes.c_void_p()
        status = self.library.green_v400_resident_jet_buffer_fused_contrast_exact(
            values.handle, weight_strings,
            weight_exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            hexadecimal_significand(bias), int(bias["exponent_2"]),
            ctypes.byref(handle),
        )
        if status != 0:
            raise RuntimeError(f"resident fused contrast failed with status {status}")
        return CompiledResidentJetBuffer(
            self, handle, values.precision_bits, 1
        )

    def export_resident_jet_buffer(self, values: CompiledResidentJetBuffer) -> dict:
        if not values.handle.value:
            raise ValueError("resident Jet2 export buffer is closed")
        native_width = self.library.green_v400_resident_jet_buffer_width(values.handle)
        if native_width != values.width:
            raise RuntimeError("resident Jet2 native width mismatch")
        output = ctypes.create_string_buffer(max(8192, 4096 * values.width))
        status = self.library.green_v400_resident_jet_buffer_export_json(
            values.handle, output, len(output)
        )
        if status != 0:
            raise RuntimeError(f"resident Jet2 export failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def benchmark_gpt2_joint_witness_cell(
        self, precision_bits: int, d_model: int, d_mlp: int,
        sequence_length: int, n_heads: int, d_head: int,
        selected_gates: int, repeats: int = 1,
    ) -> dict:
        elapsed = ctypes.c_double()
        checksum = ctypes.c_uint64()
        primitive_count = ctypes.c_uint64()
        dispatch_trace = ctypes.c_uint64()
        dispatch_events = ctypes.c_uint64()
        tag_capacity = 81 * int(repeats)
        dispatch_tags = (ctypes.c_uint8 * tag_capacity)()
        status = self.library.green_v400_benchmark_gpt2_joint_witness_cell(
            precision_bits, d_model, d_mlp, sequence_length, n_heads,
            d_head, selected_gates, repeats,
            ctypes.byref(elapsed), ctypes.byref(checksum), ctypes.byref(primitive_count),
            ctypes.byref(dispatch_trace), ctypes.byref(dispatch_events),
            dispatch_tags, tag_capacity,
        )
        if status != 0:
            raise RuntimeError(f"compiled joint-witness benchmark failed with status {status}")
        return {
            "precision_bits": int(precision_bits), "d_model": int(d_model),
            "d_mlp": int(d_mlp), "sequence_length": int(sequence_length),
            "n_heads": int(n_heads), "d_head": int(d_head),
            "selected_gates": int(selected_gates), "repeats": int(repeats),
            "elapsed_seconds": elapsed.value,
            "cells_per_second": int(repeats) / elapsed.value,
            "mpfr_primitive_count": int(primitive_count.value),
            "mpfr_primitives_per_second": int(primitive_count.value) / elapsed.value,
            "dispatch_trace_fnv1a_u64": f"{dispatch_trace.value:016x}",
            "dispatch_event_count": int(dispatch_events.value),
            "dispatch_tags": [int(dispatch_tags[index])
                              for index in range(dispatch_events.value)],
            "checksum": f"{checksum.value:016x}",
        }

    def interval_primitive(self, operation: str, interval: Interval) -> dict:
        if operation not in {"exp", "tanh", "sqrt", "inv_sqrt"}:
            raise ValueError("unsupported compiled interval primitive")
        lower_significand, lower_exponent = _binary_endpoint(interval.lower)
        upper_significand, upper_exponent = _binary_endpoint(interval.upper)
        output = ctypes.create_string_buffer(4096)
        status = self.library.green_v400_interval_primitive_exact(
            operation.encode("ascii"), interval.precision_bits,
            lower_significand, lower_exponent, upper_significand, upper_exponent,
            output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled interval primitive failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def gelu_new_jet2(self, value: Jet2, kappa, lam) -> dict:
        encoded = []
        for component in (value.value, value.first, value.second):
            encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * 6)(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        output = ctypes.create_string_buffer(8192)
        status = self.library.green_v400_gelu_new_jet2_exact(
            value.precision_bits, strings,
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            _bits_f32(kappa), _bits_f32(lam), output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled GELU jet failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def gelu_new_layer_jet2(self, values: list[Jet2], kappa, lam) -> dict:
        if not values:
            raise ValueError("compiled GELU layer requires values")
        precision = values[0].precision_bits
        if any(value.precision_bits != precision for value in values):
            raise ValueError("compiled GELU layer precision mismatch")
        encoded = []
        for value in values:
            for component in (value.value, value.first, value.second):
                encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        output = ctypes.create_string_buffer(max(8192, 4096 * len(values)))
        status = self.library.green_v400_gelu_new_layer_jet2_exact(
            precision, len(values), strings,
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            _bits_f32(kappa), _bits_f32(lam), output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled GELU layer failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def layer_norm_jet2(self, values: list[Jet2], epsilon, gamma, beta) -> dict:
        if not values:
            raise ValueError("compiled LayerNorm requires values")
        precision = values[0].precision_bits
        if any(value.precision_bits != precision for value in values):
            raise ValueError("compiled LayerNorm precision mismatch")
        gamma_bits = np.asarray(gamma, dtype="<f4").reshape(-1).view("<u4")
        beta_bits = np.asarray(beta, dtype="<f4").reshape(-1).view("<u4")
        if gamma_bits.size != len(values) or beta_bits.size != len(values):
            raise ValueError("compiled LayerNorm affine width mismatch")
        encoded = []
        for value in values:
            for component in (value.value, value.first, value.second):
                encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        output = ctypes.create_string_buffer(max(8192, 4096 * len(values)))
        status = self.library.green_v400_layer_norm_jet2_exact(
            precision, len(values), strings,
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            _bits_f32(epsilon), gamma_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            beta_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled LayerNorm jet failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def causal_attention_final_head_jet2(self, query: list[Jet2], keys: list[list[Jet2]],
                                         values: list[list[Jet2]], pivot: int = 0) -> dict:
        if not query or not keys or len(keys) != len(values):
            raise ValueError("compiled attention shape mismatch")
        head_dim, sequence_length = len(query), len(keys)
        if any(len(row) != head_dim for row in keys + values):
            raise ValueError("compiled attention head width mismatch")
        jets = query + [jet for row in keys for jet in row] + [jet for row in values for jet in row]
        precision = jets[0].precision_bits
        if any(jet.precision_bits != precision for jet in jets):
            raise ValueError("compiled attention precision mismatch")
        encoded = []
        for jet in jets:
            for component in (jet.value, jet.first, jet.second):
                encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        output = ctypes.create_string_buffer(max(8192, 4096 * head_dim))
        status = self.library.green_v400_causal_attention_final_head_jet2_exact(
            precision, sequence_length, head_dim, pivot, strings,
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled attention jet failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def causal_attention_final_all_heads_jet2(
        self, query: list[Jet2], keys: list[list[Jet2]], values: list[list[Jet2]],
        n_heads: int, pivot: int = 0,
    ) -> dict:
        if not query or not keys or len(keys) != len(values) or n_heads <= 0:
            raise ValueError("compiled all-head attention shape mismatch")
        d_model, sequence_length = len(query), len(keys)
        if d_model % n_heads or any(len(row) != d_model for row in keys + values):
            raise ValueError("compiled all-head attention width mismatch")
        head_dim = d_model // n_heads
        jets = query + [jet for row in keys for jet in row] + [jet for row in values for jet in row]
        precision = jets[0].precision_bits
        if any(jet.precision_bits != precision for jet in jets):
            raise ValueError("compiled all-head attention precision mismatch")
        encoded = []
        for jet in jets:
            for component in (jet.value, jet.first, jet.second):
                encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        output = ctypes.create_string_buffer(max(8192, 4096 * d_model))
        status = self.library.green_v400_causal_attention_final_all_heads_jet2_exact(
            precision, sequence_length, n_heads, head_dim, pivot, strings,
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled all-head attention failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def fused_contrast_jet2(self, values: list[Jet2], fusion_payload: dict) -> dict:
        if not values or fusion_payload.get("d_model") != len(values):
            raise ValueError("compiled fused contrast shape mismatch")
        precision = values[0].precision_bits
        if any(value.precision_bits != precision for value in values):
            raise ValueError("compiled fused contrast precision mismatch")
        encoded = []
        for value in values:
            for component in (value.value, value.first, value.second):
                encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        weights = fusion_payload.get("weights", [])
        if len(weights) != len(values):
            raise ValueError("compiled fused contrast weight count mismatch")

        def hexadecimal_significand(item: dict) -> bytes:
            value = int(item["significand"])
            return (("-" if value < 0 else "") + format(abs(value), "x")).encode("ascii")

        weight_strings = (ctypes.c_char_p * len(weights))(
            *(hexadecimal_significand(item) for item in weights)
        )
        weight_exponents = np.asarray([item["exponent_2"] for item in weights], dtype="<i8")
        bias = fusion_payload["bias"]
        output = ctypes.create_string_buffer(8192)
        status = self.library.green_v400_fused_contrast_jet2_exact(
            precision, len(values), strings,
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), weight_strings,
            weight_exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            hexadecimal_significand(bias), int(bias["exponent_2"]), output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled fused contrast failed with status {status}")
        return json.loads(output.value.decode("ascii"))

    def final_contrast_jet2(self, values: list[Jet2], unembed, bias,
                            suffix_ids, coefficients) -> dict:
        if not values:
            raise ValueError("compiled final contrast requires residual values")
        precision = values[0].precision_bits
        if any(value.precision_bits != precision for value in values):
            raise ValueError("compiled final contrast precision mismatch")
        unembed = np.asarray(unembed, dtype="<f4", order="C")
        bias = np.asarray(bias, dtype="<f4").reshape(-1)
        suffix_ids = np.asarray(suffix_ids, dtype="<i8").reshape(-1)
        coefficients = np.asarray(coefficients, dtype="<f8").reshape(-1)
        if (unembed.ndim != 2 or unembed.shape[0] != len(values)
                or unembed.shape[1] != bias.size or suffix_ids.size == 0
                or suffix_ids.size != coefficients.size):
            raise ValueError("compiled final contrast shape mismatch")
        encoded = []
        for value in values:
            for component in (value.value, value.first, value.second):
                encoded.extend((_binary_endpoint(component.lower), _binary_endpoint(component.upper)))
        strings = (ctypes.c_char_p * len(encoded))(*(item[0] for item in encoded))
        exponents = np.asarray([item[1] for item in encoded], dtype="<i8")
        output = ctypes.create_string_buffer(8192)
        status = self.library.green_v400_final_contrast_jet2_exact(
            precision, unembed.shape[0], unembed.shape[1], suffix_ids.size,
            strings, exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            unembed.view("<u4").ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            bias.view("<u4").ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            suffix_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            coefficients.view("<u8").ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            output, len(output),
        )
        if status != 0:
            raise RuntimeError(f"compiled final contrast failed with status {status}")
        return json.loads(output.value.decode("ascii"))
