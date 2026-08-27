"""ctypes bridge and bit-identity checks for the compiled MPFR backend."""
from __future__ import annotations

import ctypes
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import gmpy2

from green_bridge_v400_interval_jet import Jet2
from green_bridge_v400_interval import Interval


def _bits_f32(value) -> int:
    return int(np.asarray(value, dtype="<f4").view("<u4").reshape(()))


def _bits_f64_array(values) -> np.ndarray:
    return np.asarray(values, dtype="<f8").view("<u8")


def _exact_fraction(payload: dict) -> Fraction:
    raw = str(payload["significand_hex"])
    negative = raw.startswith("-")
    digits = raw[1:] if negative else raw
    significand = int(digits, 16) * (-1 if negative else 1)
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


class CompiledMPFRBackend:
    def __init__(self, library_path: Path):
        self.library_path = Path(library_path).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(self.library_path)
        self.library = ctypes.CDLL(str(self.library_path))
        self.library.green_v400_mpfr_backend_version.restype = ctypes.c_char_p
        self.version = self.library.green_v400_mpfr_backend_version().decode("ascii")
        if self.version != "green-v400-compiled-mpfr-v1":
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
