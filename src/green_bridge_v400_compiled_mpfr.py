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
        benchmark = self.library.green_v400_benchmark_affine_jet2_layer
        benchmark.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
        ]
        benchmark.restype = ctypes.c_int
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
        layer_norm = self.library.green_v400_layer_norm_jet2_exact
        layer_norm.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p, ctypes.c_uint64,
        ]
        layer_norm.restype = ctypes.c_int

    def affine_jet2(self, weights, bias, values: list[Jet2], precision_bits: int) -> dict:
        weights = np.asarray(weights, dtype="<f4").reshape(-1)
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
