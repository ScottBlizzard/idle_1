"""Exact outcome-blind TensorProgram replay with Python or compiled MPFR kernels."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import gmpy2
import numpy as np

from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import (
    Jet2, add_jet, affine_control_jet, constant_jet, sub_jet,
)
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_program import tensor_program_dispatch_signature
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_store import TensorStoreReader
from green_bridge_v400_transformer_ops import (
    affine_map_jets, attention_head_jets, gelu_new_jet, layernorm_jets,
)


def _decode_interval(payload: dict, precision: int) -> Interval:
    def rational(endpoint: dict):
        significand = int(endpoint["significand_hex"], 16)
        exponent = int(endpoint["exponent_2"])
        return (gmpy2.mpq(significand) * (gmpy2.mpq(2) ** exponent))
    return Interval.from_bounds(rational(payload["lower"]), rational(payload["upper"]), precision)


def _decode_jet(payload: dict, precision: int) -> Jet2:
    return Jet2(*(_decode_interval(payload[name], precision)
                  for name in ("value", "first", "second")))


def _point_jet(value, precision: int) -> Jet2:
    return constant_jet(Interval.point(value, precision))


def _zero(precision: int) -> Jet2:
    return _point_jet(0, precision)


def _exact_rational(value) -> gmpy2.mpq:
    fraction = Fraction.from_float(float(value))
    return gmpy2.mpq(fraction.numerator, fraction.denominator)


def _final_contrast_reference(values: list[Jet2], unembed: np.ndarray,
                              bias: np.ndarray, suffix_ids: np.ndarray,
                              coefficients: np.ndarray) -> Jet2:
    weights = [
        sum((_exact_rational(coefficients[index]) * _exact_rational(unembed[coordinate, token])
             for index, token in enumerate(suffix_ids)), gmpy2.mpq(0))
        for coordinate in range(unembed.shape[0])
    ]
    fused_bias = sum(
        (_exact_rational(coefficients[index]) * _exact_rational(bias[token])
         for index, token in enumerate(suffix_ids)), gmpy2.mpq(0)
    )
    return affine_map_jets([weights], values, [fused_bias])[0]


def execute_tensor_program_mpfr(
    program: TensorProgram, reader: TensorStoreReader, domain: Interval,
    compiled_backend: CompiledMPFRBackend | Path | None = None,
    *, return_node_values: bool = False,
    return_dispatch_trace: bool = False,
) -> dict[str, object]:
    """Replay all branch roots over one interval cell; never reads scientific labels/outcomes."""
    precision = domain.precision_bits
    if isinstance(compiled_backend, Path):
        compiled_backend = CompiledMPFRBackend(compiled_backend)
    values: dict[str, object] = {}
    dispatch_events = []
    tensor_cache: dict[str, np.ndarray] = {}
    for ordinal, node in enumerate(program.nodes):
        parents = [values[parent] for parent in node.parent_semantic_ids]
        tensors = []
        for ref in node.tensor_inputs:
            if ref.tensor_sha256 not in tensor_cache:
                tensor_cache[ref.tensor_sha256] = reader.read_semantic(ref.tensor_sha256)
            tensors.append(tensor_cache[ref.tensor_sha256])
        kernel = node.kernel_id
        shape = node.output_spec.shape
        if kernel == "affine_scatter.v1":
            base, direction = tensors
            final_position = int(node.exact_attrs["final_position"])
            output = [[_point_jet(base[row, coordinate], precision)
                       for coordinate in range(base.shape[1])]
                      for row in range(base.shape[0])]
            output[final_position] = [
                affine_control_jet(Interval.point(base[final_position, coordinate], precision),
                                   Interval.point(direction[coordinate], precision), domain)
                for coordinate in range(base.shape[1])
            ]
        elif kernel == "layer_norm.v1":
            source = parents[0]
            gamma, beta, epsilon = tensors
            output = []
            for row in source:
                if compiled_backend is None:
                    output.append(layernorm_jets(
                        row, epsilon=float(epsilon.reshape(())), gamma=gamma, beta=beta,
                    ))
                else:
                    payload = compiled_backend.layer_norm_jet2(
                        row, epsilon.reshape(()), gamma, beta,
                    )
                    output.append([_decode_jet(item, precision) for item in payload["outputs"]])
        elif kernel == "pairwise_affine.v1":
            source = parents[0]
            weight, bias = tensors
            output = []
            for row in source:
                if compiled_backend is None:
                    output.append(affine_map_jets(weight.T, row, bias))
                else:
                    output.append([
                        _decode_jet(compiled_backend.affine_jet2(
                            weight[:, coordinate], bias[coordinate], row, precision,
                        ), precision)
                        for coordinate in range(weight.shape[1])
                    ])
        elif kernel == "gelu_new.v1":
            source = parents[0]
            kappa, lam = (tensor.reshape(()) for tensor in tensors)
            output = []
            for row in source:
                if compiled_backend is None:
                    output.append([gelu_new_jet(jet, kappa=float(kappa), lam=float(lam))
                                   for jet in row])
                else:
                    output.append([_decode_jet(
                        compiled_backend.gelu_new_jet2(jet, kappa, lam), precision
                    ) for jet in row])
        elif kernel == "static_view.v1":
            operation = node.exact_attrs.get("operation")
            if operation == "tensor_constant":
                output = [[_point_jet(value, precision) for value in row] for row in tensors[0]]
            elif operation == "subtract_exact_parent_at_final_position":
                final_position = int(node.exact_attrs["final_position"])
                output = [[_zero(precision) for _ in row] for row in parents[0]]
                output[final_position] = [
                    sub_jet(left, right)
                    for left, right in zip(parents[0][final_position], parents[1][final_position])
                ]
            else:
                raise ValueError("unsupported MPFR static view operation")
        elif kernel == "causal_attention.v1":
            q, k, v = parents
            n_heads, d_head = int(node.exact_attrs["n_heads"]), int(node.exact_attrs["d_head"])
            sequence_length = len(q)
            final_position = node.exact_attrs["dependency_mask_spec"]["axis0_indices"][0]
            output = [[_zero(precision) for _ in range(n_heads * d_head)]
                      for _ in range(sequence_length)]
            for head in range(n_heads):
                start, stop = head * d_head, (head + 1) * d_head
                query = q[final_position][start:stop]
                keys = [row[start:stop] for row in k[:final_position + 1]]
                vectors = [row[start:stop] for row in v[:final_position + 1]]
                if compiled_backend is None:
                    pivot = int(node.exact_attrs["softmax_pivot"]["index"])
                    attended = attention_head_jets(
                        [query] * (final_position + 1), keys, vectors, causal=True,
                    )[-1] if pivot == 0 else None
                    if attended is None:
                        raise ValueError("Python MPFR attention currently requires fixed pivot zero")
                else:
                    pivot = int(node.exact_attrs["softmax_pivot"]["index"])
                    attended = [_decode_jet(item, precision) for item in
                                compiled_backend.causal_attention_final_head_jet2(
                                    query, keys, vectors, pivot=pivot,
                                )["outputs"]]
                output[final_position][start:stop] = attended
        elif kernel == "residual_add.v1":
            output = [[add_jet(left, right) for left, right in zip(left_row, right_row)]
                      for left_row, right_row in zip(parents[0], parents[1])]
        elif kernel == "final_contrast.v1":
            unembed, bias, suffix_ids, coefficients = tensors
            final_position = int(node.exact_attrs["final_position"])
            row = parents[0][final_position]
            if compiled_backend is None:
                output = _final_contrast_reference(
                    row, unembed, bias, suffix_ids.astype(np.int64), coefficients,
                )
            else:
                output = _decode_jet(compiled_backend.final_contrast_jet2(
                    row, unembed, bias, suffix_ids, coefficients,
                ), precision)
        elif kernel == "branch_linear_combination.v1":
            output = add_jet(sub_jet(sub_jet(parents[0], parents[1]), parents[2]), parents[3])
        else:
            raise RuntimeError(f"unsupported MPFR TensorProgram kernel: {kernel}")
        values[node.semantic_id] = output
        dispatch_events.append({
            "ordinal": ordinal,
            "semantic_id": node.semantic_id,
            "kernel_id": node.kernel_id,
            "output_spec": node.output_spec.to_dict(),
            "dependency_mask_hash": node.dependency_mask_hash,
        })
    result = {
        **{name: values[root] for name, root in program.branch_roots.items()},
        "output": values[program.output_root],
    }
    if return_node_values:
        result["node_values"] = values
    if return_dispatch_trace:
        expected = tensor_program_dispatch_signature(program.nodes)
        if dispatch_events != expected["ordered_nodes"]:
            raise RuntimeError("successful dispatcher events disagree with TensorProgram signature")
        payload = {
            "schema_version": "green-v400-successful-mpfr-dispatch-trace-v1",
            "events": dispatch_events,
        }
        result["dispatch_trace"] = {
            **payload,
            "trace_sha256": sha256_canonical(payload),
            "program_dispatch_signature_sha256": sha256_canonical(expected),
        }
    return result


def jet_exact_payload(jet: Jet2) -> dict:
    """Compact exact comparison payload used by synthetic parity audits."""
    def encode(value):
        rational = gmpy2.mpq(value)
        return [int(rational.numerator), int(rational.denominator)]
    return {
        component: {endpoint: encode(getattr(getattr(jet, component), endpoint))
                    for endpoint in ("lower", "upper")}
        for component in ("value", "first", "second")
    }
