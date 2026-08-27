"""Exact outcome-blind TensorProgram replay with Python or compiled MPFR kernels."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
import time
from typing import Callable

import gmpy2
import numpy as np

from green_bridge_v400_compiled_mpfr import (
    CompiledMPFRBackend, CompiledResidentJetBuffer,
)
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import (
    Jet2, add_jet, affine_control_jet, constant_jet, sub_jet,
)
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_program import tensor_program_dispatch_signature
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_resident_plan import ValidatedResidentPlan
from green_bridge_v400_tensor_store import TensorStoreReader
from green_bridge_v400_transformer_ops import (
    affine_map_jets, attention_head_jets, gelu_new_jet, layernorm_jets,
)


@dataclass
class ResidentStaticRowCache:
    """Cross-cell cache closed to one program, packed plan, backend, and precision."""

    program_semantic_hash: str
    resident_plan_semantic_hash: str
    backend_sha256: str
    precision_bits: int
    _entries: dict[tuple[object, ...], list[Jet2]] = field(
        default_factory=dict, repr=False
    )
    _native_rows: dict[tuple[int, int], CompiledResidentJetBuffer] = field(
        default_factory=dict, repr=False
    )
    _native_flattened_rows: dict[
        tuple[int, tuple[int, ...]], CompiledResidentJetBuffer
    ] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, program: TensorProgram, resident_plan: dict,
              compiled_backend: CompiledMPFRBackend, precision_bits: int):
        if precision_bits <= 0:
            raise ValueError("resident static-row cache precision must be positive")
        if not isinstance(resident_plan, ValidatedResidentPlan):
            raise TypeError("resident static-row cache requires a validated resident plan")
        if resident_plan.get("program_semantic_hash") != program.semantic_hash():
            raise ValueError("resident static-row cache plan/program mismatch")
        plan_hash = resident_plan.get("resident_plan_semantic_hash")
        if not isinstance(plan_hash, str) or len(plan_hash) != 64:
            raise ValueError("resident static-row cache plan hash is invalid")
        return cls(
            program.semantic_hash(), plan_hash, compiled_backend.library_sha256,
            int(precision_bits),
        )

    def validate(self, program: TensorProgram, resident_plan: dict,
                 compiled_backend: CompiledMPFRBackend, precision_bits: int) -> None:
        expected = (
            program.semantic_hash(), resident_plan.get("resident_plan_semantic_hash"),
            compiled_backend.library_sha256, int(precision_bits),
        )
        actual = (
            self.program_semantic_hash, self.resident_plan_semantic_hash,
            self.backend_sha256, self.precision_bits,
        )
        if actual != expected:
            raise ValueError("resident static-row cache identity mismatch")

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def native_entry_count(self) -> int:
        return len(self._native_rows) + len(self._native_flattened_rows)

    def close(self) -> None:
        for buffer in reversed([
            *self._native_rows.values(), *self._native_flattened_rows.values()
        ]):
            buffer.close()
        self._native_rows.clear()
        self._native_flattened_rows.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _decode_interval(payload: dict, precision: int) -> Interval:
    def rational(endpoint: dict):
        significand = int(endpoint["significand_hex"], 16)
        if significand == 0:
            return gmpy2.mpq(0)
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


def tensor_program_required_axis0_rows(program: TensorProgram) -> dict[str, tuple[int, ...]]:
    """Exact backward row liveness for the scalar joint-witness output."""
    nodes = {node.semantic_id: node for node in program.nodes}
    active = {program.output_root}
    required: dict[str, set[int]] = {}

    def require(parent_id: str, rows: set[int] | None = None) -> None:
        parent = nodes[parent_id]
        active.add(parent_id)
        if parent.output_spec.shape:
            if rows is None:
                raise ValueError("tensor parent requires explicit live rows")
            if any(row < 0 or row >= parent.output_spec.shape[0] for row in rows):
                raise ValueError("live row is outside parent output shape")
            required.setdefault(parent_id, set()).update(rows)

    rowwise = {
        "layer_norm.v1", "pairwise_affine.v1", "gelu_new.v1",
        "residual_add.v1",
    }
    for node in reversed(program.nodes):
        if node.semantic_id not in active:
            continue
        kernel = node.kernel_id
        if kernel == "branch_linear_combination.v1":
            for parent_id in node.parent_semantic_ids:
                require(parent_id)
        elif kernel == "final_contrast.v1":
            require(node.parent_semantic_ids[0], {int(node.exact_attrs["final_position"])})
        elif kernel in rowwise:
            rows = required.get(node.semantic_id)
            if rows is None:
                raise ValueError("row-wise live node has no live rows")
            for parent_id in node.parent_semantic_ids:
                require(parent_id, set(rows))
        elif kernel == "static_view.v1":
            rows = required.get(node.semantic_id)
            if rows is None:
                raise ValueError("static-view live node has no live rows")
            operation = node.exact_attrs.get("operation")
            if operation == "tensor_constant":
                if node.parent_semantic_ids:
                    raise ValueError("tensor-constant static view cannot have parents")
            elif operation == "subtract_exact_parent_at_final_position":
                final_position = int(node.exact_attrs["final_position"])
                if final_position in rows:
                    for parent_id in node.parent_semantic_ids:
                        require(parent_id, {final_position})
            else:
                raise ValueError("unsupported live-row static-view operation")
        elif kernel == "causal_attention.v1":
            rows = required.get(node.semantic_id)
            if rows is None or len(node.parent_semantic_ids) != 3:
                raise ValueError("attention live-row closure is invalid")
            q_parent, k_parent, v_parent = node.parent_semantic_ids
            require(q_parent, set(rows))
            history = {history_row for row in rows for history_row in range(row + 1)}
            require(k_parent, history)
            require(v_parent, history)
        elif kernel != "affine_scatter.v1":
            raise ValueError(f"unsupported live-row kernel: {kernel}")
    if active != set(nodes):
        raise ValueError("TensorProgram contains nodes outside scalar-output liveness")
    return {semantic_id: tuple(sorted(rows)) for semantic_id, rows in required.items()}


def execute_tensor_program_mpfr(
    program: TensorProgram, reader: TensorStoreReader, domain: Interval,
    compiled_backend: CompiledMPFRBackend | Path | None = None,
    *, resident_plan: dict | None = None,
    resident_arrays: dict[str, np.ndarray] | None = None,
    resident_static_row_cache: ResidentStaticRowCache | None = None,
    sparse_axis0_execution: bool = False,
    resident_buffer_execution: bool = False,
    return_node_values: bool = False,
    return_dispatch_trace: bool = False,
    return_runtime_metrics: bool = False,
    successful_node_callback: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    """Replay all branch roots over one interval cell; never reads scientific labels/outcomes."""
    precision = domain.precision_bits
    if isinstance(compiled_backend, Path):
        compiled_backend = CompiledMPFRBackend(compiled_backend)
    values: dict[str, object] = {}
    dispatch_events = []
    tensor_cache: dict[str, np.ndarray] = {}
    resident_by_semantic: dict[str, np.ndarray] = {}
    resident_packed_binding_reads = 0
    tensor_store_fallback_reads = 0
    resident_fused_contrast_nodes = 0
    resident_gelu_batch_rows = 0
    resident_buffer_imports = 0
    resident_buffer_exports = 0
    resident_buffer_imported_jet_count = 0
    resident_buffer_exported_jet_count = 0
    resident_buffer_nodes: dict[str, int] = {}
    resident_native_buffers: list[CompiledResidentJetBuffer] = []
    resident_python_rows: dict[int, list[Jet2]] = {}
    resident_native_cache_hits = 0
    resident_native_cache_misses = 0
    if resident_arrays is not None:
        if resident_plan is None or compiled_backend is None:
            raise ValueError("resident arrays require a resident plan and compiled backend")
        if not isinstance(resident_plan, ValidatedResidentPlan):
            raise TypeError("resident execution requires a validated resident plan")
        resident_plan.validate_runtime(program, reader, resident_arrays)
        resident_by_semantic = {
            record["tensor_semantic_sha256"]: resident_arrays[record["name"]]
            for record in resident_plan["records"]
        }
    if resident_static_row_cache is not None:
        if (resident_plan is None or resident_arrays is None
                or compiled_backend is None or not sparse_axis0_execution):
            raise ValueError(
                "resident static-row cache requires sparse packed resident execution"
            )
        resident_static_row_cache.validate(
            program, resident_plan, compiled_backend, precision
        )
    if sparse_axis0_execution and (resident_arrays is None or return_node_values):
        raise ValueError("sparse row execution requires resident arrays and root-only output")
    if resident_buffer_execution and (
        not sparse_axis0_execution or resident_arrays is None
        or resident_plan is None or compiled_backend is None or return_node_values
    ):
        raise ValueError(
            "resident-buffer execution requires sparse packed resident root-only execution"
        )
    live_rows = tensor_program_required_axis0_rows(program) if sparse_axis0_execution else {}
    static_row_cache = (
        resident_static_row_cache._entries
        if resident_static_row_cache is not None else {}
    )
    static_row_cache_initial_entry_count = len(static_row_cache)
    static_python_row_ids = {id(row) for row in static_row_cache.values()}
    cache_hits: dict[str, int] = {}
    cache_misses: dict[str, int] = {}
    static_node_identities = {
        node.semantic_id: (
            node.kernel_id,
            sha256_canonical(node.exact_attrs),
            sha256_canonical([ref.tensor_sha256 for ref in node.tensor_inputs]),
        )
        for node in program.nodes
    }

    def track_native_buffer(
        buffer: CompiledResidentJetBuffer, kernel: str | None = None,
    ) -> CompiledResidentJetBuffer:
        resident_native_buffers.append(buffer)
        if kernel is not None:
            resident_buffer_nodes[kernel] = resident_buffer_nodes.get(kernel, 0) + 1
        return buffer

    def native_row(row) -> CompiledResidentJetBuffer:
        nonlocal resident_buffer_imports, resident_buffer_imported_jet_count
        nonlocal resident_native_cache_hits, resident_native_cache_misses
        if isinstance(row, CompiledResidentJetBuffer):
            return row
        cache_key = (id(compiled_backend), id(row))
        if resident_static_row_cache is not None and id(row) in static_python_row_ids:
            cached = resident_static_row_cache._native_rows.get(cache_key)
            if cached is not None:
                resident_native_cache_hits += 1
                return cached
            resident_native_cache_misses += 1
        resident_buffer_imports += 1
        resident_buffer_imported_jet_count += len(row)
        imported = compiled_backend.resident_jet_buffer(row)
        if resident_static_row_cache is not None and id(row) in static_python_row_ids:
            resident_static_row_cache._native_rows[cache_key] = imported
            return imported
        return track_native_buffer(imported)

    def python_row(row) -> list[Jet2]:
        nonlocal resident_buffer_exports, resident_buffer_exported_jet_count
        if not isinstance(row, CompiledResidentJetBuffer):
            return row
        identity = int(row.handle.value)
        if identity in resident_python_rows:
            return resident_python_rows[identity]
        resident_buffer_exports += 1
        resident_buffer_exported_jet_count += row.width
        decoded = [
            _decode_jet(item, precision) for item in
            compiled_backend.export_resident_jet_buffer(row)["outputs"]
        ]
        resident_python_rows[identity] = decoded
        return decoded

    def native_flatten_rows(rows) -> CompiledResidentJetBuffer:
        chunks: list[CompiledResidentJetBuffer] = []
        pending_rows: list[list[Jet2]] = []

        def flush_pending() -> None:
            nonlocal resident_buffer_imports, resident_buffer_imported_jet_count
            nonlocal resident_native_cache_hits, resident_native_cache_misses
            if not pending_rows:
                return
            identities = tuple(id(row) for row in pending_rows)
            cache_key = (id(compiled_backend), identities)
            cacheable = (
                resident_static_row_cache is not None
                and all(identity in static_python_row_ids for identity in identities)
            )
            if cacheable:
                cached = resident_static_row_cache._native_flattened_rows.get(cache_key)
                if cached is not None:
                    resident_native_cache_hits += 1
                    chunks.append(cached)
                    pending_rows.clear()
                    return
                resident_native_cache_misses += 1
            flattened = [jet for row in pending_rows for jet in row]
            resident_buffer_imports += 1
            resident_buffer_imported_jet_count += len(flattened)
            imported = compiled_backend.resident_jet_buffer(flattened)
            if cacheable:
                resident_static_row_cache._native_flattened_rows[cache_key] = imported
                chunks.append(imported)
            else:
                chunks.append(track_native_buffer(imported))
            pending_rows.clear()

        for row in rows:
            if isinstance(row, CompiledResidentJetBuffer):
                flush_pending()
                chunks.append(row)
            else:
                pending_rows.append(row)
        flush_pending()
        if len(chunks) == 1:
            return chunks[0]
        return track_native_buffer(compiled_backend.resident_concat_jet2(chunks))

    def exact_row_key(row: list[Jet2]) -> tuple[object, ...]:
        row = python_row(row)
        return tuple(
            endpoint
            for jet in row
            for component in (jet.value, jet.first, jet.second)
            for endpoint in (component.lower, component.upper)
        )

    def static_row_cache_key(node, row_index: int, row: list[Jet2]):
        if not sparse_axis0_execution:
            return None
        dynamic_rows = set(node.exact_attrs["dependency_mask_spec"]["axis0_indices"])
        if row_index in dynamic_rows:
            return None
        return static_node_identities[node.semantic_id] + (precision, exact_row_key(row))

    def cached_static_row(key, kernel: str):
        if key is not None and key in static_row_cache:
            cache_hits[kernel] = cache_hits.get(kernel, 0) + 1
            return static_row_cache[key]
        if key is not None:
            cache_misses[kernel] = cache_misses.get(kernel, 0) + 1
        return None

    def store_static_row(key, row: list[Jet2]) -> None:
        if key is not None:
            static_row_cache[key] = row
            static_python_row_ids.add(id(row))
    for ordinal, node in enumerate(program.nodes):
        node_started = time.perf_counter() if successful_node_callback is not None else None
        parents = [values[parent] for parent in node.parent_semantic_ids]
        kernel = node.kernel_id
        use_resident_fusion = (
            resident_plan is not None and compiled_backend is not None
            and kernel == "final_contrast.v1"
        )
        tensors = []
        if not use_resident_fusion:
            for ref in node.tensor_inputs:
                if ref.tensor_sha256 in resident_by_semantic:
                    tensor_cache[ref.tensor_sha256] = resident_by_semantic[ref.tensor_sha256]
                    resident_packed_binding_reads += 1
                elif ref.tensor_sha256 not in tensor_cache:
                    tensor_cache[ref.tensor_sha256] = reader.read_semantic(ref.tensor_sha256)
                    tensor_store_fallback_reads += 1
                tensors.append(tensor_cache[ref.tensor_sha256])
        shape = node.output_spec.shape
        row_indices = live_rows.get(node.semantic_id, tuple(range(shape[0]))) if shape else ()
        if kernel == "affine_scatter.v1":
            base, direction = tensors
            final_position = int(node.exact_attrs["final_position"])
            output = [None] * base.shape[0]
            for row_index in row_indices:
                output[row_index] = [
                    _point_jet(base[row_index, coordinate], precision)
                    for coordinate in range(base.shape[1])
                ]
            if final_position in row_indices:
                output[final_position] = [
                    affine_control_jet(
                        Interval.point(base[final_position, coordinate], precision),
                        Interval.point(direction[coordinate], precision), domain,
                    )
                    for coordinate in range(base.shape[1])
                ]
        elif kernel == "layer_norm.v1":
            source = parents[0]
            gamma, beta, epsilon = tensors
            output = [None] * len(source)
            for row_index in row_indices:
                row = source[row_index]
                cache_key = static_row_cache_key(node, row_index, row)
                if cache_key is not None:
                    row = python_row(row)
                cached = cached_static_row(cache_key, kernel)
                if cached is not None:
                    output[row_index] = cached
                    continue
                if resident_buffer_execution and cache_key is None:
                    output[row_index] = track_native_buffer(
                        compiled_backend.resident_layer_norm_jet2(
                            native_row(row), epsilon.reshape(()), gamma, beta,
                        ), kernel,
                    )
                elif compiled_backend is None:
                    output[row_index] = layernorm_jets(
                        row, epsilon=float(epsilon.reshape(())), gamma=gamma, beta=beta,
                    )
                else:
                    payload = compiled_backend.layer_norm_jet2(
                        row, epsilon.reshape(()), gamma, beta,
                    )
                    output[row_index] = [
                        _decode_jet(item, precision) for item in payload["outputs"]
                    ]
                store_static_row(cache_key, output[row_index])
        elif kernel == "pairwise_affine.v1":
            source = parents[0]
            weight, bias = tensors
            output = [None] * len(source)
            for row_index in row_indices:
                row = source[row_index]
                cache_key = static_row_cache_key(node, row_index, row)
                if cache_key is not None:
                    row = python_row(row)
                cached = cached_static_row(cache_key, kernel)
                if cached is not None:
                    output[row_index] = cached
                    continue
                if resident_buffer_execution and cache_key is None:
                    output[row_index] = track_native_buffer(
                        compiled_backend.resident_packed_affine_layer_jet2(
                            native_row(row), weight, bias,
                        ), kernel,
                    )
                elif compiled_backend is None:
                    output[row_index] = affine_map_jets(weight.T, row, bias)
                elif resident_arrays is not None:
                    output[row_index] = [
                        _decode_jet(item, precision) for item in
                        compiled_backend.packed_affine_layer_jet2(
                            weight, bias, row
                        )["outputs"]
                    ]
                else:
                    output[row_index] = [
                        _decode_jet(compiled_backend.affine_jet2(
                            weight[:, coordinate], bias[coordinate], row, precision,
                        ), precision)
                        for coordinate in range(weight.shape[1])
                    ]
                store_static_row(cache_key, output[row_index])
        elif kernel == "gelu_new.v1":
            source = parents[0]
            kappa, lam = (tensor.reshape(()) for tensor in tensors)
            output = [None] * len(source)
            for row_index in row_indices:
                row = source[row_index]
                cache_key = static_row_cache_key(node, row_index, row)
                if cache_key is not None:
                    row = python_row(row)
                cached = cached_static_row(cache_key, kernel)
                if cached is not None:
                    output[row_index] = cached
                    continue
                if resident_buffer_execution and cache_key is None:
                    output[row_index] = track_native_buffer(
                        compiled_backend.resident_gelu_new_layer_jet2(
                            native_row(row), kappa, lam,
                        ), kernel,
                    )
                elif compiled_backend is None:
                    output[row_index] = [
                        gelu_new_jet(jet, kappa=float(kappa), lam=float(lam)) for jet in row
                    ]
                elif resident_arrays is not None:
                    payload = compiled_backend.gelu_new_layer_jet2(row, kappa, lam)
                    output[row_index] = [
                        _decode_jet(item, precision) for item in payload["outputs"]
                    ]
                    resident_gelu_batch_rows += 1
                else:
                    output[row_index] = [_decode_jet(
                        compiled_backend.gelu_new_jet2(jet, kappa, lam), precision
                    ) for jet in row]
                store_static_row(cache_key, output[row_index])
        elif kernel == "static_view.v1":
            operation = node.exact_attrs.get("operation")
            if operation == "tensor_constant":
                output = [None] * tensors[0].shape[0]
                for row_index in row_indices:
                    output[row_index] = [
                        _point_jet(value, precision) for value in tensors[0][row_index]
                    ]
            elif operation == "subtract_exact_parent_at_final_position":
                final_position = int(node.exact_attrs["final_position"])
                output = [None] * shape[0]
                for row_index in row_indices:
                    if row_index == final_position:
                        if resident_buffer_execution:
                            output[row_index] = track_native_buffer(
                                compiled_backend.resident_sub_jet2(
                                    native_row(parents[0][row_index]),
                                    native_row(parents[1][row_index]),
                                ), kernel,
                            )
                        else:
                            output[row_index] = [
                                sub_jet(left, right) for left, right in zip(
                                    parents[0][row_index], parents[1][row_index]
                                )
                            ]
                    else:
                        output[row_index] = [_zero(precision) for _ in range(shape[1])]
            else:
                raise ValueError("unsupported MPFR static view operation")
        elif kernel == "causal_attention.v1":
            q, k, v = parents
            n_heads, d_head = int(node.exact_attrs["n_heads"]), int(node.exact_attrs["d_head"])
            sequence_length = len(q)
            final_position = node.exact_attrs["dependency_mask_spec"]["axis0_indices"][0]
            if sparse_axis0_execution:
                output = [None] * sequence_length
                if tuple(row_indices) != (final_position,):
                    raise ValueError("resident attention requires exactly the final causal row")
                output[final_position] = [_zero(precision) for _ in range(n_heads * d_head)]
            else:
                output = [[_zero(precision) for _ in range(n_heads * d_head)]
                          for _ in range(sequence_length)]
            pivot = int(node.exact_attrs["softmax_pivot"]["index"])
            if resident_buffer_execution:
                output[final_position] = track_native_buffer(
                    compiled_backend.resident_causal_attention_all_heads_jet2(
                        native_row(q[final_position]),
                        native_flatten_rows(k[:final_position + 1]),
                        native_flatten_rows(v[:final_position + 1]),
                        final_position + 1, n_heads, d_head, pivot,
                    ), kernel,
                )
            else:
                for head in range(n_heads):
                    start, stop = head * d_head, (head + 1) * d_head
                    query = q[final_position][start:stop]
                    keys = [row[start:stop] for row in k[:final_position + 1]]
                    vectors = [row[start:stop] for row in v[:final_position + 1]]
                    if compiled_backend is None:
                        attended = attention_head_jets(
                            [query] * (final_position + 1), keys, vectors, causal=True,
                        )[-1] if pivot == 0 else None
                        if attended is None:
                            raise ValueError(
                                "Python MPFR attention currently requires fixed pivot zero"
                            )
                    else:
                        attended = [_decode_jet(item, precision) for item in
                                    compiled_backend.causal_attention_final_head_jet2(
                                        query, keys, vectors, pivot=pivot,
                                    )["outputs"]]
                    output[final_position][start:stop] = attended
        elif kernel == "residual_add.v1":
            output = [None] * shape[0]
            for row_index in row_indices:
                if resident_buffer_execution:
                    output[row_index] = track_native_buffer(
                        compiled_backend.resident_add_jet2(
                            native_row(parents[0][row_index]),
                            native_row(parents[1][row_index]),
                        ), kernel,
                    )
                else:
                    output[row_index] = [
                        add_jet(left, right) for left, right in zip(
                            parents[0][row_index], parents[1][row_index]
                        )
                    ]
        elif kernel == "final_contrast.v1":
            final_position = int(node.exact_attrs["final_position"])
            row = parents[0][final_position]
            if use_resident_fusion:
                resident_fused_contrast_nodes += 1
                if (resident_plan.get("exact_final_contrast_fusion_sha256")
                        != program.resource_formula["exact_final_contrast_fusion_sha256"]):
                    raise ValueError("resident exact-fusion hash disagrees with TensorProgram")
                if resident_buffer_execution:
                    output = track_native_buffer(
                        compiled_backend.resident_fused_contrast_jet2(
                            native_row(row), resident_plan["exact_final_contrast_fusion"]
                        ), kernel,
                    )
                else:
                    output = _decode_jet(compiled_backend.fused_contrast_jet2(
                        row, resident_plan["exact_final_contrast_fusion"]
                    ), precision)
            elif compiled_backend is None:
                row = python_row(row)
                unembed, bias, suffix_ids, coefficients = tensors
                output = _final_contrast_reference(
                    row, unembed, bias, suffix_ids.astype(np.int64), coefficients,
                )
            else:
                row = python_row(row)
                unembed, bias, suffix_ids, coefficients = tensors
                output = _decode_jet(compiled_backend.final_contrast_jet2(
                    row, unembed, bias, suffix_ids, coefficients,
                ), precision)
        elif kernel == "branch_linear_combination.v1":
            if resident_buffer_execution:
                first = track_native_buffer(compiled_backend.resident_sub_jet2(
                    native_row(parents[0]), native_row(parents[1])
                ))
                second = track_native_buffer(compiled_backend.resident_sub_jet2(
                    first, native_row(parents[2])
                ))
                output = track_native_buffer(compiled_backend.resident_add_jet2(
                    second, native_row(parents[3])
                ), kernel)
            else:
                output = add_jet(
                    sub_jet(sub_jet(parents[0], parents[1]), parents[2]), parents[3]
                )
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
        if successful_node_callback is not None:
            successful_node_callback({
                "ordinal": ordinal,
                "semantic_id": node.semantic_id,
                "kernel_id": node.kernel_id,
                "elapsed_seconds": time.perf_counter() - node_started,
            })
    def scalar_value(value) -> Jet2:
        if isinstance(value, CompiledResidentJetBuffer):
            if value.width != 1:
                raise RuntimeError("resident scalar root has non-scalar width")
            return python_row(value)[0]
        return value

    result = {
        **{name: scalar_value(values[root])
           for name, root in program.branch_roots.items()},
        "output": scalar_value(values[program.output_root]),
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
    if return_runtime_metrics:
        result["runtime_metrics"] = {
            "schema_version": "green-v400-resident-runtime-metrics-v1",
            "sparse_axis0_execution": sparse_axis0_execution,
            "dense_axis0_row_slot_count": sum(
                node.output_spec.shape[0] for node in program.nodes if node.output_spec.shape
            ),
            "materialized_axis0_row_count": (
                sum(map(len, live_rows.values())) if sparse_axis0_execution else
                sum(node.output_spec.shape[0] for node in program.nodes if node.output_spec.shape)
            ),
            "static_row_cache_hits_by_kernel": cache_hits,
            "static_row_cache_misses_by_kernel": cache_misses,
            "static_row_cache_entry_count": len(static_row_cache),
            "static_row_cache_initial_entry_count": static_row_cache_initial_entry_count,
            "resident_static_row_cache_enabled": resident_static_row_cache is not None,
            "resident_packed_tensor_binding_reads": resident_packed_binding_reads,
            "tensor_store_fallback_reads": tensor_store_fallback_reads,
            "resident_fused_contrast_nodes": resident_fused_contrast_nodes,
            "resident_gelu_batch_rows": resident_gelu_batch_rows,
            "resident_buffer_execution": resident_buffer_execution,
            "resident_buffer_imports": resident_buffer_imports,
            "resident_buffer_exports": resident_buffer_exports,
            "resident_buffer_imported_jet_count": resident_buffer_imported_jet_count,
            "resident_buffer_exported_jet_count": resident_buffer_exported_jet_count,
            "resident_buffer_nodes_by_kernel": resident_buffer_nodes,
            "resident_native_static_cache_hits": resident_native_cache_hits,
            "resident_native_static_cache_misses": resident_native_cache_misses,
            "resident_native_static_cache_entry_count": (
                resident_static_row_cache.native_entry_count
                if resident_static_row_cache is not None else 0
            ),
        }
    for buffer in reversed(resident_native_buffers):
        buffer.close()
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
