from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import hashlib
import os
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from green_bridge_v400_native_descriptor import (
    DESCRIPTOR_HEADER, DESCRIPTOR_HEADER_SIZE, _header_for, build_native_execution_descriptor,
    decode_canonical_binary, descriptor_payload, encode_canonical_binary,
    load_native_execution_descriptor, program_execution_identity,
)
from green_bridge_v400_compiled_mpfr import (
    CompiledMPFRBackend, CompiledNativeJointWitnessEvaluator,
)
from green_bridge_v400_certificate import certify_adaptive_cells
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr_tensor_executor import (
    execute_tensor_program_mpfr, jet_exact_payload,
)
from green_bridge_v400_resident_plan import (
    build_resident_plan, load_resident_plan_arrays,
)
from green_bridge_v400_tensor_program import NATIVE_DISPATCH_KERNEL_TAGS, TensorProgram
from green_bridge_v400_schemas import CertificatePlan, Dyadic
from test_green_bridge_v400_gpt2_program import _fixture


def _descriptor_fixture(tmp_path):
    fixture = tmp_path / "fixture"; fixture.mkdir()
    reader, _, program = _fixture(fixture)
    resident = tmp_path / "resident"; resident.mkdir()
    build_resident_plan(resident, "tiny", program, reader)
    plan, _ = load_resident_plan_arrays(resident / "tiny.json", program, reader)
    descriptor = tmp_path / "tiny.desc"
    built = build_native_execution_descriptor(descriptor, program, plan)
    return reader, program, plan, descriptor, built


def test_canonical_binary_codec_round_trip_and_rejects_noncanonical_values():
    value = {
        "z": [None, False, True, 0, -17, 2**130],
        "a": {"unicode": "Ψ"},
    }
    encoded = encode_canonical_binary(value)
    assert decode_canonical_binary(encoded) == value
    with pytest.raises(TypeError, match="unsupported type"):
        encode_canonical_binary(0.5)
    with pytest.raises(ValueError, match="trailing"):
        decode_canonical_binary(encoded + b"N")
    with pytest.raises(ValueError, match="redundant"):
        decode_canonical_binary(b"I\x00\x01\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="keys are not canonical"):
        decode_canonical_binary(b"D\x02\x00\x00\x00S\x01\x00\x00\x00zNS\x01\x00\x00\x00aN")
    with pytest.raises(ValueError, match="UTF-8"):
        decode_canonical_binary(b"S\x01\x00\x00\x00\xff")
    nested = None
    for _ in range(130):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        encode_canonical_binary(nested)


def test_native_descriptor_closes_program_plan_blob_bindings_and_liveness(tmp_path):
    _, program, plan, descriptor, built = _descriptor_fixture(tmp_path)
    replay = load_native_execution_descriptor(descriptor, program, plan)
    assert replay == built
    payload = replay["payload"]
    assert payload["contains_scientific_outcome"] is False
    assert payload["native_execution_ready"] is False
    assert payload["claim_status"] == "PASS_NATIVE_DESCRIPTOR_PREPARE_ONLY"
    assert len(payload["records"]) == 32
    assert len(payload["program_execution_identity"]["nodes"]) == 81
    assert len(payload["program_input_binding_table"]) == 150
    assert len(payload["required_axis0_rows"]) == 81
    assert payload["blob_sha256"] == plan["blob_sha256"]
    assert len(replay["descriptor_file_sha256"]) == 64
    with pytest.raises(FileExistsError, match="immutable"):
        build_native_execution_descriptor(descriptor, program, plan)


def test_execution_identity_allows_one_way_resource_rebind_without_hash_cycle(tmp_path):
    reader, program, source_plan, descriptor, built = _descriptor_fixture(tmp_path)
    rebound_payload = program.to_dict()
    rebound_payload["resource_formula"] = dict(rebound_payload["resource_formula"])
    rebound_payload["resource_formula"]["native_execution_descriptor"] = {
        "descriptor_file_sha256": built["descriptor_file_sha256"]
    }
    rebound = TensorProgram.from_dict(rebound_payload)
    assert rebound.semantic_hash() != program.semantic_hash()
    assert program_execution_identity(rebound) == program_execution_identity(program)
    assert "source_program_semantic_hash" not in built["payload"]
    rebound_root = tmp_path / "rebound"; rebound_root.mkdir()
    build_resident_plan(rebound_root, "tiny", rebound, reader)
    rebound_plan, _ = load_resident_plan_arrays(
        rebound_root / "tiny.json", rebound, reader
    )
    assert rebound_plan["blob_sha256"] == source_plan["blob_sha256"]
    assert load_native_execution_descriptor(
        descriptor, rebound, rebound_plan
    )["descriptor_file_sha256"] == built["descriptor_file_sha256"]


@pytest.mark.parametrize("mutation", ["truncate", "trailing", "payload", "reserved"])
def test_native_descriptor_rejects_file_corruption(tmp_path, mutation):
    _, program, plan, descriptor, _ = _descriptor_fixture(tmp_path)
    raw = bytearray(descriptor.read_bytes())
    if mutation == "truncate":
        raw = raw[:-1]
    elif mutation == "trailing":
        raw.extend(b"\0")
    elif mutation == "payload":
        raw[-1] ^= 1
    else:
        raw[DESCRIPTOR_HEADER_SIZE - 1] = 1
    corrupted = tmp_path / f"{mutation}.desc"
    corrupted.write_bytes(raw)
    with pytest.raises(ValueError, match="descriptor"):
        load_native_execution_descriptor(corrupted, program, plan)


@pytest.mark.parametrize("header_index", [1, 2, 3, 4, 5, 12, 13, 14, 16])
def test_native_descriptor_rejects_each_header_policy_field(tmp_path, header_index):
    _, program, plan, descriptor, _ = _descriptor_fixture(tmp_path)
    raw = bytearray(descriptor.read_bytes())
    fields = list(DESCRIPTOR_HEADER.unpack(raw[:DESCRIPTOR_HEADER_SIZE]))
    if header_index == 16:
        fields[header_index] = b"\x01" + fields[header_index][1:]
    else:
        fields[header_index] += 1
    raw[:DESCRIPTOR_HEADER_SIZE] = DESCRIPTOR_HEADER.pack(*fields)
    corrupted = tmp_path / f"header-{header_index}.desc"
    corrupted.write_bytes(raw)
    with pytest.raises(ValueError, match="descriptor"):
        load_native_execution_descriptor(corrupted, program, plan)


@pytest.mark.parametrize(
    "substitution", ["claim", "binding", "record_offset", "fusion_weight"],
)
def test_native_descriptor_rejects_semantic_substitution_with_valid_payload_hash(
    tmp_path, substitution,
):
    _, program, plan, _, _ = _descriptor_fixture(tmp_path)
    payload = descriptor_payload(program, plan)
    if substitution == "claim":
        payload["claim_status"] = "SUBSTITUTED"
    elif substitution == "binding":
        payload["program_input_binding_table"] = payload[
            "program_input_binding_table"
        ][1:]
    elif substitution == "record_offset":
        payload["records"][0]["offset"] += 64
    else:
        payload["exact_final_contrast_fusion"]["weights"][0]["exponent_2"] += 1
    encoded = encode_canonical_binary(payload)
    substituted = tmp_path / "substituted.desc"
    substituted.write_bytes(_header_for(payload, encoded) + encoded)
    with pytest.raises(ValueError, match="descriptor"):
        load_native_execution_descriptor(substituted, program, plan)


def test_compiled_native_envelope_loader_is_hash_closed_and_generation_safe(tmp_path):
    library = os.environ.get("GREEN_V400_MPFR_BACKEND")
    if not library:
        pytest.skip("compiled MPFR backend is not configured")
    reader, program, plan, descriptor, built = _descriptor_fixture(tmp_path)
    backend = CompiledMPFRBackend(Path(library))
    payload = built["payload"]
    _, resident_arrays = load_resident_plan_arrays(
        plan.manifest_path, program, reader
    )
    envelope = backend.open_native_plan_envelope(
        descriptor, plan.blob_path,
        descriptor_sha256=built["descriptor_file_sha256"],
        program_execution_sha256=payload["program_execution_semantic_hash"],
        dispatch_sha256=payload["program_dispatch_signature_sha256"],
        blob_sha256=payload["blob_sha256"],
        fusion_sha256=payload["exact_final_contrast_fusion_sha256"],
        blob_nbytes=payload["blob_nbytes"],
        fusion_weight_count=len(payload["exact_final_contrast_fusion"]["weights"]),
    )
    assert envelope.info == {
        "descriptor_nbytes": descriptor.stat().st_size,
        "blob_nbytes": payload["blob_nbytes"], "record_count": 32,
        "node_count": 81, "binding_count": 150,
        "fusion_weight_count": len(payload["exact_final_contrast_fusion"]["weights"]),
        "payload_tables_validated": True,
        "typed_plan_materialized": True,
        "liveness_row_count": sum(
            len(item["rows"]) for item in payload["required_axis0_rows"]
        ),
        "branch_root_count": 4,
    }
    trace = backend.native_plan_typed_trace(envelope)
    node_ids = [node.semantic_id for node in program.nodes]
    assert trace == {
        "kernel_tags": [NATIVE_DISPATCH_KERNEL_TAGS[node.kernel_id]
                        for node in program.nodes],
        "liveness_counts": [len(item["rows"])
                            for item in payload["required_axis0_rows"]],
        "branch_root_indices": [node_ids.index(payload["branch_roots"][name])
                                for name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B")],
        "output_root_index": node_ids.index(payload["output_root"]),
    }
    contexts = [backend.open_native_precision_context(envelope, precision)
                for precision in (384, 512)]
    expected_static_jets = payload["dimensions"]["d_model"] * (
        1 + 4 * payload["dimensions"]["sequence_length"]
    )
    for precision, context in zip((384, 512), contexts):
        assert context.info == {
            "precision_bits": precision, "static_buffer_count": 5,
            "static_jet_count": expected_static_jets,
            "node_count": 81, "binding_count": 150, "plan_retained": True,
        }
        historical_rows = payload["dimensions"]["final_position"]
        assert backend.native_precision_context_projection_info(context) == {
            "projection_buffer_count": 4 * historical_rows * 2,
            "projection_jet_count": (
                4 * historical_rows * 2 * payload["dimensions"]["d_model"]
            ),
            "historical_row_count": historical_rows, "branch_count": 4,
        }
        buffers = []
        try:
            buffers.append(backend.resident_f32_constant_buffer(
                reader.read("PAT.resid_mid")[0], precision
            ))
            buffers.append(backend.resident_layer_norm_jet2(
                buffers[-1], reader.read("layer_norm.eps").reshape(()),
                reader.read("block11.ln1.w"), reader.read("block11.ln1.b"),
            ))
            buffers.append(backend.resident_packed_affine_layer_jet2(
                buffers[-1], reader.read("block11.attn.W_K"),
                reader.read("block11.attn.b_K"),
            ))
            buffers.append(backend.resident_packed_affine_layer_jet2(
                buffers[-2], reader.read("block11.attn.W_V"),
                reader.read("block11.attn.b_V"),
            ))
            assert backend.export_native_precision_context_projection(
                context, 0
            ) == backend.export_resident_jet_buffer(buffers[-2])
            assert backend.export_native_precision_context_projection(
                context, 1
            ) == backend.export_resident_jet_buffer(buffers[-1])
        finally:
            for buffer in reversed(buffers):
                buffer.close()
        domain = Interval.from_bounds(-(2.0 ** -14), 2.0 ** -14, precision)
        native = backend.dispatch_native_precision_context_cell(context, domain)
        reference = execute_tensor_program_mpfr(
            program, reader, domain, backend, resident_plan=plan,
            resident_arrays=resident_arrays, sparse_axis0_execution=True,
            resident_buffer_execution=True,
        )
        assert native["event_count"] == 81
        assert native["kernel_tags"] == [
            NATIVE_DISPATCH_KERNEL_TAGS[node.kernel_id] for node in program.nodes
        ]
        for root_name in ("PAT_J", "PAT_B", "TAR_J", "TAR_B", "output"):
            actual = {
                component: {
                    endpoint: list(backend.exact_fraction(
                        native[root_name][component][endpoint]
                    ).as_integer_ratio())
                    for endpoint in ("lower", "upper")
                }
                for component in ("value", "first", "second")
            }
            assert actual == jet_exact_payload(reference[root_name])
    row_hash = "a" * 64
    evaluator = CompiledNativeJointWitnessEvaluator(
        backend, {384: contexts[0], 512: contexts[1]},
        certificate_row_hash=row_hash,
        expected_kernel_tags=tuple(trace["kernel_tags"]),
    )
    certificate_plan = CertificatePlan(
        "green-v400-certificate-plan-v1", row_hash, (Dyadic(1, -14),),
        "[-h,0],[0,h]", "curvature-weighted width priority dyadic bisection",
        "0x1p+100", "0x1p+100", 0, 2, 384, 512, (), False,
    )
    cells = certify_adaptive_cells(
        evaluator, Fraction(1, 2**14), 384, certificate_plan,
    )
    assert cells is not None and len(cells) == 2
    assert evaluator.dispatch_count_by_precision == {384: 2, 512: 0}
    with pytest.raises(RuntimeError, match="status 2"):
        backend.open_native_precision_context(envelope, 256)
    stale = envelope.handle
    envelope.close()
    assert backend.library.green_v400_native_plan_envelope_info_v1(
        stale, None, None, None, None, None, None
    ) == 2
    stale_contexts = [context.handle for context in contexts]
    assert all(backend.library.green_v400_native_precision_context_info_v1(
        handle, None, None, None, None, None
    ) == 0 for handle in stale_contexts)
    for context in contexts:
        context.close()
    assert all(backend.library.green_v400_native_precision_context_info_v1(
        handle, None, None, None, None, None
    ) == 2 for handle in stale_contexts)
    corrupted = tmp_path / "corrupted.desc"
    raw = bytearray(descriptor.read_bytes()); raw[-1] ^= 1
    corrupted.write_bytes(raw)
    with pytest.raises(RuntimeError, match="status 5"):
        backend.open_native_plan_envelope(
            corrupted, plan.blob_path,
            descriptor_sha256=built["descriptor_file_sha256"],
            program_execution_sha256=payload["program_execution_semantic_hash"],
            dispatch_sha256=payload["program_dispatch_signature_sha256"],
            blob_sha256=payload["blob_sha256"],
            fusion_sha256=payload["exact_final_contrast_fusion_sha256"],
            blob_nbytes=payload["blob_nbytes"],
            fusion_weight_count=len(payload["exact_final_contrast_fusion"]["weights"]),
        )
    corrupted_blob = tmp_path / "corrupted.bin"
    blob_raw = bytearray(plan.blob_path.read_bytes()); blob_raw[-1] ^= 1
    corrupted_blob.write_bytes(blob_raw)
    with pytest.raises(RuntimeError, match="status 8"):
        backend.open_native_plan_envelope(
            descriptor, corrupted_blob,
            descriptor_sha256=built["descriptor_file_sha256"],
            program_execution_sha256=payload["program_execution_semantic_hash"],
            dispatch_sha256=payload["program_dispatch_signature_sha256"],
            blob_sha256=payload["blob_sha256"],
            fusion_sha256=payload["exact_final_contrast_fusion_sha256"],
            blob_nbytes=payload["blob_nbytes"],
            fusion_weight_count=len(payload["exact_final_contrast_fusion"]["weights"]),
        )
    substituted_payload = descriptor_payload(program, plan)
    substituted_payload["records"][0]["offset"] += 64
    substituted_encoded = encode_canonical_binary(substituted_payload)
    substituted_descriptor = tmp_path / "native-semantic-substitution.desc"
    substituted_descriptor.write_bytes(
        _header_for(substituted_payload, substituted_encoded) + substituted_encoded
    )
    substituted_sha = hashlib.sha256(substituted_descriptor.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="status 11"):
        backend.open_native_plan_envelope(
            substituted_descriptor, plan.blob_path,
            descriptor_sha256=substituted_sha,
            program_execution_sha256=payload["program_execution_semantic_hash"],
            dispatch_sha256=payload["program_dispatch_signature_sha256"],
            blob_sha256=payload["blob_sha256"],
            fusion_sha256=payload["exact_final_contrast_fusion_sha256"],
            blob_nbytes=payload["blob_nbytes"],
            fusion_weight_count=len(payload["exact_final_contrast_fusion"]["weights"]),
        )
    invalid_utf8_raw = bytearray(descriptor.read_bytes())
    invalid_utf8_raw[invalid_utf8_raw.index(b"PASS_NATIVE_DESCRIPTOR_PREPARE_ONLY")] = 0xff
    invalid_fields = list(DESCRIPTOR_HEADER.unpack(
        invalid_utf8_raw[:DESCRIPTOR_HEADER_SIZE]
    ))
    invalid_fields[7] = hashlib.sha256(
        invalid_utf8_raw[DESCRIPTOR_HEADER_SIZE:]
    ).digest()
    invalid_utf8_raw[:DESCRIPTOR_HEADER_SIZE] = DESCRIPTOR_HEADER.pack(*invalid_fields)
    invalid_utf8_descriptor = tmp_path / "native-invalid-utf8.desc"
    invalid_utf8_descriptor.write_bytes(invalid_utf8_raw)
    with pytest.raises(RuntimeError, match="status 11"):
        backend.open_native_plan_envelope(
            invalid_utf8_descriptor, plan.blob_path,
            descriptor_sha256=hashlib.sha256(invalid_utf8_raw).hexdigest(),
            program_execution_sha256=payload["program_execution_semantic_hash"],
            dispatch_sha256=payload["program_dispatch_signature_sha256"],
            blob_sha256=payload["blob_sha256"],
            fusion_sha256=payload["exact_final_contrast_fusion_sha256"],
            blob_nbytes=payload["blob_nbytes"],
            fusion_weight_count=len(payload["exact_final_contrast_fusion"]["weights"]),
        )
