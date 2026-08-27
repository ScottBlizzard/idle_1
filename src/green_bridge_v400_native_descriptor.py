"""Canonical binary execution descriptor for the GREEN v4 packed native plan."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import struct

from green_bridge_v400_mpfr_tensor_executor import tensor_program_required_axis0_rows
from green_bridge_v400_resident_plan import ValidatedResidentPlan
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import TensorProgram


DESCRIPTOR_MAGIC = b"GREENV400DESC\0\0\0"
DESCRIPTOR_FORMAT_VERSION = 1
DESCRIPTOR_HEADER_SIZE = 256
DESCRIPTOR_ALIGNMENT = 64
DESCRIPTOR_ENDIAN_MARKER = 0x01020304
DESCRIPTOR_HEADER = struct.Struct("<16sIIIIQQ32s32s32s32s32sIIII32s")
MAX_DESCRIPTOR_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_CONTAINER_ITEMS = 1_000_000
MAX_SCALAR_BYTES = 16 * 1024 * 1024
MAX_RECORD_COUNT = 64
MAX_NODE_COUNT = 1024
MAX_BINDING_COUNT = 16384
MAX_FUSION_WEIGHT_COUNT = 16384
EXPECTED_RECORD_COUNT = 32
EXPECTED_NODE_COUNT = 81
EXPECTED_BINDING_COUNT = 150
_DTYPE_ITEMSIZE = {"|u1": 1, "|i1": 1, "<i2": 2, "<i4": 4,
                   "<i8": 8, "<f2": 2, "<f4": 4, "<f8": 8}


def _u32(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("descriptor length/count exceeds uint32")
    return struct.pack("<I", value)


def _encode_canonical_binary(value, depth: int) -> bytes:
    if depth > 128:
        raise ValueError("descriptor nesting exceeds resource cap")
    if value is None:
        return b"N"
    if value is False:
        return b"F"
    if value is True:
        return b"T"
    if isinstance(value, int):
        sign = 1 if value < 0 else 0
        magnitude = abs(value)
        raw = (magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "big")
               if magnitude else b"")
        if len(raw) > MAX_SCALAR_BYTES:
            raise ValueError("descriptor integer exceeds resource cap")
        return b"I" + bytes((sign,)) + _u32(len(raw)) + raw
    if isinstance(value, str):
        raw = value.encode("utf-8")
        if len(raw) > MAX_SCALAR_BYTES:
            raise ValueError("descriptor string exceeds resource cap")
        return b"S" + _u32(len(raw)) + raw
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ValueError("descriptor list exceeds resource cap")
        return b"L" + _u32(len(value)) + b"".join(
            _encode_canonical_binary(item, depth + 1) for item in value
        )
    if isinstance(value, dict):
        if len(value) > MAX_CONTAINER_ITEMS or any(
                not isinstance(key, str) for key in value):
            raise ValueError("descriptor dict keys/count are noncanonical")
        ordered = sorted(value, key=lambda key: key.encode("utf-8"))
        return b"D" + _u32(len(ordered)) + b"".join(
            _encode_canonical_binary(key, depth + 1)
            + _encode_canonical_binary(value[key], depth + 1)
            for key in ordered
        )
    raise TypeError(f"descriptor value has unsupported type: {type(value).__name__}")


def encode_canonical_binary(value) -> bytes:
    """Encode the exact JSON value domain without text parsing or floats."""
    return _encode_canonical_binary(value, 0)


class _Decoder:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def _take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.payload):
            raise ValueError("descriptor payload is truncated")
        result = self.payload[self.offset:self.offset + count]
        self.offset += count
        return result

    def _count(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def decode(self, depth: int = 0):
        if depth > 128:
            raise ValueError("descriptor nesting exceeds resource cap")
        tag = self._take(1)
        if tag == b"N":
            return None
        if tag == b"F":
            return False
        if tag == b"T":
            return True
        if tag == b"I":
            sign = self._take(1)[0]
            count = self._count()
            if sign not in (0, 1) or count > MAX_SCALAR_BYTES:
                raise ValueError("descriptor integer is noncanonical")
            raw = self._take(count)
            if (count and raw[0] == 0) or (not count and sign):
                raise ValueError("descriptor integer has redundant encoding")
            value = int.from_bytes(raw, "big")
            return -value if sign else value
        if tag == b"S":
            count = self._count()
            if count > MAX_SCALAR_BYTES:
                raise ValueError("descriptor string exceeds resource cap")
            try:
                return self._take(count).decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("descriptor string is not UTF-8") from error
        if tag == b"L":
            count = self._count()
            if count > MAX_CONTAINER_ITEMS:
                raise ValueError("descriptor list exceeds resource cap")
            return [self.decode(depth + 1) for _ in range(count)]
        if tag == b"D":
            count = self._count()
            if count > MAX_CONTAINER_ITEMS:
                raise ValueError("descriptor dict exceeds resource cap")
            result = {}
            previous = None
            for _ in range(count):
                key = self.decode(depth + 1)
                if not isinstance(key, str):
                    raise ValueError("descriptor dict key is not a string")
                encoded_key = key.encode("utf-8")
                if previous is not None and encoded_key <= previous:
                    raise ValueError("descriptor dict keys are not canonical")
                previous = encoded_key
                result[key] = self.decode(depth + 1)
            return result
        raise ValueError("descriptor payload tag is invalid")


def decode_canonical_binary(payload: bytes):
    decoder = _Decoder(payload)
    result = decoder.decode()
    if decoder.offset != len(payload):
        raise ValueError("descriptor payload has trailing bytes")
    if encode_canonical_binary(result) != payload:
        raise ValueError("descriptor payload is not canonically encoded")
    return result


def program_execution_identity(program: TensorProgram) -> dict:
    """Stable identity excluding only the self-referential descriptor binding."""
    payload = program.to_dict()
    resources = dict(payload["resource_formula"])
    resources.pop("native_execution_descriptor", None)
    payload["resource_formula"] = resources
    return payload


def descriptor_payload(program: TensorProgram,
                       resident_plan: ValidatedResidentPlan) -> dict:
    if not isinstance(resident_plan, ValidatedResidentPlan):
        raise TypeError("native descriptor requires a validated resident plan")
    resident_plan.validate_manifest_identity(program)
    execution_identity = program_execution_identity(program)
    live_rows = tensor_program_required_axis0_rows(program)
    return {
        "schema_version": "green-v400-native-execution-descriptor-payload-v1",
        "contains_scientific_outcome": False,
        "program_execution_identity": execution_identity,
        "program_execution_semantic_hash": sha256_canonical(execution_identity),
        "program_dispatch_signature_sha256": resident_plan[
            "program_dispatch_signature_sha256"
        ],
        "tensor_store_record_closure_sha256": resident_plan[
            "tensor_store_record_closure_sha256"
        ],
        "blob_name": resident_plan["blob_name"],
        "blob_nbytes": resident_plan["blob_nbytes"],
        "blob_sha256": resident_plan["blob_sha256"],
        "alignment_bytes": resident_plan["alignment_bytes"],
        "dimensions": copy.deepcopy(resident_plan["dimensions"]),
        "records": copy.deepcopy(resident_plan["records"]),
        "program_input_binding_table": copy.deepcopy(resident_plan[
            "program_input_binding_table"
        ]),
        "exact_final_contrast_fusion": copy.deepcopy(resident_plan[
            "exact_final_contrast_fusion"
        ]),
        "exact_final_contrast_fusion_sha256": resident_plan[
            "exact_final_contrast_fusion_sha256"
        ],
        "required_axis0_rows": [
            {"node_semantic_id": node.semantic_id,
             "rows": list(live_rows.get(node.semantic_id, ()))}
            for node in program.nodes
        ],
        "branch_roots": copy.deepcopy(program.branch_roots),
        "output_root": program.output_root,
        "native_runtime_policy": {
            "schema_version": "green-v400-native-runtime-policy-v1",
            "descriptor_format_version": DESCRIPTOR_FORMAT_VERSION,
            "compiled_kernel_abi": "green-v400-compiled-mpfr-v2",
            "rounding_contract": "directed-mpfr-outward-interval-jet2",
            "supported_precision_bits": [384, 512],
            "domain_schema": "closed-dyadic-interval-v1",
            "exact_successful_dispatch_event_count": EXPECTED_NODE_COUNT,
            "corruption_fallback_allowed": False,
            "blob_locator_policy": "explicit-path-plus-nbytes-sha256-record-closure",
            "record_count": EXPECTED_RECORD_COUNT,
            "node_count": EXPECTED_NODE_COUNT,
            "binding_count": EXPECTED_BINDING_COUNT,
        },
        "native_execution_ready": False,
        "claim_status": "PASS_NATIVE_DESCRIPTOR_PREPARE_ONLY",
    }


def _header_for(payload: dict, encoded: bytes) -> bytes:
    program_hash = bytes.fromhex(payload["program_execution_semantic_hash"])
    dispatch_hash = bytes.fromhex(payload["program_dispatch_signature_sha256"])
    blob_hash = bytes.fromhex(payload["blob_sha256"])
    fusion_hash = bytes.fromhex(payload["exact_final_contrast_fusion_sha256"])
    return DESCRIPTOR_HEADER.pack(
        DESCRIPTOR_MAGIC, DESCRIPTOR_FORMAT_VERSION, DESCRIPTOR_HEADER_SIZE,
        DESCRIPTOR_ENDIAN_MARKER, DESCRIPTOR_ALIGNMENT,
        DESCRIPTOR_HEADER_SIZE, len(encoded), hashlib.sha256(encoded).digest(),
        program_hash, dispatch_hash, blob_hash, fusion_hash,
        len(payload["records"]),
        len(payload["program_execution_identity"]["nodes"]),
        len(payload["program_input_binding_table"]),
        len(payload["exact_final_contrast_fusion"]["weights"]),
        b"\0" * 32,
    )


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_payload_structure(payload: dict) -> None:
    expected_keys = {
        "schema_version", "contains_scientific_outcome",
        "program_execution_identity", "program_execution_semantic_hash",
        "program_dispatch_signature_sha256", "tensor_store_record_closure_sha256",
        "blob_name", "blob_nbytes", "blob_sha256", "alignment_bytes", "dimensions",
        "records", "program_input_binding_table", "exact_final_contrast_fusion",
        "exact_final_contrast_fusion_sha256", "required_axis0_rows", "branch_roots",
        "output_root", "native_runtime_policy", "native_execution_ready", "claim_status",
    }
    if set(payload) != expected_keys:
        raise ValueError("native execution descriptor payload field mismatch")
    if (payload["schema_version"]
            != "green-v400-native-execution-descriptor-payload-v1"
            or payload["contains_scientific_outcome"] is not False
            or payload["native_execution_ready"] is not False
            or payload["claim_status"] != "PASS_NATIVE_DESCRIPTOR_PREPARE_ONLY"
            or payload["alignment_bytes"] != DESCRIPTOR_ALIGNMENT):
        raise ValueError("native execution descriptor policy mismatch")
    policy = payload["native_runtime_policy"]
    if policy != {
        "schema_version": "green-v400-native-runtime-policy-v1",
        "descriptor_format_version": DESCRIPTOR_FORMAT_VERSION,
        "compiled_kernel_abi": "green-v400-compiled-mpfr-v2",
        "rounding_contract": "directed-mpfr-outward-interval-jet2",
        "supported_precision_bits": [384, 512],
        "domain_schema": "closed-dyadic-interval-v1",
        "exact_successful_dispatch_event_count": EXPECTED_NODE_COUNT,
        "corruption_fallback_allowed": False,
        "blob_locator_policy": "explicit-path-plus-nbytes-sha256-record-closure",
        "record_count": EXPECTED_RECORD_COUNT,
        "node_count": EXPECTED_NODE_COUNT,
        "binding_count": EXPECTED_BINDING_COUNT,
    }:
        raise ValueError("native execution descriptor runtime policy mismatch")
    dimensions = payload["dimensions"]
    dimension_keys = {
        "sequence_length", "d_model", "d_mlp", "n_heads", "d_head",
        "selected_gates", "final_position", "contrast_width",
    }
    if (set(dimensions) != dimension_keys
            or any(not isinstance(dimensions[key], int) or dimensions[key] <= 0
                   for key in ("sequence_length", "d_model", "d_mlp", "n_heads",
                               "d_head", "contrast_width"))
            or dimensions["d_model"] != dimensions["n_heads"] * dimensions["d_head"]
            or not 0 <= dimensions["final_position"] < dimensions["sequence_length"]
            or not isinstance(dimensions["selected_gates"], list)
            or len(dimensions["selected_gates"])
                != len(set(dimensions["selected_gates"]))
            or any(not isinstance(index, int) or not 0 <= index < dimensions["d_mlp"]
                   for index in dimensions["selected_gates"])):
        raise ValueError("native execution descriptor dimensions are invalid")
    program_identity = payload["program_execution_identity"]
    nodes = program_identity.get("nodes") if isinstance(program_identity, dict) else None
    records = payload["records"]
    bindings = payload["program_input_binding_table"]
    fusion = payload["exact_final_contrast_fusion"]
    if (not isinstance(nodes, list) or len(nodes) != EXPECTED_NODE_COUNT
            or not isinstance(records, list) or len(records) != EXPECTED_RECORD_COUNT
            or not isinstance(bindings, list) or len(bindings) != EXPECTED_BINDING_COUNT
            or not isinstance(fusion, dict)
            or len(fusion.get("weights", ())) != dimensions["d_model"]):
        raise ValueError("native execution descriptor exact counts mismatch")
    hashes = (
        payload["program_execution_semantic_hash"],
        payload["program_dispatch_signature_sha256"],
        payload["tensor_store_record_closure_sha256"], payload["blob_sha256"],
        payload["exact_final_contrast_fusion_sha256"],
    )
    if (not all(_is_sha256(value) for value in hashes)
            or sha256_canonical(program_identity)
                != payload["program_execution_semantic_hash"]
            or sha256_canonical(fusion)
                != payload["exact_final_contrast_fusion_sha256"]):
        raise ValueError("native execution descriptor semantic hash mismatch")
    prior_end = 0
    record_names = []
    for record in records:
        shape = record.get("shape") if isinstance(record, dict) else None
        dtype = record.get("dtype") if isinstance(record, dict) else None
        if (not isinstance(shape, list) or len(shape) > 8 or dtype not in _DTYPE_ITEMSIZE
                or any(not isinstance(value, int) or value < 0 for value in shape)):
            raise ValueError("native execution descriptor record shape/dtype mismatch")
        elements = 1
        for value in shape:
            if value and elements > (1 << 63) // value:
                raise ValueError("native execution descriptor record shape overflows")
            elements *= value
        expected_offset = (prior_end + DESCRIPTOR_ALIGNMENT - 1) // DESCRIPTOR_ALIGNMENT * DESCRIPTOR_ALIGNMENT
        if (record.get("offset") != expected_offset
                or record.get("nbytes") != elements * _DTYPE_ITEMSIZE[dtype]
                or not isinstance(record.get("name"), str)
                or not _is_sha256(record.get("data_sha256"))
                or not _is_sha256(record.get("tensor_semantic_sha256"))):
            raise ValueError("native execution descriptor record closure mismatch")
        prior_end = record["offset"] + record["nbytes"]
        record_names.append(record["name"])
    if (len(record_names) != len(set(record_names))
            or prior_end != payload["blob_nbytes"]
            or not isinstance(payload["blob_name"], str)
            or Path(payload["blob_name"]).name != payload["blob_name"]):
        raise ValueError("native execution descriptor blob layout mismatch")
    node_ids = [node.get("semantic_id") for node in nodes]
    if (len(node_ids) != len(set(node_ids))
            or set(payload["branch_roots"]) != {"PAT_J", "PAT_B", "TAR_J", "TAR_B"}
            or any(root not in node_ids for root in payload["branch_roots"].values())
            or payload["output_root"] not in node_ids):
        raise ValueError("native execution descriptor root/node closure mismatch")
    seen = set()
    total_parents = 0
    total_tensor_inputs = 0
    node_by_id = {}
    for node in nodes:
        parents = node.get("parent_semantic_ids")
        tensor_inputs = node.get("tensor_inputs")
        output_shape = node.get("output_spec", {}).get("shape")
        if (not _is_sha256(node.get("semantic_id"))
                or not isinstance(node.get("kernel_id"), str)
                or not isinstance(parents, list)
                or any(parent not in seen for parent in parents)
                or not isinstance(tensor_inputs, list)
                or not isinstance(output_shape, list)
                or any(not isinstance(value, int) or value < 0 for value in output_shape)):
            raise ValueError("native execution descriptor node topology is invalid")
        total_parents += len(parents)
        total_tensor_inputs += len(tensor_inputs)
        if total_parents > 4096 or total_tensor_inputs > 4096:
            raise ValueError("native execution descriptor node resources exceed cap")
        seen.add(node["semantic_id"])
        node_by_id[node["semantic_id"]] = node
    liveness = payload["required_axis0_rows"]
    if (not isinstance(liveness, list) or len(liveness) != EXPECTED_NODE_COUNT
            or [row.get("node_semantic_id") for row in liveness] != node_ids):
        raise ValueError("native execution descriptor liveness closure mismatch")
    for node, row in zip(nodes, liveness):
        rows = row.get("rows")
        shape = node.get("output_spec", {}).get("shape")
        if (not isinstance(rows, list) or rows != sorted(set(rows))
                or not isinstance(shape, list)
                or any(not isinstance(index, int) or not shape
                       or not 0 <= index < shape[0] for index in rows)):
            raise ValueError("native execution descriptor liveness rows are invalid")
    binding_keys = []
    for binding in bindings:
        key = (binding.get("node_semantic_id"), binding.get("tensor_input_ordinal"))
        source = binding.get("source")
        if (key[0] not in node_ids or not isinstance(key[1], int) or key[1] < 0
                or binding.get("kernel_id") != node_by_id[key[0]]["kernel_id"]
                or key[1] >= len(node_by_id[key[0]]["tensor_inputs"])
                or not _is_sha256(binding.get("tensor_semantic_sha256"))
                or not isinstance(source, dict)
                or source.get("kind") not in {
                    "packed_record", "exact_final_contrast_fusion_source"
                }):
            raise ValueError("native execution descriptor binding is invalid")
        if source["kind"] == "packed_record" and (
                not isinstance(source.get("record_index"), int)
                or not 0 <= source["record_index"] < len(records)
                or source.get("record_name")
                    != records[source["record_index"]]["name"]):
            raise ValueError("native execution descriptor record binding is invalid")
        if (source["kind"] == "exact_final_contrast_fusion_source"
                and source.get("source_name")
                    not in fusion.get("input_closure", {})):
            raise ValueError("native execution descriptor fusion binding is invalid")
        binding_keys.append(key)
    if len(binding_keys) != len(set(binding_keys)):
        raise ValueError("native execution descriptor binding is duplicated")


def build_native_execution_descriptor(
    path: Path, program: TensorProgram, resident_plan: ValidatedResidentPlan,
) -> dict:
    output = Path(path)
    if output.exists():
        raise FileExistsError("native execution descriptor is immutable")
    payload = descriptor_payload(program, resident_plan)
    encoded = encode_canonical_binary(payload)
    if len(encoded) > MAX_DESCRIPTOR_PAYLOAD_BYTES:
        raise ValueError("native execution descriptor exceeds payload cap")
    content = _header_for(payload, encoded) + encoded
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("native execution descriptor temporary path exists")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        load_native_execution_descriptor(temporary, program, resident_plan)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return load_native_execution_descriptor(output, program, resident_plan)


def load_native_execution_descriptor(
    path: Path, program: TensorProgram, resident_plan: ValidatedResidentPlan,
) -> dict:
    descriptor = Path(path).resolve()
    with descriptor.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if not DESCRIPTOR_HEADER_SIZE <= size <= (
                DESCRIPTOR_HEADER_SIZE + MAX_DESCRIPTOR_PAYLOAD_BYTES):
            raise ValueError("native execution descriptor file exceeds resource bounds")
        raw = handle.read(size + 1)
        if len(raw) != size:
            raise ValueError("native execution descriptor changed during read")
    if len(raw) < DESCRIPTOR_HEADER_SIZE:
        raise ValueError("native execution descriptor is truncated")
    unpacked = DESCRIPTOR_HEADER.unpack(raw[:DESCRIPTOR_HEADER_SIZE])
    (magic, version, header_size, endian, alignment, payload_offset,
     payload_nbytes, payload_sha, program_hash, dispatch_hash, blob_hash,
     fusion_hash,
     record_count, node_count, binding_count, fusion_weight_count,
     reserved) = unpacked
    if (magic != DESCRIPTOR_MAGIC or version != DESCRIPTOR_FORMAT_VERSION
            or header_size != DESCRIPTOR_HEADER_SIZE
            or endian != DESCRIPTOR_ENDIAN_MARKER
            or alignment != DESCRIPTOR_ALIGNMENT
            or payload_offset != DESCRIPTOR_HEADER_SIZE
            or payload_nbytes > MAX_DESCRIPTOR_PAYLOAD_BYTES
            or record_count != EXPECTED_RECORD_COUNT
            or node_count != EXPECTED_NODE_COUNT
            or binding_count != EXPECTED_BINDING_COUNT
            or record_count > MAX_RECORD_COUNT
            or node_count > MAX_NODE_COUNT
            or binding_count > MAX_BINDING_COUNT
            or fusion_weight_count > MAX_FUSION_WEIGHT_COUNT
            or len(raw) != payload_offset + payload_nbytes):
        raise ValueError("native execution descriptor header/length mismatch")
    if reserved != b"\0" * 32:
        raise ValueError("native execution descriptor reserved bytes are nonzero")
    encoded = raw[payload_offset:]
    if hashlib.sha256(encoded).digest() != payload_sha:
        raise ValueError("native execution descriptor payload hash mismatch")
    payload = decode_canonical_binary(encoded)
    _validate_payload_structure(payload)
    expected = descriptor_payload(program, resident_plan)
    if payload != expected:
        raise ValueError("native execution descriptor semantic closure mismatch")
    if (program_hash.hex() != payload["program_execution_semantic_hash"]
            or dispatch_hash.hex() != payload["program_dispatch_signature_sha256"]
            or blob_hash.hex() != payload["blob_sha256"]
            or fusion_hash.hex() != payload["exact_final_contrast_fusion_sha256"]
            or record_count != len(payload["records"])
            or node_count != len(payload["program_execution_identity"]["nodes"])
            or binding_count != len(payload["program_input_binding_table"])
            or fusion_weight_count
                != len(payload["exact_final_contrast_fusion"]["weights"])):
        raise ValueError("native execution descriptor header semantic mismatch")
    return {
        "payload": payload,
        "descriptor_file_sha256": hashlib.sha256(raw).hexdigest(),
        "descriptor_payload_sha256": payload_sha.hex(),
        "descriptor_nbytes": len(raw),
    }
