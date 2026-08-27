"""Hash-closed packed constants for the performance-resident GPT-2 dispatcher."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from green_bridge_v400_final_contrast_fusion import fuse_final_contrast_exact
from green_bridge_v400_gpt2_program import GPT2TailDimensions, validate_gpt2_joint_witness_program
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader


RESIDENT_TENSOR_NAMES = (
    "physical_direction", "layer_norm.eps", "gelu.kappa", "gelu.lambda",
    "zero.d_model", "block10.ln2.w", "block10.ln2.b",
    "block10.mlp.W_in_selected", "block10.mlp.b_in_selected",
    "block10.mlp.W_out_selected",
    "block11.ln1.w", "block11.ln1.b",
    "block11.attn.W_Q", "block11.attn.b_Q",
    "block11.attn.W_K", "block11.attn.b_K",
    "block11.attn.W_V", "block11.attn.b_V",
    "block11.attn.W_O", "block11.attn.b_O",
    "block11.ln2.w", "block11.ln2.b",
    "block11.mlp.W_in", "block11.mlp.b_in",
    "block11.mlp.W_out", "block11.mlp.b_out",
    "ln_final.w", "ln_final.b",
    "PAT.resid_mid", "PAT.resid_post", "TAR.resid_mid", "TAR.resid_post",
)


_VALIDATED_PLAN_TOKEN = object()


class ValidatedResidentPlan(dict):
    """Opaque mapping returned only after manifest/blob closure validation."""

    def __init__(self, manifest: dict, manifest_path: Path, blob_path: Path, token):
        if token is not _VALIDATED_PLAN_TOKEN:
            raise TypeError("validated resident plans must be created by the loader")
        super().__init__(manifest)
        self.manifest_path = Path(manifest_path).resolve()
        self.blob_path = Path(blob_path).resolve()
        stat = self.blob_path.stat()
        self._blob_identity = (
            stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
        )

    @staticmethod
    def _immutable(*_args, **_kwargs):
        raise TypeError("validated resident plan is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def validate_runtime(self, program: TensorProgram, reader: TensorStoreReader,
                         arrays: dict[str, np.ndarray]) -> None:
        """Reject plan/array substitution at every public resident execution entry."""
        semantic_hash = self.get("resident_plan_semantic_hash")
        unhashed = dict(self)
        unhashed.pop("resident_plan_semantic_hash", None)
        if (semantic_hash != sha256_canonical(unhashed)
                or self.get("program_semantic_hash") != program.semantic_hash()
                or self.get("program_dispatch_signature_sha256")
                    != program.resource_formula["dispatcher_signature_sha256"]
                or self.get("tensor_store_record_closure_sha256")
                    != reader.manifest.record_closure_sha256
                or self.get("exact_final_contrast_fusion_sha256")
                    != sha256_canonical(self["exact_final_contrast_fusion"])):
            raise ValueError("resident execution plan identity mismatch")
        stat = self.blob_path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != self._blob_identity:
            raise ValueError("resident execution blob identity changed after validation")
        if set(arrays) != {record["name"] for record in self["records"]}:
            raise ValueError("resident execution arrays do not match validated records")
        for record in self["records"]:
            array = arrays[record["name"]]
            filename = getattr(array, "filename", None)
            if (not isinstance(array, np.memmap) or filename is None
                    or Path(filename).resolve() != self.blob_path
                    or getattr(array, "offset", None) != record["offset"]
                    or array.dtype.str != record["dtype"]
                    or tuple(array.shape) != tuple(record["shape"])
                    or array.nbytes != record["nbytes"]
                    or array.ctypes.data % self["alignment_bytes"] != 0):
                raise ValueError("resident execution array identity mismatch")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dims(program: TensorProgram) -> GPT2TailDimensions:
    value = program.resource_formula["dimensions"]
    return GPT2TailDimensions(
        value["sequence_length"], value["d_model"], value["d_mlp"],
        value["n_heads"], value["d_head"], tuple(value["selected_gates"]),
        value["final_position"], value["contrast_width"],
    )


def build_resident_plan(root: Path, name: str, program: TensorProgram,
                        reader: TensorStoreReader) -> dict:
    """Write one immutable aligned blob plus a manifest; no outcomes are read."""
    validate_gpt2_joint_witness_program(program, reader, _dims(program))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    blob_path, manifest_path = root / f"{name}.bin", root / f"{name}.json"
    if blob_path.exists() or manifest_path.exists():
        raise FileExistsError("resident plan is immutable")
    records, offset = [], 0
    temporary = blob_path.with_suffix(".bin.tmp")
    with temporary.open("xb") as handle:
        for tensor_name in RESIDENT_TENSOR_NAMES:
            array = reader.read(tensor_name)
            padding = (-offset) % 64
            if padding:
                handle.write(b"\0" * padding); offset += padding
            raw = array.tobytes(order="C")
            handle.write(raw)
            reference = reader.tensor_ref(tensor_name)
            records.append({
                "name": tensor_name, "offset": offset, "nbytes": len(raw),
                "dtype": array.dtype.str, "shape": list(array.shape),
                "data_sha256": hashlib.sha256(raw).hexdigest(),
                "tensor_semantic_sha256": reference.tensor_sha256,
            })
            offset += len(raw)
        handle.flush()
    temporary.replace(blob_path)
    fusion = fuse_final_contrast_exact(
        reader.read("unembed.W_U_full"), reader.read("unembed.b_U_full"),
        reader.read("unembed.suffix_ids"), reader.read("contrast.coefficients"),
    )
    if fusion.semantic_hash() != program.resource_formula["exact_final_contrast_fusion_sha256"]:
        raise RuntimeError("resident-plan fusion disagrees with TensorProgram")
    record_by_semantic = {
        record["tensor_semantic_sha256"]: (index, record["name"])
        for index, record in enumerate(records)
    }
    fusion_by_semantic = {
        value["semantic_sha256"]: name
        for name, value in fusion.payload()["input_closure"].items()
    }
    binding_table = []
    for node in program.nodes:
        for input_ordinal, reference in enumerate(node.tensor_inputs):
            if reference.tensor_sha256 in record_by_semantic:
                record_index, record_name = record_by_semantic[reference.tensor_sha256]
                source = {"kind": "packed_record", "record_index": record_index,
                          "record_name": record_name}
            elif reference.tensor_sha256 in fusion_by_semantic:
                source = {"kind": "exact_final_contrast_fusion_source",
                          "source_name": fusion_by_semantic[reference.tensor_sha256]}
            else:
                raise RuntimeError("TensorProgram input is absent from resident plan")
            binding_table.append({
                "node_semantic_id": node.semantic_id, "kernel_id": node.kernel_id,
                "tensor_input_ordinal": input_ordinal,
                "tensor_semantic_sha256": reference.tensor_sha256,
                "source": source,
            })
    manifest = {
        "schema_version": "green-v400-resident-packed-plan-v1",
        "contains_scientific_outcome": False,
        "program_semantic_hash": program.semantic_hash(),
        "program_dispatch_signature_sha256": program.resource_formula[
            "dispatcher_signature_sha256"
        ],
        "tensor_store_record_closure_sha256": reader.manifest.record_closure_sha256,
        "dimensions": _dims(program).to_dict(),
        "blob_name": blob_path.name,
        "blob_nbytes": blob_path.stat().st_size,
        "blob_sha256": _sha256_file(blob_path),
        "alignment_bytes": 64,
        "records": records,
        "program_input_binding_table": binding_table,
        "program_unique_tensor_ref_count": len({
            ref.tensor_sha256 for node in program.nodes for ref in node.tensor_inputs
        }),
        "exact_final_contrast_fusion": fusion.payload(),
        "exact_final_contrast_fusion_sha256": fusion.semantic_hash(),
        "full_unembedding_excluded_after_exact_fusion": True,
        "fusion_storage": "canonical_json_control_plane_pending_native_loader",
        "native_execution_ready": False,
        "claim_status": "PASS_PACKED_RESIDENT_MANIFEST_PREPARE_ONLY",
    }
    manifest["resident_plan_semantic_hash"] = sha256_canonical(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
    validate_resident_plan(manifest_path, program, reader)
    return manifest


def validate_resident_plan(manifest_path: Path, program: TensorProgram,
                           reader: TensorStoreReader) -> dict:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    semantic_hash = manifest.pop("resident_plan_semantic_hash", None)
    if semantic_hash != sha256_canonical(manifest):
        raise ValueError("resident plan semantic hash mismatch")
    manifest["resident_plan_semantic_hash"] = semantic_hash
    if (manifest.get("schema_version") != "green-v400-resident-packed-plan-v1"
            or manifest.get("contains_scientific_outcome") is not False
            or manifest.get("program_semantic_hash") != program.semantic_hash()
            or manifest.get("program_dispatch_signature_sha256")
                != program.resource_formula["dispatcher_signature_sha256"]
            or manifest.get("tensor_store_record_closure_sha256")
                != reader.manifest.record_closure_sha256
            or manifest.get("dimensions") != _dims(program).to_dict()):
        raise ValueError("resident plan program/store closure mismatch")
    blob = (path.parent / manifest["blob_name"]).resolve()
    if (blob.parent != path.parent or not blob.is_file()
            or blob.stat().st_size != manifest["blob_nbytes"]
            or _sha256_file(blob) != manifest["blob_sha256"]):
        raise ValueError("resident plan blob closure mismatch")
    if manifest.get("alignment_bytes") != 64:
        raise ValueError("resident plan alignment policy mismatch")
    prior_end = 0
    covered_semantic_hashes = set()
    with blob.open("rb") as handle:
        for expected_name, record in zip(RESIDENT_TENSOR_NAMES, manifest["records"]):
            expected_offset = (prior_end + 63) // 64 * 64
            if (record["name"] != expected_name or record["offset"] != expected_offset):
                raise ValueError("resident plan record order/alignment mismatch")
            handle.seek(prior_end)
            if handle.read(record["offset"] - prior_end) != b"\0" * (record["offset"] - prior_end):
                raise ValueError("resident plan padding is not canonical zero")
            reference = reader.tensor_ref(expected_name)
            if (record["dtype"] != reference.dtype or tuple(record["shape"]) != reference.shape
                    or record["nbytes"] != reference.nbytes
                    or record["tensor_semantic_sha256"] != reference.tensor_sha256):
                raise ValueError("resident plan tensor reference mismatch")
            handle.seek(record["offset"]); raw = handle.read(record["nbytes"])
            if hashlib.sha256(raw).hexdigest() != record["data_sha256"]:
                raise ValueError("resident plan tensor bytes mismatch")
            covered_semantic_hashes.add(record["tensor_semantic_sha256"])
            prior_end = record["offset"] + record["nbytes"]
    if len(manifest["records"]) != len(RESIDENT_TENSOR_NAMES):
        raise ValueError("resident plan tensor count mismatch")
    if prior_end != manifest["blob_nbytes"]:
        raise ValueError("resident plan has trailing bytes")
    if (manifest.get("exact_final_contrast_fusion_sha256")
            != program.resource_formula["exact_final_contrast_fusion_sha256"]
            or sha256_canonical(manifest["exact_final_contrast_fusion"])
                != manifest["exact_final_contrast_fusion_sha256"]
            or manifest.get("full_unembedding_excluded_after_exact_fusion") is not True):
        raise ValueError("resident plan exact-fusion closure mismatch")
    fusion_closure = manifest["exact_final_contrast_fusion"]["input_closure"]
    covered_semantic_hashes.update(
        value["semantic_sha256"] for value in fusion_closure.values()
    )
    program_semantic_hashes = {
        ref.tensor_sha256 for node in program.nodes for ref in node.tensor_inputs
    }
    record_by_semantic = {
        record["tensor_semantic_sha256"]: (index, record["name"])
        for index, record in enumerate(manifest["records"])
    }
    fusion_by_semantic = {
        value["semantic_sha256"]: name
        for name, value in fusion_closure.items()
    }
    expected_bindings = []
    for node in program.nodes:
        for input_ordinal, reference in enumerate(node.tensor_inputs):
            if reference.tensor_sha256 in record_by_semantic:
                record_index, record_name = record_by_semantic[reference.tensor_sha256]
                source = {"kind": "packed_record", "record_index": record_index,
                          "record_name": record_name}
            elif reference.tensor_sha256 in fusion_by_semantic:
                source = {"kind": "exact_final_contrast_fusion_source",
                          "source_name": fusion_by_semantic[reference.tensor_sha256]}
            else:
                raise ValueError("resident binding source is missing")
            expected_bindings.append({
                "node_semantic_id": node.semantic_id, "kernel_id": node.kernel_id,
                "tensor_input_ordinal": input_ordinal,
                "tensor_semantic_sha256": reference.tensor_sha256,
                "source": source,
            })
    if (covered_semantic_hashes != program_semantic_hashes
            or manifest.get("program_unique_tensor_ref_count") != len(program_semantic_hashes)
            or manifest.get("fusion_storage")
                != "canonical_json_control_plane_pending_native_loader"
            or manifest.get("native_execution_ready") is not False):
        raise ValueError("resident plan does not exactly cover the program tensor closure")
    if (manifest.get("program_input_binding_table") != expected_bindings
            or manifest.get("claim_status")
                != "PASS_PACKED_RESIDENT_MANIFEST_PREPARE_ONLY"):
        raise ValueError("resident program-input binding table mismatch")
    return manifest


def load_resident_plan_arrays(manifest_path: Path, program: TensorProgram,
                              reader: TensorStoreReader) -> tuple[dict, dict[str, np.ndarray]]:
    """Validated mmap loader; every exposed array must be actually 64-byte aligned."""
    path = Path(manifest_path).resolve()
    manifest = validate_resident_plan(path, program, reader)
    blob = path.parent / manifest["blob_name"]
    arrays = {}
    for record in manifest["records"]:
        array = np.memmap(blob, dtype=np.dtype(record["dtype"]), mode="r",
                          offset=record["offset"], shape=tuple(record["shape"]), order="C")
        if array.ctypes.data % 64 != 0:
            raise RuntimeError("resident mmap base does not satisfy 64-byte alignment")
        arrays[record["name"]] = array
    validated = ValidatedResidentPlan(
        manifest, path, blob, _VALIDATED_PLAN_TOKEN
    )
    validated.validate_runtime(program, reader, arrays)
    return validated, arrays
