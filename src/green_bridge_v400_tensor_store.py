"""Bit-exact, hash-closed tensor blobs for replayable GREEN v4 graphs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable
from fractions import Fraction

import numpy as np

from green_bridge_v400_schemas import canonical_json, sha256_canonical


SCHEMA_VERSION = "green-v400-tensor-store-v1"
TENSOR_REF_SCHEMA_VERSION = "green-v400-tensor-ref-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_exact_array(value) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.asarray(value)
    if (array.dtype.hasobject or array.dtype.kind not in "buif"
            or array.dtype.itemsize not in {1, 2, 4, 8}):
        raise ValueError("tensor dtype is outside the certified whitelist")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("nonfinite tensor constant")
    # Canonical persisted representation is little-endian.  This preserves
    # the exact IEEE/integer value bits while making hashes architecture-neutral.
    if array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return array


@dataclass(frozen=True)
class TensorRef:
    schema_version: str
    tensor_sha256: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    nbytes: int

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "tensor_sha256": self.tensor_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_order: str
    offset: int
    nbytes: int
    data_sha256: str
    semantic_sha256: str

    def tensor_ref(self) -> TensorRef:
        return TensorRef(TENSOR_REF_SCHEMA_VERSION, self.semantic_sha256,
                         self.dtype, self.shape, "C", self.nbytes)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "dtype": self.dtype, "shape": list(self.shape),
            "byte_order": self.byte_order, "offset": self.offset,
            "nbytes": self.nbytes, "data_sha256": self.data_sha256,
            "semantic_sha256": self.semantic_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorRecord":
        expected = {"name", "dtype", "shape", "byte_order", "offset", "nbytes",
                    "data_sha256", "semantic_sha256"}
        if set(payload) != expected:
            raise ValueError("tensor record schema mismatch")
        return cls(
            str(payload["name"]), str(payload["dtype"]),
            tuple(int(value) for value in payload["shape"]),
            str(payload["byte_order"]), int(payload["offset"]),
            int(payload["nbytes"]), str(payload["data_sha256"]),
            str(payload["semantic_sha256"]),
        )


@dataclass(frozen=True)
class TensorStoreManifest:
    schema_version: str
    blob_name: str
    blob_sha256: str
    blob_nbytes: int
    records: tuple[TensorRecord, ...]
    record_closure_sha256: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "blob_name": self.blob_name,
            "blob_sha256": self.blob_sha256,
            "blob_nbytes": self.blob_nbytes,
            "records": [record.to_dict() for record in self.records],
            "record_closure_sha256": self.record_closure_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorStoreManifest":
        expected = {"schema_version", "blob_name", "blob_sha256", "blob_nbytes",
                    "records", "record_closure_sha256"}
        if set(payload) != expected:
            raise ValueError("tensor store manifest schema mismatch")
        result = cls(
            str(payload["schema_version"]), str(payload["blob_name"]),
            str(payload["blob_sha256"]), int(payload["blob_nbytes"]),
            tuple(TensorRecord.from_dict(row) for row in payload["records"]),
            str(payload["record_closure_sha256"]),
        )
        if result.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported tensor store schema")
        if result.record_closure_sha256 != sha256_canonical(
                [record.to_dict() for record in result.records]):
            raise ValueError("tensor record closure mismatch")
        names = [record.name for record in result.records]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tensor name")
        return result


def write_tensor_store(root: Path, name: str,
                       tensors: Iterable[tuple[str, object]], *,
                       max_tensors: int = 100_000,
                       max_total_bytes: int = 2_147_483_648) -> TensorStoreManifest:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    blob_path = root / f"{name}.bin"
    manifest_path = root / f"{name}.json"
    if blob_path.exists() or manifest_path.exists():
        raise FileExistsError("tensor store is immutable")
    temporary_blob = blob_path.with_suffix(".bin.tmp")
    records: list[TensorRecord] = []
    offset = 0
    with temporary_blob.open("xb") as handle:
        for tensor_name, value in tensors:
            if len(records) >= max_tensors:
                raise ValueError("tensor count resource limit exceeded")
            if not tensor_name or any(record.name == tensor_name for record in records):
                raise ValueError("tensor names must be nonempty and unique")
            array = _as_exact_array(value)
            if array.ndim > 8 or any(dimension < 0 or dimension > 10_000_000
                                     for dimension in array.shape):
                raise ValueError("tensor shape resource limit exceeded")
            raw = array.tobytes(order="C")
            if offset + len(raw) > max_total_bytes:
                raise ValueError("tensor byte resource limit exceeded")
            handle.write(raw)
            byte_order = "|" if array.dtype.itemsize == 1 else "<"
            dtype = array.dtype.str
            semantic = _sha256_bytes(canonical_json({
                "dtype": dtype, "shape": list(array.shape),
                "byte_order": byte_order, "layout": "C",
            }).encode("ascii") + b"\0" + raw)
            records.append(TensorRecord(
                tensor_name, dtype, tuple(array.shape), byte_order,
                offset, len(raw), _sha256_bytes(raw), semantic,
            ))
            offset += len(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_blob, blob_path)
    record_payload = [record.to_dict() for record in records]
    manifest = TensorStoreManifest(
        SCHEMA_VERSION, blob_path.name, _sha256_file(blob_path),
        blob_path.stat().st_size, tuple(records), sha256_canonical(record_payload),
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_manifest, manifest_path)
    return manifest


class TensorStoreReader:
    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest = TensorStoreManifest.from_dict(payload)
        self.blob_path = (self.manifest_path.parent / self.manifest.blob_name).resolve()
        if self.blob_path.parent != self.manifest_path.parent:
            raise ValueError("tensor blob path escapes manifest directory")
        if (not self.blob_path.is_file()
                or self.blob_path.stat().st_size != self.manifest.blob_nbytes
                or _sha256_file(self.blob_path) != self.manifest.blob_sha256):
            raise ValueError("tensor blob closure mismatch")
        self._records = {record.name: record for record in self.manifest.records}
        self._semantic_records: dict[str, TensorRecord] = {}
        cursor = 0
        for record in sorted(self.manifest.records, key=lambda item: item.offset):
            if record.offset != cursor or record.nbytes < 0:
                raise ValueError("tensor records do not exactly partition blob")
            dtype = np.dtype(record.dtype)
            expected = int(np.prod(record.shape, dtype=np.int64)) * dtype.itemsize
            canonical_order = "|" if dtype.itemsize == 1 else "<"
            if (expected != record.nbytes or record.byte_order != canonical_order
                    or dtype.str != record.dtype):
                raise ValueError("tensor record dtype/shape mismatch")
            prior = self._semantic_records.get(record.semantic_sha256)
            if prior is not None and prior.tensor_ref() != record.tensor_ref():
                raise ValueError("semantic tensor hash collision")
            self._semantic_records.setdefault(record.semantic_sha256, record)
            cursor += record.nbytes
        if cursor != self.manifest.blob_nbytes:
            raise ValueError("tensor records leave trailing or missing bytes")

    def names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self.manifest.records)

    def read(self, name: str) -> np.ndarray:
        return self._read_record(self._records[name])

    def read_semantic(self, tensor_sha256: str) -> np.ndarray:
        return self._read_record(self._semantic_records[tensor_sha256])

    def tensor_ref(self, name: str) -> TensorRef:
        return self._records[name].tensor_ref()

    def _read_record(self, record: TensorRecord) -> np.ndarray:
        with self.blob_path.open("rb") as handle:
            handle.seek(record.offset)
            raw = handle.read(record.nbytes)
        if len(raw) != record.nbytes or _sha256_bytes(raw) != record.data_sha256:
            raise ValueError("tensor payload hash mismatch")
        semantic = _sha256_bytes(canonical_json({
            "dtype": record.dtype, "shape": list(record.shape),
            "byte_order": record.byte_order, "layout": "C",
        }).encode("ascii") + b"\0" + raw)
        if semantic != record.semantic_sha256:
            raise ValueError("tensor semantic hash mismatch")
        return np.frombuffer(raw, dtype=np.dtype(record.dtype)).reshape(record.shape).copy()


def exact_dyadic_scalar(value) -> Fraction:
    """Decode a certified scalar tensor value as its exact rational value."""
    scalar = np.asarray(value)
    if scalar.shape != () or scalar.dtype.kind not in "buif":
        raise ValueError("exact dyadic decoder requires one certified scalar")
    item = scalar.item()
    if scalar.dtype.kind == "f":
        if not np.isfinite(scalar):
            raise ValueError("nonfinite dyadic scalar")
        numerator, denominator = float(item).as_integer_ratio()
        return Fraction(numerator, denominator)
    return Fraction(int(item), 1)
