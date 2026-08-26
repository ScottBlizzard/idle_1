from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_tensor_store import TensorStoreReader, write_tensor_store


def test_bit_exact_tensor_store_round_trip(tmp_path):
    tensors = [
        ("weight", np.asarray([[0.0, -0.0], [1.5, -2.25]], dtype="<f4")),
        ("tokens", np.asarray([1, 50256, 17], dtype="<i8")),
    ]
    manifest = write_tensor_store(tmp_path, "fixture", tensors)
    reader = TensorStoreReader(tmp_path / "fixture.json")
    assert reader.names() == ("weight", "tokens")
    for name, expected in tensors:
        actual = reader.read(name)
        assert actual.dtype.str == expected.dtype.str
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    assert manifest.blob_nbytes == sum(value.nbytes for _, value in tensors)


def test_tensor_store_records_dtype_shape_endianness_and_hash(tmp_path):
    write_tensor_store(tmp_path, "fixture", [("x", np.asarray([1, 2], dtype=">i2"))])
    payload = json.loads((tmp_path / "fixture.json").read_text(encoding="utf-8"))
    record = payload["records"][0]
    assert record["dtype"] == ">i2"
    assert record["byte_order"] == ">"
    assert record["shape"] == [2]
    assert len(record["data_sha256"]) == 64
    assert len(payload["blob_sha256"]) == 64


def test_tensor_store_rejects_blob_corruption(tmp_path):
    write_tensor_store(tmp_path, "fixture", [("x", np.asarray([1.0], dtype="<f4"))])
    blob = tmp_path / "fixture.bin"
    corrupted = bytearray(blob.read_bytes())
    corrupted[-1] ^= 1
    blob.write_bytes(corrupted)
    with pytest.raises(ValueError, match="blob closure"):
        TensorStoreReader(tmp_path / "fixture.json")


def test_tensor_store_rejects_nonfinite_constants(tmp_path):
    with pytest.raises(ValueError, match="nonfinite"):
        write_tensor_store(tmp_path, "fixture", [("x", np.asarray([np.inf], dtype="<f4"))])


def test_tensor_store_is_immutable(tmp_path):
    write_tensor_store(tmp_path, "fixture", [("x", np.asarray([1], dtype="<i4"))])
    with pytest.raises(FileExistsError):
        write_tensor_store(tmp_path, "fixture", [("x", np.asarray([2], dtype="<i4"))])
