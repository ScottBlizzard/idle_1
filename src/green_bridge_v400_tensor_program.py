"""Canonical Tensor-SSA macros with deterministic scalar-DAG semantics."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any

from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_store import TensorRef


PROGRAM_SCHEMA_VERSION = "green-v400-tensor-program-v1"
NODE_SCHEMA_VERSION = "green-v400-tensor-node-v1"

KERNEL_REGISTRY = {
    "affine_scatter.v1": {"scalar_semantics": "base[i]+t*direction[i] at static indices"},
    "static_view.v1": {"scalar_semantics": "bijective or static-index reference view"},
    "pairwise_affine.v1": {"scalar_semantics": "fixed balanced sum_j weight[j]*x[j]+bias"},
    "layer_norm.v1": {"scalar_semantics": "joint mean, nonnegative squares, epsilon, affine"},
    "gelu_new.v1": {"scalar_semantics": "runtime-bit kappa/lambda tanh GELU"},
    "causal_attention.v1": {"scalar_semantics": "static mask deletion, fixed pivot, exact exp(0)=1"},
    "residual_add.v1": {"scalar_semantics": "elementwise exact addition"},
    "final_contrast.v1": {"scalar_semantics": "fixed pairwise exact-rational linear contrast"},
    "branch_linear_combination.v1": {"scalar_semantics": "PAT_J-PAT_B-TAR_J+TAR_B"},
}
KERNEL_REGISTRY_HASH = sha256_canonical(KERNEL_REGISTRY)
NATIVE_DISPATCH_KERNEL_TAGS = {
    "affine_scatter.v1": 1,
    "static_view.v1": 2,
    "pairwise_affine.v1": 3,
    "layer_norm.v1": 4,
    "gelu_new.v1": 5,
    "causal_attention.v1": 6,
    "residual_add.v1": 7,
    "final_contrast.v1": 8,
    "branch_linear_combination.v1": 9,
}


def tensor_program_dispatch_signature(nodes: tuple["TensorNode", ...]) -> dict:
    """Canonical native-dispatch trace derived from the actual Tensor-SSA order."""
    ordered_kernel_ids = [node.kernel_id for node in nodes]
    kernel_counts = dict(sorted(Counter(ordered_kernel_ids).items()))
    return {
        "schema_version": "green-v400-tensor-dispatch-signature-v2-semantic-nodes",
        "node_count": len(ordered_kernel_ids),
        "ordered_kernel_ids": ordered_kernel_ids,
        "ordered_nodes": [
            {
                "ordinal": ordinal,
                "semantic_id": node.semantic_id,
                "kernel_id": node.kernel_id,
                "output_spec": node.output_spec.to_dict(),
                "dependency_mask_hash": node.dependency_mask_hash,
            }
            for ordinal, node in enumerate(nodes)
        ],
        "kernel_counts": kernel_counts,
    }


def tensor_program_dispatch_signature_hash(nodes: tuple["TensorNode", ...]) -> str:
    return sha256_canonical(tensor_program_dispatch_signature(nodes))


def tensor_program_native_trace(nodes: tuple["TensorNode", ...]) -> dict:
    """Mirror the backend's runtime FNV-1a event trace over ordered kernels."""
    state = 14695981039346656037
    for node in nodes:
        state ^= NATIVE_DISPATCH_KERNEL_TAGS[node.kernel_id]
        state = (state * 1099511628211) & ((1 << 64) - 1)
    return {"event_count": len(nodes), "fnv1a_u64": f"{state:016x}"}


def tensor_program_native_tags(nodes: tuple["TensorNode", ...]) -> list[int]:
    return [NATIVE_DISPATCH_KERNEL_TAGS[node.kernel_id] for node in nodes]


def _assert_exact_json(value: Any, path: str = "attrs") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise ValueError(f"ordinary JSON float forbidden at {path}")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_exact_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string exact attr key at {path}")
            _assert_exact_json(item, f"{path}.{key}")
        return
    raise ValueError(f"unsupported exact attr type at {path}: {type(value).__name__}")


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]
    layout: str = "C"

    def __post_init__(self):
        if self.layout != "C" or not self.dtype.startswith(("<", "|")):
            raise ValueError("TensorSpec must use canonical C/little-endian representation")
        if len(self.shape) > 8 or any(value < 0 or value > 10_000_000 for value in self.shape):
            raise ValueError("TensorSpec shape outside certified resource bounds")

    def to_dict(self) -> dict:
        return {"dtype": self.dtype, "shape": list(self.shape), "layout": self.layout}

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorSpec":
        if set(payload) != {"dtype", "shape", "layout"}:
            raise ValueError("TensorSpec schema mismatch")
        return cls(str(payload["dtype"]), tuple(int(value) for value in payload["shape"]),
                   str(payload["layout"]))


@dataclass(frozen=True)
class TensorNode:
    schema_version: str
    semantic_id: str
    kernel_id: str
    parent_semantic_ids: tuple[str, ...]
    tensor_inputs: tuple[TensorRef, ...]
    exact_attrs: dict
    output_spec: TensorSpec
    provenance_identity: str
    dependency_mask_hash: str

    @classmethod
    def build(cls, kernel_id: str, parent_semantic_ids: tuple[str, ...],
              tensor_inputs: tuple[TensorRef, ...], exact_attrs: dict,
              output_spec: TensorSpec, provenance_identity: str,
              dependency_mask_hash: str) -> "TensorNode":
        payload = cls.semantic_payload(
            kernel_id, parent_semantic_ids, tensor_inputs, exact_attrs,
            output_spec, provenance_identity, dependency_mask_hash,
        )
        return cls(NODE_SCHEMA_VERSION, sha256_canonical(payload), kernel_id,
                   parent_semantic_ids, tensor_inputs, exact_attrs, output_spec,
                   provenance_identity, dependency_mask_hash)

    @staticmethod
    def semantic_payload(kernel_id: str, parent_semantic_ids: tuple[str, ...],
                         tensor_inputs: tuple[TensorRef, ...], exact_attrs: dict,
                         output_spec: TensorSpec, provenance_identity: str,
                         dependency_mask_hash: str) -> dict:
        return {
            "kernel_id": kernel_id,
            "parent_semantic_ids": list(parent_semantic_ids),
            "tensor_inputs": [reference.to_dict() for reference in tensor_inputs],
            "exact_attrs": exact_attrs,
            "output_spec": output_spec.to_dict(),
            "provenance_identity": provenance_identity,
            "dependency_mask_hash": dependency_mask_hash,
        }

    def __post_init__(self):
        if self.schema_version != NODE_SCHEMA_VERSION or self.kernel_id not in KERNEL_REGISTRY:
            raise ValueError("unknown TensorNode schema or kernel")
        _assert_exact_json(self.exact_attrs)
        if len(self.dependency_mask_hash) != 64:
            raise ValueError("invalid dependency mask hash")
        mask = self.exact_attrs.get("dependency_mask_spec")
        if mask is not None:
            validate_dependency_mask_spec(mask, self.output_spec)
            if sha256_canonical(mask) != self.dependency_mask_hash:
                raise ValueError("TensorNode dependency mask closure mismatch")
        expected = sha256_canonical(self.semantic_payload(
            self.kernel_id, self.parent_semantic_ids, self.tensor_inputs,
            self.exact_attrs, self.output_spec, self.provenance_identity,
            self.dependency_mask_hash,
        ))
        if self.semantic_id != expected:
            raise ValueError("TensorNode semantic id mismatch")

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "semantic_id": self.semantic_id,
                **self.semantic_payload(self.kernel_id, self.parent_semantic_ids,
                                        self.tensor_inputs, self.exact_attrs,
                                        self.output_spec, self.provenance_identity,
                                        self.dependency_mask_hash)}

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorNode":
        expected = {
            "schema_version", "semantic_id", "kernel_id", "parent_semantic_ids",
            "tensor_inputs", "exact_attrs", "output_spec", "provenance_identity",
            "dependency_mask_hash",
        }
        if set(payload) != expected:
            raise ValueError("TensorNode schema mismatch")
        return cls(
            str(payload["schema_version"]), str(payload["semantic_id"]),
            str(payload["kernel_id"]),
            tuple(str(value) for value in payload["parent_semantic_ids"]),
            tuple(TensorRef.from_dict(value) for value in payload["tensor_inputs"]),
            dict(payload["exact_attrs"]), TensorSpec.from_dict(payload["output_spec"]),
            str(payload["provenance_identity"]), str(payload["dependency_mask_hash"]),
        )


def dependency_mask_spec(depends_on_t: bool, output_spec: TensorSpec,
                         axis0_indices: tuple[int, ...] | None = None) -> dict:
    """Canonical exact element mask; axis-0 rows are dense in remaining axes."""
    import math
    scalar_count = math.prod(output_spec.shape)
    if not depends_on_t:
        if axis0_indices:
            raise ValueError("empty dependency mask cannot declare rows")
        kind, indices, dependent_count = "empty", [], 0
    elif not output_spec.shape:
        if axis0_indices is not None:
            raise ValueError("scalar dependency mask cannot declare axis-0 rows")
        kind, indices, dependent_count = "dense", [], 1
    elif axis0_indices is None:
        kind, indices, dependent_count = "dense", [], scalar_count
    else:
        indices = sorted(set(int(index) for index in axis0_indices))
        if not indices or indices != list(axis0_indices):
            raise ValueError("dependency row indices must be nonempty sorted unique integers")
        if indices[0] < 0 or indices[-1] >= output_spec.shape[0]:
            raise ValueError("dependency row is outside output shape")
        kind = "axis0_rows"
        dependent_count = len(indices) * math.prod(output_spec.shape[1:])
    return {
        "schema_version": "green-v400-dependency-mask-v2",
        "kind": kind,
        "output_spec": output_spec.to_dict(),
        "axis0_indices": indices,
        "dependent_scalar_count": dependent_count,
    }


def dependency_mask_hash(depends_on_t: bool, output_spec: TensorSpec,
                         axis0_indices: tuple[int, ...] | None = None) -> str:
    return sha256_canonical(dependency_mask_spec(depends_on_t, output_spec, axis0_indices))


def validate_dependency_mask_spec(payload: dict, output_spec: TensorSpec) -> None:
    expected_keys = {
        "schema_version", "kind", "output_spec", "axis0_indices",
        "dependent_scalar_count",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != "green-v400-dependency-mask-v2":
        raise ValueError("dependency mask schema mismatch")
    kind = payload.get("kind")
    if kind == "empty":
        expected = dependency_mask_spec(False, output_spec)
    elif kind == "dense":
        expected = dependency_mask_spec(True, output_spec)
    elif kind == "axis0_rows":
        expected = dependency_mask_spec(
            True, output_spec, tuple(payload.get("axis0_indices", ()))
        )
    else:
        raise ValueError("unknown dependency mask kind")
    if payload != expected:
        raise ValueError("dependency mask content mismatch")


def scalarization_merkle_root(nodes: tuple[TensorNode, ...]) -> str:
    """Commit to the deterministic scalar expansion of every Tensor-SSA node."""
    leaves = [{
        "semantic_id": node.semantic_id,
        "kernel_id": node.kernel_id,
        "scalar_output_count": __import__("math").prod(node.output_spec.shape),
        "dependency_mask_hash": node.dependency_mask_hash,
    } for node in nodes]
    return sha256_canonical({
        "schema_version": "green-v400-scalarization-merkle-v1",
        "leaves": leaves,
    })


@dataclass(frozen=True)
class TensorProgram:
    schema_version: str
    kernel_registry_hash: str
    model_manifest_hash: str
    nodes: tuple[TensorNode, ...]
    branch_roots: dict[str, str]
    output_root: str
    scalarization_merkle_root: str
    resource_formula: dict

    @classmethod
    def build(cls, model_manifest_hash: str, nodes: tuple[TensorNode, ...],
              branch_roots: dict[str, str], output_root: str,
              resource_formula: dict) -> "TensorProgram":
        return cls(
            PROGRAM_SCHEMA_VERSION, KERNEL_REGISTRY_HASH, model_manifest_hash,
            nodes, dict(branch_roots), output_root,
            scalarization_merkle_root(nodes), dict(resource_formula),
        )

    def __post_init__(self):
        if self.schema_version != PROGRAM_SCHEMA_VERSION:
            raise ValueError("unsupported TensorProgram schema")
        if self.kernel_registry_hash != KERNEL_REGISTRY_HASH:
            raise ValueError("kernel registry hash mismatch")
        if len(self.model_manifest_hash) != 64 or len(self.scalarization_merkle_root) != 64:
            raise ValueError("invalid TensorProgram closure hash")
        if set(self.branch_roots) != {"PAT_J", "PAT_B", "TAR_J", "TAR_B"}:
            raise ValueError("TensorProgram branch roots mismatch")
        seen = set()
        for node in self.nodes:
            if node.semantic_id in seen or any(parent not in seen for parent in node.parent_semantic_ids):
                raise ValueError("TensorProgram is not unique topological SSA")
            seen.add(node.semantic_id)
        if self.output_root not in seen or any(root not in seen for root in self.branch_roots.values()):
            raise ValueError("TensorProgram root is missing")
        if self.scalarization_merkle_root != scalarization_merkle_root(self.nodes):
            raise ValueError("TensorProgram scalarization closure mismatch")
        by_id = {node.semantic_id: node for node in self.nodes}
        for node in self.nodes:
            mask = node.exact_attrs.get("dependency_mask_spec")
            if mask is not None:
                if mask.get("output_spec") != node.output_spec.to_dict():
                    raise ValueError("TensorProgram dependency mask output mismatch")
                if sha256_canonical(mask) != node.dependency_mask_hash:
                    raise ValueError("TensorProgram dependency mask closure mismatch")
                mask_depends = mask.get("kind") != "empty"
                if bool(node.exact_attrs.get("depends_on_t", False)) != mask_depends:
                    raise ValueError("TensorProgram dependency mask boolean mismatch")
            parent_dependency = any(
                bool(by_id[parent].exact_attrs.get("depends_on_t", False))
                for parent in node.parent_semantic_ids
            )
            declared = node.exact_attrs.get("depends_on_t")
            if declared is not None and bool(declared) != parent_dependency and node.kernel_id != "affine_scatter.v1":
                raise ValueError("TensorProgram dependency declaration mismatch")
        _assert_exact_json(self.resource_formula, "resource_formula")

    def semantic_hash(self) -> str:
        return sha256_canonical(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "kernel_registry_hash": self.kernel_registry_hash,
            "model_manifest_hash": self.model_manifest_hash,
            "nodes": [node.to_dict() for node in self.nodes],
            "branch_roots": self.branch_roots,
            "output_root": self.output_root,
            "scalarization_merkle_root": self.scalarization_merkle_root,
            "resource_formula": self.resource_formula,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorProgram":
        expected = {
            "schema_version", "kernel_registry_hash", "model_manifest_hash", "nodes",
            "branch_roots", "output_root", "scalarization_merkle_root", "resource_formula",
        }
        if set(payload) != expected:
            raise ValueError("TensorProgram schema mismatch")
        return cls(
            str(payload["schema_version"]), str(payload["kernel_registry_hash"]),
            str(payload["model_manifest_hash"]),
            tuple(TensorNode.from_dict(value) for value in payload["nodes"]),
            {str(key): str(value) for key, value in payload["branch_roots"].items()},
            str(payload["output_root"]), str(payload["scalarization_merkle_root"]),
            dict(payload["resource_formula"]),
        )
