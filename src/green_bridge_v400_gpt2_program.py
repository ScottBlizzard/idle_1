"""Replayable GPT-2-small Tensor-SSA for the binding four-branch witness."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from green_bridge_v400_branch_semantics import BRANCH_ORDER, BRANCH_WEIGHTS
from green_bridge_v400_resource_plan import TailShape, plan_gpt2_tail_resources
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import (
    TensorNode, TensorProgram, TensorSpec, dependency_mask_hash,
)
from green_bridge_v400_tensor_store import TensorRef, TensorStoreReader, write_tensor_store


@dataclass(frozen=True)
class GPT2TailDimensions:
    sequence_length: int
    d_model: int
    d_mlp: int
    n_heads: int
    d_head: int
    selected_gates: tuple[int, ...]
    final_position: int
    contrast_width: int = 100

    def __post_init__(self):
        if (self.sequence_length < 1 or self.d_model < 1 or self.d_mlp < 1
                or self.n_heads < 1 or self.d_head < 1
                or self.n_heads * self.d_head != self.d_model):
            raise ValueError("invalid GPT-2 tail dimensions")
        if not 0 <= self.final_position < self.sequence_length:
            raise ValueError("final position is outside sequence")
        if (not self.selected_gates or len(set(self.selected_gates)) != len(self.selected_gates)
                or min(self.selected_gates) < 0 or max(self.selected_gates) >= self.d_mlp):
            raise ValueError("invalid selected-gate panel")
        if self.contrast_width < 1:
            raise ValueError("contrast width must be positive")

    def to_dict(self) -> dict:
        return {
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "d_mlp": self.d_mlp,
            "n_heads": self.n_heads,
            "d_head": self.d_head,
            "selected_gates": list(self.selected_gates),
            "final_position": self.final_position,
            "contrast_width": self.contrast_width,
        }


def _ref_map(reader: TensorStoreReader) -> dict[str, TensorRef]:
    return {name: reader.tensor_ref(name) for name in reader.names()}


def _spec(shape: tuple[int, ...], dtype: str = "<f4") -> TensorSpec:
    return TensorSpec(dtype, shape)


def _resource_plan(dims: GPT2TailDimensions):
    return plan_gpt2_tail_resources(TailShape(
        dims.d_model, dims.d_mlp, dims.n_heads, dims.d_head,
        dims.sequence_length, len(dims.selected_gates), dims.contrast_width,
    ))


def _node(kernel: str, parents: Iterable[TensorNode], tensors: Iterable[TensorRef],
          attrs: dict, shape: tuple[int, ...], provenance: str,
          *, depends_on_t: bool = True, dtype: str = "<f4") -> TensorNode:
    output_spec = _spec(shape, dtype)
    exact_attrs = dict(attrs) | {"depends_on_t": depends_on_t}
    return TensorNode.build(
        kernel, tuple(parent.semantic_id for parent in parents), tuple(tensors), exact_attrs,
        output_spec, provenance, dependency_mask_hash(depends_on_t, output_spec),
    )


def _tail_nodes(prefix: str, resid_post: TensorNode, refs: dict[str, TensorRef],
                dims: GPT2TailDimensions) -> tuple[list[TensorNode], TensorNode]:
    s, d, m = dims.sequence_length, dims.d_model, dims.d_mlp
    nodes: list[TensorNode] = []

    ln1 = _node("layer_norm.v1", (resid_post,),
                (refs["block11.ln1.w"], refs["block11.ln1.b"], refs["layer_norm.eps"]),
                {"axis": -1}, (s, d), f"{prefix}.block11.ln1")
    nodes.append(ln1)
    qkv = []
    for name in ("q", "k", "v"):
        projection = _node(
            "pairwise_affine.v1", (ln1,),
            (refs[f"block11.attn.W_{name.upper()}"], refs[f"block11.attn.b_{name.upper()}"]),
            {"weight_layout": "input_output", "torch_float_kernel": "linear"}, (s, d),
            f"{prefix}.block11.attn.{name}",
        )
        nodes.append(projection)
        qkv.append(projection)
    attention = _node(
        "causal_attention.v1", tuple(qkv), (),
        {"n_heads": dims.n_heads, "d_head": dims.d_head,
         "score_scale": "inverse_sqrt_d_head", "mask": "causal_delete_future",
         "softmax_pivot": "row_max_first_index"},
        (s, d), f"{prefix}.block11.attn.pattern_value",
    )
    nodes.append(attention)
    attn_out = _node(
        "pairwise_affine.v1", (attention,),
        (refs["block11.attn.W_O"], refs["block11.attn.b_O"]),
        {"weight_layout": "input_output", "torch_float_kernel": "linear"},
        (s, d), f"{prefix}.block11.attn.out",
    )
    nodes.append(attn_out)
    resid_mid = _node("residual_add.v1", (resid_post, attn_out), (), {}, (s, d),
                      f"{prefix}.block11.resid_mid")
    nodes.append(resid_mid)
    ln2 = _node("layer_norm.v1", (resid_mid,),
                (refs["block11.ln2.w"], refs["block11.ln2.b"], refs["layer_norm.eps"]),
                {"axis": -1}, (s, d), f"{prefix}.block11.ln2")
    nodes.append(ln2)
    pre = _node("pairwise_affine.v1", (ln2,),
                (refs["block11.mlp.W_in"], refs["block11.mlp.b_in"]),
                {"weight_layout": "input_output"}, (s, m), f"{prefix}.block11.mlp.pre")
    nodes.append(pre)
    post = _node("gelu_new.v1", (pre,),
                 (refs["gelu.kappa"], refs["gelu.lambda"]), {}, (s, m),
                 f"{prefix}.block11.mlp.post")
    nodes.append(post)
    mlp_out = _node("pairwise_affine.v1", (post,),
                    (refs["block11.mlp.W_out"], refs["block11.mlp.b_out"]),
                    {"weight_layout": "input_output"}, (s, d),
                    f"{prefix}.block11.mlp.out")
    nodes.append(mlp_out)
    resid_final = _node("residual_add.v1", (resid_mid, mlp_out), (), {}, (s, d),
                        f"{prefix}.block11.resid_post")
    nodes.append(resid_final)
    normalized = _node("layer_norm.v1", (resid_final,),
                       (refs["ln_final.w"], refs["ln_final.b"], refs["layer_norm.eps"]),
                       {"axis": -1}, (s, d), f"{prefix}.ln_final")
    nodes.append(normalized)
    contrast = _node(
        "final_contrast.v1", (normalized,),
        (refs["unembed.W_U_full"], refs["unembed.b_U_full"],
         refs["unembed.suffix_ids"], refs["contrast.coefficients"]),
        {"final_position": dims.final_position,
         "contrast_width": dims.contrast_width,
         "reduction": "fixed_balanced_pairwise",
         "scalarization": "exact_affine_fusion_to_residual_contrast"},
        (), f"{prefix}.final_contrast", dtype=refs["contrast.coefficients"].dtype,
    )
    nodes.append(contrast)
    return nodes, contrast


def build_gpt2_joint_witness_program(reader: TensorStoreReader,
                                     model_manifest_hash: str,
                                     dims: GPT2TailDimensions) -> TensorProgram:
    """Build the exact high-level four-branch program from a closed tensor store."""
    refs = _ref_map(reader)
    nodes: list[TensorNode] = []
    roots: dict[str, str] = {}
    k = len(dims.selected_gates)
    s, d = dims.sequence_length, dims.d_model

    for condition in ("PAT", "TAR"):
        base = _node(
            "affine_scatter.v1", (),
            (refs[f"{condition}.resid_post"], refs["physical_direction"]),
            {"final_position": dims.final_position, "control": "same_t_times_physical_direction"},
            (s, d), f"{condition}.shared_residual_bypass",
        )
        nodes.append(base)

        controlled_mid = _node(
            "affine_scatter.v1", (),
            (refs[f"{condition}.resid_mid"], refs["physical_direction"]),
            {"final_position": dims.final_position, "control": "same_t_times_physical_direction"},
            (s, d), f"{condition}.J.resid_mid",
        )
        nodes.append(controlled_mid)
        ln2 = _node("layer_norm.v1", (controlled_mid,),
                    (refs["block10.ln2.w"], refs["block10.ln2.b"], refs["layer_norm.eps"]),
                    {"axis": -1}, (s, d), f"{condition}.J.block10.ln2")
        nodes.append(ln2)
        selected_pre = _node(
            "pairwise_affine.v1", (ln2,),
            (refs["block10.mlp.W_in_selected"], refs["block10.mlp.b_in_selected"]),
                {"weight_layout": "input_output", "torch_float_kernel": "batch_addmm",
                 "selected_gates": list(dims.selected_gates)},
            (s, k), f"{condition}.J.block10.selected_pre",
        )
        nodes.append(selected_pre)
        selected_live = _node("gelu_new.v1", (selected_pre,),
                              (refs["gelu.kappa"], refs["gelu.lambda"]), {}, (s, k),
                              f"{condition}.J.block10.selected_live")
        nodes.append(selected_live)
        selected_delta = _node(
            "static_view.v1", (selected_live,), (refs[f"{condition}.selected_post"],),
            {"operation": "subtract_anchor_at_final_position",
             "final_position": dims.final_position},
            (s, k), f"{condition}.J.block10.selected_delta",
        )
        nodes.append(selected_delta)
        delta_out = _node(
            "pairwise_affine.v1", (selected_delta,),
            (refs["block10.mlp.W_out_selected"], refs["zero.d_model"]),
            {"weight_layout": "input_output", "torch_float_kernel": "batch_addmm"}, (s, d),
            f"{condition}.J.block10.selected_out_delta",
        )
        nodes.append(delta_out)
        joint_resid = _node("residual_add.v1", (base, delta_out), (), {}, (s, d),
                            f"{condition}.J.block10.resid_post")
        nodes.append(joint_resid)

        j_nodes, j_root = _tail_nodes(f"{condition}.J", joint_resid, refs, dims)
        nodes.extend(j_nodes)
        b_nodes, b_root = _tail_nodes(f"{condition}.B", base, refs, dims)
        nodes.extend(b_nodes)
        roots[f"{condition}_J"] = j_root.semantic_id
        roots[f"{condition}_B"] = b_root.semantic_id

    root_nodes = [{node.semantic_id: node for node in nodes}[roots[name]] for name in BRANCH_ORDER]
    output = _node(
        "branch_linear_combination.v1", root_nodes, (),
        {"order": list(BRANCH_ORDER), "weights": list(BRANCH_WEIGHTS),
         "reduction": "PAT_J_minus_PAT_B_minus_TAR_J_plus_TAR_B"},
        (), "binding_joint_witness_psi", dtype=refs["contrast.coefficients"].dtype,
    )
    nodes.append(output)
    plan = _resource_plan(dims)
    resource_formula = plan.to_dict() | {
        "dimensions": dims.to_dict(),
        "tensor_store_closure": reader.manifest.record_closure_sha256,
    }
    program = TensorProgram.build(
        model_manifest_hash, tuple(nodes), roots, output.semantic_id, resource_formula,
    )
    validate_gpt2_joint_witness_program(program, reader, dims)
    return program


def validate_gpt2_joint_witness_program(program: TensorProgram, reader: TensorStoreReader,
                                        dims: GPT2TailDimensions) -> None:
    if TensorProgram.from_dict(program.to_dict()) != program:
        raise ValueError("TensorProgram does not round-trip")
    for node in program.nodes:
        for reference in node.tensor_inputs:
            reader.validate_ref(reference)
    expected = _resource_plan(dims).to_dict()
    for key, value in expected.items():
        if program.resource_formula.get(key) != value:
            raise ValueError("TensorProgram resource formula mismatch")
    if program.resource_formula.get("dimensions") != dims.to_dict():
        raise ValueError("TensorProgram dimension closure mismatch")
    if program.resource_formula.get("tensor_store_closure") != reader.manifest.record_closure_sha256:
        raise ValueError("TensorProgram tensor-store closure mismatch")


def execute_tensor_program_numpy(program: TensorProgram, reader: TensorStoreReader,
                                 t: float) -> dict[str, np.ndarray]:
    """Outcome-blind floating replay oracle; MPFR execution is implemented separately."""
    values: dict[str, np.ndarray] = {}
    for node in program.nodes:
        parents = [values[parent] for parent in node.parent_semantic_ids]
        tensors = [reader.read_semantic(ref.tensor_sha256) for ref in node.tensor_inputs]
        kernel = node.kernel_id
        if kernel == "affine_scatter.v1":
            base, direction = tensors
            value = base.copy()
            value[int(node.exact_attrs["final_position"])] += np.asarray(t, dtype=value.dtype) * direction
        elif kernel == "layer_norm.v1":
            x = parents[0]
            weight, bias, epsilon = tensors
            mean = np.mean(x, axis=-1, keepdims=True)
            centered = x - mean
            variance = np.mean(centered * centered, axis=-1, keepdims=True)
            value = centered / np.sqrt(variance + epsilon.reshape(())) * weight + bias
        elif kernel == "pairwise_affine.v1":
            x = parents[0]
            weight, bias = tensors
            value = x @ weight + bias
        elif kernel == "gelu_new.v1":
            x = parents[0]
            kappa, lam = (tensor.reshape(()) for tensor in tensors)
            value = np.asarray(0.5, dtype=x.dtype) * x * (
                np.asarray(1.0, dtype=x.dtype) + np.tanh(kappa * (x + lam * x * x * x))
            )
        elif kernel == "static_view.v1":
            if node.exact_attrs.get("operation") != "subtract_anchor_at_final_position":
                raise ValueError("unsupported static view operation")
            value = parents[0].copy()
            position = int(node.exact_attrs["final_position"])
            value[position] -= tensors[0]
            value[:position] = 0
            value[position + 1:] = 0
        elif kernel == "causal_attention.v1":
            q, k, v = (parent.reshape(parent.shape[0], int(node.exact_attrs["n_heads"]),
                                      int(node.exact_attrs["d_head"])) for parent in parents)
            scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(
                np.asarray(int(node.exact_attrs["d_head"]), dtype=q.dtype)
            )
            mask = np.triu(np.ones((q.shape[0], q.shape[0]), dtype=bool), 1)
            scores = np.where(mask[None, :, :], -np.inf, scores)
            pivot = np.max(scores, axis=-1, keepdims=True)
            exponentials = np.exp(scores - pivot)
            pattern = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
            value = np.einsum("hqk,khd->qhd", pattern, v).reshape(q.shape[0], -1)
        elif kernel == "residual_add.v1":
            value = parents[0] + parents[1]
        elif kernel == "final_contrast.v1":
            weight, bias, suffix_ids, coefficients = tensors
            position = int(node.exact_attrs["final_position"])
            full_logits = parents[0][position] @ weight + bias
            value = np.asarray(full_logits[suffix_ids.astype(np.int64)] @ coefficients)
        elif kernel == "branch_linear_combination.v1":
            value = np.asarray(parents[0] - parents[1] - parents[2] + parents[3])
        else:  # pragma: no cover - registry gate
            raise ValueError(f"unsupported floating replay kernel {kernel}")
        if tuple(value.shape) != node.output_spec.shape:
            raise ValueError("TensorProgram runtime shape mismatch")
        values[node.semantic_id] = value
    return {"output": values[program.output_root], **{
        name: values[root] for name, root in program.branch_roots.items()
    }}


def execute_tensor_program_torch(program: TensorProgram, reader: TensorStoreReader,
                                 t: float, device: str, *,
                                 return_node_values: bool = False) -> dict[str, object]:
    """Model-semantics replay on the same Torch device as TransformerLens."""
    torch = __import__("torch")
    values = {}
    tensor_cache = {}

    def load(reference):
        if reference.tensor_sha256 not in tensor_cache:
            array = reader.read_semantic(reference.tensor_sha256)
            tensor_cache[reference.tensor_sha256] = torch.from_numpy(array).to(device)
        return tensor_cache[reference.tensor_sha256]

    for node in program.nodes:
        parents = [values[parent] for parent in node.parent_semantic_ids]
        tensors = [load(reference) for reference in node.tensor_inputs]
        kernel = node.kernel_id
        if kernel == "affine_scatter.v1":
            base, direction = tensors
            value = base.clone()
            value[int(node.exact_attrs["final_position"])] += value.new_tensor(t) * direction
        elif kernel == "layer_norm.v1":
            x = parents[0]
            weight, bias, epsilon = tensors
            centered = x - x.mean(dim=-1, keepdim=True)
            variance = centered.pow(2).mean(dim=-1, keepdim=True)
            value = centered / torch.sqrt(variance + epsilon.reshape(())) * weight + bias
        elif kernel == "pairwise_affine.v1":
            x = parents[0]
            weight, bias = tensors
            if node.exact_attrs.get("torch_float_kernel") == "linear":
                value = torch.nn.functional.linear(x, weight.T.contiguous(), bias)
            else:
                from transformer_lens.utilities.addmm import batch_addmm
                value = batch_addmm(bias, weight, x)
        elif kernel == "gelu_new.v1":
            x = parents[0]
            kappa, lam = (tensor.reshape(()) for tensor in tensors)
            value = x.new_tensor(0.5) * x * (
                x.new_tensor(1.0) + torch.tanh(kappa * (x + lam * x * x * x))
            )
        elif kernel == "static_view.v1":
            if node.exact_attrs.get("operation") != "subtract_anchor_at_final_position":
                raise ValueError("unsupported static view operation")
            value = torch.zeros_like(parents[0])
            position = int(node.exact_attrs["final_position"])
            value[position] = parents[0][position] - tensors[0]
        elif kernel == "causal_attention.v1":
            heads = int(node.exact_attrs["n_heads"])
            width = int(node.exact_attrs["d_head"])
            q, k, v = (parent.reshape(parent.shape[0], heads, width) for parent in parents)
            scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(width)
            mask = torch.triu(torch.ones(
                (q.shape[0], q.shape[0]), dtype=torch.bool, device=q.device
            ), diagonal=1)
            scores = scores.masked_fill(mask[None, :, :], -torch.inf)
            pattern = torch.softmax(scores, dim=-1)
            value = torch.einsum("hqk,khd->qhd", pattern, v).reshape(q.shape[0], -1)
        elif kernel == "residual_add.v1":
            value = parents[0] + parents[1]
        elif kernel == "final_contrast.v1":
            weight, bias, suffix_ids, coefficients = tensors
            position = int(node.exact_attrs["final_position"])
            full_logits = torch.nn.functional.linear(
                parents[0][position], weight.T.contiguous(), bias
            )
            logits = full_logits.index_select(0, suffix_ids.to(torch.long))
            value = (logits.to(coefficients.dtype) * coefficients).sum()
        elif kernel == "branch_linear_combination.v1":
            value = parents[0] - parents[1] - parents[2] + parents[3]
        else:  # pragma: no cover - registry gate
            raise ValueError(f"unsupported Torch replay kernel {kernel}")
        if tuple(value.shape) != node.output_spec.shape:
            raise ValueError("TensorProgram Torch runtime shape mismatch")
        values[node.semantic_id] = value
    result = {"output": values[program.output_root], **{
        name: values[root] for name, root in program.branch_roots.items()
    }}
    if return_node_values:
        result["node_values"] = values
    return result


def materialize_gpt2_joint_witness_store(root: Path, name: str, model,
                                         pat_anchor, tar_anchor, physical_direction,
                                         suffix_token_ids, contrast_coefficients,
                                         selected_gates: Iterable[int]) -> tuple[TensorStoreReader, GPT2TailDimensions]:
    """Freeze model weights and outcome-free anchors without serializing logits."""
    torch = __import__("torch")
    gates = tuple(int(value) for value in selected_gates)
    if int(pat_anchor.resid_mid.shape[0]) != 1 or int(tar_anchor.resid_mid.shape[0]) != 1:
        raise ValueError("formal TensorProgram materialization is row-wise")
    if pat_anchor.year_logits is None or tar_anchor.year_logits is None:
        pass
    output_softcap = getattr(model.cfg, "output_logits_soft_cap", None)
    if output_softcap is not None and float(output_softcap) > 0:
        raise ValueError("nonlinear output softcap is unsupported by final affine fusion")
    sequence_length = int(pat_anchor.resid_mid.shape[1])
    if tuple(tar_anchor.resid_mid.shape[1:]) != tuple(pat_anchor.resid_mid.shape[1:]):
        raise ValueError("PAT/TAR anchor shape mismatch")
    final_position = int(pat_anchor.final_positions.reshape(-1)[0].item())
    if int(tar_anchor.final_positions.reshape(-1)[0].item()) != final_position:
        raise ValueError("PAT/TAR final-position mismatch")
    dims = GPT2TailDimensions(
        sequence_length, int(model.cfg.d_model), int(model.cfg.d_mlp),
        int(model.cfg.n_heads), int(model.cfg.d_head), gates, final_position,
        int(suffix_token_ids.numel()),
    )
    block10, block11 = model.blocks[10], model.blocks[11]
    device = block10.mlp.W_in.device
    gate_index = torch.tensor(gates, dtype=torch.long, device=device)

    def cpu(value):
        return value.detach().cpu().contiguous().numpy()

    tensors = [
        ("physical_direction", cpu(physical_direction.reshape(-1))),
        ("layer_norm.eps", np.asarray(model.cfg.eps, dtype="<f4")),
        ("gelu.kappa", np.asarray(math.sqrt(2.0 / math.pi), dtype="<f4")),
        ("gelu.lambda", np.asarray(0.044715, dtype="<f4")),
        ("zero.d_model", np.zeros((dims.d_model,), dtype="<f4")),
        ("block10.ln2.w", cpu(block10.ln2.w)),
        ("block10.ln2.b", cpu(block10.ln2.b)),
        ("block10.mlp.W_in_selected", cpu(block10.mlp.W_in.index_select(1, gate_index))),
        ("block10.mlp.b_in_selected", cpu(block10.mlp.b_in.index_select(0, gate_index))),
        ("block10.mlp.W_out_selected", cpu(block10.mlp.W_out.index_select(0, gate_index))),
        ("block11.ln1.w", cpu(block11.ln1.w)),
        ("block11.ln1.b", cpu(block11.ln1.b)),
        ("block11.attn.W_Q", cpu(block11.attn.W_Q.permute(1, 0, 2).reshape(dims.d_model, dims.d_model))),
        ("block11.attn.b_Q", cpu(block11.attn.b_Q.reshape(dims.d_model))),
        ("block11.attn.W_K", cpu(block11.attn.W_K.permute(1, 0, 2).reshape(dims.d_model, dims.d_model))),
        ("block11.attn.b_K", cpu(block11.attn.b_K.reshape(dims.d_model))),
        ("block11.attn.W_V", cpu(block11.attn.W_V.permute(1, 0, 2).reshape(dims.d_model, dims.d_model))),
        ("block11.attn.b_V", cpu(block11.attn.b_V.reshape(dims.d_model))),
        ("block11.attn.W_O", cpu(block11.attn.W_O.reshape(dims.d_model, dims.d_model))),
        ("block11.attn.b_O", cpu(block11.attn.b_O)),
        ("block11.ln2.w", cpu(block11.ln2.w)),
        ("block11.ln2.b", cpu(block11.ln2.b)),
        ("block11.mlp.W_in", cpu(block11.mlp.W_in)),
        ("block11.mlp.b_in", cpu(block11.mlp.b_in)),
        ("block11.mlp.W_out", cpu(block11.mlp.W_out)),
        ("block11.mlp.b_out", cpu(block11.mlp.b_out)),
        ("ln_final.w", cpu(model.ln_final.w)),
        ("ln_final.b", cpu(model.ln_final.b)),
        ("unembed.W_U_full", cpu(model.W_U)),
        ("unembed.b_U_full", cpu(model.b_U)),
        ("unembed.suffix_ids", cpu(suffix_token_ids.to(torch.int64))),
        ("contrast.coefficients", cpu(contrast_coefficients.reshape(-1))),
    ]
    for condition, anchor in (("PAT", pat_anchor), ("TAR", tar_anchor)):
        tensors.extend((
            (f"{condition}.resid_mid", cpu(anchor.resid_mid[0])),
            (f"{condition}.resid_post", cpu(anchor.resid_post[0])),
            (f"{condition}.selected_post", cpu(anchor.post[0, final_position].index_select(0, gate_index))),
        ))
    write_tensor_store(Path(root), name, tensors)
    reader = TensorStoreReader(Path(root) / f"{name}.json")
    return reader, dims


def program_identity_payload(program: TensorProgram, dims: GPT2TailDimensions,
                             reader: TensorStoreReader) -> dict:
    return {
        "schema_version": "green-v400-gpt2-program-identity-v1",
        "program_semantic_hash": program.semantic_hash(),
        "tensor_store_manifest_hash": sha256_canonical(reader.manifest.to_dict()),
        "dimensions": dims.to_dict(),
        "branch_order": list(BRANCH_ORDER),
        "contains_scientific_outcome": False,
    }
