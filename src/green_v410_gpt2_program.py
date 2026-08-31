"""GREEN v4.1 full-downstream GPT-2 TensorProgram materialization.

This module deliberately does not reuse the v4.0 fixed block-10/11 program.
For a ``resid_post`` site after layer 0--8 it starts from the complete
controlled-hook tensor and evaluates every later block.  The matched-bypass
split is introduced only at the selected block-10 MLP posts at the final
causal token.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from green_bridge_v400_branch_semantics import BRANCH_ORDER, BRANCH_WEIGHTS
from green_bridge_v400_schemas import sha256_canonical
from green_bridge_v400_tensor_program import (
    TensorNode,
    TensorProgram,
    TensorSpec,
    dependency_mask_hash,
    dependency_mask_spec,
    tensor_program_dispatch_signature_hash,
)
from green_bridge_v400_tensor_store import TensorRef, TensorStoreReader, write_tensor_store
from green_v410_causal_cone import build_causal_cone_plan


GRAPH_SEMANTICS_ID = "GREEN_V410_FULL_DOWNSTREAM_CONE_MB_V1"


@dataclass(frozen=True)
class V410GPT2Dimensions:
    site_layer: int
    site_position: int
    sequence_length: int
    d_model: int
    d_mlp: int
    n_heads: int
    d_head: int
    selected_gates: tuple[int, ...]
    final_position: int
    contrast_width: int
    contrast_coefficient_rationals: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.site_layer not in range(9):
            raise ValueError("v4.1 site layer must be in 0..8")
        if self.sequence_length < 1 or not 0 <= self.site_position < self.sequence_length:
            raise ValueError("site position is outside the sequence")
        if self.final_position != self.sequence_length - 1:
            raise ValueError("task scalar must use the final unpadded causal token")
        if (self.d_model < 1 or self.d_mlp < 1 or self.n_heads < 1 or self.d_head < 1
                or self.n_heads * self.d_head != self.d_model):
            raise ValueError("invalid GPT-2 dimensions")
        if (not self.selected_gates
                or len(set(self.selected_gates)) != len(self.selected_gates)
                or min(self.selected_gates) < 0
                or max(self.selected_gates) >= self.d_mlp):
            raise ValueError("invalid selected-gate panel")
        if self.contrast_width < 1:
            raise ValueError("contrast width must be positive")
        if (len(self.contrast_coefficient_rationals) != self.contrast_width
                or any(
                    type(numerator) is not int or type(denominator) is not int
                    or denominator <= 0
                    for numerator, denominator in self.contrast_coefficient_rationals
                )):
            raise ValueError("exact rational task contrast is malformed")

    @property
    def dynamic_rows(self) -> tuple[int, ...]:
        return tuple(range(self.site_position, self.sequence_length))

    def to_dict(self) -> dict:
        return {
            "site_layer": self.site_layer,
            "site_position": self.site_position,
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "d_mlp": self.d_mlp,
            "n_heads": self.n_heads,
            "d_head": self.d_head,
            "selected_gates": list(self.selected_gates),
            "final_position": self.final_position,
            "contrast_width": self.contrast_width,
            "contrast_coefficient_rationals": [
                [numerator, denominator]
                for numerator, denominator in self.contrast_coefficient_rationals
            ],
        }


def _spec(shape: tuple[int, ...], dtype: str = "<f8") -> TensorSpec:
    return TensorSpec(dtype, shape)


def _node(
    kernel: str,
    parents: Iterable[TensorNode],
    tensors: Iterable[TensorRef],
    attrs: dict,
    shape: tuple[int, ...],
    provenance: str,
    *,
    depends_on_t: bool = True,
    dtype: str = "<f8",
    dynamic_axis0_indices: tuple[int, ...] | None = None,
) -> TensorNode:
    output_spec = _spec(shape, dtype)
    if depends_on_t and shape and dynamic_axis0_indices is None:
        raise ValueError("dynamic tensor node requires an exact axis-0 dependency mask")
    mask = dependency_mask_spec(depends_on_t, output_spec, dynamic_axis0_indices)
    exact_attrs = dict(attrs) | {
        "depends_on_t": depends_on_t,
        "dependency_mask_spec": mask,
    }
    return TensorNode.build(
        kernel,
        tuple(parent.semantic_id for parent in parents),
        tuple(tensors),
        exact_attrs,
        output_spec,
        provenance,
        dependency_mask_hash(depends_on_t, output_spec, dynamic_axis0_indices),
    )


def _refs(reader: TensorStoreReader) -> dict[str, TensorRef]:
    return {name: reader.tensor_ref(name) for name in reader.names()}


def _block_nodes(
    *,
    prefix: str,
    block: int,
    resid_post: TensorNode,
    refs: Mapping[str, TensorRef],
    dims: V410GPT2Dimensions,
    input_dynamic_rows: tuple[int, ...] | None,
) -> tuple[list[TensorNode], TensorNode, TensorNode]:
    """Append one complete pre-LN GPT-2 block.

    Returns all nodes, the block ``resid_post`` root, and the live MLP-post
    tensor.  The latter is used only by the block-10 matched-bypass split.
    """
    s, d, m = dims.sequence_length, dims.d_model, dims.d_mlp
    dynamic = input_dynamic_rows is not None
    if dynamic and not input_dynamic_rows:
        raise ValueError("downstream block requires a nonempty dynamic-row set")
    input_rows = tuple(input_dynamic_rows or ())
    rows = tuple(range(min(input_rows), s)) if dynamic else ()
    key = f"block{block}"
    nodes: list[TensorNode] = []

    ln1 = _node(
        "layer_norm.v1", (resid_post,),
        (refs[f"{key}.ln1.w"], refs[f"{key}.ln1.b"], refs["layer_norm.eps"]),
        {"axis": -1}, (s, d), f"{prefix}.{key}.ln1",
        depends_on_t=dynamic,
        dynamic_axis0_indices=input_rows if dynamic else None,
    )
    nodes.append(ln1)
    qkv: list[TensorNode] = []
    for name in ("q", "k", "v"):
        projection = _node(
            "pairwise_affine.v1", (ln1,),
            (refs[f"{key}.attn.W_{name.upper()}"], refs[f"{key}.attn.b_{name.upper()}"]),
            {"weight_layout": "input_output", "torch_float_kernel": "linear"},
            (s, d), f"{prefix}.{key}.attn.{name}",
            depends_on_t=dynamic,
            dynamic_axis0_indices=input_rows if dynamic else None,
        )
        nodes.append(projection)
        qkv.append(projection)
    attention = _node(
        "causal_attention.v1", tuple(qkv), (),
        {
            "n_heads": dims.n_heads,
            "d_head": dims.d_head,
            "score_scale": "inverse_sqrt_d_head",
            "mask": "causal_delete_future",
            "softmax_pivot": {"kind": "fixed_index", "index": 0},
        },
        (s, d), f"{prefix}.{key}.attn.pattern_value", depends_on_t=dynamic,
        dynamic_axis0_indices=rows if dynamic else None,
    )
    nodes.append(attention)
    attn_out = _node(
        "pairwise_affine.v1", (attention,),
        (refs[f"{key}.attn.W_O"], refs[f"{key}.attn.b_O"]),
        {"weight_layout": "input_output", "torch_float_kernel": "linear"},
        (s, d), f"{prefix}.{key}.attn.out", depends_on_t=dynamic,
        dynamic_axis0_indices=rows if dynamic else None,
    )
    nodes.append(attn_out)
    resid_mid = _node(
        "residual_add.v1", (resid_post, attn_out), (), {}, (s, d),
        f"{prefix}.{key}.resid_mid", depends_on_t=dynamic,
        dynamic_axis0_indices=rows if dynamic else None,
    )
    nodes.append(resid_mid)
    ln2 = _node(
        "layer_norm.v1", (resid_mid,),
        (refs[f"{key}.ln2.w"], refs[f"{key}.ln2.b"], refs["layer_norm.eps"]),
        {"axis": -1}, (s, d), f"{prefix}.{key}.ln2", depends_on_t=dynamic,
        dynamic_axis0_indices=rows if dynamic else None,
    )
    nodes.append(ln2)
    pre = _node(
        "pairwise_affine.v1", (ln2,),
        (refs[f"{key}.mlp.W_in"], refs[f"{key}.mlp.b_in"]),
        {"weight_layout": "input_output"}, (s, m), f"{prefix}.{key}.mlp.pre",
        depends_on_t=dynamic, dynamic_axis0_indices=rows if dynamic else None,
    )
    nodes.append(pre)
    post = _node(
        "gelu_new.v1", (pre,), (refs["gelu.kappa"], refs["gelu.lambda"]), {},
        (s, m), f"{prefix}.{key}.mlp.post", depends_on_t=dynamic,
        dynamic_axis0_indices=rows if dynamic else None,
    )
    nodes.append(post)
    mlp_out = _node(
        "pairwise_affine.v1", (post,),
        (refs[f"{key}.mlp.W_out"], refs[f"{key}.mlp.b_out"]),
        {"weight_layout": "input_output"}, (s, d), f"{prefix}.{key}.mlp.out",
        depends_on_t=dynamic, dynamic_axis0_indices=rows if dynamic else None,
    )
    nodes.append(mlp_out)
    result = _node(
        "residual_add.v1", (resid_mid, mlp_out), (), {}, (s, d),
        f"{prefix}.{key}.resid_post", depends_on_t=dynamic,
        dynamic_axis0_indices=rows if dynamic else None,
    )
    nodes.append(result)
    return nodes, result, post


def _tail_scalar_nodes(
    *,
    prefix: str,
    resid_post: TensorNode,
    refs: Mapping[str, TensorRef],
    dims: V410GPT2Dimensions,
    input_dynamic_rows: tuple[int, ...],
) -> tuple[list[TensorNode], TensorNode]:
    nodes, final_resid, _ = _block_nodes(
        prefix=prefix, block=11, resid_post=resid_post, refs=refs, dims=dims,
        input_dynamic_rows=input_dynamic_rows,
    )
    normalized = _node(
        "layer_norm.v1", (final_resid,),
        (refs["ln_final.w"], refs["ln_final.b"], refs["layer_norm.eps"]),
        {"axis": -1}, (dims.sequence_length, dims.d_model), f"{prefix}.ln_final",
        dynamic_axis0_indices=dims.dynamic_rows,
    )
    nodes.append(normalized)
    contrast = _node(
        "final_contrast.v1", (normalized,),
        (refs["unembed.W_U_full"], refs["unembed.b_U_full"],
         refs["unembed.suffix_ids"], refs["contrast.coefficients"]),
        {
            "final_position": dims.final_position,
            "contrast_width": dims.contrast_width,
            "reduction": "fixed_balanced_pairwise",
            "scalarization": "exact_affine_fusion_to_residual_contrast",
            "coefficient_rationals": [
                {"numerator": numerator, "denominator": denominator}
                for numerator, denominator in dims.contrast_coefficient_rationals
            ],
        },
        (), f"{prefix}.task_scalar", dtype="<f8",
    )
    nodes.append(contrast)
    return nodes, contrast


def build_green_v410_full_cone_program(
    reader: TensorStoreReader,
    model_manifest_hash: str,
    dims: V410GPT2Dimensions,
) -> TensorProgram:
    """Build the five-output full downstream cone for one SFC direction."""
    refs = _refs(reader)
    nodes: list[TensorNode] = []
    roots: dict[str, str] = {}
    s, d, k = dims.sequence_length, dims.d_model, len(dims.selected_gates)
    rows = dims.dynamic_rows
    final_row = (dims.final_position,)

    for condition in ("PAT", "TAR"):
        controlled = _node(
            "affine_scatter.v1", (),
            (refs[f"{condition}.controlled_hook_t0"], refs["physical_direction"]),
            {
                # The inherited kernel calls this field final_position; v4.1 binds
                # it explicitly to the actual site position in its graph validator.
                "final_position": dims.site_position,
                "control": "same_t_times_physical_direction",
            },
            (s, d), f"{condition}.site_layer{dims.site_layer}.controlled_resid_post",
            dynamic_axis0_indices=(dims.site_position,),
        )
        nodes.append(controlled)
        shared = controlled
        shared_rows = (dims.site_position,)
        for block in range(dims.site_layer + 1, 10):
            block_nodes, shared, _ = _block_nodes(
                prefix=f"{condition}.SHARED", block=block, resid_post=shared,
                refs=refs, dims=dims, input_dynamic_rows=shared_rows,
            )
            nodes.extend(block_nodes)
            shared_rows = rows

        # A second, t-independent path defines the formal gate anchor from the
        # identical controlled-hook payload.  This is essential: importing a
        # rounded checkpoint gate value as the mathematical anchor would make
        # J(0)=B(0) fail by floating re-evaluation noise.
        zero_control = _node(
            "static_view.v1", (), (refs[f"{condition}.controlled_hook_t0"],),
            {"operation": "tensor_constant"}, (s, d),
            f"{condition}.ZERO.site_layer{dims.site_layer}.controlled_resid_post",
            depends_on_t=False,
        )
        nodes.append(zero_control)
        zero_shared = zero_control
        for block in range(dims.site_layer + 1, 10):
            zero_nodes, zero_shared, _ = _block_nodes(
                prefix=f"{condition}.ZERO", block=block, resid_post=zero_shared,
                refs=refs, dims=dims, input_dynamic_rows=None,
            )
            nodes.extend(zero_nodes)

        block10_nodes, joint_resid, _ = _block_nodes(
            prefix=f"{condition}.J_AND_B", block=10, resid_post=shared,
            refs=refs, dims=dims, input_dynamic_rows=shared_rows,
        )
        nodes.extend(block10_nodes)
        zero_block10_nodes, _, _ = _block_nodes(
            prefix=f"{condition}.ZERO", block=10, resid_post=zero_shared,
            refs=refs, dims=dims, input_dynamic_rows=None,
        )
        zero_ln2_index = next(
            index for index, node in enumerate(zero_block10_nodes)
            if node.provenance_identity.endswith("block10.ln2")
        )
        # The zero-control branch exists only to define the selected gate-post
        # anchor.  Nodes after block-10 ln2 would be dead graph material.
        nodes.extend(zero_block10_nodes[:zero_ln2_index + 1])

        # Recompute only the selected columns using the identical affine/GELU
        # semantics.  This exposes an exact selected-gate delta without freezing
        # any nonselected gate or the residual bypass.
        block10_ln2 = next(
            node for node in reversed(block10_nodes)
            if node.provenance_identity.endswith("block10.ln2")
        )
        zero_block10_ln2 = next(
            node for node in reversed(zero_block10_nodes[:zero_ln2_index + 1])
            if node.provenance_identity.endswith("block10.ln2")
        )
        selected_pre = _node(
            "pairwise_affine.v1", (block10_ln2,),
            (refs["block10.mlp.W_in_selected"], refs["block10.mlp.b_in_selected"]),
            {
                "weight_layout": "input_output",
                "selected_gates": list(dims.selected_gates),
            },
            (s, k), f"{condition}.block10.selected_pre", dynamic_axis0_indices=rows,
        )
        nodes.append(selected_pre)
        selected_live = _node(
            "gelu_new.v1", (selected_pre,),
            (refs["gelu.kappa"], refs["gelu.lambda"]), {}, (s, k),
            f"{condition}.block10.selected_live", dynamic_axis0_indices=rows,
        )
        nodes.append(selected_live)
        selected_zero_pre = _node(
            "pairwise_affine.v1", (zero_block10_ln2,),
            (refs["block10.mlp.W_in_selected"], refs["block10.mlp.b_in_selected"]),
            {
                "weight_layout": "input_output",
                "selected_gates": list(dims.selected_gates),
            },
            (s, k), f"{condition}.ZERO.block10.selected_pre", depends_on_t=False,
        )
        nodes.append(selected_zero_pre)
        selected_anchor = _node(
            "gelu_new.v1", (selected_zero_pre,),
            (refs["gelu.kappa"], refs["gelu.lambda"]), {}, (s, k),
            f"{condition}.ZERO.block10.selected_anchor_t0", depends_on_t=False,
        )
        nodes.append(selected_anchor)
        selected_delta = _node(
            "static_view.v1", (selected_live, selected_anchor), (),
            {
                "operation": "subtract_exact_parent_at_final_position",
                "final_position": dims.final_position,
            },
            (s, k), f"{condition}.B.block10.selected_live_minus_anchor",
            dynamic_axis0_indices=final_row,
        )
        nodes.append(selected_delta)
        negative_delta_out = _node(
            "pairwise_affine.v1", (selected_delta,),
            (refs["block10.mlp.W_out_selected_negative"], refs["zero.d_model"]),
            {"weight_layout": "input_output"}, (s, d),
            f"{condition}.B.block10.negative_selected_out_delta",
            dynamic_axis0_indices=final_row,
        )
        nodes.append(negative_delta_out)
        bypass_resid = _node(
            "residual_add.v1", (joint_resid, negative_delta_out), (), {}, (s, d),
            f"{condition}.B.block10.resid_post", dynamic_axis0_indices=rows,
        )
        nodes.append(bypass_resid)

        j_nodes, j_root = _tail_scalar_nodes(
            prefix=f"{condition}.J", resid_post=joint_resid, refs=refs, dims=dims,
            input_dynamic_rows=rows,
        )
        b_nodes, b_root = _tail_scalar_nodes(
            prefix=f"{condition}.B", resid_post=bypass_resid, refs=refs, dims=dims,
            input_dynamic_rows=rows,
        )
        nodes.extend(j_nodes)
        nodes.extend(b_nodes)
        roots[f"{condition}_J"] = j_root.semantic_id
        roots[f"{condition}_B"] = b_root.semantic_id

    by_id = {node.semantic_id: node for node in nodes}
    output = _node(
        "branch_linear_combination.v1",
        tuple(by_id[roots[name]] for name in BRANCH_ORDER), (),
        {
            "order": list(BRANCH_ORDER),
            "weights": list(BRANCH_WEIGHTS),
            "reduction": "PAT_J_minus_PAT_B_minus_TAR_J_plus_TAR_B",
        },
        (), "GREEN_V410.full_downstream_cone.PSI", dtype="<f8",
    )
    nodes.append(output)
    cone = build_causal_cone_plan(
        site_layer=dims.site_layer,
        patch_position=dims.site_position,
        sequence_length=dims.sequence_length,
    )
    masks = [{
        "semantic_id": node.semantic_id,
        "dependency_mask_hash": node.dependency_mask_hash,
        "dependent_scalar_count": node.exact_attrs["dependency_mask_spec"][
            "dependent_scalar_count"
        ],
    } for node in nodes]
    resource_formula = {
        "schema_version": "green-v410-full-cone-resource-formula-v1",
        "graph_semantics_id": GRAPH_SEMANTICS_ID,
        "dimensions": dims.to_dict(),
        "causal_cone_plan_sha256": cone["plan_sha256"],
        "tensor_store_closure_sha256": reader.manifest.record_closure_sha256,
        "dependency_mask_closure_sha256": sha256_canonical(masks),
        "dependent_scalar_outputs_total": sum(
            row["dependent_scalar_count"] for row in masks
        ),
        "dispatcher_signature_sha256": tensor_program_dispatch_signature_hash(tuple(nodes)),
        "endpoint_material_allowed": False,
    }
    program = TensorProgram.build(
        model_manifest_hash, tuple(nodes), roots, output.semantic_id, resource_formula,
    )
    validate_green_v410_full_cone_program(program, reader, dims)
    return program


def validate_green_v410_full_cone_program(
    program: TensorProgram,
    reader: TensorStoreReader,
    dims: V410GPT2Dimensions,
) -> None:
    if TensorProgram.from_dict(program.to_dict()) != program:
        raise ValueError("v4.1 TensorProgram does not round-trip")
    if program.resource_formula.get("graph_semantics_id") != GRAPH_SEMANTICS_ID:
        raise ValueError("old or unknown graph semantics are forbidden")
    if program.resource_formula.get("endpoint_material_allowed") is not False:
        raise ValueError("endpoint material is forbidden")
    if program.resource_formula.get("dimensions") != dims.to_dict():
        raise ValueError("v4.1 dimension closure mismatch")
    if (program.resource_formula.get("tensor_store_closure_sha256")
            != reader.manifest.record_closure_sha256):
        raise ValueError("v4.1 tensor-store closure mismatch")
    scatter = [node for node in program.nodes if node.kernel_id == "affine_scatter.v1"]
    if len(scatter) != 2 or any(
        node.exact_attrs.get("final_position") != dims.site_position for node in scatter
    ):
        raise ValueError("v4.1 graph does not patch the exact site position")
    required_blocks = set(range(dims.site_layer + 1, 12))
    for condition in ("PAT", "TAR"):
        observed = {
            block for block in required_blocks
            if any(
                node.provenance_identity.startswith(f"{condition}.")
                and f".block{block}." in node.provenance_identity
                for node in program.nodes
            )
        }
        if observed != required_blocks:
            raise ValueError("v4.1 full downstream block closure is incomplete")
        for branch in ("J", "B"):
            root = program.branch_roots[f"{condition}_{branch}"]
            node = next(item for item in program.nodes if item.semantic_id == root)
            if not node.provenance_identity.endswith("task_scalar"):
                raise ValueError("v4.1 branch root is not the exact task scalar")
    for node in program.nodes:
        for reference in node.tensor_inputs:
            reader.validate_ref(reference)
        if "endpoint" in node.provenance_identity.lower():
            raise ValueError("endpoint identifier leaked into certificate graph")
    if any("endpoint" in name.lower() for name in reader.names()):
        raise ValueError("endpoint tensor leaked into certificate store")
    if tuple(program.branch_roots) != BRANCH_ORDER:
        raise ValueError("v4.1 branch order mismatch")


def _cpu_f64(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype="<f8"))


def _cpu_f32(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    # The response route is a float64 evaluation of the same frozen float32
    # checkpoint, not a separately rounded model.
    if not np.array_equal(array.astype("<f8"), np.asarray(value, dtype="<f8")):
        raise ValueError("model parameter does not round-trip through frozen float32 checkpoint")
    return array


def materialize_green_v410_full_cone_store(
    root: Path,
    name: str,
    model,
    *,
    site_layer: int,
    site_position: int,
    pat_controlled_hook_t0,
    tar_controlled_hook_t0,
    physical_direction,
    suffix_token_ids,
    contrast_coefficients,
    contrast_coefficient_rationals: Iterable[tuple[int, int]],
    selected_gates: Iterable[int],
) -> tuple[TensorStoreReader, V410GPT2Dimensions]:
    """Freeze one endpoint-free float64 full-cone tensor store.

    Model parameters are copied to float64 without mutating ``model``.  The
    controlled-hook payloads and branch-specific gate anchors must already have
    been produced by the frozen response-evaluation checkpoint.
    """
    gates = tuple(int(value) for value in selected_gates)
    pat = _cpu_f64(pat_controlled_hook_t0)
    tar = _cpu_f64(tar_controlled_hook_t0)
    if pat.ndim == 3 and pat.shape[0] == 1:
        pat = pat[0]
    if tar.ndim == 3 and tar.shape[0] == 1:
        tar = tar[0]
    if pat.ndim != 2 or tar.shape != pat.shape:
        raise ValueError("controlled-hook tensors must be matching [sequence,d_model]")
    direction = np.asarray(physical_direction, dtype="<f4").reshape(-1)
    suffix_ids = np.asarray(suffix_token_ids, dtype="<i8").reshape(-1)
    coefficients = _cpu_f64(contrast_coefficients).reshape(-1)
    rationals = tuple(
        (int(numerator), int(denominator))
        for numerator, denominator in contrast_coefficient_rationals
    )
    if suffix_ids.size != coefficients.size or suffix_ids.size < 1:
        raise ValueError("task contrast payload mismatch")
    if len(rationals) != suffix_ids.size or any(denominator <= 0 for _, denominator in rationals):
        raise ValueError("exact task contrast rational payload mismatch")
    expected_coefficients = np.asarray(
        [numerator / denominator for numerator, denominator in rationals], dtype="<f8"
    )
    if not np.array_equal(coefficients, expected_coefficients):
        raise ValueError("floating task contrast is not the exact rational oracle projection")
    dims = V410GPT2Dimensions(
        int(site_layer), int(site_position), int(pat.shape[0]), int(model.cfg.d_model),
        int(model.cfg.d_mlp), int(model.cfg.n_heads), int(model.cfg.d_head), gates,
        int(pat.shape[0] - 1), int(suffix_ids.size), rationals,
    )
    if pat.shape[1] != dims.d_model or direction.size != dims.d_model:
        raise ValueError("controlled-hook or direction width mismatch")
    if not np.array_equal(pat[dims.site_position], tar[dims.site_position]):
        raise ValueError("PAT and TAR t=0 site centers must be byte-identical")
    output_softcap = getattr(model.cfg, "output_logits_soft_cap", None)
    if output_softcap is not None and float(output_softcap) > 0:
        raise ValueError("nonlinear output softcap is unsupported")

    tensors: list[tuple[str, object]] = [
        ("physical_direction", direction),
        ("layer_norm.eps", np.asarray(model.cfg.eps, dtype="<f8")),
        ("gelu.kappa", np.asarray(math.sqrt(2.0 / math.pi), dtype="<f8")),
        ("gelu.lambda", np.asarray(0.044715, dtype="<f8")),
        ("zero.d_model", np.zeros((dims.d_model,), dtype="<f8")),
        ("PAT.controlled_hook_t0", pat),
        ("TAR.controlled_hook_t0", tar),
    ]
    for block_index in range(dims.site_layer + 1, 12):
        block = model.blocks[block_index]
        key = f"block{block_index}"
        tensors.extend([
            (f"{key}.ln1.w", _cpu_f32(block.ln1.w)),
            (f"{key}.ln1.b", _cpu_f32(block.ln1.b)),
            (f"{key}.attn.W_Q", _cpu_f32(block.attn.W_Q.permute(1, 0, 2).reshape(dims.d_model, dims.d_model))),
            (f"{key}.attn.b_Q", _cpu_f32(block.attn.b_Q.reshape(dims.d_model))),
            (f"{key}.attn.W_K", _cpu_f32(block.attn.W_K.permute(1, 0, 2).reshape(dims.d_model, dims.d_model))),
            (f"{key}.attn.b_K", _cpu_f32(block.attn.b_K.reshape(dims.d_model))),
            (f"{key}.attn.W_V", _cpu_f32(block.attn.W_V.permute(1, 0, 2).reshape(dims.d_model, dims.d_model))),
            (f"{key}.attn.b_V", _cpu_f32(block.attn.b_V.reshape(dims.d_model))),
            (f"{key}.attn.W_O", _cpu_f32(block.attn.W_O.reshape(dims.d_model, dims.d_model))),
            (f"{key}.attn.b_O", _cpu_f32(block.attn.b_O)),
            (f"{key}.ln2.w", _cpu_f32(block.ln2.w)),
            (f"{key}.ln2.b", _cpu_f32(block.ln2.b)),
            (f"{key}.mlp.W_in", _cpu_f32(block.mlp.W_in)),
            (f"{key}.mlp.b_in", _cpu_f32(block.mlp.b_in)),
            (f"{key}.mlp.W_out", _cpu_f32(block.mlp.W_out)),
            (f"{key}.mlp.b_out", _cpu_f32(block.mlp.b_out)),
        ])
    block10 = model.blocks[10]
    gate_index = __import__("torch").tensor(
        gates, dtype=__import__("torch").long, device=block10.mlp.W_in.device
    )
    selected_w_out = _cpu_f32(block10.mlp.W_out.index_select(0, gate_index))
    tensors.extend([
        ("block10.mlp.W_in_selected", _cpu_f32(block10.mlp.W_in.index_select(1, gate_index))),
        ("block10.mlp.b_in_selected", _cpu_f32(block10.mlp.b_in.index_select(0, gate_index))),
        ("block10.mlp.W_out_selected_negative", -selected_w_out),
        ("ln_final.w", _cpu_f32(model.ln_final.w)),
        ("ln_final.b", _cpu_f32(model.ln_final.b)),
        ("unembed.W_U_full", _cpu_f32(model.W_U)),
        ("unembed.b_U_full", _cpu_f32(model.b_U)),
        ("unembed.suffix_ids", suffix_ids),
        ("contrast.coefficients", coefficients),
    ])
    write_tensor_store(Path(root), name, tensors)
    reader = TensorStoreReader(Path(root) / f"{name}.json")
    return reader, dims
