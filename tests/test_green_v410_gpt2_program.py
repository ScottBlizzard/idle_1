from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green_bridge_v400_gpt2_program import execute_tensor_program_numpy
from green_v410_gpt2_program import (
    GRAPH_SEMANTICS_ID,
    build_green_v410_full_cone_program,
    materialize_green_v410_full_cone_store,
    validate_green_v410_full_cone_program,
)


GATES = (1, 4)


def _tensor(rng: np.random.Generator, shape, scale=0.08):
    return torch.tensor(rng.normal(0.0, scale, shape), dtype=torch.float32)


def _fake_model(seed=410):
    rng = np.random.default_rng(seed)
    d, m, heads, head, vocab = 4, 6, 2, 2, 7
    blocks = []
    for _ in range(12):
        blocks.append(SimpleNamespace(
            ln1=SimpleNamespace(w=_tensor(rng, (d,), 0.04) + 1, b=_tensor(rng, (d,), 0.02)),
            ln2=SimpleNamespace(w=_tensor(rng, (d,), 0.04) + 1, b=_tensor(rng, (d,), 0.02)),
            attn=SimpleNamespace(
                W_Q=_tensor(rng, (heads, d, head)),
                b_Q=_tensor(rng, (heads, head), 0.02),
                W_K=_tensor(rng, (heads, d, head)),
                b_K=_tensor(rng, (heads, head), 0.02),
                W_V=_tensor(rng, (heads, d, head)),
                b_V=_tensor(rng, (heads, head), 0.02),
                W_O=_tensor(rng, (heads, head, d)),
                b_O=_tensor(rng, (d,), 0.02),
            ),
            mlp=SimpleNamespace(
                W_in=_tensor(rng, (d, m)), b_in=_tensor(rng, (m,), 0.02),
                W_out=_tensor(rng, (m, d)), b_out=_tensor(rng, (d,), 0.02),
            ),
        ))
    return SimpleNamespace(
        cfg=SimpleNamespace(
            d_model=d, d_mlp=m, n_heads=heads, d_head=head, eps=1e-5,
            output_logits_soft_cap=None,
        ),
        blocks=blocks,
        ln_final=SimpleNamespace(w=_tensor(rng, (d,), 0.04) + 1, b=_tensor(rng, (d,), 0.02)),
        W_U=_tensor(rng, (d, vocab)),
        b_U=_tensor(rng, (vocab,), 0.02),
    )


def _np(value):
    return value.detach().cpu().numpy()


def _layer_norm(x, layer, eps):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps) * _np(layer.w) + _np(layer.b)


def _block(x, block, eps, *, anchor=None):
    heads, head = block.attn.W_Q.shape[0], block.attn.W_Q.shape[2]
    d = x.shape[-1]
    ln1 = _layer_norm(x, block.ln1, eps)
    qkv = []
    for name in ("Q", "K", "V"):
        weight = _np(getattr(block.attn, f"W_{name}")).transpose(1, 0, 2).reshape(d, d)
        bias = _np(getattr(block.attn, f"b_{name}")).reshape(d)
        qkv.append(ln1 @ weight + bias)
    q, k, v = (item.reshape(item.shape[0], heads, head) for item in qkv)
    scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(np.asarray(head, dtype=x.dtype))
    scores = np.where(np.triu(np.ones((x.shape[0], x.shape[0]), dtype=bool), 1)[None], -np.inf, scores)
    exps = np.exp(scores - scores[:, :, :1])
    pattern = exps / exps.sum(axis=-1, keepdims=True)
    attended = np.einsum("hqk,khd->qhd", pattern, v).reshape(x.shape[0], d)
    w_o = _np(block.attn.W_O).reshape(d, d)
    mid = x + attended @ w_o + _np(block.attn.b_O)
    ln2 = _layer_norm(mid, block.ln2, eps)
    pre = ln2 @ _np(block.mlp.W_in) + _np(block.mlp.b_in)
    post = 0.5 * pre * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (pre + 0.044715 * pre ** 3)))
    live_post = post.copy()
    if anchor is not None:
        post[-1, list(GATES)] = anchor
    out = post @ _np(block.mlp.W_out) + _np(block.mlp.b_out)
    return mid + out, live_post


def _forward(model, base, site_layer, site_position, direction, t, anchor=None):
    x = base.copy()
    x[site_position] += t * direction
    live10 = None
    for index in range(site_layer + 1, 12):
        x, post = _block(
            x, model.blocks[index], model.cfg.eps,
            anchor=anchor if index == 10 else None,
        )
        if index == 10:
            live10 = post
    x = _layer_norm(x, model.ln_final, model.cfg.eps)
    logits = x[-1] @ _np(model.W_U) + _np(model.b_U)
    return logits[0] - logits[1], live10


def _case(tmp_path: Path, site_layer: int):
    model = _fake_model(site_layer + 100)
    rng = np.random.default_rng(site_layer + 900)
    s, d, position = 4, model.cfg.d_model, 1
    center = rng.normal(size=d).astype(np.float64)
    pat = rng.normal(size=(s, d)).astype(np.float64)
    tar = rng.normal(size=(s, d)).astype(np.float64)
    pat[position] = center
    tar[position] = center
    raw = rng.normal(size=d)
    direction = (raw / np.linalg.norm(raw) * 0.001).astype(np.float32)
    _, pat_live = _forward(model, pat, site_layer, position, direction, 0.0)
    _, tar_live = _forward(model, tar, site_layer, position, direction, 0.0)
    reader, dims = materialize_green_v410_full_cone_store(
        tmp_path, f"case_l{site_layer}", model,
        site_layer=site_layer,
        site_position=position,
        pat_controlled_hook_t0=pat,
        tar_controlled_hook_t0=tar,
        physical_direction=direction,
        suffix_token_ids=np.asarray([0, 1], dtype=np.int64),
        contrast_coefficients=np.asarray([1.0, -1.0], dtype=np.float64),
        contrast_coefficient_rationals=((1, 1), (-1, 1)),
        selected_gates=GATES,
    )
    program = build_green_v410_full_cone_program(reader, "a" * 64, dims)
    return model, pat, tar, direction.astype(np.float64), reader, dims, program, pat_live, tar_live


@pytest.mark.parametrize("site_layer", [0, 4, 8])
def test_full_cone_numpy_parity_and_matched_bypass(tmp_path, site_layer):
    model, pat, tar, direction, reader, dims, program, pat_live, tar_live = _case(tmp_path, site_layer)
    anchors = {
        "PAT": pat_live[-1, list(GATES)],
        "TAR": tar_live[-1, list(GATES)],
    }
    for t in (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25):
        actual = execute_tensor_program_numpy(program, reader, t)
        expected = {}
        for condition, base in (("PAT", pat), ("TAR", tar)):
            expected[f"{condition}_J"], _ = _forward(
                model, base, site_layer, dims.site_position, direction, t,
            )
            expected[f"{condition}_B"], _ = _forward(
                model, base, site_layer, dims.site_position, direction, t,
                anchor=anchors[condition],
            )
        expected["output"] = (
            expected["PAT_J"] - expected["PAT_B"]
            - expected["TAR_J"] + expected["TAR_B"]
        )
        for name, value in expected.items():
            assert np.allclose(actual[name], value, rtol=2e-12, atol=2e-12), (name, t)
        if t == 0.0:
            assert actual["PAT_J"] == pytest.approx(actual["PAT_B"], abs=2e-12)
            assert actual["TAR_J"] == pytest.approx(actual["TAR_B"], abs=2e-12)


def test_full_cone_has_exact_layer_and_token_dependency_closure(tmp_path):
    _, _, _, _, reader, dims, program, _, _ = _case(tmp_path, 4)
    assert program.resource_formula["graph_semantics_id"] == GRAPH_SEMANTICS_ID
    scatter = [node for node in program.nodes if node.kernel_id == "affine_scatter.v1"]
    assert {tuple(node.exact_attrs["dependency_mask_spec"]["axis0_indices"]) for node in scatter} == {(1,)}
    first_ln1 = next(
        node for node in program.nodes
        if node.provenance_identity == "PAT.SHARED.block5.ln1"
    )
    first_attention = next(
        node for node in program.nodes
        if node.provenance_identity == "PAT.SHARED.block5.attn.pattern_value"
    )
    assert first_ln1.exact_attrs["dependency_mask_spec"]["axis0_indices"] == [1]
    assert first_attention.exact_attrs["dependency_mask_spec"]["axis0_indices"] == [1, 2, 3]
    provenance = "\n".join(node.provenance_identity for node in program.nodes)
    for block in range(5, 12):
        assert f"block{block}." in provenance
    assert "selected_live_minus_anchor" in provenance
    assert "negative_selected_out_delta" in provenance
    assert all("endpoint" not in name.lower() for name in reader.names())


def test_old_graph_semantics_is_rejected(tmp_path):
    _, _, _, _, reader, dims, program, _, _ = _case(tmp_path, 8)
    object.__setattr__(
        program,
        "resource_formula",
        deepcopy(program.resource_formula) | {
            "graph_semantics_id": "GREEN_V400_FIXED_BLOCK8_FINAL_TOKEN"
        },
    )
    with pytest.raises(ValueError, match="old or unknown"):
        validate_green_v410_full_cone_program(program, reader, dims)


def test_mpfr_full_cone_t0_encloses_exact_matched_bypass_identity(tmp_path):
    pytest.importorskip("gmpy2")
    from green_bridge_v400_interval import Interval
    from green_bridge_v400_mpfr_tensor_executor import (
        execute_tensor_program_mpfr,
        tensor_program_required_axis0_rows,
    )

    _, _, _, _, reader, dims, program, _, _ = _case(tmp_path, 8)
    live_rows = tensor_program_required_axis0_rows(program)
    first_attention = next(
        node for node in program.nodes
        if node.provenance_identity == "PAT.SHARED.block9.attn.pattern_value"
    )
    # The final scalar needs every causal key/value history row, and the new
    # executor must no longer collapse this attention to one query row.
    assert live_rows[first_attention.semantic_id] == tuple(range(dims.sequence_length))
    result = execute_tensor_program_mpfr(
        program, reader, Interval.point(0, 80), sparse_axis0_execution=True,
    )
    psi = result["output"].value
    assert psi.lower <= 0 <= psi.upper


def test_mpfr_full_cone_preloaded_constants_are_hash_closed(tmp_path):
    pytest.importorskip("gmpy2")
    from green_bridge_v400_interval import Interval
    from green_bridge_v400_mpfr_tensor_executor import (
        execute_tensor_program_mpfr,
        jet_exact_payload,
        preload_tensor_program_arrays,
    )

    _, _, _, _, reader, _, program, _, _ = _case(tmp_path, 8)
    closure = preload_tensor_program_arrays(program, reader)
    domain = Interval.from_bounds("-1/4", "1/4", 80)
    reference = execute_tensor_program_mpfr(
        program, reader, domain, sparse_axis0_execution=True,
    )
    cached = execute_tensor_program_mpfr(
        program, reader, domain, sparse_axis0_execution=True,
        return_runtime_metrics=True, preloaded_tensors=closure,
    )
    assert jet_exact_payload(cached["output"]) == jet_exact_payload(reference["output"])
    assert cached["runtime_metrics"]["tensor_store_fallback_reads"] == 0
    assert cached["runtime_metrics"]["preloaded_tensor_reads"] > 0
    with pytest.raises(TypeError, match="immutable"):
        closure.clear()


def test_mpfr_task_contrast_uses_exact_nondyadic_rationals():
    gmpy2 = pytest.importorskip("gmpy2")
    from green_bridge_v400_interval import Interval
    from green_bridge_v400_interval_jet import constant_jet
    from green_bridge_v400_mpfr_tensor_executor import _final_contrast_reference

    result = _final_contrast_reference(
        [constant_jet(Interval.point(1, 128))],
        np.asarray([[1.0]], dtype="<f4"),
        np.asarray([0.0], dtype="<f4"),
        np.asarray([0], dtype="<i8"),
        np.asarray([1.0 / 3.0], dtype="<f8"),
        [{"numerator": 1, "denominator": 3}],
    )
    exact = gmpy2.mpq(1, 3)
    dyadic = gmpy2.mpq(*(float(1.0 / 3.0).as_integer_ratio()))
    assert result.value.lower <= exact <= result.value.upper
    assert not (result.value.lower <= dyadic <= result.value.upper)
