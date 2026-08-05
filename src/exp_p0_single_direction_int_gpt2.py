"""Single clean--corrupt response direction baseline for GPT-2 IRS.

This computes the cross-difference induced by the one mediator displacement used
by the prompt contrast.  It is the fair single-direction interaction baseline
against which multi-direction, target-admissible IRS must add value.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from exp_p0_irs_gpt2 import run_injections
from exp_p0_reference_crossfit import (
    SEED,
    build_context_matched_records,
    cache_residuals,
    finite_or_none,
    ioi_logit_diff,
    records_to_batch,
    sample_records,
    valid_single_token_names,
)
from exp_p0_within_site_mechanism import NM_HEADS, nmh_attention
from interventional_response import compare_signatures, forward_signature
from validity_crossfit import CrossFitSiteReference


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    names = valid_single_token_names(model)
    rng = np.random.RandomState(args.seed)
    initial, signature = sample_records(
        model, names, max(args.n_contexts, 8), rng, None,
        unique_on="corrupt_key", forbidden=set(),
    )
    del initial
    eval_records, fit_records, cal_records = build_context_matched_records(
        model, names, rng, signature, args.n_contexts, args.n_eval_io,
        args.n_cf_fit, args.n_cf_cal,
    )
    io_position = signature[1]
    clean_tokens = records_to_batch(eval_records, "clean", args.device)
    corrupt_tokens = records_to_batch(eval_records, "corrupt", args.device)
    io_ids = torch.tensor([r["io_id"] for r in eval_records], device=args.device)
    s_ids = torch.tensor([r["s_id"] for r in eval_records], device=args.device)
    context_ids = np.array([r["context_id"] for r in eval_records])
    fit_context_ids = np.array([r["context_id"] for r in fit_records])
    cal_context_ids = np.array([r["context_id"] for r in cal_records])

    with torch.no_grad():
        clean_logit = ioi_logit_diff(model(clean_tokens), io_ids, s_ids).float().cpu().numpy()
        corrupt_logit = ioi_logit_diff(model(corrupt_tokens), io_ids, s_ids).float().cpu().numpy()
    logit_denom = float(clean_logit.mean() - corrupt_logit.mean())
    clean_nmh = nmh_attention(model, clean_tokens, io_position)
    corrupt_nmh = nmh_attention(model, corrupt_tokens, io_position)
    nmh_denom = float(clean_nmh.mean() - corrupt_nmh.mean())

    n_layers = args.max_layer + 1
    clean_cache = cache_residuals(model, clean_tokens, n_layers, args.cache_chunk_size)
    corrupt_cache = cache_residuals(model, corrupt_tokens, n_layers, args.cache_chunk_size)
    fit_cache = cache_residuals(
        model, records_to_batch(fit_records, "clean", args.device),
        n_layers, args.cache_chunk_size,
    )
    cal_cache = cache_residuals(
        model, records_to_batch(cal_records, "clean", args.device),
        n_layers, args.cache_chunk_size,
    )

    layer_results = []
    prompt_rows = []
    for layer in range(n_layers):
        key = (layer, io_position)
        clean_h = clean_cache[key]
        corrupt_h = corrupt_cache[key]
        deltas = (corrupt_h - clean_h)[:, None, :]
        lengths = np.linalg.norm(deltas[:, 0, :], axis=1)
        if np.any(lengths <= 0):
            raise RuntimeError(f"zero clean-corrupt displacement at layer {layer}")
        clean_center = run_injections(
            model, clean_tokens, io_ids, s_ids, layer, io_position,
            clean_h[:, None, :], args.forward_chunk_size,
        )[:, 0]
        patch_center = run_injections(
            model, corrupt_tokens, io_ids, s_ids, layer, io_position,
            clean_h[:, None, :], args.forward_chunk_size,
        )[:, 0]
        clean_endpoint = run_injections(
            model, clean_tokens, io_ids, s_ids, layer, io_position,
            corrupt_h[:, None, :], args.forward_chunk_size,
        )
        patch_endpoint = run_injections(
            model, corrupt_tokens, io_ids, s_ids, layer, io_position,
            corrupt_h[:, None, :], args.forward_chunk_size,
        )
        target_signature = forward_signature(clean_center, clean_endpoint, deltas)
        patch_signature = forward_signature(patch_center, patch_endpoint, deltas)
        comparison = compare_signatures(patch_signature, target_signature)
        raw_cross_difference = (
            (patch_endpoint[:, 0] - patch_center)
            - (clean_endpoint[:, 0] - clean_center)
        )

        endpoint_p = np.empty(len(clean_h), dtype=np.float64)
        for context_id in np.unique(context_ids):
            q_mask = context_ids == context_id
            scorer = CrossFitSiteReference(
                fit_cache[key][fit_context_ids == context_id],
                cal_cache[key][cal_context_ids == context_id],
                knn_k=args.knn_k,
                proj_rank=args.proj_rank,
            )
            endpoint_p[q_mask] = scorer.score(corrupt_h[q_mask])["overlap_conformal"]

        center_value = torch.tensor(clean_h, dtype=model.cfg.dtype, device=args.device)

        def center_patch(act, hook, value=center_value):
            result = act.clone()
            result[:, io_position, :] = value
            return result

        patched_nmh = nmh_attention(
            model, corrupt_tokens, io_position,
            resid_hook=(f"blocks.{layer}.hook_resid_post", center_patch),
        )
        restoration = (patch_center - corrupt_logit) / logit_denom
        nmh_recovery = (patched_nmh - corrupt_nmh) / nmh_denom
        layer_results.append({
            "layer": layer,
            "mean_restoration": float(restoration.mean()),
            "mean_nmh_recovery": float(nmh_recovery.mean()),
            "single_direction_normalized_rmse": comparison.normalized_rmse,
            "single_direction_rmse": comparison.rmse,
            "mean_abs_raw_cross_difference": float(np.mean(np.abs(raw_cross_difference))),
            "mean_clean_corrupt_activation_distance": float(lengths.mean()),
            "mean_endpoint_conformal": float(endpoint_p.mean()),
            "endpoint_accept_rate": float(np.mean(endpoint_p > args.support_alpha)),
        })
        for i in range(len(clean_h)):
            prompt_rows.append({
                "layer": layer,
                "context_id": int(context_ids[i]),
                "restoration": float(restoration[i]),
                "nmh_recovery": float(nmh_recovery[i]),
                "single_direction_rmse": float(comparison.per_item_rmse[i]),
                "single_direction_normalized_rmse": float(
                    comparison.per_item_normalized_rmse[i]
                ),
                "abs_raw_cross_difference": float(abs(raw_cross_difference[i])),
                "activation_distance": float(lengths[i]),
                "endpoint_conformal": float(endpoint_p[i]),
            })
        print(
            f"L{layer} R={restoration.mean():.3f} A={nmh_recovery.mean():.3f} "
            f"single_dir={comparison.normalized_rmse:.3f} "
            f"endpoint_accept={np.mean(endpoint_p > args.support_alpha):.3f}",
            flush=True,
        )

    output = finite_or_none({
        "experiment": "p0_single_clean_corrupt_interaction_direction_gpt2",
        "model": args.model,
        "seed": args.seed,
        "site": {"position": io_position, "layers": list(range(n_layers))},
        "layer_results": layer_results,
        "prompt_rows": prompt_rows,
        "elapsed_seconds": time.time() - started,
        "configuration": vars(args),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exp_p0_single_direction_{args.model}_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    parser.add_argument("--n-contexts", type=int, default=8)
    parser.add_argument("--n-eval-io", type=int, default=10)
    parser.add_argument("--n-cf-fit", type=int, default=48)
    parser.add_argument("--n-cf-cal", type=int, default=32)
    parser.add_argument("--knn-k", type=int, default=12)
    parser.add_argument("--proj-rank", type=int, default=32)
    parser.add_argument("--support-alpha", type=float, default=0.1)
    parser.add_argument("--cache-chunk-size", type=int, default=64)
    parser.add_argument("--forward-chunk-size", type=int, default=64)
    parser.add_argument("--max-layer", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
