"""P0 IRS validation on the temporally eligible GPT-2 IOI protocol.

For each IO-position residual-stream site, the clean target computation and the
patched-corrupt computation receive exactly the same activation center and the
same context-matched reference-chord probes.  IRS measures whether their
downstream logit responses agree despite zero-order behavioral restoration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer

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
from interventional_response import (
    compare_signatures,
    forward_signature,
    reference_chord_probes,
)
from validity_crossfit import CrossFitSiteReference


@torch.no_grad()
def run_injections(
    model,
    tokens: torch.Tensor,
    io_ids: torch.Tensor,
    s_ids: torch.Tensor,
    layer: int,
    position: int,
    injections: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Run [prompt, probe, d_model] injections and return prompt/probe logit diff."""
    injections = np.asarray(injections)
    if injections.ndim != 3 or injections.shape[0] != len(tokens):
        raise ValueError("injections must have shape [prompt, probe, d_model]")
    n_prompts, n_probes, width = injections.shape
    repeated_tokens = tokens.repeat_interleave(n_probes, dim=0)
    repeated_io = io_ids.repeat_interleave(n_probes)
    repeated_s = s_ids.repeat_interleave(n_probes)
    flat_injections = torch.tensor(
        injections.reshape(-1, width), dtype=model.cfg.dtype, device=tokens.device
    )
    outputs = []
    for start in range(0, len(repeated_tokens), chunk_size):
        stop = min(start + chunk_size, len(repeated_tokens))
        values = flat_injections[start:stop]

        def patch(act, hook, value=values, pos=position):
            result = act.clone()
            result[:, pos, :] = value
            return result

        logits = model.run_with_hooks(
            repeated_tokens[start:stop],
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch)],
        )
        outputs.append(
            ioi_logit_diff(logits, repeated_io[start:stop], repeated_s[start:stop])
            .float().cpu().numpy()
        )
    return np.concatenate(outputs).reshape(n_prompts, n_probes)


def build_context_probes_and_support(
    centers: np.ndarray,
    context_ids: np.ndarray,
    fit_activations: np.ndarray,
    fit_context_ids: np.ndarray,
    cal_activations: np.ndarray,
    cal_context_ids: np.ndarray,
    args,
    seed: int,
) -> tuple[np.ndarray, dict]:
    probes = np.empty(
        (len(centers), args.n_probes, centers.shape[1]), dtype=np.float64
    )
    center_p = np.empty(len(centers), dtype=np.float64)
    endpoint_p = np.empty((len(centers), args.n_probes), dtype=np.float64)
    for context_id in np.unique(context_ids):
        query_mask = context_ids == context_id
        fit_mask = fit_context_ids == context_id
        cal_mask = cal_context_ids == context_id
        context_centers = centers[query_mask]
        context_fit = fit_activations[fit_mask]
        scorer = CrossFitSiteReference(
            context_fit,
            cal_activations[cal_mask],
            knn_k=args.knn_k,
            proj_rank=args.proj_rank,
        )
        context_probes = reference_chord_probes(
            context_centers,
            context_fit,
            args.n_probes,
            args.interpolation,
            args.nearest_pool,
            np.random.RandomState(seed + int(context_id)),
        )
        endpoints = context_centers[:, None, :] + context_probes
        probes[query_mask] = context_probes
        center_p[query_mask] = scorer.score(context_centers)["overlap_conformal"]
        endpoint_p[query_mask] = scorer.score(
            endpoints.reshape(-1, centers.shape[1])
        )["overlap_conformal"].reshape(len(context_centers), args.n_probes)
    return probes, {
        "center_conformal": center_p,
        "endpoint_conformal": endpoint_p,
    }


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    if args.model != "gpt2":
        raise ValueError("P0 IRS protocol is frozen for gpt2 before extension")
    names = valid_single_token_names(model)
    rng = np.random.RandomState(args.seed)

    initial_records, signature = sample_records(
        model, names, max(args.n_contexts, 8), rng, None,
        unique_on="corrupt_key", forbidden=set(),
    )
    # initial_records are used only to freeze token-position signature.
    del initial_records
    eval_records, matched_fit_records, matched_cal_records = build_context_matched_records(
        model,
        names,
        rng,
        signature,
        args.n_contexts,
        args.n_eval_io,
        args.n_cf_fit,
        args.n_cf_cal,
    )
    io_position = signature[1]
    eval_clean = records_to_batch(eval_records, "clean", args.device)
    eval_corrupt = records_to_batch(eval_records, "corrupt", args.device)
    io_ids = torch.tensor([r["io_id"] for r in eval_records], device=args.device)
    s_ids = torch.tensor([r["s_id"] for r in eval_records], device=args.device)
    context_ids = np.array([r["context_id"] for r in eval_records])
    fit_context_ids = np.array([r["context_id"] for r in matched_fit_records])
    cal_context_ids = np.array([r["context_id"] for r in matched_cal_records])

    with torch.no_grad():
        clean_logit = ioi_logit_diff(model(eval_clean), io_ids, s_ids).float().cpu().numpy()
        corrupt_logit = ioi_logit_diff(model(eval_corrupt), io_ids, s_ids).float().cpu().numpy()
    logit_denom = float(clean_logit.mean() - corrupt_logit.mean())
    clean_nmh = nmh_attention(model, eval_clean, io_position)
    corrupt_nmh = nmh_attention(model, eval_corrupt, io_position)
    nmh_denom = float(clean_nmh.mean() - corrupt_nmh.mean())

    n_layers = args.max_layer + 1
    print("Caching clean target and context-matched references", flush=True)
    clean_cache = cache_residuals(model, eval_clean, n_layers, args.cache_chunk_size)
    fit_cache = cache_residuals(
        model,
        records_to_batch(matched_fit_records, "clean", args.device),
        n_layers,
        args.cache_chunk_size,
    )
    cal_cache = cache_residuals(
        model,
        records_to_batch(matched_cal_records, "clean", args.device),
        n_layers,
        args.cache_chunk_size,
    )

    layer_results = []
    prompt_rows = []
    for layer in range(n_layers):
        key = (layer, io_position)
        centers = clean_cache[key]
        center_injections = centers[:, None, :]
        patched_center = run_injections(
            model, eval_corrupt, io_ids, s_ids, layer, io_position,
            center_injections, args.forward_chunk_size,
        )[:, 0]
        target_center = run_injections(
            model, eval_clean, io_ids, s_ids, layer, io_position,
            center_injections, args.forward_chunk_size,
        )[:, 0]

        probes, support = build_context_probes_and_support(
            centers,
            context_ids,
            fit_cache[key],
            fit_context_ids,
            cal_cache[key],
            cal_context_ids,
            args,
            args.seed + 10000 * (layer + 1),
        )
        endpoints = centers[:, None, :] + probes
        patched_endpoint = run_injections(
            model, eval_corrupt, io_ids, s_ids, layer, io_position,
            endpoints, args.forward_chunk_size,
        )
        target_endpoint = run_injections(
            model, eval_clean, io_ids, s_ids, layer, io_position,
            endpoints, args.forward_chunk_size,
        )
        patched_signature = forward_signature(patched_center, patched_endpoint, probes)
        target_signature = forward_signature(target_center, target_endpoint, probes)
        comparison = compare_signatures(patched_signature, target_signature)

        def center_patch(act, hook, value=torch.tensor(
            centers, dtype=model.cfg.dtype, device=args.device
        )):
            result = act.clone()
            result[:, io_position, :] = value
            return result

        patched_nmh = nmh_attention(
            model,
            eval_corrupt,
            io_position,
            resid_hook=(f"blocks.{layer}.hook_resid_post", center_patch),
        )
        restoration = (patched_center - corrupt_logit) / logit_denom
        nmh_recovery = (patched_nmh - corrupt_nmh) / nmh_denom
        endpoint_accept = support["endpoint_conformal"] > args.support_alpha
        rho, p_value = spearmanr(
            comparison.per_item_normalized_rmse,
            nmh_recovery,
        )
        layer_result = {
            "layer": layer,
            "n_prompts": len(centers),
            "n_probes": args.n_probes,
            "mean_restoration": float(restoration.mean()),
            "mean_nmh_recovery": float(nmh_recovery.mean()),
            "irs_rmse": comparison.rmse,
            "irs_normalized_rmse": comparison.normalized_rmse,
            "irs_mean_cosine": comparison.mean_cosine,
            "mean_center_conformal": float(support["center_conformal"].mean()),
            "center_accept_rate": float(np.mean(
                support["center_conformal"] > args.support_alpha
            )),
            "mean_endpoint_conformal": float(support["endpoint_conformal"].mean()),
            "endpoint_accept_rate": float(endpoint_accept.mean()),
            "prompt_spearman_irs_vs_nmh": float(rho),
            "prompt_spearman_p": float(p_value),
            "target_center_replay_max_abs_error": float(
                np.max(np.abs(target_center - clean_logit))
            ),
        }
        layer_results.append(layer_result)
        for i in range(len(centers)):
            prompt_rows.append({
                "layer": layer,
                "context_id": int(context_ids[i]),
                "restoration": float(restoration[i]),
                "nmh_recovery": float(nmh_recovery[i]),
                "irs_rmse": float(comparison.per_item_rmse[i]),
                "irs_normalized_rmse": float(
                    comparison.per_item_normalized_rmse[i]
                ),
                "center_conformal": float(support["center_conformal"][i]),
                "endpoint_accept_rate": float(endpoint_accept[i].mean()),
            })
        print(
            f"L{layer} R={restoration.mean():.3f} A={nmh_recovery.mean():.3f} "
            f"IRS={comparison.normalized_rmse:.3f} "
            f"endpoint_accept={endpoint_accept.mean():.3f}",
            flush=True,
        )

    layer_r = np.array([row["mean_restoration"] for row in layer_results])
    layer_a = np.array([row["mean_nmh_recovery"] for row in layer_results])
    layer_irs = np.array([row["irs_normalized_rmse"] for row in layer_results])
    rho_layer, p_layer = spearmanr(layer_irs, layer_a)
    stable = [
        row for row in layer_results
        if row["mean_restoration"] > 0.8 and row["mean_nmh_recovery"] < 0.5
    ]
    output = finite_or_none({
        "experiment": "p0_interventional_response_signature_gpt2",
        "model": args.model,
        "seed": args.seed,
        "site": {
            "position": io_position,
            "layers": list(range(n_layers)),
            "hook": "resid_post",
            "nmh_heads": NM_HEADS,
            "eligibility": "all measured NMH heads strictly downstream",
        },
        "probe_law": {
            "type": "context-matched clean-reference forward chord",
            "interpolation": args.interpolation,
            "nearest_pool": args.nearest_pool,
            "paired_across_target_and_patch": True,
        },
        "global_logit_denominator": logit_denom,
        "global_nmh_denominator": nmh_denom,
        "layer_results": layer_results,
        "prompt_rows": prompt_rows,
        "layer_spearman_irs_vs_nmh": {
            "rho": float(rho_layer), "p_value": float(p_layer)
        },
        "high_R_low_A_layers": [row["layer"] for row in stable],
        "elapsed_seconds": time.time() - started,
        "configuration": vars(args),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exp_p0_irs_{args.model}_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    parser.add_argument("--n-contexts", type=int, default=4)
    parser.add_argument("--n-eval-io", type=int, default=4)
    parser.add_argument("--n-cf-fit", type=int, default=48)
    parser.add_argument("--n-cf-cal", type=int, default=32)
    parser.add_argument("--n-probes", type=int, default=8)
    parser.add_argument("--interpolation", type=float, default=0.25)
    parser.add_argument("--nearest-pool", type=int, default=12)
    parser.add_argument("--knn-k", type=int, default=12)
    parser.add_argument("--proj-rank", type=int, default=32)
    parser.add_argument("--support-alpha", type=float, default=0.1)
    parser.add_argument("--cache-chunk-size", type=int, default=64)
    parser.add_argument("--forward-chunk-size", type=int, default=64)
    parser.add_argument("--max-layer", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
