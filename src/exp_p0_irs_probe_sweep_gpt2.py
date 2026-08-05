"""Probe interpolation and count robustness for GPT-2 IRS."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer

from exp_p0_irs_gpt2 import build_context_probes_and_support, run_injections
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


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    etas = [float(value) for value in args.interpolations.split(",")]
    counts = [int(value) for value in args.probe_counts.split(",")]
    if max(counts) != args.n_probes:
        raise ValueError("n_probes must equal the maximum probe count")
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
    fit_cache = cache_residuals(
        model, records_to_batch(fit_records, "clean", args.device),
        n_layers, args.cache_chunk_size,
    )
    cal_cache = cache_residuals(
        model, records_to_batch(cal_records, "clean", args.device),
        n_layers, args.cache_chunk_size,
    )

    rows = []
    layer_mechanism = []
    for layer in range(n_layers):
        key = (layer, io_position)
        centers = clean_cache[key]
        center_injections = centers[:, None, :]
        patch_center = run_injections(
            model, corrupt_tokens, io_ids, s_ids, layer, io_position,
            center_injections, args.forward_chunk_size,
        )[:, 0]
        target_center = run_injections(
            model, clean_tokens, io_ids, s_ids, layer, io_position,
            center_injections, args.forward_chunk_size,
        )[:, 0]
        center_value = torch.tensor(centers, dtype=model.cfg.dtype, device=args.device)

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
        layer_mechanism.append({
            "layer": layer,
            "mean_restoration": float(restoration.mean()),
            "mean_nmh_recovery": float(nmh_recovery.mean()),
        })

        for eta in etas:
            args.interpolation = eta
            probes, support = build_context_probes_and_support(
                centers, context_ids, fit_cache[key], fit_context_ids,
                cal_cache[key], cal_context_ids, args,
                args.seed + 10000 * (layer + 1),
            )
            endpoints = centers[:, None, :] + probes
            patch_endpoint = run_injections(
                model, corrupt_tokens, io_ids, s_ids, layer, io_position,
                endpoints, args.forward_chunk_size,
            )
            target_endpoint = run_injections(
                model, clean_tokens, io_ids, s_ids, layer, io_position,
                endpoints, args.forward_chunk_size,
            )
            patch_sig = forward_signature(patch_center, patch_endpoint, probes)
            target_sig = forward_signature(target_center, target_endpoint, probes)
            for count in counts:
                comparison = compare_signatures(
                    patch_sig[:, :count], target_sig[:, :count]
                )
                rows.append({
                    "layer": layer,
                    "interpolation": eta,
                    "n_probes": count,
                    "irs_normalized_rmse": comparison.normalized_rmse,
                    "irs_rmse": comparison.rmse,
                    "irs_cosine": comparison.mean_cosine,
                    "endpoint_accept_rate": float(np.mean(
                        support["endpoint_conformal"][:, :count] > args.support_alpha
                    )),
                })
            print(
                f"L{layer} eta={eta:g} IRS{max(counts)}="
                f"{rows[-1]['irs_normalized_rmse']:.3f} "
                f"accept={rows[-1]['endpoint_accept_rate']:.3f}",
                flush=True,
            )

    primary = {
        row["layer"]: row["irs_normalized_rmse"] for row in rows
        if row["interpolation"] == args.primary_interpolation
        and row["n_probes"] == args.primary_probe_count
    }
    setting_results = []
    nmh_by_layer = {r["layer"]: r["mean_nmh_recovery"] for r in layer_mechanism}
    for eta in etas:
        for count in counts:
            subset = [
                row for row in rows
                if row["interpolation"] == eta and row["n_probes"] == count
            ]
            subset.sort(key=lambda row: row["layer"])
            values = np.array([row["irs_normalized_rmse"] for row in subset])
            primary_values = np.array([primary[row["layer"]] for row in subset])
            nmh_values = np.array([nmh_by_layer[row["layer"]] for row in subset])
            rank_primary, _ = spearmanr(values, primary_values)
            rank_nmh, _ = spearmanr(values, nmh_values)
            setting_results.append({
                "interpolation": eta,
                "n_probes": count,
                "spearman_vs_primary_layer_ranking": float(rank_primary),
                "spearman_vs_nmh": float(rank_nmh),
                "min_endpoint_accept_rate": float(np.min([
                    row["endpoint_accept_rate"] for row in subset
                ])),
            })

    output = finite_or_none({
        "experiment": "p0_irs_probe_sweep_gpt2",
        "model": args.model,
        "seed": args.seed,
        "layer_mechanism": layer_mechanism,
        "rows": rows,
        "setting_results": setting_results,
        "elapsed_seconds": time.time() - started,
        "configuration": vars(args),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exp_p0_irs_probe_sweep_{args.model}_seed{args.seed}.json"
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
    parser.add_argument("--n-probes", type=int, default=16)
    parser.add_argument("--probe-counts", default="2,4,8,16")
    parser.add_argument("--interpolations", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--primary-interpolation", type=float, default=0.25)
    parser.add_argument("--primary-probe-count", type=int, default=8)
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
