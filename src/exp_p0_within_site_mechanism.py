"""Within-site, temporally eligible mechanism validation for the July audit.

The old pooled NMH AUROC compares early IO sites with late last-token sites. This
experiment instead holds position fixed at IO, analyzes each layer separately, and
uses only residual-post patches that occur before every measured Name Mover Head.
Outcomes are retained per prompt rather than averaged into one value per site.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
from validity_crossfit import CrossFitSiteReference


NM_HEADS = [(9, 9), (10, 0)]


def nmh_attention(model, tokens, io_position: int, resid_hook=None) -> np.ndarray:
    captured = {}

    def make_capture(layer, head):
        def capture(pattern, hook):
            captured[(layer, head)] = pattern[:, head, -1, io_position].detach().cpu().numpy()
        return capture

    hooks = [
        (f"blocks.{layer}.attn.hook_pattern", make_capture(layer, head))
        for layer, head in NM_HEADS
    ]
    if resid_hook is not None:
        hooks.append(resid_hook)
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)
    return np.mean(np.stack([captured[head] for head in NM_HEADS]), axis=0)


def partial_residual_spearman(overlap, outcome, restoration, context_ids):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    context = encoder.fit_transform(np.asarray(context_ids)[:, None])
    restoration_scaled = StandardScaler().fit_transform(np.asarray(restoration)[:, None])
    controls = np.column_stack([context, restoration_scaled])
    overlap_resid = overlap - Ridge(alpha=1.0).fit(controls, overlap).predict(controls)
    outcome_resid = outcome - Ridge(alpha=1.0).fit(controls, outcome).predict(controls)
    rho, p_value = spearmanr(overlap_resid, outcome_resid)
    return float(rho), float(p_value)


def run(args):
    started = time.time()
    os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY", "0")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    if args.model != "gpt2":
        raise ValueError("this first temporally matched audit is preregistered for gpt2")
    names = valid_single_token_names(model)
    rng = np.random.RandomState(args.seed)

    corrupt_fit_records, signature = sample_records(
        model, names, args.n_fit, rng, None, unique_on="corrupt_key", forbidden=set()
    )
    forbidden = {record["corrupt_key"] for record in corrupt_fit_records}
    corrupt_cal_records, signature = sample_records(
        model, names, args.n_cal, rng, signature,
        unique_on="corrupt_key", forbidden=forbidden,
    )
    eval_records, matched_fit_records, matched_cal_records = build_context_matched_records(
        model, names, rng, signature, args.n_contexts, args.n_eval_io,
        args.n_cf_fit, args.n_cf_cal,
    )
    io_position = signature[1]
    eval_clean = records_to_batch(eval_records, "clean", args.device)
    eval_corrupt = records_to_batch(eval_records, "corrupt", args.device)
    io_ids = torch.tensor([record["io_id"] for record in eval_records], device=args.device)
    s_ids = torch.tensor([record["s_id"] for record in eval_records], device=args.device)
    context_ids = np.array([record["context_id"] for record in eval_records])

    with torch.no_grad():
        clean_logit = ioi_logit_diff(model(eval_clean), io_ids, s_ids).detach().cpu().numpy()
        corrupt_logit = ioi_logit_diff(model(eval_corrupt), io_ids, s_ids).detach().cpu().numpy()
    global_logit_denom = float(clean_logit.mean() - corrupt_logit.mean())
    clean_nmh = nmh_attention(model, eval_clean, io_position)
    corrupt_nmh = nmh_attention(model, eval_corrupt, io_position)
    global_nmh_denom = float(clean_nmh.mean() - corrupt_nmh.mean())

    max_layer = min(args.max_layer, min(layer for layer, _ in NM_HEADS) - 1)
    n_layers = max_layer + 1
    cache_specs = {
        "corrupt_fit": (corrupt_fit_records, "corrupt"),
        "corrupt_cal": (corrupt_cal_records, "corrupt"),
        "matched_fit": (matched_fit_records, "clean"),
        "matched_cal": (matched_cal_records, "clean"),
        "eval_clean": (eval_records, "clean"),
    }
    caches = {}
    for label, (records, token_key) in cache_specs.items():
        print(f"Caching {label}: n={len(records)}", flush=True)
        caches[label] = cache_residuals(
            model, records_to_batch(records, token_key, args.device), n_layers, args.chunk_size
        )

    layer_results = []
    all_rows = []
    for layer in range(n_layers):
        key = (layer, io_position)
        query = caches["eval_clean"][key]
        injection = torch.tensor(query, dtype=model.cfg.dtype, device=args.device)

        def patch(act, hook, position=io_position, value=injection):
            result = act.clone()
            result[:, position, :] = value
            return result

        hook_spec = (f"blocks.{layer}.hook_resid_post", patch)
        with torch.no_grad():
            patched_logits = model.run_with_hooks(eval_corrupt, fwd_hooks=[hook_spec])
            patched_logit = ioi_logit_diff(patched_logits, io_ids, s_ids).detach().cpu().numpy()
        patched_nmh = nmh_attention(model, eval_corrupt, io_position, resid_hook=hook_spec)

        restoration = (patched_logit - corrupt_logit) / global_logit_denom
        nmh_recovery = (patched_nmh - corrupt_nmh) / global_nmh_denom

        corrupt_ref = CrossFitSiteReference(
            caches["corrupt_fit"][key], caches["corrupt_cal"][key],
            knn_k=args.knn_k, proj_rank=args.proj_rank,
        )
        corrupt_score = corrupt_ref.score(query)

        matched_overlap = np.empty(len(query))
        for context_id in range(args.n_contexts):
            fit_mask = np.array([
                record["context_id"] == context_id for record in matched_fit_records
            ])
            cal_mask = np.array([
                record["context_id"] == context_id for record in matched_cal_records
            ])
            query_mask = context_ids == context_id
            matched_ref = CrossFitSiteReference(
                caches["matched_fit"][key][fit_mask], caches["matched_cal"][key][cal_mask],
                knn_k=args.knn_k, proj_rank=args.proj_rank,
            )
            matched_overlap[query_mask] = matched_ref.score(query[query_mask])["overlap_z"]

        rho_corrupt, p_corrupt = partial_residual_spearman(
            corrupt_score["overlap_z"], nmh_recovery, restoration, context_ids
        )
        rho_matched, p_matched = partial_residual_spearman(
            matched_overlap, nmh_recovery, restoration, context_ids
        )
        layer_results.append({
            "layer": layer,
            "n": len(query),
            "mean_restoration": float(restoration.mean()),
            "mean_nmh_recovery": float(nmh_recovery.mean()),
            "mean_corrupt_overlap": float(corrupt_score["overlap_z"].mean()),
            "mean_matched_overlap": float(matched_overlap.mean()),
            "partial_spearman_corrupt_overlap_vs_nmh": rho_corrupt,
            "partial_spearman_corrupt_p": p_corrupt,
            "partial_spearman_matched_overlap_vs_nmh": rho_matched,
            "partial_spearman_matched_p": p_matched,
        })
        for index in range(len(query)):
            all_rows.append({
                "layer": layer,
                "context_id": int(context_ids[index]),
                "restoration": float(restoration[index]),
                "nmh_recovery": float(nmh_recovery[index]),
                "corrupt_overlap": float(corrupt_score["overlap_z"][index]),
                "matched_overlap": float(matched_overlap[index]),
            })
        print(
            f"layer={layer} R={restoration.mean():.3f} A={nmh_recovery.mean():.3f} "
            f"rho_corrupt={rho_corrupt:.3f} rho_matched={rho_matched:.3f}",
            flush=True,
        )

    # Pooled fixed-effect residual association across all prompt-level observations.
    layer_values = np.array([row["layer"] for row in all_rows])
    context_values = np.array([row["context_id"] for row in all_rows])
    rest_values = np.array([row["restoration"] for row in all_rows])
    nmh_values = np.array([row["nmh_recovery"] for row in all_rows])
    categorical = OneHotEncoder(sparse_output=False).fit_transform(
        np.column_stack([layer_values, context_values])
    )
    controls = np.column_stack([
        categorical, StandardScaler().fit_transform(rest_values[:, None])
    ])
    pooled = {}
    for label in ("corrupt_overlap", "matched_overlap"):
        overlap = np.array([row[label] for row in all_rows])
        overlap_resid = overlap - Ridge(alpha=1.0).fit(controls, overlap).predict(controls)
        nmh_resid = nmh_values - Ridge(alpha=1.0).fit(controls, nmh_values).predict(controls)
        rho, p_value = spearmanr(overlap_resid, nmh_resid)
        pooled[label] = {"spearman_rho": float(rho), "p_value": float(p_value)}

    output = finite_or_none({
        "experiment": "p0_within_site_temporally_eligible_mechanism",
        "model": args.model,
        "seed": args.seed,
        "site_definition": {
            "position": io_position,
            "layers": list(range(n_layers)),
            "hook": "resid_post",
            "nmh_heads": NM_HEADS,
            "eligibility": "every measured NMH head is strictly downstream of every patch",
        },
        "n_prompts_per_site": len(eval_records),
        "global_logit_denominator": global_logit_denom,
        "global_nmh_denominator": global_nmh_denom,
        "layer_results": layer_results,
        "pooled_fixed_effects": pooled,
        "rows": all_rows,
        "elapsed_seconds": time.time() - started,
        "configuration": vars(args),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exp_p0_within_site_mechanism_{args.model}_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-fit", type=int, default=256)
    parser.add_argument("--n-cal", type=int, default=128)
    parser.add_argument("--n-contexts", type=int, default=8)
    parser.add_argument("--n-eval-io", type=int, default=10)
    parser.add_argument("--n-cf-fit", type=int, default=24)
    parser.add_argument("--n-cf-cal", type=int, default=12)
    parser.add_argument("--knn-k", type=int, default=12)
    parser.add_argument("--proj-rank", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-layer", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
