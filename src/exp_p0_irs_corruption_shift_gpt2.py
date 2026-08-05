"""Corruption-shift stress test for IRS versus a single mediator direction.

The clean target computation and clean-reference IRS probe law are frozen.  We
then compare two downstream contexts that differ only in the name replacing the
IO token: the conventional duplicate-S corruption and an independent third-name
(pABC-style) corruption.  A single-direction score is recomputed from each
clean--corrupt displacement, whereas IRS uses the same admissible probe law.
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

from exp_p0_irs_gpt2 import (
    build_context_probes_and_support,
    run_injections,
)
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
from exp_p0_within_site_mechanism import nmh_attention
from interventional_response import compare_signatures, forward_signature
from validity_crossfit import CrossFitSiteReference


def make_third_name_corruption(
    model,
    clean_tokens: torch.Tensor,
    records: list[dict],
    names: list[str],
    io_position: int,
    rng: np.random.RandomState,
) -> tuple[torch.Tensor, list[str]]:
    """Replace the clean IO occurrence with a distinct, single-token third name."""
    result = clean_tokens.clone()
    third_names: list[str] = []
    for row, record in enumerate(records):
        io_name, s_name = record["clean_key"][:2]
        candidates = [name for name in names if name not in {io_name, s_name}]
        third_name = str(rng.choice(candidates))
        third_id = int(model.to_tokens(
            f" {third_name}", prepend_bos=False
        )[0, 0])
        result[row, io_position] = third_id
        third_names.append(third_name)
    return result, third_names


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    if args.model != "gpt2":
        raise ValueError("corruption-shift P0 is frozen for gpt2")
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
    duplicate_s = records_to_batch(eval_records, "corrupt", args.device)
    pabc, third_names = make_third_name_corruption(
        model, clean_tokens, eval_records, names, io_position,
        np.random.RandomState(args.seed + 991),
    )
    changed = (clean_tokens != duplicate_s).nonzero(as_tuple=False)
    if len(changed) != len(eval_records) or not torch.all(changed[:, 1] == io_position):
        raise RuntimeError("duplicate-S corruption is not a one-token IO replacement")

    corruptions = {"duplicate_s": duplicate_s, "third_name_pabc": pabc}
    io_ids = torch.tensor([r["io_id"] for r in eval_records], device=args.device)
    s_ids = torch.tensor([r["s_id"] for r in eval_records], device=args.device)
    context_ids = np.array([r["context_id"] for r in eval_records])
    fit_context_ids = np.array([r["context_id"] for r in fit_records])
    cal_context_ids = np.array([r["context_id"] for r in cal_records])

    with torch.no_grad():
        clean_logit = ioi_logit_diff(
            model(clean_tokens), io_ids, s_ids
        ).float().cpu().numpy()
    clean_nmh = nmh_attention(model, clean_tokens, io_position)
    baselines = {}
    for label, tokens in corruptions.items():
        with torch.no_grad():
            corrupt_logit = ioi_logit_diff(
                model(tokens), io_ids, s_ids
            ).float().cpu().numpy()
        corrupt_nmh = nmh_attention(model, tokens, io_position)
        baselines[label] = {
            "logit": corrupt_logit,
            "nmh": corrupt_nmh,
            "logit_denominator": float(clean_logit.mean() - corrupt_logit.mean()),
            "nmh_denominator": float(clean_nmh.mean() - corrupt_nmh.mean()),
        }
        if abs(baselines[label]["logit_denominator"]) < 1e-6:
            raise RuntimeError(f"degenerate logit denominator for {label}")
        if abs(baselines[label]["nmh_denominator"]) < 1e-6:
            raise RuntimeError(f"degenerate NMH denominator for {label}")

    n_layers = args.max_layer + 1
    clean_cache = cache_residuals(
        model, clean_tokens, n_layers, args.cache_chunk_size
    )
    corrupt_cache = {
        label: cache_residuals(model, tokens, n_layers, args.cache_chunk_size)
        for label, tokens in corruptions.items()
    }
    fit_cache = cache_residuals(
        model, records_to_batch(fit_records, "clean", args.device),
        n_layers, args.cache_chunk_size,
    )
    cal_cache = cache_residuals(
        model, records_to_batch(cal_records, "clean", args.device),
        n_layers, args.cache_chunk_size,
    )

    rows = []
    for layer in range(n_layers):
        key = (layer, io_position)
        centers = clean_cache[key]
        center_injections = centers[:, None, :]
        target_center = run_injections(
            model, clean_tokens, io_ids, s_ids, layer, io_position,
            center_injections, args.forward_chunk_size,
        )[:, 0]
        probes, support = build_context_probes_and_support(
            centers, context_ids, fit_cache[key], fit_context_ids,
            cal_cache[key], cal_context_ids, args,
            args.seed + 10000 * (layer + 1),
        )
        endpoints = centers[:, None, :] + probes
        target_endpoint = run_injections(
            model, clean_tokens, io_ids, s_ids, layer, io_position,
            endpoints, args.forward_chunk_size,
        )
        target_irs = forward_signature(target_center, target_endpoint, probes)

        for label, tokens in corruptions.items():
            patched_center = run_injections(
                model, tokens, io_ids, s_ids, layer, io_position,
                center_injections, args.forward_chunk_size,
            )[:, 0]
            patched_endpoint = run_injections(
                model, tokens, io_ids, s_ids, layer, io_position,
                endpoints, args.forward_chunk_size,
            )
            patch_irs = forward_signature(patched_center, patched_endpoint, probes)
            irs_comparison = compare_signatures(patch_irs, target_irs)

            corrupt_h = corrupt_cache[label][key]
            single_delta = (corrupt_h - centers)[:, None, :]
            clean_single_endpoint = run_injections(
                model, clean_tokens, io_ids, s_ids, layer, io_position,
                corrupt_h[:, None, :], args.forward_chunk_size,
            )
            patch_single_endpoint = run_injections(
                model, tokens, io_ids, s_ids, layer, io_position,
                corrupt_h[:, None, :], args.forward_chunk_size,
            )
            target_single = forward_signature(
                target_center, clean_single_endpoint, single_delta
            )
            patch_single = forward_signature(
                patched_center, patch_single_endpoint, single_delta
            )
            single_comparison = compare_signatures(patch_single, target_single)

            single_p = np.empty(len(centers), dtype=np.float64)
            for context_id in np.unique(context_ids):
                query = context_ids == context_id
                scorer = CrossFitSiteReference(
                    fit_cache[key][fit_context_ids == context_id],
                    cal_cache[key][cal_context_ids == context_id],
                    knn_k=args.knn_k, proj_rank=args.proj_rank,
                )
                single_p[query] = scorer.score(
                    corrupt_h[query]
                )["overlap_conformal"]

            center_value = torch.tensor(
                centers, dtype=model.cfg.dtype, device=args.device
            )

            def patch_center_hook(act, hook, value=center_value):
                result = act.clone()
                result[:, io_position, :] = value
                return result

            patched_nmh = nmh_attention(
                model, tokens, io_position,
                resid_hook=(
                    f"blocks.{layer}.hook_resid_post", patch_center_hook
                ),
            )
            base = baselines[label]
            restoration = (
                patched_center - base["logit"]
            ) / base["logit_denominator"]
            nmh_recovery = (
                patched_nmh - base["nmh"]
            ) / base["nmh_denominator"]
            row = {
                "corruption": label,
                "layer": layer,
                "mean_restoration": float(restoration.mean()),
                "mean_nmh_recovery": float(nmh_recovery.mean()),
                "irs_normalized_rmse": irs_comparison.normalized_rmse,
                "single_direction_normalized_rmse": single_comparison.normalized_rmse,
                "irs_endpoint_accept_rate": float(np.mean(
                    support["endpoint_conformal"] > args.support_alpha
                )),
                "single_endpoint_accept_rate": float(np.mean(
                    single_p > args.support_alpha
                )),
            }
            rows.append(row)
            print(
                f"{label} L{layer} R={row['mean_restoration']:.3f} "
                f"A={row['mean_nmh_recovery']:.3f} IRS="
                f"{row['irs_normalized_rmse']:.3f} single="
                f"{row['single_direction_normalized_rmse']:.3f}",
                flush=True,
            )

    stability = {}
    for metric in ("irs_normalized_rmse", "single_direction_normalized_rmse"):
        a = np.array([
            row[metric] for row in rows if row["corruption"] == "duplicate_s"
        ])
        b = np.array([
            row[metric] for row in rows if row["corruption"] == "third_name_pabc"
        ])
        rho, p_value = spearmanr(a, b)
        stability[metric] = {
            "spearman_across_corruptions": float(rho),
            "p_value": float(p_value),
            "mean_absolute_change": float(np.mean(np.abs(a - b))),
            "relative_l2_change": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        }
    for label in corruptions:
        subset = [row for row in rows if row["corruption"] == label]
        nmh = np.array([row["mean_nmh_recovery"] for row in subset])
        for metric in ("irs_normalized_rmse", "single_direction_normalized_rmse"):
            rho, p_value = spearmanr(
                np.array([row[metric] for row in subset]), nmh
            )
            stability[f"{label}_{metric}_vs_nmh"] = {
                "rho": float(rho), "p_value": float(p_value)
            }

    output = finite_or_none({
        "experiment": "p0_irs_corruption_shift_gpt2",
        "model": args.model,
        "seed": args.seed,
        "corruptions": {
            "duplicate_s": "replace IO occurrence by repeated subject name",
            "third_name_pabc": "replace IO occurrence by independent third name",
        },
        "third_names": third_names,
        "baselines": {
            label: {
                "logit_denominator": value["logit_denominator"],
                "nmh_denominator": value["nmh_denominator"],
            }
            for label, value in baselines.items()
        },
        "rows": rows,
        "stability": stability,
        "elapsed_seconds": time.time() - started,
        "configuration": vars(args),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exp_p0_irs_corruption_shift_{args.model}_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parent.parent / "outputs"
    ))
    parser.add_argument("--n-contexts", type=int, default=8)
    parser.add_argument("--n-eval-io", type=int, default=10)
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
