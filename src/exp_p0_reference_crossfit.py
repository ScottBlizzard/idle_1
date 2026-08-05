"""P0 experiment from the 2026-07-12 audit.

For the same clean->corrupt IOI residual-stream patches, compare four explicitly
defined reference distributions using unique prompts and cross-fitted calibration:

1. corrupt observational;
2. clean/source observational;
3. 50/50 clean-corrupt mixture;
4. semantic counterfactual, matched on target name and nuisance context.

The script reports overlap diagnostics without interpreting any one reference as
the uniquely correct causal-validity estimand.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from transformer_lens import HookedTransformer

from validity_crossfit import CrossFitSiteReference


SEED = 20260712
KNN_K = 12
PROJ_RANK = 32
RESTORE_THRESHOLD = 0.1
HISTORICAL_OVERLAP_THRESHOLD = 0.3
SCALE_FLOORS = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)

NAME_CANDIDATES = """
Aaron Adam Adrian Alan Albert Alex Alexander Alice Amanda Amber Amy Andrea Andrew
Angela Anna Anthony Arthur Ashley Austin Barbara Benjamin Betty Bill Bob Brandon
Brian Brittany Bruce Bryan Carl Carol Carolyn Catherine Charles Charlotte Cheryl
Christian Christina Christine Christopher Cindy Claire Daniel Danielle David
Deborah Dennis Diana Diane Donald Donna Dorothy Douglas Dylan Edward Elizabeth
Emily Emma Eric Ethan Evelyn Frank Gabriel Gary George Grace Gregory Hannah Harold
Heather Helen Henry Holly Howard Ian Iris Isabella Jack Jacob James Jane Janet
Jason Jeffrey Jennifer Jeremy Jerry Jesse Jessica Joan Joe John Jonathan Jordan
Joseph Joshua Joyce Julia Julie Justin Karen Katherine Kathleen Kathryn Katie Kate
Keith Kelly Kenneth Kevin Kimberly Laura Lauren Lawrence Linda Lisa Logan Luke
Madison Margaret Maria Marie Marilyn Mark Martha Martin Mary Matthew Megan Melissa
Michael Michelle Mike Nancy Natalie Nathan Nicholas Nicole Noah Olivia Pamela
Patricia Patrick Paul Peter Philip Rachel Rebecca Richard Robert Roger Ronald Rose
Roy Russell Ruth Ryan Samantha Samuel Sandra Sara Sarah Scott Sean Sharon Shirley
Sophia Stephanie Stephen Steven Susan Teresa Terry Thomas Timothy Tom Tyler Victoria
Vincent Walter Wayne William Zachary
""".split()

LEADS = "Earlier Yesterday Recently Eventually Suddenly Later Today Meanwhile".split()
PLACES = "park store office school library station garden museum hotel theater market cafe".split()
ITEMS = "note book drink ball gift letter menu key ticket box package card".split()
PROMPT = "{lead}, {IO} and {S} visited the {place}. Afterwards, {S} handed the {item} to"


def valid_single_token_names(model) -> list[str]:
    valid = []
    seen_ids = set()
    for name in NAME_CANDIDATES:
        tokens = model.to_tokens(f" {name}", prepend_bos=False)
        if tokens.shape[-1] == 1:
            token_id = int(tokens[0, 0])
            if token_id not in seen_ids:
                valid.append(name)
                seen_ids.add(token_id)
    return valid


def encode_record(model, io_name: str, s_name: str, lead: str, place: str, item: str):
    clean_text = PROMPT.format(lead=lead, IO=io_name, S=s_name, place=place, item=item)
    corrupt_text = PROMPT.format(lead=lead, IO=s_name, S=s_name, place=place, item=item)
    clean = model.to_tokens(clean_text)[0].cpu()
    corrupt = model.to_tokens(corrupt_text)[0].cpu()
    if len(clean) != len(corrupt):
        return None
    io_id = int(model.to_tokens(f" {io_name}", prepend_bos=False)[0, 0])
    s_id = int(model.to_tokens(f" {s_name}", prepend_bos=False)[0, 0])
    clean_list = clean.tolist()
    io_positions = [i for i, token in enumerate(clean_list) if token == io_id]
    s_positions = [i for i, token in enumerate(clean_list) if token == s_id]
    if len(io_positions) != 1 or len(s_positions) != 2:
        return None
    signature = (len(clean_list), io_positions[0], s_positions[0], s_positions[1])
    return {
        "clean": clean,
        "corrupt": corrupt,
        "io_id": io_id,
        "s_id": s_id,
        "signature": signature,
        "clean_key": (io_name, s_name, lead, place, item),
        "corrupt_key": (s_name, lead, place, item),
        "context_key": (s_name, lead, place, item),
    }


def sample_records(
    model,
    names: list[str],
    n: int,
    rng: np.random.RandomState,
    signature: tuple[int, int, int, int] | None,
    *,
    unique_on: str,
    forbidden: set,
) -> tuple[list[dict], tuple[int, int, int, int]]:
    records = []
    used = set(forbidden)
    max_attempts = max(2000, n * 200)
    for _ in range(max_attempts):
        io_name, s_name = rng.choice(names, 2, replace=False)
        lead = str(rng.choice(LEADS))
        place = str(rng.choice(PLACES))
        item = str(rng.choice(ITEMS))
        record = encode_record(model, io_name, s_name, lead, place, item)
        if record is None:
            continue
        if signature is None:
            signature = record["signature"]
        if record["signature"] != signature:
            continue
        key = record[unique_on]
        if key in used:
            continue
        used.add(key)
        records.append(record)
        if len(records) == n:
            return records, signature
    raise RuntimeError(
        f"could only construct {len(records)}/{n} records for {unique_on}; "
        f"signature={signature}, names={len(names)}"
    )


def build_context_matched_records(
    model,
    names: list[str],
    rng: np.random.RandomState,
    signature: tuple[int, int, int, int],
    n_contexts: int,
    n_eval_io: int,
    n_fit_io: int,
    n_cal_io: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    needed_names = 1 + n_eval_io + n_fit_io + n_cal_io
    if len(names) < needed_names:
        raise RuntimeError(
            f"semantic matching needs {needed_names} valid names, found {len(names)}"
        )
    eval_records, fit_records, cal_records = [], [], []
    used_contexts = set()
    attempts = 0
    while len(used_contexts) < n_contexts and attempts < n_contexts * 500:
        attempts += 1
        s_name = str(rng.choice(names))
        lead = str(rng.choice(LEADS))
        place = str(rng.choice(PLACES))
        item = str(rng.choice(ITEMS))
        context = (s_name, lead, place, item)
        if context in used_contexts:
            continue
        io_pool = [name for name in names if name != s_name]
        rng.shuffle(io_pool)
        chosen = io_pool[: n_eval_io + n_fit_io + n_cal_io]
        context_records = []
        for io_name in chosen:
            record = encode_record(model, io_name, s_name, lead, place, item)
            if record is None or record["signature"] != signature:
                context_records = []
                break
            record["context_id"] = len(used_contexts)
            context_records.append(record)
        if len(context_records) != len(chosen):
            continue
        used_contexts.add(context)
        eval_records.extend(context_records[:n_eval_io])
        fit_records.extend(context_records[n_eval_io:n_eval_io + n_fit_io])
        cal_records.extend(context_records[n_eval_io + n_fit_io:])
    if len(used_contexts) != n_contexts:
        raise RuntimeError(f"could only construct {len(used_contexts)}/{n_contexts} contexts")
    return eval_records, fit_records, cal_records


def records_to_batch(records: list[dict], key: str, device: str) -> torch.Tensor:
    return torch.stack([record[key] for record in records]).to(device)


def token_uniques(records: list[dict], key: str) -> int:
    rows = np.stack([record[key].numpy() for record in records])
    return int(np.unique(rows, axis=0).shape[0])


def cache_residuals(model, tokens: torch.Tensor, n_layers: int, chunk_size: int) -> dict:
    result: dict[tuple[int, int], list[np.ndarray]] = {}
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start:start + chunk_size]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                chunk, names_filter=lambda name: "hook_resid_post" in name
            )
        for layer in range(n_layers):
            activations = cache[f"blocks.{layer}.hook_resid_post"].float().cpu().numpy()
            for pos in range(activations.shape[1]):
                result.setdefault((layer, pos), []).append(activations[:, pos, :])
        del cache
        torch.cuda.empty_cache()
    return {key: np.concatenate(parts, axis=0) for key, parts in result.items()}


def ioi_logit_diff(logits, io_ids, s_ids):
    final = logits[:, -1, :]
    rows = torch.arange(len(io_ids), device=final.device)
    return final[rows, io_ids] - final[rows, s_ids]


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def summarize_scores(ref: CrossFitSiteReference, query: np.ndarray) -> dict:
    base = ref.score(query, scale_floor=1e-6)
    summary = {
        name: float(np.mean(base[name]))
        for name in (
            "knn", "recon", "maha", "knn_z", "recon_z", "maha_z",
            "z_sum", "overlap_z", "overlap_ecdf",
        )
    }
    summary["scale_floor_sensitivity"] = {
        f"{floor:.0e}": float(np.mean(ref.score(query, scale_floor=floor)["overlap_z"]))
        for floor in SCALE_FLOORS
    }
    diag = ref.diagnostics()
    summary["diagnostics"] = {
        "n_fit": diag["n_fit"],
        "n_calibration": diag["n_calibration"],
        "unique_fit_activations": diag["unique_fit_activations"],
        "unique_calibration_activations": diag["unique_calibration_activations"],
        "effective_rank": diag["effective_rank"],
        "selected_rank": diag["selected_rank"],
        "condition_number": diag["condition_number"],
        "calibration": diag["calibration"],
    }
    return summary


def run(args) -> Path:
    started = time.time()
    device = args.device
    print(f"Loading {args.model} on {device}", flush=True)
    os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY", "0")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    names = valid_single_token_names(model)
    print(f"Valid single-token names: {len(names)}", flush=True)
    if len(names) < 50:
        raise RuntimeError("fewer than 50 single-token names; cannot build matched references")

    rng = np.random.RandomState(args.seed)
    signature = None
    forbidden_corrupt: set = set()
    corrupt_fit_records, signature = sample_records(
        model, names, args.n_fit, rng, signature,
        unique_on="corrupt_key", forbidden=forbidden_corrupt,
    )
    forbidden_corrupt.update(record["corrupt_key"] for record in corrupt_fit_records)
    corrupt_cal_records, signature = sample_records(
        model, names, args.n_cal, rng, signature,
        unique_on="corrupt_key", forbidden=forbidden_corrupt,
    )

    forbidden_clean: set = set()
    clean_fit_records, signature = sample_records(
        model, names, args.n_fit, rng, signature,
        unique_on="clean_key", forbidden=forbidden_clean,
    )
    forbidden_clean.update(record["clean_key"] for record in clean_fit_records)
    clean_cal_records, signature = sample_records(
        model, names, args.n_cal, rng, signature,
        unique_on="clean_key", forbidden=forbidden_clean,
    )

    eval_records, matched_fit_records, matched_cal_records = build_context_matched_records(
        model, names, rng, signature, args.n_contexts, args.n_eval_io,
        args.n_cf_fit, args.n_cf_cal,
    )
    print(
        f"Prompt signature={signature}; fit/cal/eval={args.n_fit}/{args.n_cal}/{len(eval_records)}",
        flush=True,
    )

    eval_clean = records_to_batch(eval_records, "clean", device)
    eval_corrupt = records_to_batch(eval_records, "corrupt", device)
    io_ids = torch.tensor([record["io_id"] for record in eval_records], device=device)
    s_ids = torch.tensor([record["s_id"] for record in eval_records], device=device)
    context_ids = np.array([record["context_id"] for record in eval_records])

    with torch.no_grad():
        clean_ld = float(ioi_logit_diff(model(eval_clean), io_ids, s_ids).mean())
        corrupt_ld = float(ioi_logit_diff(model(eval_corrupt), io_ids, s_ids).mean())
    denominator = clean_ld - corrupt_ld
    if abs(denominator) < 0.2:
        raise RuntimeError(f"clean-corrupt denominator is too small: {denominator}")
    print(f"clean_ld={clean_ld:.4f}, corrupt_ld={corrupt_ld:.4f}, denom={denominator:.4f}")

    n_layers = min(model.cfg.n_layers, args.max_layers or model.cfg.n_layers)
    cache_specs = {
        "corrupt_fit": (corrupt_fit_records, "corrupt"),
        "corrupt_cal": (corrupt_cal_records, "corrupt"),
        "clean_fit": (clean_fit_records, "clean"),
        "clean_cal": (clean_cal_records, "clean"),
        "matched_fit": (matched_fit_records, "clean"),
        "matched_cal": (matched_cal_records, "clean"),
        "eval_clean": (eval_records, "clean"),
    }
    caches = {}
    for label, (records, token_key) in cache_specs.items():
        print(f"Caching {label}: n={len(records)}", flush=True)
        tokens = records_to_batch(records, token_key, device)
        caches[label] = cache_residuals(model, tokens, n_layers, args.chunk_size)
        del tokens

    seq_len = signature[0]
    sites = []
    for layer in range(n_layers):
        for pos in range(seq_len):
            key = (layer, pos)
            clean_query = caches["eval_clean"][key]
            clean_injection = torch.tensor(clean_query, dtype=model.cfg.dtype, device=device)

            def patch_hook(act, hook, position=pos, injection=clean_injection):
                patched = act.clone()
                patched[:, position, :] = injection
                return patched

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    eval_corrupt,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)],
                )
            patched_ld = float(ioi_logit_diff(patched_logits, io_ids, s_ids).mean())
            restoration = (patched_ld - corrupt_ld) / denominator
            if restoration <= args.restore_threshold:
                continue

            condition_arrays = {
                "corrupt_observational": (
                    caches["corrupt_fit"][key], caches["corrupt_cal"][key]
                ),
                "clean_source": (caches["clean_fit"][key], caches["clean_cal"][key]),
                "mixture": (
                    np.concatenate([
                        caches["corrupt_fit"][key][: args.n_fit // 2],
                        caches["clean_fit"][key][: args.n_fit - args.n_fit // 2],
                    ]),
                    np.concatenate([
                        caches["corrupt_cal"][key][: args.n_cal // 2],
                        caches["clean_cal"][key][: args.n_cal - args.n_cal // 2],
                    ]),
                ),
            }
            references = {}
            for label, (fit_ref, cal_ref) in condition_arrays.items():
                ref = CrossFitSiteReference(
                    fit_ref, cal_ref, knn_k=args.knn_k, proj_rank=args.proj_rank
                )
                references[label] = summarize_scores(ref, clean_query)

            matched_parts = []
            matched_refs = []
            matched_diags = []
            for context_id in range(args.n_contexts):
                fit_mask = np.array([
                    record["context_id"] == context_id for record in matched_fit_records
                ])
                cal_mask = np.array([
                    record["context_id"] == context_id for record in matched_cal_records
                ])
                query_mask = context_ids == context_id
                ref = CrossFitSiteReference(
                    caches["matched_fit"][key][fit_mask],
                    caches["matched_cal"][key][cal_mask],
                    knn_k=args.knn_k,
                    proj_rank=args.proj_rank,
                )
                matched_parts.append((query_mask, ref.score(clean_query[query_mask])))
                matched_refs.append((query_mask, ref))
                matched_diags.append(ref.diagnostics())

            matched_summary = {}
            for metric in (
                "knn", "recon", "maha", "knn_z", "recon_z", "maha_z",
                "z_sum", "overlap_z", "overlap_ecdf",
            ):
                values = np.concatenate([scores[metric] for _, scores in matched_parts])
                matched_summary[metric] = float(values.mean())
            matched_summary["scale_floor_sensitivity"] = {}
            for floor in SCALE_FLOORS:
                floor_values = [
                    ref.score(clean_query[query_mask], scale_floor=floor)["overlap_z"]
                    for query_mask, ref in matched_refs
                ]
                matched_summary["scale_floor_sensitivity"][f"{floor:.0e}"] = float(
                    np.concatenate(floor_values).mean()
                )
            matched_summary["diagnostics"] = {
                "n_contexts": args.n_contexts,
                "effective_rank_min": min(d["effective_rank"] for d in matched_diags),
                "effective_rank_max": max(d["effective_rank"] for d in matched_diags),
                "selected_rank_min": min(d["selected_rank"] for d in matched_diags),
                "selected_rank_max": max(d["selected_rank"] for d in matched_diags),
            }
            references["matched_semantic_counterfactual"] = matched_summary

            sites.append({
                "layer": layer,
                "pos": pos,
                "restoration": float(restoration),
                "references": references,
            })
        print(f"Layer {layer}: cumulative retained sites={len(sites)}", flush=True)

    reference_labels = [
        "corrupt_observational", "clean_source", "mixture",
        "matched_semantic_counterfactual",
    ]
    summary = {}
    for label in reference_labels:
        overlap = np.array([site["references"][label]["overlap_z"] for site in sites])
        ecdf = np.array([site["references"][label]["overlap_ecdf"] for site in sites])
        summary[label] = {
            "mean_overlap_z": float(overlap.mean()) if len(overlap) else None,
            "median_overlap_z": float(np.median(overlap)) if len(overlap) else None,
            "mean_overlap_ecdf": float(ecdf.mean()) if len(ecdf) else None,
            "n_below_historical_0_3": int((overlap < HISTORICAL_OVERLAP_THRESHOLD).sum()),
        }

    output = {
        "experiment": "p0_reference_crossfit",
        "interpretation_guardrail": (
            "Scores measure overlap with the named reference distribution; no reference "
            "is assumed to be the uniquely correct causal-validity estimand."
        ),
        "model": args.model,
        "seed": args.seed,
        "n_layers_scanned": n_layers,
        "sequence_signature": list(signature),
        "clean_logit_diff": clean_ld,
        "corrupt_logit_diff": corrupt_ld,
        "denominator": denominator,
        "prompt_audit": {
            "valid_single_token_names": len(names),
            "corrupt_fit_nominal": len(corrupt_fit_records),
            "corrupt_fit_unique_tokens": token_uniques(corrupt_fit_records, "corrupt"),
            "corrupt_cal_nominal": len(corrupt_cal_records),
            "corrupt_cal_unique_tokens": token_uniques(corrupt_cal_records, "corrupt"),
            "clean_fit_unique_tokens": token_uniques(clean_fit_records, "clean"),
            "clean_cal_unique_tokens": token_uniques(clean_cal_records, "clean"),
            "eval_unique_clean_tokens": token_uniques(eval_records, "clean"),
            "matched_contexts": args.n_contexts,
            "matched_fit_per_context": args.n_cf_fit,
            "matched_cal_per_context": args.n_cf_cal,
        },
        "configuration": vars(args),
        "n_retained_sites": len(sites),
        "reference_summary": summary,
        "sites": sites,
        "elapsed_seconds": time.time() - started,
    }
    output = finite_or_none(output)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "_").replace("-", "_")
    output_path = output_dir / f"exp_p0_reference_crossfit_{tag}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    print(f"Saved {output_path}; elapsed={time.time() - started:.1f}s", flush=True)
    return output_path


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
    parser.add_argument("--knn-k", type=int, default=KNN_K)
    parser.add_argument("--proj-rank", type=int, default=PROJ_RANK)
    parser.add_argument("--restore-threshold", type=float, default=RESTORE_THRESHOLD)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-layers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
