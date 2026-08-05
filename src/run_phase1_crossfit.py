"""Re-audit the known-ground-truth gated task with cross-fitted overlap scores."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import config as C
import model as M
import patching as P
import task
from run_phase1 import POSITIONS, POS_NAME, set_seed, site_label
from validity_crossfit import CrossFitSiteReference
from exp_p0_reference_crossfit import finite_or_none


def cache_reference(model, sites, n, gate, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens, _ = task.sample_batch(n, gate=gate, generator=generator, device=device)
    _, activations = P.cache_sites(model, tokens, sites)
    return {site: value.detach().cpu().numpy() for site, value in activations.items()}, tokens.cpu().numpy()


def summarize(ref, query):
    scores = ref.score(query)
    return {
        "overlap_z_mean": float(scores["overlap_z"].mean()),
        "overlap_ecdf_mean": float(scores["overlap_ecdf"].mean()),
        "overlap_conformal_mean": float(scores["overlap_conformal"].mean()),
        "z_sum_mean": float(scores["z_sum"].mean()),
        "knn_z_mean": float(scores["knn_z"].mean()),
        "recon_z_mean": float(scores["recon_z"].mean()),
        "maha_z_mean": float(scores["maha_z"].mean()),
        "scale_floor_overlap": {
            f"{floor:.0e}": float(ref.score(query, scale_floor=floor)["overlap_z"].mean())
            for floor in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)
        },
        "diagnostics": ref.diagnostics(),
        "per_example_z_sum": scores["z_sum"].tolist(),
        "per_example_ecdf": scores["overlap_ecdf"].tolist(),
        "per_example_conformal": scores["overlap_conformal"].tolist(),
    }


def run(args):
    started = time.time()
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    exp = C.ExpCfg(seed=args.seed)
    exp.model.seed = args.seed
    model = M.build_model(exp.model, device)
    checkpoint = Path(args.checkpoint)
    training_info = None
    if args.train:
        training_info = M.train_model(model, exp.train, device, seed=args.seed)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
    else:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    acc_off = M.eval_acc(model, "off", 2048, device)
    acc_on = M.eval_acc(model, "on", 2048, device)
    if min(acc_off, acc_on) < 0.95:
        raise RuntimeError(f"checkpoint did not learn both paths: off={acc_off}, on={acc_on}")

    sites = P.all_sites(model, POSITIONS)
    off_fit, off_fit_tokens = cache_reference(
        model, sites, args.n_fit, "off", args.seed + 11, device
    )
    off_cal, off_cal_tokens = cache_reference(
        model, sites, args.n_cal, "off", args.seed + 12, device
    )
    on_fit, on_fit_tokens = cache_reference(
        model, sites, args.n_fit, "on", args.seed + 21, device
    )
    on_cal, on_cal_tokens = cache_reference(
        model, sites, args.n_cal, "on", args.seed + 22, device
    )

    triple = task.make_triple(
        args.n_eval, device=device,
        generator=torch.Generator(device=device).manual_seed(args.seed + 31),
    )
    records, meta = P.restoration_sweep(model, triple, sites)
    rows = []
    prompt_level_labels = []
    prompt_level_scores = []
    prompt_level_conformal_scores = []
    for record in records:
        site = record["site"]
        query = record["injected"].detach().cpu().numpy()
        reference_arrays = {
            "target_natural_gate_off": (off_fit[site], off_cal[site]),
            "source_gate_on": (on_fit[site], on_cal[site]),
            "broad_gate_mixture": (
                np.concatenate([off_fit[site][: args.n_fit // 2], on_fit[site][: args.n_fit - args.n_fit // 2]]),
                np.concatenate([off_cal[site][: args.n_cal // 2], on_cal[site][: args.n_cal - args.n_cal // 2]]),
            ),
        }
        reference_scores = {}
        for label, (fit_ref, cal_ref) in reference_arrays.items():
            scorer = CrossFitSiteReference(
                fit_ref, cal_ref, knn_k=args.knn_k, proj_rank=args.proj_rank
            )
            reference_scores[label] = summarize(scorer, query)
        row = {
            "label": site_label(record),
            "hook_type": record["hook_type"],
            "layer": record["layer"],
            "pos": record["pos"],
            "pos_name": POS_NAME[record["pos"]],
            "source": record["source"],
            "known_target_support": record["source"] == "clean",
            "restoration": float(record["restoration"]),
            "references": reference_scores,
        }
        rows.append(row)
        if record["restoration"] >= args.restore_threshold:
            z_values = reference_scores["target_natural_gate_off"]["per_example_z_sum"]
            prompt_level_scores.extend(z_values)
            prompt_level_conformal_scores.extend([
                -value for value in
                reference_scores["target_natural_gate_off"]["per_example_conformal"]
            ])
            prompt_level_labels.extend([
                1 if record["source"] == "donor" else 0
            ] * len(z_values))

    auc = (
        float(roc_auc_score(prompt_level_labels, prompt_level_scores))
        if len(set(prompt_level_labels)) == 2 else None
    )
    conformal_auc = (
        float(roc_auc_score(prompt_level_labels, prompt_level_conformal_scores))
        if len(set(prompt_level_labels)) == 2 else None
    )
    high = [row for row in rows if row["restoration"] >= args.restore_threshold]
    high_clean = [row for row in high if row["source"] == "clean"]
    high_donor = [row for row in high if row["source"] == "donor"]
    output = {
        "experiment": "phase1_known_ground_truth_crossfit",
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "trained_in_this_run": args.train,
        "training_info": training_info,
        "acc_off": acc_off,
        "acc_on": acc_on,
        "meta": meta,
        "prompt_audit": {
            "off_fit_nominal": args.n_fit,
            "off_fit_unique_tokens": int(np.unique(off_fit_tokens, axis=0).shape[0]),
            "off_cal_nominal": args.n_cal,
            "off_cal_unique_tokens": int(np.unique(off_cal_tokens, axis=0).shape[0]),
            "on_fit_unique_tokens": int(np.unique(on_fit_tokens, axis=0).shape[0]),
            "on_cal_unique_tokens": int(np.unique(on_cal_tokens, axis=0).shape[0]),
        },
        "ground_truth": (
            "Deployment/target condition fixes gate=off, so clean donors are on target "
            "support and gate=on donors are outside target support by construction."
        ),
        "n_high_clean": len(high_clean),
        "n_high_donor": len(high_donor),
        "prompt_level_auroc_target_off_z_for_donor": auc,
        "prompt_level_auroc_target_off_negative_conformal_for_donor": conformal_auc,
        "mean_target_off_ecdf_high_clean": (
            float(np.mean([
                row["references"]["target_natural_gate_off"]["overlap_ecdf_mean"]
                for row in high_clean
            ])) if high_clean else None
        ),
        "mean_target_off_ecdf_high_donor": (
            float(np.mean([
                row["references"]["target_natural_gate_off"]["overlap_ecdf_mean"]
                for row in high_donor
            ])) if high_donor else None
        ),
        "mean_target_off_conformal_high_clean": (
            float(np.mean([
                row["references"]["target_natural_gate_off"]["overlap_conformal_mean"]
                for row in high_clean
            ])) if high_clean else None
        ),
        "mean_target_off_conformal_high_donor": (
            float(np.mean([
                row["references"]["target_natural_gate_off"]["overlap_conformal_mean"]
                for row in high_donor
            ])) if high_donor else None
        ),
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "configuration": vars(args),
    }
    output = finite_or_none(output)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"phase1_crossfit_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"high clean/donor={len(high_clean)}/{len(high_donor)} AUROC={auc} "
        f"conformal_AUROC={conformal_auc} "
        f"ECDF clean/donor={output['mean_target_off_ecdf_high_clean']}/"
        f"{output['mean_target_off_ecdf_high_donor']}",
        flush=True,
    )
    print(f"Saved {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(Path(C.outputs_dir()) / "phase1_model.pt"))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--out-dir", default=C.outputs_dir())
    parser.add_argument("--n-fit", type=int, default=2000)
    parser.add_argument("--n-cal", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--proj-rank", type=int, default=8)
    parser.add_argument("--restore-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
