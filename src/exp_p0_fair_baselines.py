"""Fair baseline comparison requested by the July audit.

All supervised methods receive the same balanced train/validation/test split and
the same tuning budget. Component-space models use three raw overlap diagnostics;
activation-space models use the full residual vector. Labels denote held-out
reference-condition membership versus clean-source shift, not causal validity.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer

from exp_p0_reference_crossfit import (
    SEED,
    cache_residuals,
    finite_or_none,
    records_to_batch,
    sample_records,
    valid_single_token_names,
)
from validity_crossfit import CrossFitSiteReference


CANONICAL_SITES = [(layer, 3) for layer in range(11)] + [(9, 16), (10, 16), (11, 16)]


def split_class_indices(n: int, rng: np.random.RandomState):
    order = rng.permutation(n)
    n_train = n // 2
    n_val = n // 4
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def build_balanced_split(valid, shifted, indices):
    valid_idx, shifted_idx = indices
    x = np.concatenate([valid[valid_idx], shifted[shifted_idx]], axis=0)
    y = np.concatenate([
        np.zeros(len(valid_idx), dtype=int), np.ones(len(shifted_idx), dtype=int)
    ])
    return x, y


def tune_and_test(kind, x_train, y_train, x_val, y_val, x_test, y_test, seed):
    if kind == "logistic":
        candidates = [0.01, 0.1, 1.0, 10.0]
        def make_model(value):
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(C=value, max_iter=2000, random_state=seed),
            )
        hyper_name = "C"
    elif kind == "mlp":
        candidates = [1e-4, 1e-3, 1e-2, 1e-1]
        def make_model(value):
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(32,), activation="relu", solver="adam",
                    alpha=value, batch_size=64, learning_rate_init=1e-3,
                    early_stopping=True, validation_fraction=0.2,
                    n_iter_no_change=12, max_iter=250, random_state=seed,
                ),
            )
        hyper_name = "alpha"
    else:
        raise ValueError(kind)

    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for value in candidates:
            model = make_model(value)
            model.fit(x_train, y_train)
            val_score = model.predict_proba(x_val)[:, 1]
            val_auc = float(roc_auc_score(y_val, val_score))
            candidate = (val_auc, -candidates.index(value), value, model)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        _, _, value, model = best
        # Refit with the selected hyperparameter on train+validation.
        x_fit = np.concatenate([x_train, x_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        model = make_model(value)
        model.fit(x_fit, y_fit)
        test_score = model.predict_proba(x_test)[:, 1]
    return {
        "test_auroc": float(roc_auc_score(y_test, test_score)),
        "selected_hyperparameter": {hyper_name: value},
        "validation_auroc": float(best[0]),
    }


def run(args):
    started = time.time()
    os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY", "0")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    if args.model != "gpt2":
        raise ValueError("canonical sites in this audit are defined for gpt2")
    names = valid_single_token_names(model)
    rng = np.random.RandomState(args.seed)

    ref_fit, signature = sample_records(
        model, names, args.n_fit, rng, None, unique_on="corrupt_key", forbidden=set()
    )
    used_corrupt = {record["corrupt_key"] for record in ref_fit}
    ref_cal, signature = sample_records(
        model, names, args.n_cal, rng, signature,
        unique_on="corrupt_key", forbidden=used_corrupt,
    )
    used_corrupt.update(record["corrupt_key"] for record in ref_cal)
    valid_records, signature = sample_records(
        model, names, args.n_supervised, rng, signature,
        unique_on="corrupt_key", forbidden=used_corrupt,
    )
    shifted_records, signature = sample_records(
        model, names, args.n_supervised, rng, signature,
        unique_on="clean_key", forbidden=set(),
    )

    cache_specs = {
        "ref_fit": (ref_fit, "corrupt"),
        "ref_cal": (ref_cal, "corrupt"),
        "valid": (valid_records, "corrupt"),
        "shifted": (shifted_records, "clean"),
    }
    caches = {}
    for label, (records, token_key) in cache_specs.items():
        print(f"Caching {label}: n={len(records)}", flush=True)
        caches[label] = cache_residuals(
            model, records_to_batch(records, token_key, args.device), 12, args.chunk_size
        )
    del model
    torch.cuda.empty_cache()

    split_rng = np.random.RandomState(args.seed + 9000)
    valid_split = split_class_indices(args.n_supervised, split_rng)
    shifted_split = split_class_indices(args.n_supervised, split_rng)
    split_names = ("train", "validation", "test")
    site_results = []
    sites = CANONICAL_SITES[: args.max_sites or len(CANONICAL_SITES)]
    for layer, pos in sites:
        key = (layer, pos)
        scorer = CrossFitSiteReference(
            caches["ref_fit"][key], caches["ref_cal"][key],
            knn_k=args.knn_k, proj_rank=args.proj_rank,
        )
        valid_acts = caches["valid"][key]
        shifted_acts = caches["shifted"][key]
        valid_raw = scorer.raw_metrics(valid_acts)
        shifted_raw = scorer.raw_metrics(shifted_acts)
        valid_components = np.column_stack([
            valid_raw["knn"], valid_raw["recon"], valid_raw["maha"]
        ])
        shifted_components = np.column_stack([
            shifted_raw["knn"], shifted_raw["recon"], shifted_raw["maha"]
        ])

        component_splits = {}
        activation_splits = {}
        for split_index, split_name in enumerate(split_names):
            indices = (valid_split[split_index], shifted_split[split_index])
            component_splits[split_name] = build_balanced_split(
                valid_components, shifted_components, indices
            )
            activation_splits[split_name] = build_balanced_split(
                valid_acts, shifted_acts, indices
            )

        test_indices = (valid_split[2], shifted_split[2])
        valid_ivs = scorer.score(valid_acts[test_indices[0]])["z_sum"]
        shifted_ivs = scorer.score(shifted_acts[test_indices[1]])["z_sum"]
        ivs_scores = np.concatenate([valid_ivs, shifted_ivs])
        ivs_labels = np.concatenate([
            np.zeros(len(valid_ivs), dtype=int), np.ones(len(shifted_ivs), dtype=int)
        ])
        methods = {
            "crossfit_ivs": {
                "test_auroc": float(roc_auc_score(ivs_labels, ivs_scores)),
                "training_required": False,
            }
        }
        for space, splits in (
            ("components", component_splits), ("activations", activation_splits)
        ):
            for kind in ("logistic", "mlp"):
                result = tune_and_test(
                    kind,
                    *splits["train"],
                    *splits["validation"],
                    *splits["test"],
                    args.seed,
                )
                result["training_required"] = True
                result["feature_space"] = space
                methods[f"{space}_{kind}"] = result
        site_results.append({"layer": layer, "pos": pos, "methods": methods})
        print(
            f"site=({layer},{pos}) " + " ".join(
                f"{name}={values['test_auroc']:.3f}" for name, values in methods.items()
            ),
            flush=True,
        )

    method_names = list(site_results[0]["methods"])
    summary = {
        name: {
            "mean_test_auroc": float(np.mean([
                site["methods"][name]["test_auroc"] for site in site_results
            ])),
            "min_test_auroc": float(min(
                site["methods"][name]["test_auroc"] for site in site_results
            )),
            "max_test_auroc": float(max(
                site["methods"][name]["test_auroc"] for site in site_results
            )),
        }
        for name in method_names
    }
    output = finite_or_none({
        "experiment": "p0_fair_baseline_comparison",
        "model": args.model,
        "seed": args.seed,
        "label_definition": (
            "0=held-out corrupt/reference-condition activation; "
            "1=clean/source-shift activation"
        ),
        "causal_guardrail": (
            "This benchmark tests distribution classification only; labels are not "
            "mechanism bypass or causal-validity ground truth."
        ),
        "split": {
            "per_class_total": args.n_supervised,
            "train_per_class": len(valid_split[0]),
            "validation_per_class": len(valid_split[1]),
            "test_per_class": len(valid_split[2]),
            "identical_across_methods": True,
        },
        "tuning_budget": {
            "logistic_C": [0.01, 0.1, 1.0, 10.0],
            "mlp_alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        },
        "summary": summary,
        "sites": site_results,
        "configuration": vars(args),
        "elapsed_seconds": time.time() - started,
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exp_p0_fair_baselines_{args.model}_seed{args.seed}.json"
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
    parser.add_argument("--n-supervised", type=int, default=240)
    parser.add_argument("--knn-k", type=int, default=12)
    parser.add_argument("--proj-rank", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-sites", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
