"""Frozen, resumable server runner for the GPT-2 matched-bypass bridge.

The command line exposes hardware scheduling only.  Scientific constants are
imported from ``green_bridge_spec`` and cannot be overridden.  Confirmation
records are inaccessible until the development decision and its source hashes
have been atomically frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Sequence

import numpy as np

from analyze_green_bridge import BASELINES, development_decision, freeze_confirmation, confirmation_decision, spearman
from green_bridge_dataset import (
    ConfirmationLock,
    PairRecord,
    build_donor_records,
    build_evaluation_records,
    plan_payload,
    split_records,
    write_plan,
)
from green_bridge_spec import (
    DIMENSIONS,
    FROZEN_SPEC,
    GATE04_AMENDMENT_ID,
    GATE04_HOLDOUT_PAIR_SLICE,
    GATE04_LEGACY_PAIR_SLICE,
    HF_ATTN_IMPLEMENTATION,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    SELECTED_GATES,
    THRESHOLDS,
    TRANSFORMER_LENS_COMMIT,
    canonical_json,
    frozen_spec_hash,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from green_bridge_numerics import (
    active_contraction_bound,
    cell_error_bound,
    certified_null_bound,
    richardson_numerical_bounds,
    sum_item_error_bounds,
)
from green_bridge_tail import GreenBridgeTail, TailAnchor, capture_tail_anchor, gather_year_logits
from green_bridge_path_target import TargetAnchor, finite_path_effect, target_jvp
from matched_bypass_gate import (
    GateJet,
    cosine,
    extrapolate_gate_jet,
    identify_gate,
    symmetric_relative_change,
)


SOURCE_FILES = (
    "src/green_bridge_spec.py",
    "src/green_bridge_dataset.py",
    "src/matched_bypass_gate.py",
    "src/green_bridge_numerics.py",
    "src/green_bridge_tail.py",
    "src/green_bridge_path_target.py",
    "src/exp_green_bridge_gpt2.py",
    "src/analyze_green_bridge.py",
    "src/test_green_bridge_contract.py",
)
PROTOCOL_FILES = (
    "analysis/GPTPRO_GREEN_BRIDGE_20260805.md",
    "analysis/GREEN_SERVER_GATE04_20260805.md",
    "analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md",
    "requirements-green-bridge.lock",
)
EXPECTED_PACKAGES = {
    "torch": "2.7.1",
    "transformer-lens": "3.6.0",
    "transformers": "5.13.0",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "pandas": "2.2.3",
    "pyarrow": "19.0.1",
}
TL_SOURCE_SHA256 = {
    "HookedTransformer.py": "f80ee1ec42039a287a2b9366c75f98eec23ff33c6e941ffeee03f0374eb20af3",
    "HookedRootModule.py": "e7144971a973ec2d63bf7400db6443caba5d03f22f310f6789d52fa4a56ad245",
    "components/mlps/mlp.py": "615cb178d3ce65d8784af18dec86fbfe2b3957ddc02d3b99bdd2d45aa6759b32",
    "utilities/addmm.py": "f9e72f6a3d6c508814fa8e69918c20e1cb72cbc9ae7bcb1a1abb2476e246bc38",
}
FORWARD_COUNTS = {
    "mixed_per_tensor_item": 1682,
    "first_order_per_tensor_item": 1682,
    "factorial_per_tensor_item": 16,
    "tensor_items_total": 384,
    "energy_items_total": 384,
    "tail_total": 1_303_648,
    "jvp_total": 1_152,
    "full_model_total": 4_544,
    "conservative_units": 1_333_216,
}


class GreenStop(RuntimeError):
    def __init__(self, gate: str, detail: str):
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def torch_module():
    import torch
    return torch


def terminal_stop(output_root: Path, gate: str, detail: str) -> None:
    payload = {
        "schema_version": "green-bridge-terminal-v1",
        "verdict": "STOP",
        "first_failed_gate": gate,
        "detail": detail,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(output_root / "result.json", payload)
    raise GreenStop(gate, detail)


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def first_order_directions() -> np.ndarray:
    seed = int.from_bytes(
        hashlib.sha256(b"idle1-gt-bridge-20260805:first-order").digest()[:8], "big"
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    vectors = [np.eye(4, dtype=np.float64)[index] for index in range(4)]
    while len(vectors) < 200:
        value = rng.standard_normal(4)
        value /= np.linalg.norm(value)
        first = np.flatnonzero(np.abs(value) > 0)[0]
        if value[first] < 0:
            value = -value
        if any(abs(float(value @ old)) > 0.999999 for old in vectors):
            continue
        vectors.append(value)
    return np.stack(vectors)


def configure_runtime(device: str) -> dict:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise GreenStop(
            "01_ENVIRONMENT",
            "CUBLAS_WORKSPACE_CONFIG must equal :4096:8",
        )
    torch = torch_module()
    versions = {}
    for package, expected in EXPECTED_PACKAGES.items():
        actual = importlib.metadata.version(package)
        versions[package] = actual
        if package == "torch":
            matches = actual == expected or actual == expected + "+cu126"
        else:
            matches = actual == expected
        if not matches:
            raise GreenStop("01_ENVIRONMENT", f"{package}={actual}, expected {expected}")
    if platform.python_version() != "3.11.13":
        raise GreenStop("01_ENVIRONMENT", f"Python={platform.python_version()}, expected 3.11.13")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise GreenStop("01_ENVIRONMENT", "a CUDA device is required")
    if torch.version.cuda != "12.6" or not torch.__version__.startswith("2.7.1"):
        raise GreenStop("01_ENVIRONMENT", f"torch={torch.__version__}, CUDA={torch.version.cuda}")
    import transformer_lens
    tl_root = Path(transformer_lens.__file__).resolve().parent
    actual_source = {name: sha256_file(tl_root / name) for name in TL_SOURCE_SHA256}
    if actual_source != TL_SOURCE_SHA256:
        raise GreenStop("01_ENVIRONMENT", "TransformerLens source does not match frozen commit 4a4dc26")
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    np.random.seed(20260805)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise GreenStop("01_ENVIRONMENT", "deterministic algorithms are disabled")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(torch.device(device)),
        "transformer_lens_source_sha256": actual_source,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def load_models(device: str, tokenizer=None):
    torch = torch_module()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval().to(device)
    hf_model.config.use_cache = False
    if getattr(hf_model.config, "_attn_implementation", None) != "eager":
        raise GreenStop(
            "03_MODEL_CONFIG",
            "Hugging Face attention implementation is not eager",
        )
    model = HookedTransformer.from_pretrained_no_processing(
        "gpt2", hf_model=hf_model, tokenizer=tokenizer, device=device, dtype=torch.float32,
        default_prepend_bos=False,
    ).eval()
    cfg = model.cfg
    observed = {
        "n_layers": int(cfg.n_layers), "d_model": int(cfg.d_model),
        "n_heads": int(cfg.n_heads), "d_mlp": int(cfg.d_mlp),
        "normalization_type": str(cfg.normalization_type),
        "act_fn": str(cfg.act_fn), "eps": float(cfg.eps),
        "hf_attention_implementation": getattr(
            hf_model.config, "_attn_implementation", None
        ),
        "hf_use_cache": bool(hf_model.config.use_cache),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    required = asdict(DIMENSIONS)
    for name in ("n_layers", "d_model", "n_heads", "d_mlp"):
        if observed[name] != required[name]:
            raise GreenStop("03_MODEL_CONFIG", f"{name}={observed[name]}")
    if observed["normalization_type"] != "LN" or abs(observed["eps"] - 1e-5) > 1e-12:
        raise GreenStop("03_MODEL_CONFIG", f"normalization mismatch: {observed}")
    if "gelu" not in observed["act_fn"].lower():
        raise GreenStop("03_MODEL_CONFIG", f"activation mismatch: {observed['act_fn']}")
    return tokenizer, hf_model, model, observed


def tokenize_one(tokenizer, prompt: str, device: str):
    torch = torch_module()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return torch.tensor([ids], dtype=torch.long, device=device)


def validate_tokenizer(tokenizer, records: Sequence[PairRecord]) -> tuple[list[int], dict]:
    suffix_ids = []
    for suffix in range(100):
        text = f"{suffix:02d}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != text:
            raise GreenStop("02_TOKENIZATION", f"suffix {text} is not an exact single token")
        suffix_ids.append(ids[0])
    if len(set(suffix_ids)) != 100:
        raise GreenStop("02_TOKENIZATION", "suffix token IDs are not unique")

    lengths = set()
    for row in records:
        for suffix in (row.y, row.y_prime):
            year = tokenizer.encode(f" {row.century:02d}{suffix:02d}", add_special_tokens=False)
            century = tokenizer.encode(f" {row.century:02d}", add_special_tokens=False)
            if len(year) != 2 or year[1] != suffix_ids[suffix] or len(century) != 1:
                raise GreenStop("02_TOKENIZATION", f"year token contract failed for {row.century}{suffix:02d}")
        clean = tokenizer.encode(row.clean_prompt, add_special_tokens=False)
        corrupt = tokenizer.encode(row.corrupt_prompt, add_special_tokens=False)
        if len(clean) != len(corrupt) or clean[-1] != tokenizer.encode(
            f" {row.century:02d}", add_special_tokens=False
        )[0]:
            raise GreenStop("02_TOKENIZATION", f"prompt contract failed for {row.pair_digest}")
        lengths.add(len(clean))
    return suffix_ids, {"suffix_ids": suffix_ids, "sequence_lengths": sorted(lengths)}


def token_pair_allowed(tokenizer, first: str, second: str) -> bool:
    ids_first = tokenizer.encode(first, add_special_tokens=False)
    ids_second = tokenizer.encode(second, add_special_tokens=False)
    if len(ids_first) != len(ids_second):
        return False
    for prompt, prompt_ids in ((first, ids_first), (second, ids_second)):
        match = re.search(r"from the year (\d{2})(\d{2}) to the year (\d{2})$", prompt)
        if match is None or match.group(1) != match.group(3):
            return False
        century, suffix = match.group(1), match.group(2)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        year_ids = tokenizer.encode(" " + century + suffix, add_special_tokens=False)
        century_ids = tokenizer.encode(" " + century, add_special_tokens=False)
        if len(suffix_ids) != 1 or len(year_ids) != 2 or year_ids[1] != suffix_ids[0]:
            return False
        if len(century_ids) != 1 or prompt_ids[-1] != century_ids[0]:
            return False
    return True


def write_initial_manifest(
    output_root: Path,
    environment: dict,
    model_cfg: dict,
    split_payload: dict,
    gate04_panel: dict,
) -> dict:
    execution_commit = git_text("rev-parse", "HEAD")
    directions = first_order_directions()
    direction_path = output_root / "first_order_directions.npy"
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(direction_path, directions, allow_pickle=False)
    payload = {
        "schema_version": "green-bridge-manifest-v1.1",
        "theory_base_commit": "126556f",
        "execution_commit": execution_commit,
        "repository_dirty_at_launch": bool(git_text("status", "--porcelain")),
        "frozen_spec": FROZEN_SPEC,
        "frozen_spec_sha256": frozen_spec_hash(),
        "source_sha256": source_hashes(),
        "requirements_sha256": sha256_file(PROJECT_ROOT / "requirements-green-bridge.lock"),
        "protocol_sha256": {
            name: sha256_file(PROJECT_ROOT / name) for name in PROTOCOL_FILES
        },
        "splits_sha256": split_payload["records_sha256"],
        "first_order_directions_sha256": sha256_file(direction_path),
        "environment": environment,
        "model_config": model_cfg,
        "forward_counts": FORWARD_COUNTS,
        "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
        "amendment": {
            "id": GATE04_AMENDMENT_ID,
            "decision_document": "analysis/GPTPRO_GREEN_GATE04_DECISION_20260805.md",
            "decision_base_commit": "0c81e05",
            "theory_base_commit": "126556f",
            "previous_terminal_gate": "04_HF_TL",
            "previous_observed_max_abs_year_logit_error": 0.0001526,
            "previous_development_responses_observed": False,
            "previous_confirmation_responses_observed": False,
            "confirmation_was_locked": True,
            "amendment_scope": [
                "HF-versus-TransformerLens preflight fidelity audit",
                "conformance repair for frozen Richardson numerical propagation",
            ],
            "scientific_design_changed": False,
            "second_threshold_amendment_allowed": False,
        },
        "gate04": {
            "audit_version": "hf-tl-fidelity-v2",
            "hf_attention_implementation": HF_ATTN_IMPLEMENTATION,
            "transformer_lens_processing": "none",
            "dtype": "float32",
            "batch_size": 1,
            "use_cache": False,
            "prompt_selection": {
                "population": "donor",
                "ordering": "ascending pair_digest",
                "excluded_legacy_pair_ranks": {
                    "start_inclusive": 0, "stop_exclusive": 16,
                },
                "audited_holdout_pair_ranks": {
                    "start_inclusive": 16, "stop_exclusive": 32,
                },
                "prompts_per_pair": ["clean", "corrupt"],
                "n_pairs": 16,
                "n_prompts": 32,
                **gate04_panel,
            },
            "parameter_mapping": {
                "converter": "transformer_lens.pretrained.weight_conversions.gpt2.convert_gpt2_weights",
                "mapped_tensors_must_be_bitwise_equal": True,
                "allowed_extra_parameter": "unembed.b_U",
                "allowed_extra_parameter_must_be_zero": True,
            },
            "thresholds": gate04_thresholds(),
            "downstream_error": {
                "enters_epsilon_y": False,
                "reporting_only_after_gate_pass": True,
            },
        },
        "same_transformerlens_audits": {
            "no_op_max_abs": THRESHOLDS.no_op_max_abs,
            "tail_max_abs": THRESHOLDS.tail_max_abs,
            "tail_derivative_relative": THRESHOLDS.tail_derivative_relative,
            "center_rms": THRESHOLDS.center_rms,
            "center_max_abs": THRESHOLDS.center_max_abs,
            "hook_untouched_max": THRESHOLDS.hook_untouched_max,
        },
        "numerical_error_contract": {
            "version": "frozen-richardson-propagation-v1",
            "epsilon_y_source": "same-TransformerLens duplicate audit only",
            "eta_G": "3*epsilon_y/h2",
            "eta_C": "64*epsilon_y/(3*h2^2)",
            "eta_J": "3*epsilon_y/h1",
            "eta_H": "17*epsilon_y/(3*h1*h2)",
            "active_tensor_snr_uses_epsilon_P_F": True,
            "certified_null_bound_enters_theta_error": True,
        },
        "preserved": {
            "matched_bypass_theorem": True,
            "selected_gates": True,
            "resid_mid_site": True,
            "matched_control": True,
            "independent_target": True,
            "residual_bypass_subtraction": True,
            "basis_design": True,
            "radii": True,
            "finite_population": True,
            "baselines": True,
            "development_rules": True,
            "confirmation_lock": True,
            "confirmation_rules": True,
        },
        "confirmation_open": False,
    }
    write_json_atomic(output_root / "manifest.json", payload)
    return payload


def selected_position(anchor: TailAnchor, field: str):
    torch = torch_module()
    value = getattr(anchor, field)
    rows = torch.arange(value.shape[0], device=value.device)
    return value[rows, anchor.final_positions]


def capture_item_systems(model, tokenizer, suffix_ids, record: PairRecord, device: str, anchor_cache=None):
    torch = torch_module()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    clean_tokens = tokenize_one(tokenizer, record.clean_prompt, device)
    corrupt_tokens = tokenize_one(tokenizer, record.corrupt_prompt, device)
    cached = {} if anchor_cache is None else anchor_cache.get(record.pair_digest, {})
    target = cached.get("tar") or capture_tail_anchor(
        model, clean_tokens, suffix_tensor, system="tar"
    )
    corrupt = cached.get("cor") or capture_tail_anchor(
        model, corrupt_tokens, suffix_tensor, system="cor"
    )
    patched = cached.get("pat") or capture_tail_anchor(
        model, corrupt_tokens, suffix_tensor, system="pat", block8_patch=target.mlp8_out
    )
    return {"tar": target, "cor": corrupt, "pat": patched}, clean_tokens, corrupt_tokens


def _anchor_to_plain(anchor: TailAnchor) -> dict:
    fields = ("resid_mid", "pre", "post", "resid_post", "year_logits", "final_positions", "mlp8_out")
    return {name: (None if getattr(anchor, name) is None else getattr(anchor, name).detach().cpu()) for name in fields} | {
        "system": anchor.system
    }


def load_anchor_cache(path: Path, device: str) -> dict:
    torch = torch_module()
    if not path.is_file():
        return {}
    plain = torch.load(path, map_location=device, weights_only=True)
    return {
        digest: {system: TailAnchor(**values) for system, values in systems.items()}
        for digest, systems in plain.items()
    }


def merge_anchor_caches(*caches: dict) -> dict:
    merged = {}
    for cache in caches:
        for digest, systems in cache.items():
            merged.setdefault(digest, {}).update(systems)
    return merged


def _repeat_anchor(anchor: TailAnchor, count: int) -> TailAnchor:
    torch = torch_module()
    index = torch.zeros(count, dtype=torch.long, device=anchor.resid_mid.device)
    def repeat(value):
        return None if value is None else value.index_select(0, index)
    return TailAnchor(
        resid_mid=repeat(anchor.resid_mid), pre=repeat(anchor.pre), post=repeat(anchor.post),
        resid_post=repeat(anchor.resid_post), year_logits=repeat(anchor.year_logits),
        final_positions=repeat(anchor.final_positions), system=anchor.system,
        mlp8_out=repeat(anchor.mlp8_out),
    )


def margin_vector(clean_suffix: int, device: str):
    torch = torch_module()
    value = torch.empty(100, dtype=torch.float64, device=device)
    value[: clean_suffix + 1] = -1.0 / (clean_suffix + 1)
    value[clean_suffix + 1 :] = 1.0 / (99 - clean_suffix)
    return value


def margin(logits, contrast):
    return (logits.double() * contrast).sum(dim=-1)


def gate04_thresholds() -> dict:
    return {
        "raw_year_logits": {
            "max_abs": THRESHOLDS.hf_tl_raw_year_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_raw_year_pooled_rms,
        },
        "centered_year_logits": {
            "max_abs": THRESHOLDS.hf_tl_centered_year_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_centered_year_pooled_rms,
        },
        "task_margin": {
            "max_abs": THRESHOLDS.hf_tl_margin_max_abs,
            "rms": THRESHOLDS.hf_tl_margin_rms,
        },
        "resid_mid": {
            "max_abs": THRESHOLDS.hf_tl_resid_mid_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_resid_mid_pooled_rms,
        },
        "selected_pre": {
            "max_abs": THRESHOLDS.hf_tl_selected_pre_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_selected_pre_pooled_rms,
        },
        "selected_post": {
            "max_abs": THRESHOLDS.hf_tl_selected_post_max_abs,
            "pooled_rms": THRESHOLDS.hf_tl_selected_post_pooled_rms,
        },
    }


def gate04_record_panels(records):
    ranked = sorted(records, key=lambda row: row.pair_digest)
    legacy = ranked[slice(*GATE04_LEGACY_PAIR_SLICE)]
    holdout = ranked[slice(*GATE04_HOLDOUT_PAIR_SLICE)]
    legacy_ids = {row.pair_digest for row in legacy}
    holdout_ids = {row.pair_digest for row in holdout}
    if len(legacy) != 16 or len(holdout) != 16:
        raise GreenStop("04_HF_TL_FIDELITY", "Gate-04 panel has wrong size")
    if len(legacy_ids) != 16 or len(holdout_ids) != 16 or legacy_ids & holdout_ids:
        raise GreenStop("04_HF_TL_FIDELITY", "Gate-04 panels overlap or contain duplicates")
    return legacy, holdout


def gate04_panel_metadata(legacy, holdout) -> dict:
    ordered_keys = [
        [row.pair_digest, system]
        for row in holdout
        for system in ("clean", "corrupt")
    ]
    return {
        "legacy_pair_digests": [row.pair_digest for row in legacy],
        "holdout_pair_digests": [row.pair_digest for row in holdout],
        "ordered_prompt_keys": ordered_keys,
        "ordered_prompt_keys_sha256": sha256_text(canonical_json(ordered_keys)),
    }


def pooled_error_metrics(errors) -> dict:
    arrays = [np.asarray(error, dtype=np.float64).reshape(-1) for error in errors]
    if not arrays or any(array.size == 0 for array in arrays):
        raise ValueError("pooled error metrics require nonempty arrays")
    total_count = sum(array.size for array in arrays)
    sum_squares = sum(float(array @ array) for array in arrays)
    return {
        "max_abs": max(float(np.max(np.abs(array))) for array in arrays),
        "pooled_rms": float(math.sqrt(sum_squares / total_count)),
    }


def centered_year_error(hf_year, tl_year) -> np.ndarray:
    hf = np.asarray(hf_year, dtype=np.float64)
    tl = np.asarray(tl_year, dtype=np.float64)
    return (hf - hf.mean()) - (tl - tl.mean())


def task_margin_error(hf_year, tl_year, clean_suffix: int) -> float:
    contrast = np.empty(100, dtype=np.float64)
    contrast[: clean_suffix + 1] = -1.0 / (clean_suffix + 1)
    contrast[clean_suffix + 1 :] = 1.0 / (99 - clean_suffix)
    return float(contrast @ (
        np.asarray(hf_year, dtype=np.float64) - np.asarray(tl_year, dtype=np.float64)
    ))


def all_gate04_submetrics_pass(metrics: dict, thresholds: dict) -> bool:
    for name, limits in thresholds.items():
        observed = metrics[name]
        for statistic, limit in limits.items():
            if observed[statistic] > limit:
                return False
    return True


def capture_hf_gate04(hf_model, tokens):
    torch = torch_module()
    captured = {}

    def resid_mid_pre_hook(_module, args):
        captured["resid_mid"] = args[0].detach()

    def pre_hook(_module, _args, output):
        captured["pre"] = output.detach()

    def post_pre_hook(_module, args):
        captured["post"] = args[0].detach()

    handles = [
        hf_model.transformer.h[10].ln_2.register_forward_pre_hook(
            resid_mid_pre_hook
        ),
        hf_model.transformer.h[10].mlp.c_fc.register_forward_hook(pre_hook),
        hf_model.transformer.h[10].mlp.c_proj.register_forward_pre_hook(
            post_pre_hook
        ),
    ]
    try:
        with torch.inference_mode():
            logits = hf_model(
                input_ids=tokens,
                use_cache=False,
                return_dict=True,
            ).logits
    finally:
        for handle in handles:
            handle.remove()
    required = {"resid_mid", "pre", "post"}
    if set(captured) != required:
        raise GreenStop(
            "04_HF_TL_FIDELITY",
            f"incomplete Hugging Face capture: {sorted(captured)}",
        )
    return logits, captured


def _exact_tensor_equal(actual, expected) -> bool:
    try:
        torch = torch_module()
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            return bool(torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    except ImportError:
        pass
    return bool(np.array_equal(np.asarray(actual), np.asarray(expected)))


def _nonzero_count(value) -> int:
    try:
        torch = torch_module()
        if isinstance(value, torch.Tensor):
            return int(torch.count_nonzero(value.detach()).item())
    except ImportError:
        pass
    return int(np.count_nonzero(np.asarray(value)))


def weight_mapping_report(expected_state, actual_state, named_parameters=None) -> dict:
    named_parameters = actual_state if named_parameters is None else named_parameters
    missing, shapes, dtypes, values = [], [], [], []
    for key, expected in expected_state.items():
        if key not in actual_state:
            missing.append(key)
            continue
        actual = actual_state[key]
        if tuple(actual.shape) != tuple(expected.shape):
            shapes.append(key)
            continue
        if actual.dtype != expected.dtype:
            dtypes.append(key)
            continue
        if not _exact_tensor_equal(actual, expected):
            values.append(key)
    bias = actual_state.get("unembed.b_U")
    bias_nonzero = 0 if bias is None else _nonzero_count(bias)
    unexpected_nonzero = sorted(
        key for key, value in named_parameters.items()
        if key not in expected_state
        and key != "unembed.b_U"
        and _nonzero_count(value) != 0
    )
    mismatch_count = (
        len(missing) + len(shapes) + len(dtypes) + len(values)
        + len(unexpected_nonzero) + int(bias_nonzero != 0)
    )
    return {
        "mapped_tensor_count": len(expected_state),
        "missing_keys": missing,
        "shape_mismatches": shapes,
        "dtype_mismatches": dtypes,
        "value_mismatches": values,
        "unexpected_nonzero_keys": unexpected_nonzero,
        "unembed_b_U_present": bias is not None,
        "unembed_b_U_nonzero_count": bias_nonzero,
        "mismatch_count": mismatch_count,
        "passed": mismatch_count == 0,
    }


def audit_converted_weights(hf_model, model) -> dict:
    from transformer_lens.pretrained.weight_conversions.gpt2 import convert_gpt2_weights
    expected_tl_state = convert_gpt2_weights(hf_model, model.cfg)
    result = weight_mapping_report(
        expected_tl_state,
        model.state_dict(),
        dict(model.named_parameters()),
    )
    if not result["passed"]:
        raise GreenStop("04_HF_TL_WEIGHT_MAP", canonical_json(result))
    return result


def hf_tl_audit(tokenizer, hf_model, model, legacy, holdout, suffix_ids, device: str) -> dict:
    torch = torch_module()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    panel = gate04_panel_metadata(legacy, holdout)
    weight_mapping = audit_converted_weights(hf_model, model)
    families = {
        name: [] for name in (
            "raw_year_logits", "centered_year_logits", "resid_mid",
            "selected_pre", "selected_post",
        )
    }
    margin_errors, per_prompt, references = [], [], {}
    for row in holdout:
        for system, prompt in (("clean", row.clean_prompt), ("corrupt", row.corrupt_prompt)):
            tokens = tokenize_one(tokenizer, prompt, device)
            if tokens.shape[0] != 1:
                raise GreenStop("04_HF_TL_FIDELITY", "Gate-04 batch size is not one")
            hf_logits, hf_capture = capture_hf_gate04(hf_model, tokens)
            position = tokens.shape[1] - 1
            positions = torch.tensor([position], device=device)
            hf_year = gather_year_logits(model, hf_logits, positions, suffix_tensor)[0]
            with torch.inference_mode():
                anchor = capture_tail_anchor(model, tokens, suffix_tensor, system="hf-tl")
            tl_year = anchor.year_logits[0]
            raw = hf_year.detach().double().cpu().numpy() - tl_year.detach().double().cpu().numpy()
            centered = centered_year_error(
                hf_year.detach().double().cpu().numpy(),
                tl_year.detach().double().cpu().numpy(),
            )
            resid = (
                hf_capture["resid_mid"][0, position].detach().double().cpu().numpy()
                - anchor.resid_mid[0, position].detach().double().cpu().numpy()
            )
            selected = list(SELECTED_GATES)
            pre = (
                hf_capture["pre"][0, position, selected].detach().double().cpu().numpy()
                - anchor.pre[0, position, selected].detach().double().cpu().numpy()
            )
            post = (
                hf_capture["post"][0, position, selected].detach().double().cpu().numpy()
                - anchor.post[0, position, selected].detach().double().cpu().numpy()
            )
            margin_error = task_margin_error(
                hf_year.detach().double().cpu().numpy(),
                tl_year.detach().double().cpu().numpy(),
                row.y,
            )
            prompt_errors = {
                "raw_year_logits": raw,
                "centered_year_logits": centered,
                "resid_mid": resid,
                "selected_pre": pre,
                "selected_post": post,
            }
            for name, error in prompt_errors.items():
                families[name].append(error)
            margin_errors.append(margin_error)
            per_prompt.append({
                "pair_digest": row.pair_digest,
                "system": system,
                "clean_suffix": row.y,
                **{
                    name + "_max_abs": float(np.max(np.abs(error)))
                    for name, error in prompt_errors.items()
                },
                "task_margin_error": margin_error,
                "task_margin_abs_error": abs(margin_error),
            })
            references[row.pair_digest + "|" + system] = {
                "year_logits": tl_year.detach().double().cpu().numpy().tolist(),
                "mlp8_out": anchor.mlp8_out[0].detach().float().cpu().numpy().tolist(),
            }
    metrics = {name: pooled_error_metrics(errors) for name, errors in families.items()}
    margin_array = np.asarray(margin_errors, dtype=np.float64)
    metrics["task_margin"] = {
        "max_abs": float(np.max(np.abs(margin_array))),
        "rms": float(np.sqrt(np.mean(margin_array**2))),
    }
    thresholds = gate04_thresholds()
    passed = all_gate04_submetrics_pass(metrics, thresholds)
    result = {
        "audit_version": "hf-tl-fidelity-v2",
        "hf_attention_implementation": getattr(
            hf_model.config, "_attn_implementation", None
        ),
        "batch_size": 1,
        **panel,
        "n_pairs": len(holdout),
        "n_prompts": len(per_prompt),
        "weight_mapping": weight_mapping,
        "metrics": metrics,
        "thresholds": thresholds,
        "per_prompt": per_prompt,
        "hf_tl_error_enters_epsilon_y": False,
        "passed": passed,
        "tl_references": references,
    }
    if not passed:
        raise GreenStop("04_HF_TL_FIDELITY", canonical_json(metrics))
    return result


def no_op_audit(model, tokenizer, holdout, suffix_ids, device: str, references: dict) -> dict:
    torch = torch_module()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    errors = []
    with torch.inference_mode():
        for row in holdout:
            for system, prompt in (("clean", row.clean_prompt), ("corrupt", row.corrupt_prompt)):
                tokens = tokenize_one(tokenizer, prompt, device)
                cached = references[row.pair_digest + "|" + system]
                reference = torch.as_tensor(
                    cached["year_logits"], dtype=torch.float64, device=device
                )
                block8 = torch.as_tensor(cached["mlp8_out"], dtype=torch.float32, device=device)[None]
                replay = capture_tail_anchor(
                    model, tokens, suffix_tensor, system=system, block8_patch=block8
                )
                errors.append(float((reference - replay.year_logits[0].double()).abs().max()))
    result = {"n": len(errors), "max_abs": max(errors), "errors": errors}
    if result["max_abs"] > THRESHOLDS.no_op_max_abs:
        raise GreenStop("05_HOOK_NOOP", f"max error {result['max_abs']:.3e}")
    return result


def canonical_basis(chords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.linalg import svd
    old_threads = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        _, singular, vt = svd(
            np.asarray(chords, dtype=np.float64), full_matrices=False,
            lapack_driver="gesvd", overwrite_a=False, check_finite=True,
        )
    finally:
        for name, value in old_threads.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    basis = vt[:4].T.copy()
    for column in range(4):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return basis, singular


def donor_anchors(model, tokenizer, suffix_ids, records: Sequence[PairRecord], device: str) -> dict:
    """Collect only CPU float64 statistics; donor activations never enter scores."""
    chords, clean_pre, corrupt_pre, rms_anchor, metadata = [], [], [], [], []
    torch = torch_module()
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    audit_digests = {
        row.pair_digest for row in sorted(
            records, key=lambda row: sha256_text("tail-audit|" + row.pair_digest)
        )[:32]
    }
    audit_anchors = {}
    with torch.inference_mode():
        for index, row in enumerate(records):
            clean = capture_tail_anchor(
                model, tokenize_one(tokenizer, row.clean_prompt, device), suffix_tensor, system="donor-clean"
            )
            corrupt = capture_tail_anchor(
                model, tokenize_one(tokenizer, row.corrupt_prompt, device), suffix_tensor, system="donor-corrupt"
            )
            if row.pair_digest in audit_digests:
                audit_anchors[row.pair_digest] = _anchor_to_plain(clean)
            clean_resid = selected_position(clean, "resid_mid").double().cpu().numpy()[0]
            corrupt_resid = selected_position(corrupt, "resid_mid").double().cpu().numpy()[0]
            clean_gate = selected_position(clean, "pre").double().cpu().numpy()[0, list(SELECTED_GATES)]
            corrupt_gate = selected_position(corrupt, "pre").double().cpu().numpy()[0, list(SELECTED_GATES)]
            chords.append(clean_resid - corrupt_resid)
            clean_pre.append(clean_gate)
            corrupt_pre.append(corrupt_gate)
            rms_anchor.extend([
                float(np.sqrt(np.mean(clean_resid**2))),
                float(np.sqrt(np.mean(corrupt_resid**2))),
            ])
            metadata.append({"noun": row.noun, "role": row.role, "pair_digest": row.pair_digest})
            if (index + 1) % 64 == 0:
                print(f"donors {index + 1}/{len(records)}", flush=True)
    return {
        "chords": np.stack(chords), "clean_pre": np.stack(clean_pre),
        "corrupt_pre": np.stack(corrupt_pre), "rms_anchor": np.array(rms_anchor),
        "metadata": metadata,
        "audit_anchors": audit_anchors,
    }


def build_basis_and_radii(donor: dict, output_root: Path) -> tuple[np.ndarray, dict]:
    roles = np.array([row["role"] for row in donor["metadata"]])
    nouns = np.array([row["noun"] for row in donor["metadata"]])
    basis_mask = roles == "basis"
    radius_mask = roles == "radius"
    basis, singular = canonical_basis(donor["chords"][basis_mask])
    if singular[3] / singular[4] < 1.10 or singular[3] / singular[0] < 1e-4:
        raise GreenStop(
            "08_BASIS_SPECTRUM",
            f"sigma4/sigma5={singular[3]/singular[4]:.4g}, sigma4/sigma1={singular[3]/singular[0]:.4g}",
        )
    angles, leave_bases = {}, {}
    for noun in sorted(set(nouns[basis_mask])):
        leave, _ = canonical_basis(donor["chords"][basis_mask & (nouns != noun)])
        smallest = np.linalg.svd(basis.T @ leave, compute_uv=False).min()
        angle = math.degrees(math.acos(float(np.clip(smallest, -1.0, 1.0))))
        angles[noun] = angle
        leave_bases[noun] = leave
        if angle > 15.0:
            raise GreenStop("08_BASIS_STABILITY", f"leave-{noun} angle={angle:.3f}")

    radius_chords = donor["chords"][radius_mask]
    projected = radius_chords @ basis
    sigma_x = float(np.median(np.linalg.norm(projected, axis=1) / 2.0))
    h1 = 0.20 * sigma_x
    residual_floor = 2.0**-10 * float(np.median(donor["rms_anchor"]))
    if h1 < residual_floor:
        raise GreenStop("09_RADIUS_FLOOR", f"h1={h1:.4e} < {residual_floor:.4e}")
    clean = donor["clean_pre"][radius_mask]
    corrupt = donor["corrupt_pre"][radius_mask]
    pooled = np.concatenate([clean, corrupt], axis=0)
    medians = np.median(pooled, axis=0)
    mad = np.median(np.abs(pooled - medians), axis=0)
    gate_sigma = np.maximum(1.4826 * mad, np.median(np.abs(clean - corrupt), axis=0))
    h2 = 0.20 * gate_sigma
    gate_floor = 2.0**-10 * np.maximum(1.0, np.median(np.abs(pooled), axis=0))
    if np.any(h2 < gate_floor):
        failed = np.flatnonzero(h2 < gate_floor).tolist()
        raise GreenStop("09_RADIUS_FLOOR", f"gate radius floor failed slots {failed}")

    radius_leave = {}
    for noun in sorted(set(nouns[radius_mask])):
        keep = radius_mask & (nouns != noun)
        projected_leave = donor["chords"][keep] @ basis
        sx = float(np.median(np.linalg.norm(projected_leave, axis=1) / 2.0))
        cp, xp = donor["clean_pre"][keep], donor["corrupt_pre"][keep]
        pp = np.concatenate([cp, xp], axis=0)
        pm = np.median(pp, axis=0)
        gs = np.maximum(1.4826 * np.median(np.abs(pp - pm), axis=0), np.median(np.abs(cp - xp), axis=0))
        change = max(abs(sx - sigma_x) / sigma_x, float(np.max(np.abs(gs - gate_sigma) / gate_sigma)))
        radius_leave[noun] = change
        if change > 0.20:
            raise GreenStop("09_RADIUS_STABILITY", f"leave-{noun} change={change:.3f}")

    np.savez(
        output_root / "donor_basis.npz", U=basis, singular_values=singular,
        basis_chords=donor["chords"][basis_mask],
        leave_one_names=np.array(sorted(leave_bases)),
        leave_one_bases=np.stack([leave_bases[name] for name in sorted(leave_bases)]),
    )
    payload = {
        "sigma_x": sigma_x, "h1": h1, "h2": h2.tolist(),
        "residual_floor": residual_floor, "gate_floor": gate_floor.tolist(),
        "basis_singular_values": singular.tolist(), "leave_one_basis_angles": angles,
        "leave_one_radius_change": radius_leave,
    }
    write_json_atomic(output_root / "radii.json", payload)
    return basis, payload


def target_basis_stability(model, tokenizer, suffix_ids, output_root: Path, radii: dict, records, device: str) -> dict:
    torch = torch_module()
    completed = output_root / "target_basis_stability.json"
    cache_path = output_root / ".development_anchor_cache.pt"
    if completed.is_file() and cache_path.is_file():
        return json.loads(completed.read_text(encoding="utf-8"))
    archive = np.load(output_root / "donor_basis.npz", allow_pickle=False)
    full_U = archive["U"]
    names = archive["leave_one_names"].tolist()
    bases = [full_U, *list(archive["leave_one_bases"])]
    selected_records = []
    for cid in sorted({row.cell_id for row in records}):
        energy = [row for row in records if row.cell_id == cid and row.role == "energy"]
        selected_records.append(min(energy, key=lambda row: sha256_text("basis-target|" + row.pair_digest)))
    cached = []
    serialized = {}
    for row in selected_records:
        anchors, _, _ = capture_item_systems(model, tokenizer, suffix_ids, row, device)
        cached.append((row, anchors))
        serialized[row.pair_digest] = {
            system: _anchor_to_plain(anchor) for system, anchor in anchors.items()
        }
    torch.save(serialized, cache_path)
    vectors = []
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        for basis_np in bases:
            basis = torch.as_tensor(basis_np, dtype=torch.float32, device=device)
            values = []
            for row, anchors in cached:
                chord = (selected_position(anchors["tar"], "resid_mid") - selected_position(anchors["cor"], "resid_mid"))[0]
                q = (basis.T @ chord).double()
                direction = radii["h1"] * q / torch.clamp(torch.linalg.vector_norm(q), min=1e-30)
                effects = {}
                for system in ("tar", "pat"):
                    anchor = anchors[system]
                    target_anchor = TargetAnchor(anchor.resid_mid, anchor.pre, anchor.post, anchor.final_positions, system)
                    effects[system] = float(finite_path_effect(
                        model, target_anchor, basis, suffix_tensor, direction.float()[None], [row.y], rho=1.0
                    )[0].item())
                values.append(abs(effects["pat"] - effects["tar"]))
            vectors.append(np.array(values, dtype=np.float64))
    full = vectors[0]
    results = {}
    for name, value in zip(names, vectors[1:]):
        symmetric = np.abs(full - value) / np.maximum((np.abs(full) + np.abs(value)) / 2, 0.05)
        row = {"spearman": spearman(full, value), "median_change": float(np.median(symmetric))}
        results[name] = row
        if row["spearman"] < 0.90 or row["median_change"] > 0.20:
            raise GreenStop("08_TARGET_BASIS_STABILITY", f"leave-{name}: {row}")
    payload = {"n_cells": len(selected_records), "leave_one": results}
    write_json_atomic(completed, payload)
    return payload


def full_hook_endpoint(
    model, tokens, suffix_ids, anchor: TailAnchor, U, x, z, *, mode: str,
    gate_slot: int | None = None, block8_patch=None, subtract_residual_bypass: bool = False,
):
    """Independent full-model implementation used only for tail equivalence audits."""
    torch = torch_module()
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=tokens.device)
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    positions = torch.full_like(rows, tokens.shape[1] - 1)
    residual_delta = x @ U.T
    gate_ids = torch.as_tensor(SELECTED_GATES, dtype=torch.long, device=tokens.device)
    counts = {"x": 0, "z": 0, "post": 0, "subtract": 0, "patch": 0}

    def assert_untouched(before, after, zero_declared, label):
        difference = (after - before).clone()
        zero_declared(difference)
        maximum = float(difference.abs().max().item())
        if maximum > THRESHOLDS.hook_untouched_max:
            raise GreenStop("05_HOOK_UNTOUCHED", f"{label} changed an undeclared entry by {maximum}")

    def patch_hook(value, hook):
        counts["patch"] += 1
        result = value.clone()
        result[rows, positions] = block8_patch
        assert_untouched(value, result, lambda d: d.__setitem__((rows, positions), 0), "patch")
        return result

    def x_hook(value, hook):
        counts["x"] += 1
        result = value.clone()
        result[rows, positions] += residual_delta.to(result.dtype)
        assert_untouched(value, result, lambda d: d.__setitem__((rows, positions), 0), "x")
        return result

    def z_hook(value, hook):
        counts["z"] += 1
        result = value.clone()
        if mode == "joint":
            result[rows[:, None], positions[:, None], gate_ids[None]] += z.to(result.dtype)
        elif mode == "path":
            result[rows, positions, SELECTED_GATES[gate_slot]] += z.reshape(-1).to(result.dtype)
        if mode == "joint":
            assert_untouched(
                value, result,
                lambda d: d.__setitem__((rows[:, None], positions[:, None], gate_ids[None]), 0), "z-joint"
            )
        elif mode == "path":
            gate = SELECTED_GATES[gate_slot]
            assert_untouched(value, result, lambda d: d.__setitem__((rows, positions, gate), 0), "z-path")
        else:
            assert_untouched(value, result, lambda d: None, "z-control")
        return result

    def post_hook(value, hook):
        counts["post"] += 1
        result = value.clone()
        result[rows, positions, :] = anchor.post[rows, anchor.final_positions, :]
        if mode in {"path", "joint"}:
            if mode == "path":
                gate = SELECTED_GATES[gate_slot]
                result[rows, positions, gate] = value[rows, positions, gate]
            else:
                result[rows[:, None], positions[:, None], gate_ids[None]] = value[
                    rows[:, None], positions[:, None], gate_ids[None]
                ]
        else:
            gate = SELECTED_GATES[gate_slot]
            controlled = anchor.pre[rows, anchor.final_positions, gate] + z.reshape(-1)
            result[rows, positions, gate] = model.blocks[10].mlp.act_fn(controlled)
        assert_untouched(value, result, lambda d: d.__setitem__((rows, positions), 0), "post")
        return result

    def subtract_hook(value, hook):
        counts["subtract"] += 1
        result = value.clone()
        result[rows, positions] -= residual_delta.to(result.dtype)
        assert_untouched(value, result, lambda d: d.__setitem__((rows, positions), 0), "subtract")
        return result

    hooks = []
    if block8_patch is not None:
        hooks.append(("blocks.8.hook_mlp_out", patch_hook))
    hooks.extend([
        ("blocks.10.hook_resid_mid", x_hook),
        ("blocks.10.mlp.hook_pre", z_hook),
        ("blocks.10.mlp.hook_post", post_hook),
    ])
    if subtract_residual_bypass:
        hooks.append(("blocks.10.hook_resid_post", subtract_hook))
    try:
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    finally:
        model.reset_hooks()
    required = {"x": 1, "z": 1, "post": 1, "subtract": int(subtract_residual_bypass), "patch": int(block8_patch is not None)}
    if counts != required:
        raise GreenStop("05_HOOK_INVOCATION", f"hook counts {counts}, expected {required}")
    return gather_year_logits(model, logits, positions, suffix_tensor)


def tail_audit(model, tokenizer, suffix_ids, U_np, radii: dict, records, device: str, cached_anchors: dict) -> dict:
    torch = torch_module()
    U = torch.as_tensor(U_np, dtype=torch.float32, device=device)
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    tail = GreenBridgeTail(model, U, suffix_tensor)
    errors, derivative_errors = [], []
    kinds = ["center"] * 8 + ["x"] * 8 + ["z"] * 8 + ["path"] * 4 + ["control"] * 4
    ranked = sorted(records, key=lambda row: sha256_text("tail-audit|" + row.pair_digest))[:32]
    with torch.inference_mode():
        for index, (row, kind) in enumerate(zip(ranked, kinds)):
            tokens = tokenize_one(tokenizer, row.clean_prompt, device)
            plain = cached_anchors[row.pair_digest]
            anchor = TailAnchor(**{
                key: (value.to(device) if hasattr(value, "to") else value)
                for key, value in plain.items()
            })
            x = torch.zeros((1, 4), dtype=torch.float32, device=device)
            z = torch.zeros(1, dtype=torch.float32, device=device)
            gate_slot = index % 10
            mode = "path"
            if kind == "x":
                x[0, index % 4] = radii["h1"]
            elif kind == "z":
                z[0] = radii["h2"][gate_slot]
            elif kind in {"path", "control"}:
                mode = kind
                x[0, index % 4] = radii["h1"]
                z[0] = radii["h2"][gate_slot]
            manual = tail.evaluate(anchor, x, z, mode=mode, gate_slot=gate_slot)
            full = full_hook_endpoint(
                model, tokens, suffix_ids, anchor, U, x, z, mode=mode, gate_slot=gate_slot
            )
            errors.append(float((manual.double() - full.double()).abs().max().item()))
            manual_delta = manual.double() - anchor.year_logits.double()
            full_delta = full.double() - anchor.year_logits.double()
            relative = float(
                torch.linalg.vector_norm(manual_delta - full_delta).item()
                / max(float(torch.linalg.vector_norm(full_delta).item()), 1e-5)
            )
            derivative_errors.append(relative)
    result = {
        "n": 32, "condition_types": kinds, "max_abs": max(errors), "errors": errors,
        "derivative_relative_max": max(derivative_errors),
        "derivative_relative_errors": derivative_errors,
    }
    if result["max_abs"] > THRESHOLDS.tail_max_abs:
        raise GreenStop("06_MANUAL_TAIL", f"max error {result['max_abs']:.3e}")
    if result["derivative_relative_max"] > THRESHOLDS.tail_derivative_relative:
        raise GreenStop("06_MANUAL_TAIL_DERIVATIVE", f"relative error {result['derivative_relative_max']:.3e}")
    return result


def _jet_at_radius(tail: GreenBridgeTail, anchor: TailAnchor, gate_slot: int, hx: float, hz: float, center) -> GateJet:
    """Evaluate the exact 42-condition one-gate design in two batched forwards."""
    torch = torch_module()
    device = anchor.resid_mid.device
    path_x, path_z = [], []
    # Two z-axis endpoints.
    for sz in (1.0, -1.0):
        path_x.append(np.zeros(4)); path_z.append(sz * hz)
    # Eight x-axis endpoints.
    for axis in range(4):
        for sx in (1.0, -1.0):
            value = np.zeros(4); value[axis] = sx * hx
            path_x.append(value); path_z.append(0.0)
    # Sixteen path corners, ordered by axis then (++,+-,-+,--).
    for axis in range(4):
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            value = np.zeros(4); value[axis] = sx * hx
            path_x.append(value); path_z.append(sz * hz)
    control_x, control_z = [], []
    for axis in range(4):
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            value = np.zeros(4); value[axis] = sx * hx
            control_x.append(value); control_z.append(sz * hz)
    px = torch.as_tensor(np.stack(path_x), dtype=torch.float32, device=device)
    pz = torch.as_tensor(path_z, dtype=torch.float32, device=device)
    cx = torch.as_tensor(np.stack(control_x), dtype=torch.float32, device=device)
    cz = torch.as_tensor(control_z, dtype=torch.float32, device=device)
    path = tail.evaluate(_repeat_anchor(anchor, 26), px, pz, mode="path", gate_slot=gate_slot).double()
    control = tail.evaluate(_repeat_anchor(anchor, 16), cx, cz, mode="control", gate_slot=gate_slot).double()
    G = (path[0] - path[1]) / (2 * hz)
    C = (path[0] - 2 * center + path[1]) / (hz * hz)
    J, HP, HC = [], [], []
    for axis in range(4):
        J.append((path[2 + 2 * axis] - path[3 + 2 * axis]) / (2 * hx))
        p = path[10 + 4 * axis:14 + 4 * axis]
        c = control[4 * axis:4 + 4 * axis]
        HP.append((p[0] - p[1] - p[2] + p[3]) / (4 * hx * hz))
        HC.append((c[0] - c[1] - c[2] + c[3]) / (4 * hx * hz))
    return GateJet(
        G.cpu().numpy(), C.cpu().numpy(), torch.stack(J).cpu().numpy(),
        torch.stack(HP).cpu().numpy(), torch.stack(HC).cpu().numpy(),
    )


def whitebox_A(model, anchor: TailAnchor, U, gate_slot: int) -> np.ndarray:
    """Architecture-derived audit only; its return value never enters P."""
    gate = SELECTED_GATES[gate_slot]
    resid = selected_position(anchor, "resid_mid")[0].detach().double().cpu().numpy()
    basis = U.detach().double().cpu().numpy()
    weight = model.blocks[10].ln2.w.detach().double().cpu().numpy()
    w_in = model.blocks[10].mlp.W_in[:, gate].detach().double().cpu().numpy()
    centered = resid - resid.mean()
    variance = float(np.mean(centered**2) + model.cfg.eps)
    scale = math.sqrt(variance)
    dimension = len(resid)
    projector = (
        np.eye(dimension, dtype=np.float64)
        - np.ones((dimension, dimension), dtype=np.float64) / dimension
        - np.outer(centered, centered) / (dimension * variance)
    )
    jacobian_ln = (weight[:, None] * projector) / scale
    return w_in @ jacobian_ln @ basis


def classify_gate(full, half, rich, wb_A, contrast_norm: float, delta_norm: float, epsilon_y: float, hx: float, hz: float) -> tuple[str, dict]:
    numerical = richardson_numerical_bounds(
        rich, half, epsilon_y=epsilon_y, h1=hx, h2=hz
    )
    rich_gate_response_norm = float(np.linalg.norm(rich.G))
    full_gate_response_norm = float(np.linalg.norm(full.G))
    half_gate_response_norm = float(np.linalg.norm(half.G))
    rich_curvature_norm = float(np.linalg.norm(rich.C))
    wb_norm = np.linalg.norm(wb_A)
    full_id = half_id = rich_id = None
    if numerical.inverse_admissible:
        try:
            full_id, half_id, rich_id = (
                identify_gate(full), identify_gate(half), identify_gate(rich)
            )
        except ValueError:
            pass
    wb_error = math.inf if rich_id is None else float(np.linalg.norm(rich_id.A - wb_A))
    wb_ok = (
        rich_id is not None
        and wb_error <= THRESHOLDS.whitebox_a_relative_max * max(wb_norm, 1e-6)
    )
    if rich_id is not None and wb_norm < 1e-6:
        wb_ok = wb_error <= THRESHOLDS.whitebox_a_small_absolute_max
    active = (
        rich_id is not None
        and rich_curvature_norm / 10 >= THRESHOLDS.curvature_rms_min
        and rich_curvature_norm >= THRESHOLDS.curvature_snr_min * numerical.epsilon_C
        and rich_gate_response_norm / 10 >= THRESHOLDS.gate_response_rms_min
        and rich_gate_response_norm >= THRESHOLDS.gate_response_snr_min * numerical.epsilon_G
        and rich_id.factorization_residual <= THRESHOLDS.factorization_residual_max
        and wb_ok
        and cosine(full_id.P, half_id.P) >= THRESHOLDS.tensor_cosine_min
        and symmetric_relative_change(full_id.P, half_id.P) <= THRESHOLDS.tensor_symmetric_change_max
        and np.linalg.norm(rich_id.P - half_id.P) / max(np.linalg.norm(rich_id.P), 1e-8)
            <= THRESHOLDS.richardson_change_max
        and np.linalg.norm(rich_id.P) >= THRESHOLDS.tensor_snr_min * numerical.epsilon_P_F
    )
    null_bound = certified_null_bound(
        contrast_norm, delta_norm, rich_gate_response_norm,
        numerical.epsilon_G, wb_norm,
    )
    full_bound = contrast_norm * delta_norm * full_gate_response_norm * wb_norm
    half_bound = contrast_norm * delta_norm * half_gate_response_norm * wb_norm
    null = (
        rich_gate_response_norm <= 5 * numerical.epsilon_G
        and null_bound <= 0.005
        and abs(full_bound - half_bound) <= 0.005
    )
    label = "active-identified" if active else ("certified-target-null" if null else "invalid")
    audit = {
        "label": label, "curvature_norm": rich_curvature_norm,
        "gate_response_norm": rich_gate_response_norm,
        "factorization_residual": (
            None if rich_id is None else rich_id.factorization_residual
        ),
        "whitebox_error": None if not np.isfinite(wb_error) else float(wb_error),
        "whitebox_norm": float(wb_norm),
        "full_half_cosine": (
            None if rich_id is None else cosine(full_id.P, half_id.P)
        ),
        "full_half_change": (
            None if rich_id is None else symmetric_relative_change(full_id.P, half_id.P)
        ),
        "richardson_change": (
            None if rich_id is None else float(
                np.linalg.norm(rich_id.P - half_id.P)
                / max(np.linalg.norm(rich_id.P), 1e-8)
            )
        ),
        "null_bound": float(null_bound),
        "null_full_half_bound_change": float(abs(full_bound - half_bound)),
        "inverse_admissible": numerical.inverse_admissible,
        "eta_G": numerical.eta_G,
        "eta_C": numerical.eta_C,
        "eta_J": numerical.eta_J,
        "eta_H": numerical.eta_H,
        "epsilon_G": numerical.epsilon_G,
        "epsilon_C": numerical.epsilon_C,
        "epsilon_delta_H": numerical.epsilon_delta_H.tolist(),
        "A_max": [float(value) if np.isfinite(value) else None for value in numerical.A_max],
        "epsilon_A": [float(value) if np.isfinite(value) else None for value in numerical.epsilon_A],
        "epsilon_P": [float(value) if np.isfinite(value) else None for value in numerical.epsilon_P],
        "epsilon_P_F": (
            numerical.epsilon_P_F if np.isfinite(numerical.epsilon_P_F) else None
        ),
    }
    return label, {
        "audit": audit,
        "full": full_id,
        "half": half_id,
        "rich": rich_id,
        "numerical": numerical,
    }


def mixed_system(tail, model, anchor, U, radii, delta, contrast, epsilon_y: float) -> dict:
    torch = torch_module()
    zero_x = torch.zeros((1, 4), dtype=torch.float32, device=anchor.resid_mid.device)
    zero_z = torch.zeros(1, dtype=torch.float32, device=anchor.resid_mid.device)
    center = tail.evaluate(anchor, zero_x, zero_z, mode="path", gate_slot=0)[0].double()
    center_error = center - anchor.year_logits[0].double()
    center_rms = float(torch.sqrt(torch.mean(center_error**2)).item())
    center_max = float(center_error.abs().max().item())
    if center_rms > THRESHOLDS.center_rms or center_max > THRESHOLDS.center_max_abs:
        return {"theta": 0.0, "theta_full": 0.0, "theta_half": 0.0, "active_gates": 0,
                "all_valid": False, "bypass_disagreement": None, "gates": [],
                "admissible": False, "center_rms": center_rms, "center_max": center_max,
                "theta_error": None}
    gates, theta = [], 0.0
    theta_full, theta_half = 0.0, 0.0
    direct, gate_error_bounds = [], []
    for gate_slot in range(10):
        full = _jet_at_radius(tail, anchor, gate_slot, radii["h1"], radii["h2"][gate_slot], center)
        half = _jet_at_radius(tail, anchor, gate_slot, radii["h1"] / 2, radii["h2"][gate_slot] / 2, center)
        rich = extrapolate_gate_jet(full, half)
        wb = whitebox_A(model, anchor, U, gate_slot)
        label, values = classify_gate(
            full, half, rich, wb, float(np.linalg.norm(contrast)), float(np.linalg.norm(delta)),
            epsilon_y, radii["h1"], radii["h2"][gate_slot],
        )
        if label == "active-identified":
            for key, accumulator in (("rich", "theta"), ("full", "theta_full"), ("half", "theta_half")):
                contraction = contrast @ (delta @ values[key].P)
                if accumulator == "theta": theta += contraction
                elif accumulator == "theta_full": theta_full += contraction
                else: theta_half += contraction
            direct.append(values["rich"].D)
            gate_error_bounds.append(active_contraction_bound(
                float(np.linalg.norm(contrast)),
                float(np.linalg.norm(delta)),
                values["numerical"].epsilon_P_F,
            ))
        elif label == "certified-target-null":
            gate_error_bounds.append(values["audit"]["null_bound"])
        gates.append(values["audit"])
    active = sum(row["label"] == "active-identified" for row in gates)
    complete = all(row["label"] != "invalid" for row in gates)
    bypass = None
    if direct:
        stack = np.stack(direct)
        mean = stack.mean(axis=0)
        bypass = float(np.sqrt(np.mean((stack - mean) ** 2)) / max(np.sqrt(np.mean(mean**2)), 1e-12))
    return {
        "theta": float(theta), "theta_full": float(theta_full), "theta_half": float(theta_half),
        "theta_error": sum_item_error_bounds(gate_error_bounds) if complete else None,
        "active_gates": active, "all_valid": complete,
        "bypass_disagreement": bypass, "gates": gates,
        "admissible": (
            complete
            and active >= THRESHOLDS.active_gates_min
            and bypass is not None
            and bypass <= THRESHOLDS.bypass_disagreement_max
        ),
        "center_rms": center_rms, "center_max": center_max,
    }


def joint_margins(tail, anchor, xs: np.ndarray, zs: np.ndarray, contrast) -> np.ndarray:
    torch = torch_module()
    device = anchor.resid_mid.device
    device_name = torch.cuda.get_device_name(device)
    batch_limit = 512 if "4090" in device_name else 1024
    outputs = []
    for start in range(0, len(xs), batch_limit):
        stop = min(start + batch_limit, len(xs))
        x = torch.as_tensor(xs[start:stop], dtype=torch.float32, device=device)
        z = torch.as_tensor(zs[start:stop], dtype=torch.float32, device=device)
        logits = tail.evaluate(_repeat_anchor(anchor, stop - start), x, z, mode="joint")
        outputs.append(margin(logits, contrast).detach().cpu().numpy())
    return np.concatenate(outputs)


def first_order_system(tail, anchor, radii, directions: np.ndarray, contrast) -> dict:
    xs, zs, descriptors = [], [], []
    for radius_name, rho in (("full", 1.0), ("half", 0.5)):
        for kind in ("x", "z"):
            count = 200 if kind == "x" else 10
            for axis in range(count):
                for sign in (1.0, -1.0):
                    x = np.zeros(4); z = np.zeros(10)
                    if kind == "x": x = sign * rho * radii["h1"] * directions[axis]
                    else: z[axis] = sign * rho * radii["h2"][axis]
                    xs.append(x); zs.append(z); descriptors.append((radius_name, kind, axis, sign, rho))
    values = joint_margins(tail, anchor, np.stack(xs), np.stack(zs), contrast)
    response = {"full": {"x": np.zeros(200), "z": np.zeros(10)}, "half": {"x": np.zeros(200), "z": np.zeros(10)}}
    endpoints = {}
    for value, descriptor in zip(values, descriptors):
        radius_name, kind, axis, sign, rho = descriptor
        endpoints[radius_name, kind, axis, sign] = value
    for radius_name, rho in (("full", 1.0), ("half", 0.5)):
        for kind, count in (("x", 200), ("z", 10)):
            for axis in range(count):
                response[radius_name][kind][axis] = (
                    endpoints[radius_name, kind, axis, 1.0]
                    - endpoints[radius_name, kind, axis, -1.0]
                ) / (2 * rho)
    rich = {
        kind: (4 * response["half"][kind] - response["full"][kind]) / 3
        for kind in ("x", "z")
    }
    return {"response": response, "rich": rich}


def factorial_system(tail, anchor, delta, zeta, contrast) -> dict:
    xs, zs, descriptors = [], [], []
    for radius_name, rho in (("full", 1.0), ("half", 0.5)):
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            xs.append(sx * rho * delta); zs.append(sz * rho * zeta)
            descriptors.append((radius_name, rho, sx, sz))
    values = joint_margins(tail, anchor, np.stack(xs), np.stack(zs), contrast)
    endpoints = {(name, sx, sz): value for value, (name, _, sx, sz) in zip(values, descriptors)}
    rows = {}
    for name, rho in (("full", 1.0), ("half", 0.5)):
        pp, pm = endpoints[name, 1, 1], endpoints[name, 1, -1]
        mp, mm = endpoints[name, -1, 1], endpoints[name, -1, -1]
        rows[name] = {
            "single": (pp - mm) / (2 * rho),
            "pie": (pp - pm - mp + mm) / (4 * rho * rho),
            "bx": (pp + pm - mp - mm) / (4 * rho),
            "bz": (pp - pm + mp - mm) / (4 * rho),
        }
    return {key: (4 * rows["half"][key] - rows["full"][key]) / 3 for key in rows["full"]} | {
        "raw": rows
    }


def tensor_item(model, tokenizer, suffix_ids, U_np, radii, record, device, epsilon_y, directions) -> dict:
    torch = torch_module()
    anchors, _, _ = capture_item_systems(model, tokenizer, suffix_ids, record, device)
    U = torch.as_tensor(U_np, dtype=torch.float32, device=device)
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    tail = GreenBridgeTail(model, U, suffix_tensor)
    contrast_t = margin_vector(record.y, device)
    contrast = contrast_t.cpu().numpy()
    chord = (selected_position(anchors["tar"], "resid_mid") - selected_position(anchors["cor"], "resid_mid"))[0]
    q = (U.T @ chord).double().cpu().numpy()
    q_norm = float(np.linalg.norm(q))
    direction_valid = q_norm >= THRESHOLDS.projected_chord_sigma_min * radii["sigma_x"]
    delta = radii["h1"] * q / max(q_norm, 1e-30)

    mixed = {}
    with torch.inference_mode():
        for system in ("tar", "pat"):
            mixed[system] = mixed_system(tail, model, anchors[system], U, radii, delta, contrast, epsilon_y)

    pre_tar = selected_position(anchors["tar"], "pre")[0, list(SELECTED_GATES)].double().cpu().numpy()
    pre_cor = selected_position(anchors["cor"], "pre")[0, list(SELECTED_GATES)].double().cpu().numpy()
    v = (pre_tar - pre_cor) / np.asarray(radii["h2"])
    zeta = np.asarray(radii["h2"]) * v / max(float(np.linalg.norm(v)), 1e-30)
    fo, factorial = {}, {}
    with torch.inference_mode():
        for system in ("tar", "pat"):
            fo[system] = first_order_system(tail, anchors[system], radii, directions, contrast_t)
            factorial[system] = factorial_system(tail, anchors[system], delta, zeta, contrast_t)
    center_tar = float(margin(anchors["tar"].year_logits, contrast_t)[0].item())
    center_pat = float(margin(anchors["pat"].year_logits, contrast_t)[0].item())
    first_order_score = math.sqrt(
        float(np.mean((fo["pat"]["rich"]["x"] - fo["tar"]["rich"]["x"]) ** 2))
        + float(np.mean((fo["pat"]["rich"]["z"] - fo["tar"]["rich"]["z"]) ** 2))
    )
    admissible = direction_valid and mixed["tar"]["admissible"] and mixed["pat"]["admissible"]
    return {
        "pair_digest": record.pair_digest, "cell_id": record.cell_id,
        "split": record.split, "distance_bin": record.distance_bin, "orientation": record.orientation,
        "admissible": admissible, "q_norm": q_norm,
        "theta_tar": mixed["tar"]["theta"], "theta_pat": mixed["pat"]["theta"],
        "theta_full_tar": mixed["tar"]["theta_full"], "theta_full_pat": mixed["pat"]["theta_full"],
        "theta_half_tar": mixed["tar"]["theta_half"], "theta_half_pat": mixed["pat"]["theta_half"],
        "theta_error_tar": mixed["tar"].get("theta_error", float("inf")),
        "theta_error_pat": mixed["pat"].get("theta_error", float("inf")),
        "behavioral": abs(center_pat - center_tar),
        "single": abs(factorial["pat"]["single"] - factorial["tar"]["single"]),
        "first_order": first_order_score,
        "pie": abs(factorial["pat"]["pie"] - factorial["tar"]["pie"]),
        "cancellation_dx": factorial["pat"]["bx"] - factorial["tar"]["bx"],
        "cancellation_dz": factorial["pat"]["bz"] - factorial["tar"]["bz"],
        "mixed_audit": {system: {key: value for key, value in mixed[system].items() if key != "gates"} | {"gates": mixed[system]["gates"]} for system in mixed},
    }


def energy_item(model, tokenizer, suffix_ids, U_np, radii, record, device, anchor_cache=None) -> dict:
    torch = torch_module()
    anchors, _, _ = capture_item_systems(
        model, tokenizer, suffix_ids, record, device, anchor_cache=anchor_cache
    )
    U = torch.as_tensor(U_np, dtype=torch.float32, device=device)
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    chord = (selected_position(anchors["tar"], "resid_mid") - selected_position(anchors["cor"], "resid_mid"))[0]
    q = (U.T @ chord).double().cpu().numpy()
    q_norm = float(np.linalg.norm(q))
    direction_valid = q_norm >= THRESHOLDS.projected_chord_sigma_min * radii["sigma_x"]
    delta_np = radii["h1"] * q / max(q_norm, 1e-30)
    direction = torch.as_tensor(delta_np[None], dtype=torch.float32, device=device)
    systems = {}
    for name in ("tar", "pat", "cor"):
        anchor = anchors[name]
        target_anchor = TargetAnchor(
            resid_mid=anchor.resid_mid, pre=anchor.pre, post=anchor.post,
            final_positions=anchor.final_positions, system=name,
        )
        full = finite_path_effect(
            model, target_anchor, U, suffix_tensor, direction, [record.y], rho=1.0
        )[0]
        half = finite_path_effect(
            model, target_anchor, U, suffix_tensor, direction, [record.y], rho=0.5
        )[0]
        jvp = target_jvp(model, target_anchor, U, suffix_tensor, direction, [record.y])[0]
        full_v, half_v, jvp_v = float(full.item()), float(half.item()), float(jvp.item())
        rich_v = (4 * half_v - full_v) / 3
        absolute = abs(rich_v - jvp_v)
        relative = absolute / max(abs(jvp_v), 0.05)
        locality = abs(full_v - rich_v)
        admissible = (
            absolute <= 0.01 and relative <= 0.05
            and locality <= max(0.02, 0.25 * abs(rich_v))
        )
        systems[name] = {
            "full": full_v, "half": half_v, "jvp": jvp_v,
            "richardson": rich_v,
            "jvp_absolute_error": absolute, "jvp_relative_error": relative,
            "locality_error": locality, "admissible": admissible,
        }
    return {
        "pair_digest": record.pair_digest, "cell_id": record.cell_id,
        "split": record.split, "distance_bin": record.distance_bin, "orientation": record.orientation,
        "q_norm": q_norm, "admissible": direction_valid and all(row["admissible"] for row in systems.values()),
        "systems": systems,
    }


def append_journal(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_journal(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_parquet(path: Path, rows: list[dict]) -> None:
    import pandas as pd
    flattened = []
    for row in rows:
        record = {}
        for key, value in row.items():
            record[key] = canonical_json(value) if isinstance(value, (dict, list)) else value
        flattened.append(record)
    pd.DataFrame(flattened).to_parquet(path, index=False, engine="pyarrow")


def aggregate_cells(tensor_rows: list[dict], energy_rows: list[dict], *, dev_sd: float | None = None) -> tuple[dict, float]:
    cells = []
    cell_ids = sorted({row["cell_id"] for row in tensor_rows} | {row["cell_id"] for row in energy_rows})
    provisional = []
    for cid in cell_ids:
        tensor = [row for row in tensor_rows if row["cell_id"] == cid and row["admissible"]]
        energy = [row for row in energy_rows if row["cell_id"] == cid and row["admissible"]]
        survived = len(tensor) >= 6 and len(energy) >= 6
        if not survived:
            provisional.append({"cell_id": cid, "survived": False, "n_tensor": len(tensor), "n_energy": len(energy)})
            continue
        mean = lambda values: float(np.mean(list(values)))
        theta_tar = mean(row["theta_tar"] for row in tensor)
        theta_pat = mean(row["theta_pat"] for row in tensor)
        target_tar = mean(row["systems"]["tar"]["full"] for row in energy)
        target_pat = mean(row["systems"]["pat"]["full"] for row in energy)
        target_cor = mean(row["systems"]["cor"]["full"] for row in energy)
        row = {
            "cell_id": cid, "distance_bin": tensor[0]["distance_bin"], "survived": True,
            "n_tensor": len(tensor), "n_energy": len(energy),
            "mixed": abs(theta_pat - theta_tar),
            "mixed_full": abs(mean(r["theta_full_pat"] for r in tensor) - mean(r["theta_full_tar"] for r in tensor)),
            "mixed_half": abs(mean(r["theta_half_pat"] for r in tensor) - mean(r["theta_half_tar"] for r in tensor)),
            "target": abs(target_pat - target_tar),
            "conditioning_gap": abs(target_tar - target_cor),
            "baselines": {name: mean(r[name] for r in tensor) for name in BASELINES},
            "cancellation_dx": mean(r["cancellation_dx"] for r in tensor),
            "cancellation_dz": mean(r["cancellation_dz"] for r in tensor),
        }
        # Conservative cell error from duplicated-logit floor, used only as the
        # preregistered SNR gate; no observed target is fed into the predictor.
        row["error_bound"] = max(
            1e-7,
            cell_error_bound(
                (r["theta_error_tar"] for r in tensor),
                (r["theta_error_pat"] for r in tensor),
            ),
        )
        row["snr"] = row["mixed"] / row["error_bound"]
        provisional.append(row)
    surviving = [row for row in provisional if row["survived"]]
    if dev_sd is None:
        dev_sd = float(np.std([row["conditioning_gap"] for row in surviving], ddof=1)) if len(surviving) > 1 else 0.0
    for row in surviving:
        row["conditioned"] = (
            row["conditioning_gap"] >= THRESHOLDS.conditioning_absolute
            or row["conditioning_gap"] >= THRESHOLDS.conditioning_dev_sd * dev_sd
        )
    cells.extend(provisional)
    return {"cells": cells, "conditioning_dev_sd": dev_sd}, dev_sd


def run_split(model, tokenizer, suffix_ids, U, radii, records, split: str, output_root: Path, device: str, epsilon_y: float, dev_sd: float | None = None) -> dict:
    tensor_records = [row for row in records if row.role == "tensor"]
    energy_records = [row for row in records if row.role == "energy"]
    tensor_journal = output_root / f".{split}_tensor.journal.jsonl"
    energy_journal = output_root / f".{split}_energy.journal.jsonl"
    tensor_rows, energy_rows = read_journal(tensor_journal), read_journal(energy_journal)
    tensor_done = {row["pair_digest"] for row in tensor_rows}
    energy_done = {row["pair_digest"] for row in energy_rows}
    directions = first_order_directions()
    anchor_cache = merge_anchor_caches(
        load_anchor_cache(output_root / ".development_anchor_cache.pt", device)
        if split == "development" else {},
        load_anchor_cache(output_root / f".{split}_noise_anchor_cache.pt", device),
    )
    if split == "development" and not (output_root / "throughput_preflight.json").is_file():
        sample_tensor = sorted(
            tensor_records, key=lambda row: sha256_text("preflight-tensor|" + row.pair_digest)
        )[:math.ceil(0.02 * len(tensor_records))]
        sample_energy = sorted(
            energy_records, key=lambda row: sha256_text("preflight-energy|" + row.pair_digest)
        )[:math.ceil(0.02 * len(energy_records))]
        tensor_start = time.perf_counter()
        for record in sample_tensor:
            if record.pair_digest not in tensor_done:
                row = tensor_item(model, tokenizer, suffix_ids, U, radii, record, device, epsilon_y, directions)
                append_journal(tensor_journal, row); tensor_rows.append(row); tensor_done.add(record.pair_digest)
        tensor_seconds = time.perf_counter() - tensor_start
        energy_start = time.perf_counter()
        for record in sample_energy:
            if record.pair_digest not in energy_done:
                row = energy_item(model, tokenizer, suffix_ids, U, radii, record, device, anchor_cache)
                append_journal(energy_journal, row); energy_rows.append(row); energy_done.add(record.pair_digest)
        energy_seconds = time.perf_counter() - energy_start
        forecast_hours = (
            tensor_seconds / len(sample_tensor) * len(tensor_records)
            + energy_seconds / len(sample_energy) * len(energy_records)
        ) / 3600
        memory_gb = torch_module().cuda.max_memory_allocated() / 2**30
        ceiling, cap = (20.0, 24.0) if "4090" in torch_module().cuda.get_device_name() else (32.0, 40.0)
        write_json_atomic(output_root / "throughput_preflight.json", {
            "fraction": 0.02, "tensor_sample": len(sample_tensor), "energy_sample": len(sample_energy),
            "tensor_seconds": tensor_seconds, "energy_seconds": energy_seconds,
            "forecast_hours": forecast_hours, "peak_allocated_gb": memory_gb,
            "memory_ceiling_gb": ceiling, "hard_hours": cap, "outputs_reused": True,
        })
        if memory_gb > ceiling or forecast_hours > cap:
            raise GreenStop("10_THROUGHPUT_MEMORY", f"forecast={forecast_hours:.2f}h memory={memory_gb:.2f}GB")
    for index, record in enumerate(tensor_records):
        if record.pair_digest in tensor_done:
            continue
        row = tensor_item(model, tokenizer, suffix_ids, U, radii, record, device, epsilon_y, directions)
        append_journal(tensor_journal, row); tensor_rows.append(row)
        print(f"{split} tensor {len(tensor_rows)}/{len(tensor_records)} admissible={row['admissible']}", flush=True)
    for record in energy_records:
        if record.pair_digest in energy_done:
            continue
        row = energy_item(model, tokenizer, suffix_ids, U, radii, record, device, anchor_cache)
        append_journal(energy_journal, row); energy_rows.append(row)
        print(f"{split} energy {len(energy_rows)}/{len(energy_records)} admissible={row['admissible']}", flush=True)
    prefix = "dev" if split == "development" else "confirm"
    write_parquet(output_root / f"{prefix}_tensor_scores.parquet", tensor_rows)
    write_parquet(output_root / f"{prefix}_energy_targets.parquet", energy_rows)
    payload, derived_sd = aggregate_cells(tensor_rows, energy_rows, dev_sd=dev_sd)
    return payload | {"tensor_rows": len(tensor_rows), "energy_rows": len(energy_rows)}, derived_sd


def duplicate_noise_audit(model, tokenizer, suffix_ids, U_np, radii, records, device: str, phase: str, output_root: Path) -> dict:
    torch = torch_module()
    suffix_tensor = torch.as_tensor(suffix_ids, dtype=torch.long, device=device)
    U = torch.as_tensor(U_np, dtype=torch.float32, device=device)
    tail = GreenBridgeTail(model, U, suffix_tensor)
    n_full = 64 if phase == "development" else 32
    ranked = sorted(records, key=lambda row: sha256_text(f"noise-{phase}|" + row.pair_digest))
    errors, serialized, captured = [], {}, {}
    with torch.inference_mode():
        for row in ranked[:n_full]:
            tokens = tokenize_one(tokenizer, row.clean_prompt, device)
            first = capture_tail_anchor(model, tokens, suffix_tensor, system="noise")
            second = capture_tail_anchor(model, tokens, suffix_tensor, system="noise")
            errors.append(float((first.year_logits.double() - second.year_logits.double()).abs().max().item()))
            serialized[row.pair_digest] = {"tar": _anchor_to_plain(replace(first, system="tar"))}
            captured[row.pair_digest] = first
        for index, row in enumerate(ranked[:32]):
            tokens = tokenize_one(tokenizer, row.clean_prompt, device)
            anchor = captured[row.pair_digest]
            x = torch.zeros((1, 4), dtype=torch.float32, device=device)
            z = torch.zeros(1, dtype=torch.float32, device=device)
            x[0, index % 4] = radii["h1"] * (1 if index % 2 else -1)
            z[0] = radii["h2"][index % 10] * (1 if index % 3 else -1)
            first = tail.evaluate(anchor, x, z, mode="path", gate_slot=index % 10)
            second = tail.evaluate(anchor, x, z, mode="path", gate_slot=index % 10)
            errors.append(float((first.double() - second.double()).abs().max().item()))
    torch.save(serialized, output_root / f".{phase}_noise_anchor_cache.pt")
    maximum = max([0.0, *errors])
    result = {"phase": phase, "n_full": n_full, "n_tail": 32, "max_abs": maximum, "errors": errors}
    return result


def load_split_file(output_root: Path, filename: str = "splits.json") -> list[PairRecord]:
    payload = json.loads((output_root / filename).read_text(encoding="utf-8"))
    return [PairRecord(**row) for row in payload["records"]]


def load_frozen_numeric(output_root: Path, device: str):
    basis = np.load(output_root / "donor_basis.npz", allow_pickle=False)["U"]
    radii = json.loads((output_root / "radii.json").read_text(encoding="utf-8"))
    tokenizer, hf_model, model, cfg = load_models(device)
    suffix_ids = json.loads((output_root / "model_fingerprint.json").read_text(encoding="utf-8"))["tokenizer"]["suffix_ids"]
    return basis, radii, tokenizer, hf_model, model, suffix_ids, cfg


def prepare(output_root: Path, device: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    environment = configure_runtime(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    pair_allowed = lambda first, second: token_pair_allowed(tokenizer, first, second)
    evaluation = build_evaluation_records(pair_allowed)
    donors = build_donor_records(pair_allowed)
    legacy_gate04, holdout_gate04 = gate04_record_panels(donors)
    gate04_panel = gate04_panel_metadata(legacy_gate04, holdout_gate04)
    split_payload = write_plan(output_root / "splits.json", evaluation + donors)
    # Development gets its own physically separate view so its process never
    # parses confirmation prompt strings before the frozen analysis exists.
    write_json_atomic(
        output_root / "development_splits.json",
        plan_payload([row for row in evaluation if row.split == "development"]),
    )
    tokenizer, hf_model, model, cfg = load_models(device, tokenizer=tokenizer)
    suffix_ids, tokenizer_meta = validate_tokenizer(tokenizer, evaluation + donors)
    manifest = write_initial_manifest(
        output_root, environment, cfg, split_payload, gate04_panel
    )
    fingerprint = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
        "config": cfg, "tokenizer": tokenizer_meta,
        "embedding_sha256": hashlib.sha256(model.W_E.detach().cpu().numpy().tobytes()).hexdigest(),
        "unembedding_sha256": hashlib.sha256(model.W_U.detach().cpu().numpy().tobytes()).hexdigest(),
    }
    write_json_atomic(output_root / "model_fingerprint.json", fingerprint)
    hf_audit = hf_tl_audit(
        tokenizer, hf_model, model, legacy_gate04, holdout_gate04,
        suffix_ids, device,
    )
    noop = no_op_audit(
        model, tokenizer, holdout_gate04, suffix_ids, device,
        hf_audit["tl_references"],
    )
    write_json_atomic(output_root / "hook_audit.json", {"hf_vs_tl": hf_audit, "no_op_patch": noop})
    del hf_model
    torch_module().cuda.empty_cache()
    donor = donor_anchors(model, tokenizer, suffix_ids, donors, device)
    U, radii = build_basis_and_radii(donor, output_root)
    tail_result = tail_audit(
        model, tokenizer, suffix_ids, U, radii, donors, device, donor["audit_anchors"]
    )
    write_json_atomic(output_root / "tail_audit.json", tail_result)
    manifest["artifact_sha256"] = {
        name: sha256_file(output_root / name)
        for name in ("model_fingerprint.json", "splits.json", "donor_basis.npz", "radii.json", "hook_audit.json", "tail_audit.json")
    }
    manifest["prepare_complete"] = True
    write_json_atomic(output_root / "manifest.json", manifest)


def verify_freeze(output_root: Path, require_confirmation: bool = False) -> dict:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise GreenStop("17_MANIFEST_FREEZE", "manifest.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["frozen_spec_sha256"] != frozen_spec_hash():
        raise GreenStop("17_MANIFEST_FREEZE", "frozen spec hash changed")
    if manifest["source_sha256"] != source_hashes():
        raise GreenStop("17_MANIFEST_FREEZE", "source hashes changed after launch")
    if require_confirmation:
        frozen_path = output_root / "frozen_analysis.json"
        if not frozen_path.is_file() or not manifest.get("confirmation_open", False):
            raise GreenStop("17_MANIFEST_FREEZE", "confirmation lock is closed")
    return manifest


def development_phase(output_root: Path, device: str) -> None:
    manifest = verify_freeze(output_root)
    U, radii, tokenizer, hf_model, model, suffix_ids, _ = load_frozen_numeric(output_root, device)
    del hf_model
    records = load_split_file(output_root, "development_splits.json")
    development = split_records(records, "development")
    target_basis_stability(model, tokenizer, suffix_ids, output_root, radii, development, device)
    noise = duplicate_noise_audit(
        model, tokenizer, suffix_ids, U, radii, development, device, "development", output_root
    )
    epsilon_y = max(1e-7, noise["max_abs"])
    noise["epsilon_y_dev"] = epsilon_y
    write_json_atomic(output_root / "noise_audit_dev.json", noise)
    payload, dev_sd = run_split(
        model, tokenizer, suffix_ids, U, radii, development, "development",
        output_root, device, epsilon_y,
    )
    write_json_atomic(output_root / "dev_cells.json", payload)
    decision = development_decision(payload)
    write_json_atomic(output_root / "dev_result.json", decision)
    if decision["n_surviving_cells"] < THRESHOLDS.development_cells_min:
        raise GreenStop("12_DEVELOPMENT_SURVIVAL", str(decision))
    if decision["n_conditioned_cells"] < THRESHOLDS.development_cells_min:
        raise GreenStop("13_DEVELOPMENT_CONDITIONING", str(decision))
    if decision["n_snr_cells"] < THRESHOLDS.development_snr_cells_min:
        raise GreenStop("14_DEVELOPMENT_SNR", str(decision))
    if decision["verdict"] != "OPEN_CONFIRMATION":
        raise GreenStop("16_DEVELOPMENT_GAIN", str(decision))
    frozen = freeze_confirmation(payload, output_root / "frozen_analysis.json")
    frozen["source_sha256"] = source_hashes()
    frozen["manifest_preconfirmation_sha256"] = sha256_file(output_root / "manifest.json")
    frozen["conditioning_dev_sd"] = dev_sd
    write_json_atomic(output_root / "frozen_analysis.json", frozen)
    manifest["confirmation_open"] = True
    manifest["frozen_analysis_sha256"] = sha256_file(output_root / "frozen_analysis.json")
    manifest["development_artifact_sha256"] = {
        name: sha256_file(output_root / name) for name in (
            "noise_audit_dev.json", "dev_tensor_scores.parquet", "dev_energy_targets.parquet",
            "dev_result.json", "frozen_analysis.json", "target_basis_stability.json",
        )
    }
    write_json_atomic(output_root / "manifest.json", manifest)


def confirmation_phase(output_root: Path, device: str) -> None:
    manifest = verify_freeze(output_root, require_confirmation=True)
    frozen = json.loads((output_root / "frozen_analysis.json").read_text(encoding="utf-8"))
    if frozen["source_sha256"] != source_hashes():
        raise GreenStop("17_MANIFEST_FREEZE", "frozen source hash mismatch")
    U, radii, tokenizer, hf_model, model, suffix_ids, _ = load_frozen_numeric(output_root, device)
    del hf_model
    records = load_split_file(output_root)
    confirmation = split_records(
        records, "confirmation", confirmation_lock=ConfirmationLock(output_root / "frozen_analysis.json")
    )
    noise = duplicate_noise_audit(
        model, tokenizer, suffix_ids, U, radii, confirmation, device, "confirmation", output_root
    )
    dev_noise = json.loads((output_root / "noise_audit_dev.json").read_text(encoding="utf-8"))["epsilon_y_dev"]
    permitted = max(2 * dev_noise, 2e-6)
    noise["permitted_max_abs"] = permitted
    write_json_atomic(output_root / "noise_audit_confirm.json", noise)
    if noise["max_abs"] > permitted:
        raise GreenStop("18_CONFIRMATION_NOISE", f"{noise['max_abs']} > {permitted}")
    payload, _ = run_split(
        model, tokenizer, suffix_ids, U, radii, confirmation, "confirmation",
        output_root, device, dev_noise, dev_sd=float(frozen["conditioning_dev_sd"]),
    )
    write_json_atomic(output_root / "confirm_cells.json", payload)
    result = confirmation_decision(payload, frozen)
    dev_cells = json.loads((output_root / "dev_cells.json").read_text(encoding="utf-8"))["cells"]
    total_survival = sum(row.get("survived", False) for row in dev_cells + payload["cells"])
    result["total_surviving_cells"] = total_survival
    if total_survival < THRESHOLDS.total_cells_technical_min:
        result["verdict"] = "FAIL_TOTAL_SURVIVAL"
        result["first_failed_gate"] = "19_TOTAL_SURVIVAL"
    elif result["verdict"] != "ORAL_RESULT_PASS":
        if result.get("n_cells", 0) < THRESHOLDS.confirmation_technical_min:
            result["first_failed_gate"] = "19_CONFIRMATION_SURVIVAL"
        elif result.get("n_conditioned", 0) < THRESHOLDS.confirmation_oral_min:
            result["first_failed_gate"] = "19_CONFIRMATION_CONDITIONING"
        else:
            result["first_failed_gate"] = "20_CONFIRMATORY_THRESHOLD"
    else:
        result["first_failed_gate"] = None
    result["schema_version"] = "green-bridge-terminal-v1"
    write_json_atomic(output_root / "result.json", result)
    finalize_hashes(output_root)


def finalize_hashes(output_root: Path) -> None:
    paths = sorted(path for path in output_root.iterdir() if path.is_file() and path.name != "sha256sums.txt")
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    (output_root / "sha256sums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "development", "confirmation", "all"), required=True)
    parser.add_argument("--device", default="cuda:0", help="hardware placement only")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="artifact location only")
    args = parser.parse_args()
    try:
        if args.phase in {"prepare", "all"}:
            prepare(args.output_root, args.device)
        if args.phase in {"development", "all"}:
            development_phase(args.output_root, args.device)
        if args.phase in {"confirmation", "all"}:
            confirmation_phase(args.output_root, args.device)
    except GreenStop as exc:
        terminal_stop(args.output_root, exc.gate, exc.detail)


if __name__ == "__main__":
    main()
