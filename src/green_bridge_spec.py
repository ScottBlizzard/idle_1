"""Frozen constants and hashing utilities for the GPT-2 green-bridge run.

This file is the executable counterpart of
``analysis/GPTPRO_GREEN_BRIDGE_20260805.md``.  Scientific constants must not be
overridden from the command line.  Hardware-only batch sizes may be lowered on
out-of-memory, but sites, gates, splits, radii, thresholds, and analysis rules
are immutable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "green_bridge"
SCHEMA_VERSION = "green-bridge-v1.2"
THEORY_BASE_COMMIT = "126556f"
GATE04_AMENDMENT_ID = "GPTPRO-GREEN-GATE04-v2-20260805"
GATE08_AMENDMENT_ID = "GPTPRO-GREEN-GATE08-v2-20260805"
HF_ATTN_IMPLEMENTATION = "eager"
GATE04_LEGACY_PAIR_SLICE = (0, 16)
GATE04_HOLDOUT_PAIR_SLICE = (16, 32)
LEGACY_SALT = "idle1-gt-bridge-20260805"
BASIS_V2_SALT = "idle1-gt-bridge-basis-v2-20260805"
SALT = LEGACY_SALT
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TRANSFORMER_LENS_COMMIT = "4a4dc26c750475b29e6f54b362c2aab988702c9c"

PROMPT = "<|endoftext|> The {noun} lasted from the year {cc:02d}{y:02d} to the year {cc:02d}"
EVALUATION_NOUNS = (
    "campaign", "dynasty", "reign", "siege", "treaty", "warfare",
    "expedition", "kingdom",
)
EVALUATION_CENTURIES = (12, 14, 16)
LEGACY_DONOR_NOUNS = (
    "invasion", "insurgency", "rivalry", "hostility", "raids", "sanctions",
    "domination", "confrontation", "pilgrimage", "journey", "voyage",
    "operation", "outbreak", "reforms", "relationship", "modernization",
)
LEGACY_DONOR_CENTURIES = (11, 13, 15, 17)
DONOR_NOUNS = LEGACY_DONOR_NOUNS
DONOR_CENTURIES = LEGACY_DONOR_CENTURIES
BASIS_V2_DONOR_NOUNS = (
    "rebellion", "revolution", "occupation", "blockade", "crusade",
    "migration", "settlement", "construction", "administration", "regime",
    "competition", "partnership", "transition", "expansion", "uprising",
    "conflict",
)
BASIS_V2_DONOR_CENTURIES = (11, 13, 15, 17)
BASIS_V2_DONOR_SELECTION_ORDER = (
    ("near", "basis_fit", 2, 2),
    ("far", "basis_fit", 2, 2),
    ("near", "basis_holdout", 1, 1),
    ("far", "basis_holdout", 1, 1),
    ("near", "radius_v2", 2, 2),
    ("far", "radius_v2", 2, 2),
)
BASIS_V2_FIT_PAIRS = 512
BASIS_V2_HOLDOUT_PAIRS = 256
BASIS_V2_RADIUS_PAIRS = 512
BASIS_V2_BOOTSTRAP_REPLICATES = 256
BASIS_V2_BOOTSTRAP_QUANTILE = 0.95
FIRST_ORDER_RESIDUAL_DIRECTIONS = 250
DISTANCE_BINS = {"near": (8, 16), "far": (40, 56)}
SUFFIX_MIN = 5
SUFFIX_MAX = 94
SELECTED_GATES = (2326, 1138, 2287, 606, 2848, 2305, 46, 2659, 946, 1616)
OUTPUT_SUFFIXES = tuple(range(100))


@dataclass(frozen=True)
class Dimensions:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_mlp: int = 3072
    residual_rank: int = 5
    selected_gates: int = 10
    output_dimension: int = 100


@dataclass(frozen=True)
class Thresholds:
    hook_untouched_max: float = 1e-7
    hf_tl_raw_year_max_abs: float = 3.0e-4
    hf_tl_raw_year_pooled_rms: float = 7.5e-5
    hf_tl_centered_year_max_abs: float = 2.5e-4
    hf_tl_centered_year_pooled_rms: float = 6.0e-5
    hf_tl_margin_max_abs: float = 2.0e-4
    hf_tl_margin_rms: float = 5.0e-5
    hf_tl_resid_mid_max_abs: float = 1.0e-4
    hf_tl_resid_mid_pooled_rms: float = 2.0e-5
    hf_tl_selected_pre_max_abs: float = 5.0e-4
    hf_tl_selected_pre_pooled_rms: float = 1.0e-4
    hf_tl_selected_post_max_abs: float = 5.0e-4
    hf_tl_selected_post_pooled_rms: float = 1.0e-4
    no_op_max_abs: float = 2e-5
    tail_max_abs: float = 2e-5
    tail_derivative_relative: float = 1e-4
    center_rms: float = 2e-6
    center_max_abs: float = 2e-5
    basis_fit_gap_min: float = 1.10
    basis_holdout_gap_min: float = 1.10
    basis_rank_floor: float = 1e-4
    basis_angle_max_degrees: float = 15.0
    basis_holdout_efficiency_min: float = 0.90
    basis_bootstrap_q95_max_degrees: float = 15.0
    curvature_rms_min: float = 5e-4
    curvature_snr_min: float = 20.0
    gate_response_rms_min: float = 5e-4
    gate_response_snr_min: float = 20.0
    factorization_residual_max: float = 0.15
    whitebox_a_relative_max: float = 0.05
    whitebox_a_small_absolute_max: float = 1e-4
    tensor_cosine_min: float = 0.95
    tensor_symmetric_change_max: float = 0.25
    richardson_change_max: float = 0.25
    tensor_snr_min: float = 20.0
    bypass_disagreement_max: float = 0.15
    active_gates_min: int = 3
    valid_items_per_cell_min: int = 6
    projected_chord_sigma_min: float = 0.10
    development_cells_min: int = 15
    confirmation_technical_min: int = 28
    confirmation_oral_min: int = 29
    total_cells_technical_min: int = 40
    cells_per_bin_min: int = 14
    conditioning_absolute: float = 0.10
    conditioning_dev_sd: float = 0.25
    development_snr_cells_min: int = 10
    development_snr_min: float = 3.0
    development_stop_below: float = 0.05
    confirmation_open_gain_min: float = 0.10
    confirmation_relative_gain_min: float = 0.20
    confirmation_relative_lcb_min: float = 0.10
    confirmation_absolute_gain_min: float = 0.01
    per_bin_relative_gain_min: float = 0.10
    per_bin_absolute_gain_min: float = 0.005
    cancellation_size_min: int = 8
    cancellation_class_min: int = 3
    cancellation_bin_min: int = 3
    cancellation_main_effect_min: float = 0.05
    cancellation_target_threshold: float = 0.10
    cancellation_auroc_min: float = 0.80
    cancellation_auroc_lcb_min: float = 0.70
    half_radius_spearman_min: float = 0.90
    half_radius_change_max: float = 0.20


DIMENSIONS = Dimensions()
THRESHOLDS = Thresholds()


FROZEN_SPEC: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "theory_base_commit": THEORY_BASE_COMMIT,
    "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
    "transformer_lens_commit": TRANSFORMER_LENS_COMMIT,
    "prompt": PROMPT,
    "evaluation_nouns": EVALUATION_NOUNS,
    "evaluation_centuries": EVALUATION_CENTURIES,
    "donor_nouns": DONOR_NOUNS,
    "donor_centuries": DONOR_CENTURIES,
    "basis_v2_donor_nouns": BASIS_V2_DONOR_NOUNS,
    "basis_v2_donor_centuries": BASIS_V2_DONOR_CENTURIES,
    "basis_v2_donor_selection_order": BASIS_V2_DONOR_SELECTION_ORDER,
    "distance_bins": DISTANCE_BINS,
    "suffix_range": (SUFFIX_MIN, SUFFIX_MAX),
    "selected_gates": SELECTED_GATES,
    "dimensions": asdict(DIMENSIONS),
    "thresholds": asdict(THRESHOLDS),
    "sites": {
        "patch": "blocks.8.hook_mlp_out",
        "x": "blocks.10.hook_resid_mid",
        "z": "blocks.10.mlp.hook_pre",
        "gate": "blocks.10.mlp.hook_post",
        "target_bypass_subtraction": "blocks.10.hook_resid_post",
    },
    "radii": {"full": 1.0, "half": 0.5, "multiplier": 0.20},
    "bootstrap": {"replicates": 100_000, "seed": 20260805},
    "gate04_amendment": {
        "id": GATE04_AMENDMENT_ID,
        "hf_attention_implementation": HF_ATTN_IMPLEMENTATION,
        "legacy_pair_slice": GATE04_LEGACY_PAIR_SLICE,
        "holdout_pair_slice": GATE04_HOLDOUT_PAIR_SLICE,
        "prompts_per_pair": ("clean", "corrupt"),
        "prompt_count": 32,
        "batch_size": 1,
        "parameter_mapping_exact": True,
        "hf_tl_error_enters_epsilon_y": False,
    },
    "numerical_error_contract": "frozen-richardson-propagation-v1",
    "gate08_amendment": {
        "id": GATE08_AMENDMENT_ID,
        "residual_rank": 5,
        "basis_object": "projector-covariant",
        "fit_pairs": BASIS_V2_FIT_PAIRS,
        "holdout_pairs": BASIS_V2_HOLDOUT_PAIRS,
        "radius_pairs": BASIS_V2_RADIUS_PAIRS,
        "bootstrap_replicates": BASIS_V2_BOOTSTRAP_REPLICATES,
        "bootstrap_quantile": BASIS_V2_BOOTSTRAP_QUANTILE,
        "first_order_residual_directions": FIRST_ORDER_RESIDUAL_DIRECTIONS,
        "attempt_index": 1,
        "retry_allowed": False,
        "rank6_fallback": False,
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_spec_hash() -> str:
    return sha256_text(canonical_json(FROZEN_SPEC))


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a same-directory temporary file then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def assert_scientific_override_free(arguments: dict[str, Any]) -> None:
    forbidden = {
        "model", "revision", "layer", "gates", "rank", "radius", "threshold",
        "corruption", "bootstrap_seed", "confirmation_retries",
    }
    present = forbidden.intersection(arguments)
    if present:
        raise ValueError(f"scientific command-line overrides are prohibited: {sorted(present)}")
